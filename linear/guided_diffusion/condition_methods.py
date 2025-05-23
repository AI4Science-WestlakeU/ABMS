from abc import ABC, abstractmethod
import torch
from torch.autograd.functional import jacobian
from util.distance_utils import mapping_base_on_hypersphere_constraint

__CONDITIONING_METHOD__ = {}


def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls

    return wrapper


def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)


class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
        # self.sample = sampler

    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)

    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement - Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError

        return norm_grad, norm

    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass


@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t


@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        print(f'mcg_scale:{self.scale}')

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale

        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm


@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        print(f'dps_scale:{self.scale}')

    def conditioning(self, x_prev, x_t, x_t_mean, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        r = torch.linalg.norm(x_t - x_t_mean, dim=[1, 2, 3])  # (bs)
        return x_t, norm

# Implement of Diffusion with Spherical Gaussian Constraint(DSG)
@register_conditioning_method(name='DSG')
class DSG(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.interval = kwargs.get('interval', 1.)
        self.guidance_scale = kwargs.get('guidance_scale', 1.)
        print(f'interval: {self.interval}')
        print(f'guidance_scale: {self.guidance_scale}')

    def conditioning(self, x_prev, x_t, x_t_mean, x_0_hat, measurement, idx, **kwargs):
        eps = 1e-8
        if idx % self.interval == 0:
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
            grad_norm = torch.linalg.norm(grad, dim=[1, 2, 3])

            b, c, h, w = x_t.shape
            r = torch.sqrt(torch.tensor(c * h * w)) * kwargs.get('sigma_t', 1.)[0, 0, 0, 0]
            guidance_rate = self.guidance_scale

            d_star = -r * grad / (grad_norm + eps)
            d_sample = x_t - x_t_mean
            mix_direction = d_sample + guidance_rate * (d_star - d_sample)
            mix_direction_norm = torch.linalg.norm(mix_direction, dim=[1, 2, 3])
            mix_step = mix_direction / (mix_direction_norm + eps) * r

            return x_t_mean + mix_step, norm

        else:
            # use the code below for print loss in unconditional step and commit it for saving time
            # difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            # norm = torch.linalg.norm(difference)
            # return x_t, norm

            return x_t, torch.zeros(1)

@register_conditioning_method(name='ours')
class OurConditioning(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.guidance_scale = kwargs.get('guidance_scale', 1e-4)
        self.interval = kwargs.get('interval', 1.)
        print(f'interval: {self.interval}')
        print(f'guidance_scale: {self.guidance_scale}')

    @staticmethod
    def projected_gradient_update(noise, noise_grad, lr):
        raw_grad = noise_grad.data
        grad_mean = torch.mean(raw_grad, dim=[1, 2, 3], keepdim=True)
        projected_grad = raw_grad - grad_mean
        cov = torch.mean(noise * projected_grad, dim=[1, 2, 3], keepdim=True)
        projected_grad -= cov * noise
        noise = noise - lr * projected_grad
        with torch.no_grad():
            noise_mean = noise.mean(dim=[1, 2, 3], keepdim=True)
            noise = noise - noise_mean
            noise_std = noise.std(dim=[1, 2, 3], keepdim=True)
            noise = noise / noise_std
        return noise
    
    def noise_grad(self, noise, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=noise)[0]
            return norm, norm_grad

        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement - Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=noise)[0]
            return norm, norm_grad
        
        else:
            raise NotImplementedError

    def conditioning(self, x_t_mean, x_t, idx, measurement, sigma_t, func, **kwargs):
        if idx % self.interval == 0 and idx>0:
            noise = torch.randn_like(x_t_mean)
            noise.requires_grad_(True)
            x_ = x_t_mean + sigma_t * noise  
            # func is lambda x, t: self.p_mean_variance(model, x, t)["pred_xstart"] from gaussian_diffusion.py
            t = idx-1 if idx>1 else 0
            t = torch.tensor([t] * x_.shape[0], device=x_.device)
            x_0_hat = func(x_, t) 
            
            kwargs.pop('x_0_hat', None)
            norm, noise_grad = self.noise_grad(noise=noise, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
            
            b, c, h, w = x_t.shape
            r = torch.sqrt(torch.tensor(c * h * w))*self.guidance_scale
            noise_grad_norm = torch.linalg.norm(noise_grad, dim=[1, 2, 3])
            d_star = -r * noise_grad / (noise_grad_norm + 1e-20)
            with torch.no_grad():
                noise += d_star
                ## noise norm ##
                # noise_mean = noise.mean(dim=[1, 2, 3], keepdim=True)
                # noise = noise - noise_mean
                # noise_std = noise.std(dim=[1, 2, 3], keepdim=True)
                # noise = noise / noise_std
            x_t_ = x_t_mean + sigma_t * noise
            
            return x_t_, norm
        else:
            return x_t, torch.zeros(1)
        
        
@register_conditioning_method(name='ours_full')
class OurConditioningFull(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.guidance_scale = kwargs.get('guidance_scale', 1e-4)
        self.interval = kwargs.get('interval', 1.)
        self.n_guidance_directions = kwargs.get('n_guidance_directions', 10)
        print(f'interval: {self.interval}')
        print(f'guidance_scale: {self.guidance_scale}')
        print(f'n_guidance_directions: {self.n_guidance_directions}')

    @staticmethod
    def projected_gradient_update(noise, noise_grad, lr):
        raw_grad = noise_grad.data
        grad_mean = torch.mean(raw_grad, dim=[1, 2, 3], keepdim=True)
        projected_grad = raw_grad - grad_mean
        cov = torch.mean(noise * projected_grad, dim=[1, 2, 3], keepdim=True)
        projected_grad -= cov * noise
        noise = noise - lr * projected_grad
        with torch.no_grad():
            noise_mean = noise.mean(dim=[1, 2, 3], keepdim=True)
            noise = noise - noise_mean
            noise_std = noise.std(dim=[1, 2, 3], keepdim=True)
            noise = noise / noise_std
        return noise
    
    def noise_grad(self, noise, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=noise)[0]
            return norm, norm_grad

        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement - Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=noise)[0]
            return norm, norm_grad
        
        else:
            raise NotImplementedError
    

    def xt_grad(self, xt, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]
            return norm, norm_grad

        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement - Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]
            return norm, norm_grad
        
        else:
            raise NotImplementedError
    
    def conditioning(self, x_t_mean, x_t, idx, measurement, sigma_t, func, **kwargs):
        if idx % self.interval == 0 and idx>1:
            d_star_list = []
            for i in range(self.n_guidance_directions):
                noise = torch.randn_like(x_t_mean)
                noise.requires_grad_(True)
                x_ = x_t_mean + sigma_t * noise  
                # func is lambda x, t: self.p_mean_variance(model, x, t)["pred_xstart"] from gaussian_diffusion.py
                t = idx-1 if idx>1 else 0
                t = torch.tensor([t] * x_.shape[0], device=x_.device)
                x_0_hat = func(x_, t) 
                
                kwargs.pop('x_0_hat', None)
                norm, noise_grad = self.noise_grad(noise=noise, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
                
                b, c, h, w = x_t.shape
                r = torch.sqrt(torch.tensor(c * h * w))*self.guidance_scale
                noise_grad_norm = torch.linalg.norm(noise_grad, dim=[1, 2, 3])
                d_star = -r * noise_grad / (noise_grad_norm + 1e-20)
                d_star_list.append(d_star)
            
            d_star_list = torch.cat(d_star_list, dim=0)
            d_star_mean = torch.mean(d_star_list, dim=0)
            
            noise = torch.randn_like(x_t_mean)
            x_t_ = x_t_mean + sigma_t * (noise + d_star_mean)
                
            return x_t_, norm
            
        else:
            return x_t, torch.zeros(1)
    
    def conditioning_(self, x_t_mean, x_t, idx, measurement, sigma_t, func, **kwargs):
        x_t_pre = kwargs.get('x_prev', None)
        if idx % self.interval == 0 and idx>0:
            d_star_list = []
            for i in range(self.n_guidance_directions):
                noise = torch.randn_like(x_t_mean)
                noise.requires_grad_(True)
                x_ = x_t_mean + sigma_t * noise  
                # func is lambda x, t: self.p_mean_variance(model, x, t)["pred_xstart"] from gaussian_diffusion.py
                t = idx-1 if idx>1 else 0
                t = torch.tensor([t] * x_.shape[0], device=x_.device)
                x_0_hat = func(x_, t) 
                
                kwargs.pop('x_0_hat', None)
                norm, xt_grad = self.xt_grad(xt=x_t_pre, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
                
                b, c, h, w = x_t.shape
                r = torch.sqrt(torch.tensor(c * h * w))*self.guidance_scale
                xt_grad_norm = torch.linalg.norm(xt_grad, dim=[1, 2, 3])
                d_star = -r * xt_grad / (xt_grad_norm + 1e-20)
                d_star_list.append(d_star)
            
            d_star_list = torch.cat(d_star_list, dim=0)
            d_star_mean = torch.mean(d_star_list, dim=0)
            
            noise = torch.randn_like(x_t_mean)
            x_t_ = x_t_mean + sigma_t * (noise + d_star_mean)
                
            return x_t_, norm
            
        else:
            return x_t, torch.zeros(1)
        
    