import torch
from tqdm import tqdm
import torchvision.utils as tvu
import torchvision
import os

from .clip.base_clip import CLIPEncoder
from .face_parsing.model import FaceParseTool
from .anime2sketch.model import FaceSketchTool
from .landmark.model import FaceLandMarkTool
from .arcface.model import IDLoss


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def arcface_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=None, stop=100, guidance_rate=0.05, ref_path=None):
    idloss = IDLoss(ref_path=ref_path).cuda()
    
    if rho_scale is None:
        assert guidance_rate is not None
        rho_scale = guidance_rate

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')

        xt.requires_grad = True

        if cls_fn == None:
            et = model(xt, t)
        else:
            print("use class_num")
            class_num = 281
            classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda")) * class_num
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]

        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

        residual = idloss.get_residual(x0_t)
        norm = torch.linalg.norm(residual)
        # print(f'norm:{norm}')
        norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

        eta = 0.5
        c1 = (1 - at_next).sqrt() * eta
        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et

        # use guided gradient
        rho = at.sqrt() * rho_scale
        if not i <= stop:
            xt_next -= rho * norm_grad
            # print(f'norm:{norm}')

        x0_t = x0_t.detach()
        xt_next = xt_next.detach()

        x0_preds.append(x0_t.to('cpu'))
        xs.append(xt_next.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]
    # return xs, x0_preds


def arcface_ddim_diffusion_dsg(x, seq, model, b, cls_fn=None, rho_scale=1.0, stop=100, guidance_rate=0.05, ref_path=None):
    idloss = IDLoss(ref_path=ref_path).cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        # print(f'i:{i}')
        
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')

        xt.requires_grad = True

        if cls_fn == None:
            et = model(xt, t)
        else:
            print("use class_num")
            class_num = 281
            classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda")) * class_num
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]

        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

        residual = idloss.get_residual(x0_t)
        norm = torch.linalg.norm(residual)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]
        # print(f'norm:{norm}')

        eta = 1
        # c1 = (1 - at_next).sqrt() * eta
        # c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
        # xt_next_mean = at_next.sqrt() * x0_t + c2 * et
        # xt_next = xt_next_mean + c1 * torch.randn_like(x0_t)
        # xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et

        c1 = (
                eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        )
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        xt_next_mean = at_next.sqrt() * x0_t + c2 * et
        xt_next = xt_next_mean + c1 * torch.randn_like(x)

        # use guided gradient
        rho = at.sqrt() * rho_scale
        # if not i <= stop:

        ## DSG
        if not i <= stop and i % 10 == 0:
            eps = 1e-20
            grad_norm = torch.linalg.norm(norm_grad, dim=[1, 2, 3])
            grad2 = norm_grad / (grad_norm + eps)

            # r = torch.linalg.norm(xt_next - xt_next_mean, dim=[1, 2, 3])  # (bs)
            batch, ch, h, w = xt_next.shape
            import math
            r = math.sqrt(ch*h*w) * c1
            # print(f'r:{r} r2:{r2}')
            d_star = -r * grad2
            d_sample = xt_next - xt_next_mean
            mix_direction = d_sample + guidance_rate * (d_star - d_sample)
            mix_direction_norm = torch.linalg.norm(mix_direction, dim=[1, 2, 3])
            xt_next = xt_next_mean + mix_direction / (mix_direction_norm + eps) * r

        x0_t = x0_t.detach()
        xt_next = xt_next.detach()

        x0_preds.append(x0_t.to('cpu'))
        xs.append(xt_next.to('cpu'))

    x = [((y + 1.0) / 2.0).clamp(0.0, 1.0) for y in x]

    # for i in [-1]:  # range(len(x)):
    # for i in range(len(xs)):
    #     for j in range(x[i].size(0)):
    #         tvu.save_image(
    #             x[i][j], os.path.join(self.args.image_folder, f"{index + j}_{i}.png")
    #         )

    return [xs[-1]], [x0_preds[-1]]
    # return xs, x0_preds

# NOTE: add ours method
def arcface_ddim_diffusion_ours(x, seq, model, b, cls_fn=None, rho_scale=1.0, stop=100, guidance_rate=0.05, ref_path=None):
    idloss = IDLoss(ref_path=ref_path).cuda()
    
    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        # print(f'i:{i}')
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        xt.requires_grad = True
    
        if cls_fn == None:
            et = model(xt, t)
        else:
            print("use class_num")
            class_num = 281
            classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda")) * class_num
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]
            
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

        eta = 1
        c1 = (
                eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        )
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        xt_next_mean = at_next.sqrt() * x0_t + c2 * et
        xt_next = xt_next_mean + c1 * torch.randn_like(x) 

        x0_t = x0_t.detach()
        xt_next = xt_next.detach()
        
        xt_next_mean = xt_next_mean.detach()
    
        if not i <= stop and i % 10 == 0:
            eta = 1
            sigma_t = c1
            noise = torch.randn_like(x)
            noise.requires_grad_(True)
            tid_next = i if j == -1 else j
            t_ = (torch.ones(n) * tid_next).to(x.device)
            xt_next_ = xt_next_mean + sigma_t * noise

            if cls_fn == None:
                et_next_ = model(xt_next_, t_)
            else:
                print("use class_num")
                class_num = 281
                classes = torch.ones(xt_next_.size(0), dtype=torch.long, device=torch.device("cuda")) * class_num
                et_next_ = model(xt_next_, t_, classes)
                et_next_ = et_next_[:, :3]
                et_next_ = et_next_ - (1 - at_next).sqrt()[0, 0, 0, 0] * cls_fn(x, t_, classes)

            if et_next_.size(1) == 6:
                et_next_ = et_next_[:, :3]
                
            x_0_hat = (xt_next_ - et_next_ * (1 - at_next).sqrt()) / at_next.sqrt()

            residual = idloss.get_residual(x_0_hat)
            norm = torch.linalg.norm(residual)
            noise_grad = torch.autograd.grad(outputs=norm, inputs=noise)[0]
            
            bz, c, h, w = xt_next.shape
            r = torch.sqrt(torch.tensor(c * h * w)) * guidance_rate
            noise_grad_norm = torch.linalg.norm(noise_grad, dim=[1, 2, 3])
            d_star = -r * noise_grad / (noise_grad_norm + 1e-10)
            with torch.no_grad():
                noise += d_star
            xt_next = xt_next_mean + sigma_t * noise
            xt_next = xt_next.detach()

        x0_preds.append(x0_t.to('cpu'))
        xs.append(xt_next.to('cpu'))
    
    x = [((y + 1.0) / 2.0).clamp(0.0, 1.0) for y in x] 

    # for i in [-1]:  # range(len(x)):
    # for i in range(len(xs)):
    #     for j in range(x[i].size(0)):
    #         tvu.save_image(
    #             x[i][j], os.path.join(self.args.image_folder, f"{index + j}_{i}.png")
    #         )

    return [xs[-1]], [x0_preds[-1]]
    # return xs, x0_preds
