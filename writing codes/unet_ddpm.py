from basemodel import *
import pdb

class HParams():
    def __init__(self):
        self.bs = 64
        self.epo = 2000001
        self.lr = 0.0001
        self.min_lr = 0.00001
        self.lr_decay = 0.9995
        self.grad_clip = 1.

        # DDPM
        self.time_emb_size = 16
        self.char_emb_size = 256
        self.time_step = 1000 #200
        self.beta_0 = 1e-4  #0.002
        self.beta_T = 2*1e-2 #0.1


hp = HParams()

#### cross attention ####
class Cross_attention(nn.Module):
    def __init__(self, dim1, dim2, q_dim, k_dim, v_dim):
        super().__init__()
        self.q_func = nn.Linear(dim1, q_dim)
        self.k_func = nn.Linear(dim2, k_dim)
        self.v_func = nn.Linear(dim2, v_dim)

        self.out_func = nn.Linear(v_dim, dim1)

    def forward(self, x, f):
        x = x.permute(0, 2, 1) # b L dim1
        f = f.permute(0, 2, 1) # b L dim2
        q = self.q_func(x)
        k = self.k_func(f)
        v = self.v_func(f)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) / (256 ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        output = torch.bmm(attn_weights, v)
        output = self.out_func(output)
        output += x
        output = output.permute(0, 2, 1)

        return output

#### network #######
class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, dim=128):
        super().__init__()
        self.en_block1 = EncoderBlock(in_channels, dim)
        self.en_block2 = EncoderBlock(dim*2, dim*2)
        self.en_block3 = EncoderBlock(dim*4, dim*4)
        self.en_block4 = EncoderBlock(dim*8, dim*8)
        self.downblock1 = EncoderBlock(dim, dim*2, 2)
        self.downblock2 = EncoderBlock(dim*2, dim*4, 2)
        self.downblock3 = EncoderBlock(dim*4, dim*8, 2)

        self.de_block1 = EncoderBlock(dim*4, dim*4)
        self.de_block2 = EncoderBlock(dim*2, dim*2)
        self.de_block3 = EncoderBlock(dim, in_channels)
        self.de_block4 = EncoderBlock(in_channels, out_channels)
        self.upblock1 = DecoderBlock(dim*8, dim*4, 2)
        self.upblock2 = DecoderBlock(dim*4, dim*2, 2)
        self.upblock3 = DecoderBlock(dim*2, dim, 2)

        #self.cross_attention1 = Cross_attention(dim, 128, 256, 256, 256)
        self.cross_attention1 = Cross_attention(in_channels, 128, 256, 256, 256)
        self.cross_attention2 = Cross_attention(dim*2, 256, 256, 256, 256)
        self.cross_attention3 = Cross_attention(dim*4, 512, 256, 256, 256)
        self.cross_attention4 = Cross_attention(dim*8, 1024, 256, 256, 256)

        self.elu = nn.ELU()

    def forward(self, x, style_features=None):
        # encoder
        en1 = self.elu(self.en_block1(x))
        #en1 = self.cross_attention1(en1, style_features[0])
        en2 = self.elu(self.downblock1(en1))

        en3 = self.elu(self.en_block2(en2))
        #en3 = self.cross_attention2(en3, style_features[1])
        en4 = self.elu(self.downblock2(en3))

        en5 = self.elu(self.en_block3(en4))
        #en5 = self.cross_attention3(en5, style_features[2])
        en6 = self.elu(self.downblock3(en5))

        # decoder
        de1 = self.elu(self.en_block4(en6))
        de1 = self.cross_attention4(de1, style_features[3])
        de2 = self.elu(self.upblock1(de1))

        de3 = self.elu(self.de_block1(de2 + en5))
        de3 = self.cross_attention3(de3, style_features[2])
        de4 = self.elu(self.upblock2(de3 + en4))

        de5 = self.elu(self.de_block2(de4 + en3))
        de5 = self.cross_attention2(de5, style_features[1])
        de6 = self.elu(self.upblock3(de5 + en2))

        de7 = self.elu(self.de_block3(de6 + en1))
        de7 = self.cross_attention1(de7, style_features[0])
        output = self.de_block4(de7)

        return output
    

#### DDPM ####
class DDPM(nn.Module):
    def __init__(self, input_channel=hp.time_emb_size + hp.char_emb_size + 4, output_channel=3):
        super(DDPM, self).__init__()
        self.time_emb = SinusoidalPositionEmbeddings(hp.time_emb_size)
        self.denoiser = Unet(input_channel, output_channel)

        self.beta = cosine_schedule(hp.beta_T, hp.beta_0, hp.time_step).to(device)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, -1).to(device)

        dict_size = 4055
        self.char_embeddings = nn.Embedding(dict_size, hp.char_emb_size)

        ## loss
        self.loss = nn.MSELoss()

    def get_positions(self, durations):
        positions = []
        for duration in durations:
            concatenated_partitions = []
            for d in duration:
                partition = torch.linspace(0, 1, d)
                concatenated_partitions.append(partition)
            concatenated_partitions = torch.cat(concatenated_partitions).unsqueeze(0).unsqueeze(0) #1,1,length
            positions.append(concatenated_partitions)

        batch_positions = torch.cat(positions).to(device) # b, 1, length

        return batch_positions

    def get_contents_times(self, inputs, t, tags, durations):
        # tags: b, 500   return: b, dim, 500
        # inputs : b, 64, 500 + contents + time_emb ---- b 256 500
        # t : tensor

        #### content embeddings ####
        tags = torch.from_numpy(np.array(tags)).to(device)
        content_features = self.char_embeddings(tags).permute(0, 2, 1)
        bs, dim, L = content_features.size()

        #### time embeddings ####
        bs, _, length = inputs.shape
        inputs_t = self.time_emb(t).unsqueeze(2)  # 1, dim, 1
        if inputs_t.shape[0] != bs:
            inputs_t = inputs_t.repeat(bs, 1, length)
        else:
            inputs_t = inputs_t.repeat(1, 1, length)

        #### postion emb for every char ####
        batch_positions = self.get_positions(durations) # 
        #print(batch_positions)
        inputs = torch.cat([inputs, batch_positions, content_features, inputs_t], 1)

        return inputs

    def pred_noise(self, inputs, style_features=None):
        outputs = self.denoiser(inputs, style_features)
        return outputs

    def sample_x_t(self, x, t):
        noise = torch.randn_like(x).to(device)
        x_t = torch.zeros_like(x).to(device)
        #print(x.shape, t.shape)
        if t.shape[0] == 1:
            x_t = torch.sqrt(self.alpha_cumprod[t]) * x + torch.sqrt(1 - self.alpha_cumprod[t]) * noise
            return x_t, noise

        for i in range(x.shape[0]):
            x_t[i] = torch.sqrt(self.alpha_cumprod[t[i]]) * x[i] + torch.sqrt(1 - self.alpha_cumprod[t[i]]) * noise[i]
        return x_t, noise
    
    def remove_noise(self, x_t, t, predict_noise):
        temp = x_t - (self.beta[t] / torch.sqrt(1 - self.alpha_cumprod[t])) * predict_noise
        return temp / torch.sqrt(self.alpha[t])
    
    def estimate_x0(self, x_t, t, predict_noise):
        x0_pred = (x_t - torch.sqrt(1 - self.alpha_cumprod[t]) * predict_noise) / torch.sqrt(self.alpha_cumprod[t])
        return x0_pred
    
    def generate(self, tags, durations, style_features=None, var=0.9): 
        # tags: b, len
        bs = len(tags)
        length = len(tags[0])

        char_ids = []
        for i in range(bs):
            char_ids.append(tags[i][1])
        # print(char_ids)

        x = torch.randn(bs, 3, length).to(device)
        for i in range(hp.time_step):
            t = hp.time_step - 1 - i
            #if t%300 == 0: 
            #    print(t)
            t = torch.tensor([t]).to(device)

            inputs = self.get_contents_times(x, t=t, tags=tags, durations=durations)
            predict_noise = self.pred_noise(inputs, style_features)
            x = self.remove_noise(x, t, predict_noise)
            
            if t > 15:
               sigma = torch.sqrt(0.8 * self.beta[t] * (1 - self.alpha_cumprod[t-1]) / (1 - self.alpha_cumprod[t]))
               x += sigma * torch.randn_like(x)        

        return x
    
    def loss_function(self, x, tags, durations, style_features=None):
        loss = torch.Tensor([0.]).to(device)
        x = x.permute(0, 2, 1) # b channel L
        bs = x.shape[0]

        for i in range(12):
            t = torch.randint(0, hp.time_step, (bs,)).to(device)
            #t = torch.randint(0, 100, (1,)).to(device)
            if t.shape[0] == 1:
                print(f'time step: {t.item()}')
            x_t, noise = self.sample_x_t(x, t)
            x_t = self.get_contents_times(x_t, t, tags, durations)
            predict_noise = self.pred_noise(x_t, style_features)
            loss += self.loss(predict_noise, noise)
        
        loss /= 12
        return loss
    
    def dps(self, xt, classifier, char_ids, lr, t, predict_noise):
        mean = self.remove_noise(xt, t, predict_noise)
        xt.requires_grad_(True)

        x0_pred = self.estimate_x0(xt, t, predict_noise)
        p = classifier(x0_pred) 
        p = torch.softmax(p, dim=1)
        p = p[np.arange(len(char_ids)), char_ids].mean()
        loss = -p
        loss.backward()

        with torch.no_grad():
            mean -= lr * xt.grad

        return mean
    
    def optimize_x_t_1(self, x, t, tags, durations, classifier, char_ids, style_features, lr):
        sigma = torch.sqrt(0.8 * self.beta[t] * (1 - self.alpha_cumprod[t-1]) / (1 - self.alpha_cumprod[t]))
        z = torch.randn_like(x)
        x_ = x + sigma * z 
        x_.requires_grad_(True)

        inputs = self.get_contents_times(x_, t=t-1, tags=tags, durations=durations)
        predict_noise = self.pred_noise(inputs, style_features)
        x0_pred = self.estimate_x0(x_,t-1,predict_noise)

        p = classifier(x0_pred)
        p = torch.softmax(p, dim=1)
        p = p[np.arange(len(char_ids)), char_ids].mean()
        loss = -p
        loss.backward()

        b, n, l = x.shape
        radius = torch.sqrt(torch.tensor(l * (n))) * lr *sigma
        grad = x_.grad 
        grad_norm = torch.linalg.norm(grad, dim=[1, 2], keepdim=True)
        d_star = -radius * grad / (grad_norm)
        
        x = x_ + d_star

        return x
    
    def optimize_x_t_1_full(self, x, t, tags, durations, classifier, char_ids, style_features, lr, n=3):
        sigma = torch.sqrt(0.8 * self.beta[t] * (1 - self.alpha_cumprod[t-1]) / (1 - self.alpha_cumprod[t]))
        x = x.detach().clone()  
        x.requires_grad_(True)  
        
        x0_pred_list = []
        
        for _ in range(n):
            z = torch.randn_like(x)  
            x_noisy = x + sigma * z  
            
            inputs = self.get_contents_times(x_noisy, t=t-1, tags=tags, durations=durations)
            predict_noise = self.pred_noise(inputs, style_features)
            x0_pred = self.estimate_x0(x_noisy, t-1, predict_noise)
            x0_pred_list.append(x0_pred)
        
        mean_x0_pred = torch.stack(x0_pred_list).mean(dim=0)
        
        p = classifier(mean_x0_pred)
        p = torch.softmax(p, dim=1)
        p = p[torch.arange(len(char_ids)), char_ids].mean()
        loss = -p
        
        loss.backward()
        
        b, n_dim, l = x.shape
        radius = torch.sqrt(torch.tensor(l * n_dim)) * lr * sigma
        grad = x.grad 
        grad_norm = torch.linalg.norm(grad, dim=[1, 2], keepdim=True)
        d_star = -radius * grad / (grad_norm) 
        
        x = x + d_star.detach()
        
        return x
    
    def optimize_with_xt_2_grad(self, x, t, tags, durations, classifier, char_ids, style_features, lr):
        sigma_t = torch.sqrt(0.8 * self.beta[t] * (1 - self.alpha_cumprod[t-1]) / (1 - self.alpha_cumprod[t]))
        z = torch.randn_like(x)
        xt_1 = x + sigma_t * z

        inputs = self.get_contents_times(xt_1, t=t-1, tags=tags, durations=durations)
        predict_noise = self.pred_noise(inputs, style_features)
        x_mean = self.remove_noise(xt_1, t-1, predict_noise) #预测xt-2均值

        sigma_t_1 = torch.sqrt(0.8 * self.beta[t-1] * (1 - self.alpha_cumprod[t-2]) / (1 - self.alpha_cumprod[t-1]))
        z = torch.randn_like(x_mean)
        xt_2 = x_mean + sigma_t_1 * z
        xt_2.requires_grad_(True)
        xt_2.retain_grad()

        inputs = self.get_contents_times(xt_2, t=t-2, tags=tags, durations=durations)
        predict_noise = self.pred_noise(inputs, style_features)
        x0_pred = self.estimate_x0(xt_2, t-2, predict_noise)

        p = classifier(x0_pred)
        p = torch.softmax(p, dim=1)
        p = p[np.arange(len(char_ids)), char_ids].mean()
        loss = -p
        loss.backward()

        b, n, l = x.shape
        radius = torch.sqrt(torch.tensor(l * (n))) * lr * sigma_t
        grad = xt_2.grad
        grad_norm = torch.linalg.norm(grad, dim=[1, 2], keepdim=True)
        d_star = -radius * grad / (grad_norm)
        with torch.no_grad():
            x = xt_1 + d_star

        return x

    def dsg(self, xt, classifier, char_ids, lr, t, predict_noise):
        mean = self.remove_noise(xt, t, predict_noise)
        xt.requires_grad_(True)
        x0_pred = self.estimate_x0(xt, t, predict_noise)
        p = classifier(x0_pred) 
        p = torch.softmax(p, dim=1)
        p = p[np.arange(len(char_ids)), char_ids].mean()
        loss = -p
        loss.backward()

        grad = xt.grad
        grad_norm = torch.linalg.norm(grad, dim=[1, 2], keepdim=True)

        #eps = 1e-8
        b, n, l = xt.shape
        sigma = torch.sqrt(self.beta[t] * (1 - self.alpha_cumprod[t-1]) / (1 - self.alpha_cumprod[t]))
        radius = torch.sqrt(torch.tensor(l * (n))) * sigma

        guidance_rate = lr
        d_star = -radius * grad / (grad_norm)
        d_sample = torch.randn_like(mean) * sigma
        mix_direction = d_sample + guidance_rate * (d_star - d_sample)
        mix_direction_norm = torch.linalg.norm(mix_direction, dim=[1, 2], keepdim=True) 
        mix_step = mix_direction / (mix_direction_norm) * radius
        mean += mix_step

        return mean
    
    def dsg_new(self, xt, classifier, char_ids, lr, t, predict_noise):
        mean = self.remove_noise(xt, t, predict_noise) #预测xt-1均值
        xt.requires_grad_(True)
        x0_pred = self.estimate_x0(xt, t, predict_noise)
        p = classifier(x0_pred) 
        p = torch.softmax(p, dim=1)
        p = p[np.arange(len(char_ids)), char_ids].mean()
        loss = -p
        loss.backward()

        grad = xt.grad
        grad_norm = torch.linalg.norm(grad, dim=[1, 2], keepdim=True)

        b, n, l = xt.shape
        sigma = torch.sqrt(0.8 * self.beta[t] * (1 - self.alpha_cumprod[t-1]) / (1 - self.alpha_cumprod[t]))
        radius = torch.sqrt(torch.tensor(l * (n))) * sigma * lr

        d_star = -radius * grad / (grad_norm)
        d_sample = torch.randn_like(mean) * sigma
        mean = mean + d_star + d_sample

        return mean
    
    def projected_gradient_update(self, z, lr):
        raw_grad = z.grad.data  # 输入梯度形状: (b, 3, len)

        grad_mean = torch.mean(raw_grad, dim=(1,2), keepdim=True) 
        projected_grad = raw_grad - grad_mean 
        cov = torch.mean(z * projected_grad, dim=(1,2), keepdim=True) 
        projected_grad -= cov * z
        z = z - lr * projected_grad
        with torch.no_grad():
            z_mean = torch.mean(z, dim=(1,2), keepdim=True) 
            z = z - z_mean
            z_std = torch.std(z, dim=(1,2), keepdim=True) + 1e-10  
            z = z / z_std

        return z
  
    def optimize_add_noise_(self, x, t, tags, durations, classifier, char_ids, style_features, lr):
        sigma = torch.sqrt(0.8 * self.beta[t] * (1 - self.alpha_cumprod[t-1]) / (1 - self.alpha_cumprod[t]))
        z = torch.randn_like(x)
        z.requires_grad_(True)
        x_ = x + sigma * z 

        inputs = self.get_contents_times(x_, t=t-1, tags=tags, durations=durations)
        predict_noise = self.pred_noise(inputs, style_features)
        x0_pred = self.estimate_x0(x_,t-1,predict_noise)

        # noise update
        p = classifier(x0_pred)
        p = torch.softmax(p, dim=1)
        p = p[np.arange(len(char_ids)), char_ids].mean()
        loss = -p
        loss.backward()

        with torch.no_grad():
            z = self.projected_gradient_update(z, lr)

            '''z -= lr * z.grad
            z_mean = z.mean()
            z_std = z.std()
            z = (z - z_mean) / z_std'''
        
        x += sigma * z

        return x 
    
    def optimize_add_noise(self, x, t, tags, durations, classifier, char_ids, style_features, lr):
        sigma = torch.sqrt(0.8 * self.beta[t] * (1 - self.alpha_cumprod[t-1]) / (1 - self.alpha_cumprod[t]))
        z = torch.randn_like(x)
        z.requires_grad_(True)
        x_ = x + sigma * z 

        inputs = self.get_contents_times(x_, t=t-1, tags=tags, durations=durations)
        predict_noise = self.pred_noise(inputs, style_features)
        x0_pred = self.estimate_x0(x_,t-1,predict_noise)

        # noise update
        p = classifier(x0_pred)
        p = torch.softmax(p, dim=1)
        p = p[np.arange(len(char_ids)), char_ids].mean()
        loss = -p
        loss.backward()

        b, n, l = x.shape
        radius = torch.sqrt(torch.tensor(l * (n))) * lr
        grad = z.grad

        grad_norm = torch.linalg.norm(grad, dim=[1, 2], keepdim=True)
        d_star = -radius * grad / (grad_norm)
        with torch.no_grad():
            z += d_star
        x = x + sigma * z

        return x
    
    def optimize_add_noise_full(self, x, t, tags, durations, classifier, char_ids, style_features, lr, n=5):
        sigma = torch.sqrt(0.8 * self.beta[t] * (1 - self.alpha_cumprod[t-1]) / (1 - self.alpha_cumprod[t]))
        z_list = []
        
        for _ in range(n):
            z = torch.randn_like(x)
            z.requires_grad_(True)
            x_ = x + sigma * z  
            
            inputs = self.get_contents_times(x_, t=t-1, tags=tags, durations=durations)
            predict_noise = self.pred_noise(inputs, style_features)
            x0_pred = self.estimate_x0(x_, t-1, predict_noise)
            
            p = classifier(x0_pred)
            p = torch.softmax(p, dim=1)
            p = p[np.arange(len(char_ids)), char_ids].mean()
            loss = -p
            loss.backward(retain_graph=True)  
            
            b, n_dim, l = x.shape  
            radius = torch.sqrt(torch.tensor(l * n_dim)) * lr
            grad = z.grad
            grad_norm = torch.linalg.norm(grad, dim=[1, 2], keepdim=True)
            d_star = -radius * grad / (grad_norm)
            
            with torch.no_grad():
                z += d_star
            z_list.append(z)  
            
            z.grad = None  
        
        z_avg = torch.stack(z_list).mean(dim=0)
        x = x + sigma * z_avg
        
        return x

    def guided_generate(self, tags, durations, classifier=None, style_features=None): 
        # tags: b, len
        bs = len(tags)
        length = len(tags[0])

        char_ids = []
        for i in range(bs):
            char_ids.append(tags[i][1])

        x = torch.randn(bs, 3, length).to(device)
        for i in range(hp.time_step):
            t = hp.time_step - 1 - i
            t = torch.tensor([t]).to(device)

            inputs = self.get_contents_times(x, t=t, tags=tags, durations=durations)
            predict_noise = self.pred_noise(inputs, style_features)

            ## mean of xt-1
            # x = self.remove_noise(x, t, predict_noise)
            if 15 < t < 0:
                with torch.enable_grad():
                    ## DPS
                    #x = self.dps(x, classifier, char_ids, 2., t, predict_noise)

                    ## DPS + DSG
                    x = self.dsg(x, classifier, char_ids, 0.5, t, predict_noise)

                    ### DSG new
                    #x = self.dsg_new(x, classifier, char_ids, 0.5, t, predict_noise)
            else:
                x = self.remove_noise(x, t, predict_noise)
            
            if t > 15:
                if t<300:
                    with torch.enable_grad():
                        ### guidance###
                        #x = self.optimize_add_noise(x, t, tags, durations, classifier, char_ids, style_features, lr=0.1)
                        x = self.optimize_x_t_1(x, t, tags, durations, classifier, char_ids, style_features, lr=0.5)
                        #x = self.optimize_x_t_1_full(x, t, tags, durations, classifier, char_ids, style_features, lr=0.5, n=5)
                        #x = self.optimize_add_noise_full(x, t, tags, durations, classifier, char_ids, style_features, lr=0.5, n=5)
                else:
                    sigma = torch.sqrt(0.8 * self.beta[t] * (1 - self.alpha_cumprod[t-1]) / (1 - self.alpha_cumprod[t]))
                    x += sigma * torch.randn_like(x)

        return x