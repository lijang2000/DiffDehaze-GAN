import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import shutil

import yaml
from tqdm import tqdm
import dataloader.DehazeDataloader
import utils.util
from Model.DM.GAN_Generator import NCSNpp
from dataloader.DehazeDataloader import *
from Model.DM.Discriminator import *
from EMA import EMA
from Model.Dis.Dis_normal.Dis6channel import *
from utils.psnr_ssim import calculate_psnr_ssim
from utils.util import ReplayBuffer
from utils.vggLoss import PerceptualLoss
from Model.DCP.DCP_G import *
from Model.Refine.UNet_Refine import *
from utils.contraLoss import *

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))

def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    return t.to(device)

def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var  
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas ** 0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas

class Diffusion_Coefficients():
    def __init__(self, args, device):
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1
        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)

class Posterior_Coefficients():
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)
        
        self.betas = self.betas.type(torch.float32)[1:]  
        self.alphas = 1 - self.betas  
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)  
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32, device=device), self.alphas_cumprod[:-1]), 0)  
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)  
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
                (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

def q_sample(coeff_clear, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    x_t = extract(coeff_clear.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff_clear.sigmas_cum, t, x_start.shape) * noise

    return x_t

def q_sample_pairs(coeff_clear, x_start, t):
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff_clear, x_start, t)
    x_t_plus_one = extract(coeff_clear.a_s, t + 1, x_start.shape) * x_t + \
                   extract(coeff_clear.sigmas, t + 1, x_start.shape) * noise

    return x_t, x_t_plus_one

def sample_posterior(coefficients, x_0, x_t, t):
    def q_posterior(x_0, x_t, t):
        mean = (
                extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (1 - (t == 0).type(torch.float32))
        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0, x_t, t)
    return sample_x_pos

def sample_from_model(coefficients, netG_clear, n_time, x_init, T, opt, x_haze, dcp_T):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            t_time = t

            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = netG_clear(x, t_time, latent_z, x_haze, dcp_T)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()

    return x

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def create_img(coeff, pos_coeff, dcp_J, dcp_T, netG, nz, batch_size):
    t = torch.randint(0, args.num_timesteps, (dcp_J.size(0),), device=device)
    x_t, x_tp1 = q_sample_pairs(coeff, dcp_J, t)
    latent_z = torch.randn(batch_size, nz, device=device)
    x_0_predict = netG(x_tp1.detach(), t, latent_z, dcp_J, dcp_T)
    x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
    return x_0_predict, x_pos_sample, x_t, x_tp1

def init_net(args, device):
    netG = NCSNpp(args).to(device)
    netD = Discriminator_large(nc=2 * args.num_channels, ngf=args.ngf,
                               t_emb_dim=args.t_emb_dim, act=nn.LeakyReLU(0.2)).to(device)
    netD_exam = Discriminator().to(device)
    return netG, netD, netD_exam

def init_optim(netG, netD, netD_exam, args):
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizer_exam = optim.Adam(netD_exam.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    return optimizerG, optimizerD, optimizer_exam


def init_sche(optG, optD, opt_exam, args):
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optG, args.num_epoch, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optD, args.num_epoch, eta_min=1e-5)
    scheduler_exam = torch.optim.lr_scheduler.CosineAnnealingLR(opt_exam, args.num_epoch, eta_min=1e-5)
    return schedulerG, schedulerD, scheduler_exam





def save_content(epoch, global_step, args, netG, netD, optimizerG, optimizerD, schedulerG, schedulerD,
                 net_exam, optimizer_exam, scheduler_exam, netRefine, optimizer_Refine, scheduler_Refine, exp_path=''):
    print('saving...')
    
    content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,

               'netG_dict': netG.state_dict(),
               'optimizerG': optimizerG.state_dict(),
               'schedulerG': schedulerG.state_dict(),

               'netD_dict': netD.state_dict(),
               'optimizerD': optimizerD.state_dict(),
               'schedulerD': schedulerD.state_dict(),

               'net_exam': net_exam.state_dict(),
               'optimizer_exam': optimizer_exam.state_dict(),
               'scheduler_exam': scheduler_exam.state_dict(),

               'net_refine': netRefine.state_dict(),
               'optimizer_refine': optimizer_Refine.state_dict(),
               'scheduler_refine': scheduler_Refine.state_dict()}

    torch.save(content, os.path.join(exp_path, 'results', 'contents', 'content.pth'))  



def train(rank, gpu, args, config, Len):
    loss_l1 = nn.L1Loss()
    loss_gan = nn.MSELoss()
    loss_vgg = PerceptualLoss()
    loss_TV = utils.util.TVLoss()
    loss_cont = ContrastLoss()

    label_real_real = torch.tensor([0], dtype=torch.long, requires_grad=False).to('cuda', dtype=torch.float)
    label_fake_fake = torch.tensor([1], dtype=torch.long, requires_grad=False).to('cuda', dtype=torch.float)
    label_real_fake = torch.tensor([2], dtype=torch.long, requires_grad=False).to('cuda', dtype=torch.float)
    label_fake_real = torch.tensor([3], dtype=torch.long, requires_grad=False).to('cuda', dtype=torch.float)
    fake_clear_buffer = ReplayBuffer()

    torch.manual_seed(args.seed + rank)  
    torch.cuda.manual_seed(args.seed + rank)  
    torch.cuda.manual_seed_all(args.seed + rank)  
    device = gpu
    batch_size = config.training.batch_size
    nz = args.nz  

    
    DATASET = dataloader.DehazeDataloader.Dehaze(new_config)
    train_loader, val_loader = DATASET.get_loaders()


    print("load network...")
    netG_clear, netD_clear, netD_exam_clear = init_net(args, device)
    
    net_DCP = DCPDehazeGenerator().to(device)
    
    net_RefineT = UnetTransGenerator(input_nc=4,
                                     output_nc=1, num_downs=8, ngf=6, norm_layer=nn.BatchNorm2d,
                                     use_dropout=False, r=10, eps=1e-3).to(device)

    
    print("load optimizer...")
    optimizerG_clear, optimizerD_clear, optimizer_exam_clear = init_optim(netG_clear, netD_clear, netD_exam_clear, args)
    optim_RefineT = torch.optim.Adam(net_RefineT.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))

    
    if args.use_ema:
        optimizerG_clear = EMA(optimizerG_clear, ema_decay=args.ema_decay)

    
    print("load scheduler...")
    schedulerG_clear, schedulerD_clear, scheduler_exam_clear = init_sche(optimizerG_clear, optimizerD_clear,
                                                                         optimizer_exam_clear, args)
    scheduler_Refine = torch.optim.lr_scheduler.CosineAnnealingLR(optim_RefineT, args.num_epoch, eta_min=1e-5)

    exp_path = ""
    resume_path = args.resume_path

    coeff_clear = Diffusion_Coefficients(args, device)
    pos_coeff_clear = Posterior_Coefficients(args, device)
    T_clear = get_time_schedule(args, device)

    
    if args.resume:
        
        checkpoint_file = os.path.join(resume_path, 'content_clear.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        
        netG_clear.load_state_dict(checkpoint['netG_dict'])
        optimizerG_clear.load_state_dict(checkpoint['optimizerG'])
        schedulerG_clear.load_state_dict(checkpoint['schedulerG'])
        
        netD_clear.load_state_dict(checkpoint['netD_dict'])
        optimizerD_clear.load_state_dict(checkpoint['optimizerD'])
        schedulerD_clear.load_state_dict(checkpoint['schedulerD'])
        
        net_RefineT.load_state_dict(checkpoint['net_refine'])
        optim_RefineT.load_state_dict(checkpoint['optimizer_refine'])
        scheduler_Refine.load_state_dict(checkpoint['scheduler_refine'])
        global_step = checkpoint['global_step']

        print("=> load (epoch {})".format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    PSNR_MAX = 0
    SSIM_MAX = 0

    for epoch in range(init_epoch, args.num_epoch + 1):
        with tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epoch}", unit="batch") as pbar:
            for iteration, (x, y) in enumerate(pbar):

                for p in netD_clear.parameters():
                    p.requires_grad = True

                
                x_haze = x[:, :3, :, :].to(device)
                x_clear = x[:, 3:, :, :].to(device)

                
                t = torch.randint(0, args.num_timesteps, (x_haze.size(0),), device=device)  
                x_t, x_tp1 = q_sample_pairs(coeff_clear, x_haze, t)  
                latent_z = torch.randn(batch_size, nz, device=device)  

                
                net_RefineT.zero_grad()
                dcp_J, dcp_T, dcp_A = net_DCP(x_haze)  
                refine_T, out_T = net_RefineT(torch.cat((x_haze, dcp_T), 1))  
                shape = refine_T.shape
                map_A = (dcp_A).reshape((batch_size, 3, 1, 1)).repeat(1, 1, shape[2], shape[3])
                refine_T_map = refine_T.repeat(1, 3, 1, 1)
                New_J = net_DCP.reconstruct_image(x_haze, refine_T_map, map_A)

                refine_I = utils.util.synthesize_fog(New_J, refine_T_map, map_A)  
                loss_rec = loss_l1(refine_I, x_haze)  
                loss_TV_T = loss_TV(out_T)
                contrastive_loss_hIc = loss_cont(p=x_haze, a=refine_I, n=x_clear)*0.0001  
                total_rec = (loss_rec + contrastive_loss_hIc + loss_TV_T) / 3
                total_rec.backward(retain_graph=True)
                optim_RefineT.step()

                
                netD_clear.zero_grad()
                x_t.requires_grad = True  
                D_real = netD_clear(x_t, t, x_tp1.detach()).view(-1)  
                errD_real = F.softplus(-D_real)
                errD_real = errD_real.mean()
                errD_real.backward(retain_graph=True)
                if args.lazy_reg is None:
                    
                    grad_real = torch.autograd.grad(
                        outputs=D_real.sum(), inputs=x_t, create_graph=True, allow_unused=True)[0]
                    grad_penalty = (
                            grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad_penalty = args.r1_gamma / 2 * grad_penalty
                    grad_penalty.backward()
                else:
                    if global_step % args.lazy_reg == 0:
                        grad_real = torch.autograd.grad(
                            outputs=D_real.sum(), inputs=x_t, create_graph=True
                        )[0]
                        grad_penalty = (
                                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                        ).mean()
                        grad_penalty = args.r1_gamma / 2 * grad_penalty
                        grad_penalty.backward()

                
                x_0_predict = netG_clear(x_tp1.detach(), t, latent_z, New_J, refine_T_map)  
                x_pos_sample = sample_posterior(pos_coeff_clear, x_0_predict, x_tp1, t)  
                output = netD_clear(x_pos_sample, t, x_tp1.detach()).view(-1)
                errD_fake = F.softplus(output)
                errD_fake = errD_fake.mean()
                errD_fake.backward(retain_graph=True)
                optimizerD_clear.step()
                errD = errD_real + errD_fake
                
                for p in netD_clear.parameters():
                    p.requires_grad = False

                
                
                x_0_predict, x_pos_sample, x_t, x_tp1 = create_img(
                    coeff_clear, pos_coeff_clear, New_J, refine_T_map, netG_clear, nz, batch_size)

                
                optimizer_exam_clear.zero_grad()
                
                fake_clear = fake_clear_buffer.push_and_pop(x_0_predict)

                
                real_real = torch.cat((x_clear, x_clear), 1)
                fake_fake = torch.cat((fake_clear, fake_clear), 1)
                real_fake = torch.cat((x_clear, fake_clear), 1)
                fake_real = torch.cat((fake_clear, x_clear), 1)

                pred_RR = netD_exam_clear(real_real)
                loss_rr = loss_gan(pred_RR, label_real_real)
                pred_FF = netD_exam_clear(fake_fake)
                loss_ff = loss_gan(pred_FF, label_fake_fake)
                pred_RF = netD_exam_clear(real_fake)
                loss_rf = loss_gan(pred_RF, label_real_fake)
                pred_FR = netD_exam_clear(fake_real)
                loss_fr = loss_gan(pred_FR, label_fake_real)

                loss_D = (loss_rr + loss_ff + loss_rf + loss_fr) / 4
                loss_D.backward(retain_graph=True)
                optimizer_exam_clear.step()

                
                optimizerG_clear.zero_grad()
                
                
                
                output1 = netD_clear(x_pos_sample, t, x_tp1.detach()).view(-1)
                errG = F.softplus(-output1)
                errG = errG.mean()

                
                
                
                dcp_J, dcp_T, dcp_A = net_DCP(x_haze)  
                refine_T, out_T = net_RefineT(torch.cat((x_haze, dcp_T), 1))  
                map_A = (dcp_A).reshape((batch_size, 3, 1, 1)).repeat(1, 1, shape[2], shape[3])
                refine_T_map = refine_T.repeat(1, 3, 1, 1)
                New_J = net_DCP.reconstruct_image(x_haze, refine_T_map, map_A)
                refine_I = utils.util.synthesize_fog(x_0_predict, refine_T_map,
                                                     map_A)  

                cyc_loss = loss_l1(x_haze, refine_I)

                vgg_loss = loss_vgg(x_0_predict, New_J)/(1+epoch*0.005)

                RA_FA = torch.cat((x_clear, x_0_predict), 1)
                predRF = netD_exam_clear(RA_FA)
                pred_lossRF = loss_gan(predRF, label_real_fake)

                FA_RA = torch.cat((x_0_predict, x_clear), 1)
                predFR = netD_exam_clear(FA_RA)
                pred_lossFR = loss_gan(predFR, label_fake_real)

                FA_FA = torch.cat((x_0_predict, x_0_predict), 1)
                predFF = netD_exam_clear(FA_FA)
                pred_lossFF = loss_gan(predFF, label_fake_fake)
                pred_loss = -(pred_lossRF + pred_lossFF + pred_lossFR) / 3

                contrastive_loss_cph = loss_cont(p=x_clear, a=x_0_predict, n=x_haze)
                contrastive_loss_hIc = loss_cont(p=x_haze, a=refine_I, n=x_clear)
                contrastive_loss = (contrastive_loss_cph + contrastive_loss_hIc) / 2 * 0.0001

                total = (errG + cyc_loss + pred_loss + contrastive_loss + vgg_loss) / 5
                total.backward()
                optimizerG_clear.step()

                global_step += 1

                pbar.set_description(
                    f"Epoch {epoch}/{args.num_epoch} | loss1: {errD.item():.4f} | loss2: {loss_D.item():.4f} | loss3 {total.item():.4f}: ①{errG.item():.4f}, ②{cyc_loss.item():.4f}, ③{pred_loss.item():.4f}, ④{contrastive_loss.item():.4f}")

        if not args.no_lr_decay:
            schedulerG_clear.step()
            schedulerD_clear.step()
            scheduler_exam_clear.step()
            scheduler_Refine.step()
            
            if args.save_content:
                if epoch % args.save_content_every == 0 or epoch == 0:
                    save_content(epoch, global_step, args,
                                 netG_clear, netD_clear,
                                 optimizerG_clear, optimizerD_clear,
                                 schedulerG_clear, schedulerD_clear,
                                 netD_exam_clear, optimizer_exam_clear, scheduler_exam_clear,
                                 net_RefineT, optim_RefineT, scheduler_Refine, '')

            
            
            
            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizerG_clear.swap_parameters_with_ema(store_params_in_ema=True)  
                    torch.save(netG_clear.state_dict(),
                               os.path.join(exp_path, 'results', 'ckpts',
                                            'netG_clear_{}.pth'.format(epoch)))  
                    optimizerG_clear.swap_parameters_with_ema(store_params_in_ema=True)  
                else:
                    torch.save(netG_clear.state_dict(),
                               os.path.join(exp_path, 'results', 'ckpts',
                                            'netG_clear_{}.pth'.format(epoch)))  

                torch.save(net_RefineT.state_dict(),
                           os.path.join(exp_path, 'results', 'ckpts', 'net_RefineT_{}.pth'.format(epoch)))

            if epoch % args.val_step == 0:
                fake_sample_list = []
                fake_DCP_list = []
                T = []
                Ref_T = []
                psnr = 0
                ssim = 0
                L = Len
                for i, (x, y) in enumerate(tqdm(val_loader, desc="Validation Progress")):
                    x_h = x[:, :3, :, :].to(device)
                    x_c = x[:, 3:, :, :].to(device)
                    x_t = torch.randn_like(x_h)
                    dcp_J, dcp_T, dcp_A = net_DCP(x_h)
                    refine_T, out_T = net_RefineT(torch.cat((x_h, dcp_T), 1))
                    refine_T_map = refine_T.repeat(1, 3, 1, 1)
                    map_A = (dcp_A).reshape((batch_size, 3, 1, 1)).repeat(1, 1, shape[2], shape[3])
                    New_J = net_DCP.reconstruct_image(x_h, refine_T_map, map_A)
                    fake_sample = sample_from_model(pos_coeff_clear, netG_clear,
                                                    args.num_timesteps, x_t, T_clear, args, New_J, refine_T_map)
                    fake_sample_list.append(fake_sample)
                    fake_DCP_list.append(New_J)
                    T.append(dcp_T)
                    Ref_T.append(refine_T_map)
                    pt, st = calculate_psnr_ssim(x_c, fake_sample)
                    psnr += pt
                    ssim += st

                if (epoch > 10 and psnr > PSNR_MAX) or epoch == 0:
                
                    PSNR_MAX = psnr
                    os.mkdir(os.path.join(exp_path, 'results', "img", "epoch_{}_psnr".format(epoch)))
                    os.mkdir(os.path.join(exp_path, 'results', "img", "epoch_{}_psnr".format(epoch), "clear"))
                    os.mkdir(os.path.join(exp_path, 'results', "img", "epoch_{}_psnr".format(epoch), "clear_DCP"))
                    os.mkdir(os.path.join(exp_path, 'results', "img", "epoch_{}_psnr".format(epoch), "T"))
                    os.mkdir(os.path.join(exp_path, 'results', "img", "epoch_{}_psnr".format(epoch), "RefineT"))
                    for i in range(len(fake_sample_list)):
                        torchvision.utils.save_image(fake_sample_list[i],
                                                     os.path.join(exp_path, 'results', "img",
                                                                  "epoch_{}_psnr".format(epoch),"clear",
                                                                  'clear_{}.png'.format(i)), normalize=True)
                    for i in range(len(fake_DCP_list)):
                        torchvision.utils.save_image(fake_DCP_list[i],
                                                     os.path.join(exp_path, 'results', "img",
                                                                  "epoch_{}_psnr".format(epoch),"clear_DCP",
                                                                  'DCP_{}.png'.format(i)), normalize=True)

                    for i in range(len(T)):
                        torchvision.utils.save_image(T[i],
                                                     os.path.join(exp_path, 'results', "img",
                                                                  "epoch_{}_psnr".format(epoch),"T",
                                                                  'T_{}.png'.format(i)), normalize=True)

                    for i in range(len(Ref_T)):
                        torchvision.utils.save_image(Ref_T[i],
                                                     os.path.join(exp_path, 'results', "img",
                                                                  "epoch_{}_psnr".format(epoch),"RefineT",
                                                                  'RefineT_{}.png'.format(i)), normalize=True)

                elif epoch > 10 and ssim > SSIM_MAX:
                    SSIM_MAX = ssim
                    os.mkdir(os.path.join(exp_path, 'results', "img", "epoch_{}_ssim".format(epoch)))
                    os.mkdir(os.path.join(exp_path, 'results', "img", "epoch_{}_ssim".format(epoch), "clear"))
                    os.mkdir(os.path.join(exp_path, 'results', "img", "epoch_{}_ssim".format(epoch), "clear_DCP"))
                    os.mkdir(os.path.join(exp_path, 'results', "img", "epoch_{}_ssim".format(epoch), "T"))
                    os.mkdir(os.path.join(exp_path, 'results', "img", "epoch_{}_ssim".format(epoch), "RefineT"))
                    for i in range(len(fake_sample_list)):
                        torchvision.utils.save_image(fake_sample_list[i],
                                                     os.path.join(exp_path, 'results', "img",
                                                                  "epoch_{}_ssim".format(epoch),"clear",
                                                                  'clear_{}.png'.format(i)), normalize=True)
                    for i in range(len(fake_DCP_list)):
                        torchvision.utils.save_image(fake_DCP_list[i],
                                                     os.path.join(exp_path, 'results', "img",
                                                                  "epoch_{}_ssim".format(epoch),"clear_DCP",
                                                                  'DCP_{}.png'.format(i)), normalize=True)

                    for i in range(len(T)):
                        torchvision.utils.save_image(T[i],
                                                     os.path.join(exp_path, 'results', "img",
                                                                  "epoch_{}_ssim".format(epoch),"T",
                                                                  'T_{}.png'.format(i)), normalize=True)

                    for i in range(len(Ref_T)):
                        torchvision.utils.save_image(Ref_T[i],
                                                     os.path.join(exp_path, 'results', "img",
                                                                  "epoch_{}_ssim".format(epoch),"RefineT",
                                                                  'RefineT_{}.png'.format(i)), normalize=True)

                print(f"psnr: {psnr / L:.4f}, ssim: {ssim / L:.4f}")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024, help='seed used for initialization')  
    parser.add_argument('--resume', action='store_true', default=False)  
    parser.add_argument("--val_step", type=int, default=1)
    parser.add_argument("--resume_path", type=str, default=r'\results\contents')
    parser.add_argument('--image_size', type=int, default=256, help='size of image')  
    parser.add_argument('--num_channels', type=int, default=3, help='channel of image')  
    parser.add_argument('--numG_channels', type=int, default=9, help='channel of image')  

    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                        help='progressive type for output')  
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')  
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')  
    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')  
    parser.add_argument('--fir', action='store_false', default=True, help='FIR')  
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1], help='FIR kernel')  
    parser.add_argument('--not_use_tanh', action='store_true', default=False)  
    parser.add_argument('--num_channels_dae', type=int, default=128,
                        help='number of initial channels in denosing model')  
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1, 1, 2, 2, 4, 4], help='channel multiplier')
    parser.add_argument('--n_mlp', type=int, default=3, help='number of mlp layers for z')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,), help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0., help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                        help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True, help='noise conditional')
    parser.add_argument('--skip_rescale', action='store_false', default=True, help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan', help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--beta_min', type=float, default=0.1, help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=40., help='beta_max for diffusion')
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--centered', action='store_false', default=True, help='-1,1 scale')

    
    parser.add_argument('--exp', default=r'\results',
                        help='name of experiment')  
    parser.add_argument('--num_epoch', type=int, default=200)  
    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')  
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate d')  
    parser.add_argument('--nz', type=int, default=100)  
    parser.add_argument('--num_timesteps', type=int, default=8)  
    parser.add_argument('--z_emb_dim', type=int, default=256)  
    parser.add_argument('--t_emb_dim', type=int, default=256)  
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')  
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam')  
    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='use EMA or not')  
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')

    parser.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None, help='lazy regulariation.')  
    parser.add_argument('--no_lr_decay', action='store_true', default=False)
    parser.add_argument('--save_content', action='store_true', default=True)  
    parser.add_argument('--save_content_every', type=int, default=20,
                        help='save content for resuming every x epochs')  
    parser.add_argument('--save_ckpt_every', type=int, default=50, help='save ckpt every x epochs')  

    
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')  
    parser.add_argument('--num_process_per_node', type=int, default=1, help='number of gpus')  
    parser.add_argument('--node_rank', type=int, default=0, help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')  
    parser.add_argument('--master_address', type=str, default='127.0.0.1', help='address for master')  

    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node  
    with open(os.path.join("\config", "IO.yml"), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    size = args.num_process_per_node
    device = torch.device('cuda:{}'.format(0))
    L=80
    train(0, device, args, new_config,L)
