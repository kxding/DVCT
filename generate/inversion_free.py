from diffusers import StableDiffusionPipeline
import torch
from tqdm import tqdm
import numpy as np
# class Inversion(DiffusionPipeline):
#     def __init__(self, model):
#         self.model = model
#         self.tokenizer = self.model.

def get_cos_weight(len, device):
    x = np.linspace(0, np.pi / 2, len)
    weight_np = np.cos(x)
    weight = torch.from_numpy(weight_np).float().unsqueeze(1)
    weight = weight.to(device)
    return weight
def get_sin_weight(len, device):
    x = np.linspace(0, np.pi / 2, len)
    weight_np = np.sin(x)
    weight = torch.from_numpy(weight_np).float().unsqueeze(1)
    weight = weight.to(device)
    return weight
def get_linear_weight(len, device):
    x = np.linspace(0, np.pi / 2, len)
    weight_np = np.cos(x)
    weight = torch.from_numpy(weight_np).float().unsqueeze(1)
    weight = weight.to(device)
    return weight

def inversion_free_add_noise(
        model,
        latents,
        time_step: int,
        noise=None
):
    if noise is None:
        noise = torch.randn_like(latents)
    alpha = model.scheduler.alphas_cumprod[time_step]
    noised_latents = alpha ** 0.5 * latents + (1 - alpha) ** 0.5 * noise
    return noised_latents


def inversion_free_get_delta(
        model,
        z_0,
        z_t,
        t,
):
    alpha = model.scheduler.alphas_cumprod[t]
    beta = 1 - alpha
    delta = (z_t - alpha ** 0.5 * z_0) / beta ** 0.5
    return delta


def inversion_free_denoise(
        model,
        z_t,
        t,
        noise_pred,
):
    alpha = model.scheduler.alphas_cumprod[t]
    beta = 1 - alpha
    z_0 = (z_t - beta ** 0.5 * noise_pred) / alpha ** 0.5
    return z_0


def gen_direction_weights(
        inclination="src-pose",
        mode="linear",
        time_step=100,
):
    if inclination == "src-pose":
        if mode == "linear":
            _delta_weights_src = torch.linspace(1, 0, time_step)
            _delta_weights_ref = torch.linspace(0, 1, time_step)
        elif mode == "cosine":
            _delta_weights_src = torch.cos(
                torch.linspace(0, 1, time_step) * 3.14159265
            ) / 2 + 0.5
            _delta_weights_ref = 1 - _delta_weights_src
        else:
            raise ValueError("Invalid mode")
    elif inclination == "tar-pose":
        if mode == "linear":
            _delta_weights_src = torch.linspace(0, 1, time_step)
            _delta_weights_ref = torch.linspace(1, 0, time_step)
        elif mode == "cosine":
            _delta_weights_src = torch.cos(
                torch.linspace(1, 0, time_step) * 3.14159265
            ) / 2 + 0.5
            _delta_weights_ref = 1 - _delta_weights_src
        else:
            raise ValueError("Invalid mode")
    elif inclination == "none":
        _delta_weights_src = torch.ones(time_step)
        _delta_weights_ref = torch.ones(time_step)
    else:
        raise ValueError("Invalid inclination")
    return _delta_weights_src, _delta_weights_ref


def diffusion_noise_pred(
        model,
        noised_latent,
        time_step,
        embedding,
        cfg_guidance,
):
    noised_latents = torch.concat([noised_latent] * 2)
    noise_pred = model.unet(noised_latents, time_step, encoder_hidden_states=embedding)['sample']
    noise_pred_uc, noise_pred_c = noise_pred.chunk(2)
    return noise_pred_uc + cfg_guidance * (noise_pred_c - noise_pred_uc)

@torch.no_grad()
def gen_inversion_free(
    model,
    src_latent,
    tar_latent,
    src_embeddings,
    ref_embeddings,
    num_inference_steps,
    tar_weight=None,
    src_weight=None,
    return_all=False,
):
    tar_latent_predict = src_latent.clone().detach()
    model.scheduler.set_timesteps(num_inference_steps)
    if return_all:
        all_latents = [tar_latent_predict]
        
    if src_weight is None:
        src_weight = get_cos_weight(num_inference_steps, model.device)
        # tar_weight = 1 - src_weight
    else :
        src_weight = torch.tensor([src_weight] * num_inference_steps).to(model.device)
    if tar_weight is None:
        tar_weight = get_cos_weight(num_inference_steps, model.device)
    else:
        tar_weight = torch.tensor([tar_weight] * num_inference_steps).to(model.device)
    
    noise_pred = torch.randn_like(src_latent)   
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-num_inference_steps:])):
        src_latent_t = inversion_free_add_noise(model, src_latent, t, noise_pred)
        tar_latent_predict_t = inversion_free_add_noise(model, tar_latent_predict, t, noise_pred)
        src_noise_pred = model.unet(src_latent_t, t, encoder_hidden_states=src_embeddings)["sample"]
        tar_noise_pred = model.unet(tar_latent_predict_t, t, encoder_hidden_states=ref_embeddings)["sample"]
        delta_src = inversion_free_get_delta(model, src_latent, src_latent_t, t)
        delta_tar = inversion_free_get_delta(model, tar_latent, tar_latent_predict_t, t)
 
        noise_pred = tar_noise_pred  - tar_weight[i] * (tar_noise_pred - delta_tar) - src_weight[i] * (src_noise_pred - delta_src)
        tar_latent_predict = inversion_free_denoise(model, tar_latent_predict_t, t, noise_pred)
        if return_all:
            all_latents.append(tar_latent_predict)
            
    if return_all:
        return all_latents
    else:
        return torch.cat([src_latent, tar_latent_predict])
