from diffusers import LatentConsistencyModelPipeline
import torch
from tqdm import tqdm
from utils import register_attention_control
import copy


def inversion_free_add_noise(
        model,
        latents,
        time_step: int,
        noise=None
):
    # 实现 DDCM 的 scheduler 支持一步加噪到任何时间步
    if noise is None:
        noise = torch.randn_like(latents)
    alpha = model.scheduler.alphas_cumprod[time_step]
    noised_latents = alpha ** 0.5 * latents.to(model.device) + (1 - alpha) ** 0.5 * noise.to(model.device)
    return noised_latents


def inversion_free_get_delta(
        model,
        z_0,
        z_t,
        t,
):
    alpha = model.scheduler.alphas_cumprod[t]
    beta = 1 - alpha
    delta = (z_t.to(model.device) - alpha ** 0.5 * z_0.to(model.device)) / beta ** 0.5
    return delta


def inversion_free_denoise(
        model,
        z_t,
        t,
        noise_pred,
):
    alpha = model.scheduler.alphas_cumprod[t]
    beta = 1 - alpha
    z_0 = (z_t.to(model.device) - beta ** 0.5 * noise_pred) / alpha ** 0.5
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
    elif inclination == "none-tar":
        _delta_weights_src = torch.ones(time_step)
        _delta_weights_ref = torch.zeros(time_step)
    elif inclination == "none-src":
        _delta_weights_src = torch.zeros(time_step)
        _delta_weights_ref = torch.ones(time_step)
    elif inclination == "none":
        _delta_weights_src = torch.zeros(time_step)
        _delta_weights_ref = torch.zeros(time_step)
    elif inclination == "both":
        _delta_weights_src = torch.ones(time_step)
        _delta_weights_ref = torch.ones(time_step)
    else:
        raise ValueError("Invalid inclination")
    return _delta_weights_src, _delta_weights_ref


@torch.no_grad()
def diffusion_noise_pred(
        model: LatentConsistencyModelPipeline,
        noised_latent,
        time_step,
        embedding,
        cfg_guidance,
):
    noised_latents = torch.concat([noised_latent] * 2).to(model.device)
    noise_pred = model.unet(noised_latents, time_step, encoder_hidden_states=embedding)['sample']
    noise_pred_uc_src, noise_pred_uc_tar, noise_pred_c_src, noise_pred_c_tar = noise_pred.chunk(4)
    noise_pred_src = noise_pred_uc_src + cfg_guidance * (noise_pred_c_src - noise_pred_uc_src)
    noise_pred_tar = noise_pred_uc_tar + cfg_guidance * (noise_pred_c_tar - noise_pred_uc_tar)
    return noise_pred_src, noise_pred_tar


@torch.no_grad()
def gen_inversion_free(
        model: LatentConsistencyModelPipeline,
        src_origin_latent,
        tar_origin_latent,
        src_embeddings,
        tar_embeddings,
        num_inference_steps=100,
        inclination="src-pose",
        mode="cosine",
        cfg_guidance=0.7,
        src_coef=0.5,
        tar_coef=0.5,
        return_all=False,
        controller=None,
        model_2=None,
):
    # add null embedding
    null_text_input = model.tokenizer(
        [""], padding="max_length",
        max_length=model.tokenizer.model_max_length,
        return_tensors="pt", truncation=True,
    )
    null_embeddings = model.text_encoder(null_text_input.input_ids.to(model.device))[0]
    context = torch.concat([null_embeddings, null_embeddings, src_embeddings, tar_embeddings]).to(model.device)

    # generate weights
    src_weights, tar_weights = gen_direction_weights(inclination, mode, num_inference_steps)

    # set latents for iterations
    src_latent = src_origin_latent.to(model.device)
    tar_latent = src_latent.clone().detach()

    # set the alphas for DDCM
    model.scheduler.set_timesteps(num_inference_steps)
    if model_2 is not None:
        model_2 = model_2.to(model.device)
        model_2.scheduler.set_timesteps(num_inference_steps)

    model_src = model
    model_tar = model_src if model_2 is None else model_2

    if controller is not None:
        register_attention_control(model_src, controller)
        if model_2 is not None:
            register_attention_control(model_tar, controller)

    # Init
    init_noised_tar_latent = init_noised_src_latent = torch.randn_like(tar_latent)
    init_time_step = model.scheduler.timesteps[-num_inference_steps]
    init_src_delta = inversion_free_get_delta(model, src_origin_latent, init_noised_src_latent, init_time_step)
    init_tar_delta = inversion_free_get_delta(model, tar_origin_latent, init_noised_tar_latent, init_time_step)
    init_noised_latents = torch.cat([init_noised_src_latent, init_noised_tar_latent])
    init_src_pred, init_tar_pred = diffusion_noise_pred(model, init_noised_latents, init_time_step, context,
                                                        cfg_guidance)
    init_src_direction = init_src_pred - init_src_delta
    init_tar_direction = init_tar_pred - init_tar_delta
    init_noise_pred = (init_tar_pred - src_weights[0] * src_coef * init_src_direction
                       - tar_weights[0] * tar_coef * init_tar_direction)
    tar_latent = inversion_free_denoise(model, init_noised_tar_latent, init_time_step, init_noise_pred)
    if controller is not None:
        tar_latent = controller.step_callback(torch.cat([src_latent, tar_latent]))[1:]
    all_latents = []
    if return_all:
        all_latents = [tar_latent]

    # Iterations
    # tar_noise = src_noise = torch.randn_like(tar_latent)
    for i, step in enumerate(tqdm(model.scheduler.timesteps[-num_inference_steps + 1:])):
        tar_noise = src_noise = torch.randn_like(tar_latent)
        src_weight_t = src_weights[1 + i]
        tar_weight_t = tar_weights[1 + i]
        noised_src_latent_t = inversion_free_add_noise(model, src_latent, step, src_noise)
        noised_tar_latent_t = inversion_free_add_noise(model, tar_latent, step, tar_noise)
        src_pred_t, tar_pred_t = diffusion_noise_pred(model, torch.cat([noised_src_latent_t, noised_tar_latent_t]),
                                                      step, context, cfg_guidance)
        src_delta_t = inversion_free_get_delta(model, src_origin_latent, noised_src_latent_t, step)
        tar_delta_t = inversion_free_get_delta(model, tar_origin_latent, noised_tar_latent_t, step)
        src_direction_t = src_pred_t - src_delta_t
        tar_direction_t = tar_pred_t - tar_delta_t
        noise_pred_t = (tar_pred_t - src_weight_t * src_coef * src_direction_t
                        - tar_weight_t * tar_coef * tar_direction_t)
        tar_latent = inversion_free_denoise(model, noised_tar_latent_t, step, noise_pred_t)
        if controller is not None:
            tar_latent = controller.step_callback(tar_latent)
        # tar_noise = tar_pred_t
        # src_noise = src_pred_t
        if return_all:
            all_latents.append(tar_latent)

    if return_all:
        return all_latents
    else:
        return torch.cat([src_latent.to(model.device), tar_latent])
