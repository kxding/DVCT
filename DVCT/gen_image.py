import torch.cuda

from ptp_inversion import DDIMScheduler, StableDiffusionPipeline
from multi_token_clip import MultiTokenCLIPTokenizer
from transformers import CLIPTextModel
import os
from PIL import Image
from datetime import datetime
import time

import ddim_inversion
import torch
from torch.optim.sgd import SGD
from tqdm import tqdm


def add_tokens(tokenizer, text_encoder, placeholder_token, num_vec_per_token=1, initializer_token=None, use_neg=False):
    """
    Add tokens to the tokenizer and set the initial value of token embeddings
    """
    tokenizer.add_placeholder_tokens(placeholder_token, num_vec_per_token=num_vec_per_token)
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    placeholder_token_ids = tokenizer.encode(placeholder_token, add_special_tokens=False)
    print(f"number of placeholder tokens are: {len(placeholder_token_ids)}")
    if initializer_token:
        token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
        for i, placeholder_token_id in enumerate(placeholder_token_ids):

            token_embeds[placeholder_token_id] = token_embeds[token_ids[i * len(token_ids) // num_vec_per_token]]
            if use_neg:
                token_embeds[placeholder_token_id] += torch.randn_like(token_embeds[placeholder_token_id]) * 1e-3
    else:
        for i, placeholder_token_id in enumerate(placeholder_token_ids):
            token_embeds[placeholder_token_id] = torch.randn_like(token_embeds[placeholder_token_id])


def load_multitoken_tokenizer(tokenizer, text_encoder, pos_learned_embeds_dict, pos_placeholder_token):
    num_vec_pos_token = pos_learned_embeds_dict[pos_placeholder_token].shape[0]
    add_tokens(tokenizer, text_encoder, pos_placeholder_token, num_vec_per_token=num_vec_pos_token)
    pos_placeholder_token_ids = tokenizer.encode(pos_placeholder_token, add_special_tokens=False)
    token_embeds = text_encoder.get_input_embeddings().weight.data
    for i, placeholder_token_id in enumerate(pos_placeholder_token_ids):
        token_embeds[placeholder_token_id] = pos_learned_embeds_dict[pos_placeholder_token][i]


def init_model(
        pretrained_model_name_or_path,
        seed=0,
):
    # backward_embeds_dict_name = "%s.bin" % str(train_step)
    # backward_placeholder_token = "<%s>" % prompt_name_backward
    # backward_embeds_dict_path = os.path.join(model_id, backward_embeds_dict_name)
    # backward_learned_embeds_dict = torch.load(backward_embeds_dict_path)
    # get reference embedding

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder",
                                                 revision=False)
    tokenizer = MultiTokenCLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")

    # load_multitoken_tokenizer(tokenizer, text_encoder, backward_learned_embeds_dict, backward_placeholder_token)
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False)
    ldm_stable = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, scheduler=scheduler,
                                                         tokenizer=tokenizer, text_encoder=text_encoder).to(device)
    ldm_stable.safety_checker = lambda images, clip_input: (images, False)

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    # controller arguments

    return generator, ldm_stable, device


def gen_image(
        pretrained_model_name_or_path,
        output_img_path_name,
        prompt_name_backward,
        model_id,
        train_step,
        backward_prompts,
        src_reversed_latent_list=None,
        src_embedding_list=None,
        seed=0,
        guidance_scale=7.5,
        no_controller=False,
        use_direct_inversion=False,
        forward_prompt="",
        cross_attention_injection_ratio=0.2,
        self_attention_injection_ratio=0.9
):
    generator, tokenizer, inversion, ldm_stable, device = init_model(pretrained_model_name_or_path,
                                                                     prompt_name_backward, model_id, train_step, seed, )
    # ptp_utils.register_attention_control(inversion.model, None)

    # make controller
    place_holder_prompt = "<s2>"
    cross_replace_steps = {'default_': cross_attention_injection_ratio, }
    blend_word = None
    eq_params = {"words": (place_holder_prompt,),
                 "values": (0.5,)}  # amplify attention to the word "watercolor" by 5
    controller = make_controller(["", "<s2>"], tokenizer, False, cross_replace_steps,
                                 self_attention_injection_ratio, blend_word, eq_params)

    # use controller or not
    if no_controller:
        controller = None

    # generate picture
    latents = inversion.backward_diffusion_ptp(
        backward_prompts,
        controller=controller,
        latent_list=src_reversed_latent_list,
        num_inference_steps=50, guidance_scale=guidance_scale,
        uncond_embeddings=src_embedding_list,
        use_direct_inversion=use_direct_inversion,
        generator=generator,
    )

    # save picture
    images = ptp_utils.latent2image(ldm_stable.vae, latents.detach())
    print("saving image", output_img_path_name)
    output_img_list = [Image.fromarray(images[1, :, :, :])]
    output_img_list[0].save(output_img_path_name)