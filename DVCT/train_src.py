import torch.cuda

from ptp_inversion import *
from multi_token_clip import MultiTokenCLIPTokenizer
from transformers import CLIPTextModel
import os
from PIL import Image
from datetime import datetime
import time

import ddim_inversion


def has_valid_extension(filename, valid_extensions):
    """
    Check if a file has a valid extension.
    Args:
        filename (str): The filename to check.
        valid_extensions (list): A list of valid extensions.

    """
    ext = os.path.splitext(filename)[1][1:]
    return ext.lower() in valid_extensions


def train_src_embedding(
        pretrained_model_name_or_path,
        src_image_path,
        forward_prompt="",
        guidance_scale=7.5,
):
    # choose the picture in the directory
    extensions = ["jpg", "png"]
    src_image_path_name = ""
    file_name = ""
    for file in os.listdir(src_image_path):
        if has_valid_extension(file, extensions):
            src_image_path_name = os.path.join(src_image_path, file)
            file_name = file.split('.')[0]

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder",
                                                 revision=False)
    tokenizer = MultiTokenCLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False)
    ldm_stable = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, scheduler=scheduler,
                                                         tokenizer=tokenizer, text_encoder=text_encoder).to(device)
    ldm_stable.safety_checker = lambda images, clip_input: (images, False)
    inversion = Inversion(ldm_stable)
    ptp_utils.register_attention_control(inversion.model, None)
    src_image_gt = load_512(src_image_path_name)
    src_text_embeddings = inversion.get_text_embedding(forward_prompt)
    src_image_latent = inversion.image2latent(src_image_gt)

    # add noise to src pictures
    src_reversed_latent_list = inversion.forward_diffusion(src_image_latent,
                                                           text_embeddings=src_text_embeddings,
                                                           num_inference_steps=50,
                                                           return_all=True
                                                           )

    # use null-text inversion to train src_embedding.
    uncond_embeddings_list, _ = inversion.null_optimization_path(
        src_reversed_latent_list,
        src_text_embeddings,
        num_inner_steps=50,
        epsilon=1e-5,  # 0.001
        guidance_scale=guidance_scale,
        num_ddim_steps=50
    )

    return uncond_embeddings_list, src_reversed_latent_list, file_name
