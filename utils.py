from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionPipeline


@torch.no_grad()
def img2latent(img_path, model: StableDiffusionPipeline, device):
    image = Image.open(img_path).convert('RGB')
    image = np.array(image) / 255
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    image = 2 * image - 1
    latent = model.vae.encode(image)['z']
    latent = 0.18215 * latent
    return latent


@torch.no_grad()
def latent2img(latent, model: StableDiffusionPipeline):
    latent = 1 / 0.18215 * latent
    image = model.vae.decode(latent)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image[0])
