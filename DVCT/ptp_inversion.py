# This code references and modifies https://github.com/google/prompt-to-prompt

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
import torch
import copy
from diffusers import StableDiffusionPipeline, DDIMScheduler,DiffusionPipeline
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image
from transformers import CLIPTokenizer
import torch
from delta_scheduler import get_delta_weights
from ptp_utils import get_cos_weight, get_sin_weight, get_linear_weight

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# tokenizer= CLIPTokenizer.from_pretrained(
#      "/home/chengbin04/workdir/text2img/diffuser_models/Protogen_Infinity_Official_Release",
#      subfolder="tokenizer")
MAX_NUM_WORDS = 77
LOW_RESOURCE=False

def default_score_estimation(model,latents,context,t,guidance_scale,do_classifier_free_guidance):
    latent_model_input = (
        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        ) 
    noise_pred = model.unet(latent_model_input, t, encoder_hidden_states=context)["sample"]
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
    return noise_pred


class LocalBlend:
    
    def get_mask(self,x_t, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        return mask
    
    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:
           
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(x_t,maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(x_t,maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], tokenizer:CLIPTokenizer, words: [List[List[str]]], substruct_words=None, start_blend=0.2, th=(.3, .3),num_ddim_steps=50):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * num_ddim_steps)
        self.counter = 0 
        self.th=th


        
        
class EmptyControl:
    
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

    
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class SpatialReplace(EmptyControl):
    
    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float,num_ddim_steps=50):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * num_ddim_steps)
        

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                if self.cur_step >= len(self.cross_replace_alpha):
                    alpha_words = self.cross_replace_alpha[-1]
                else:
                    alpha_words = self.cross_replace_alpha[self.cur_step]

                # alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, tokenizer:CLIPTokenizer, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts,tokenizer:CLIPTokenizer, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts,tokenizer, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

class AttentionRefine(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):
        
        # attn_base.shape: [8, 4096/1024/256/64, 77]
        # att_replace.shape: [1, 8, 4096/1024/256/64, 77] 其第三维数值和attn_base的第二维数值是相同的
        # self.mapper.shape: [1, 77]
        
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        
        
        #print(self.alphas)
        # self.alphas是一个[1,1,1,77]的数组，其值中0的部分代表要替换的向量
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace
    def __init__(self, prompts,tokenizer:CLIPTokenizer, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts,tokenizer,num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts,tokenizer:CLIPTokenizer, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts,tokenizer,num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller
        


def get_equalizer(text: str,tokenizer:CLIPTokenizer, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def make_controller(prompts: List[str], tokenizer:CLIPTokenizer, is_replace_controller: bool, cross_replace_steps: Dict[str, float], self_replace_steps: float, blend_words=None, equilizer_params=None,num_ddim_steps=50) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts,tokenizer,blend_words)
    if is_replace_controller:
        controller = AttentionReplace(prompts,tokenizer, num_ddim_steps, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    else:
        controller = AttentionRefine(prompts,tokenizer, num_ddim_steps, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1],tokenizer, equilizer_params["words"], equilizer_params["values"])
        controller = AttentionReweight(prompts, tokenizer,num_ddim_steps, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb, controller=controller)
    return controller


def show_cross_attention(attention_store: AttentionStore,tokenizer:CLIPTokenizer, res: int, from_where: List[str], select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0))
    

def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))
    
def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


class Inversion(DiffusionPipeline):
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None,guidance_scale=7.5):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else guidance_scale
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt
        
    @torch.no_grad()
    def get_text_embedding(self, prompt):
        text_input_ids = self.model.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        
        text_embeddings = self.model.text_encoder(text_input_ids.to(device))[0]
        return text_embeddings
    

    @torch.no_grad()
    def ddim_loop(self, latent,num_ddim_steps):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent
    

    @torch.no_grad()
    def forward_diffusion(self, latents,
                        text_embeddings,
                        guidance_scale=1,
                        num_inference_steps=50,
                        score_estimation_fun=default_score_estimation,
                        return_all=False
                        ):

        do_classifier_free_guidance = guidance_scale > 1.0
        self.model.scheduler.set_timesteps(num_inference_steps)
        
        if return_all:
            all_latents = [latents]
            
        for i in tqdm(range(num_inference_steps)):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = score_estimation_fun(self.model,
                                        latents,text_embeddings,
                                        t,guidance_scale,
                                        do_classifier_free_guidance
                                        )
            latents = self.next_step(noise_pred, t, latents)
            if return_all:
                all_latents.append(latents)
                
        if return_all:
            return all_latents
        else:
            return latents
        
    @torch.no_grad()
    def backward_diffusion(
        self,
        latents,
        text_embeddings,
        guidance_scale=1,
        num_inference_steps=50,
        score_estimation_fun=default_score_estimation,
        return_all=False
    ):
        do_classifier_free_guidance = guidance_scale > 1.0
        self.scheduler.set_timesteps(num_inference_steps)
        all_latents=[]
        
        if return_all:
            all_latents = [latents]

        for i in range(num_inference_steps):
            t = self.model.scheduler.timesteps[i]
            noise_pred = score_estimation_fun(self.model,
                                        latents,text_embeddings,
                                        t,guidance_scale,
                                        do_classifier_free_guidance
                                        )
            latents = self.model.scheduler.step(noise_pred, t, latents)["prev_sample"]
            if return_all:
                all_latents.append(latents)
            
        if return_all:
            return all_latents
        else:
            return latents
    
    # @torch.no_grad()
    # def backward_diffusion_ptp(self,
    #     prompt:  List[str],
    #     controller,
    #     num_inference_steps: int = 50,
    #     guidance_scale: Optional[float] = 7.5,
    #     generator: Optional[torch.Generator] = None,
    #     latent: Optional[torch.FloatTensor] = None,
    #     uncond_embeddings=None,
    #     start_time=50,
    #     return_all=False
    # ):
    #     model=self.model
    #     batch_size = len(prompt)
    #     ptp_utils.register_attention_control(model, controller)
    #     height = width = 512
        
    #     text_input = model.tokenizer(
    #         prompt,
    #         padding="max_length",
    #         max_length=model.tokenizer.model_max_length,
    #         truncation=True,
    #         return_tensors="pt",
    #     )
    #     text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    #     max_length = text_input.input_ids.shape[-1]
    #     if uncond_embeddings is None:
    #         uncond_input = model.tokenizer(
    #             [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    #         )
    #         uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    #     else:
    #         uncond_embeddings_ = None

    #     latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
        
    #     model.scheduler.set_timesteps(num_inference_steps)
        
    #     if return_all:
    #         all_latents = [latents] 
        
    #     for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
    #         if uncond_embeddings_ is None:
    #             context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
    #         else:
    #             context = torch.cat([uncond_embeddings_, text_embeddings])
    #         latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
    #         if return_all:
    #             all_latents.append(latents)
            
    #     if return_all:
    #         return all_latents
    #     else:
    #         return latents

    @torch.no_grad()
    def offset_calculate(self, latents, uncond_embeddings_list, cond_embeddings, num_ddim_steps=50, guidance_scale=7.5):
        noise_loss_list = []
        latents_noised = self.forward_diffusion(latents, uncond_embeddings_list[0], guidance_scale, num_inference_steps=num_ddim_steps)
        latent_cur = latents_noised
        for i in range(num_ddim_steps):          
            ind = 0 if (len(latents) - i - 2) < 0 else (len(latents) - i - 2)
            latent_prev = torch.concat([latents[ind]]*latent_cur.unsqueeze(0).shape[0])
            # print("latent_prev = ", latent_prev.shape)
            # print("latents = ", latents.shape)
            # print("latent_cur = ", latent_cur.shape)
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                # latent_cur = latent_cur.unsqueeze(0)
                # noise_pred = self.get_noise_pred_single(latent_cur.unsqueeze(0), t, self.context[0])
                noise_pred = self.get_noise_pred_single(torch.concat([latent_cur] * 2).unsqueeze(0).reshape(2, 4, 64, 64), t, self.context)

                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_w_guidance = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred_w_guidance, t, latent_cur)
                # print("latents_prev_rec = ", latents_prev_rec.shape)
                # print("latent_cur = ", latent_cur.shape)
                loss = latent_prev - latents_prev_rec

            noise_loss_list.append(loss.detach())
            latent_cur = latents_prev_rec + loss
            
        return noise_loss_list

    # @torch.no_grad()
    # def backward_diffusion_ptp(self,
    #     prompt: List[str],
    #     controller,
    #     num_inference_steps: int = 50,
    #     guidance_scale: Optional[float] = 7.5,
    #     generator: Optional[torch.Generator] = None,
    #     latent_list=None,
    #     uncond_embeddings=None,
    #     cond_embeddings=None,
    #     start_time=50,
    #     return_all=False,
    #     use_direct_inversion=False,
    #     inclination="none",
    #     mode="linear",
    #     delta_coef_ref=1.0,
    #     delta_coef_src=1.0,
    # ):
    #     latent = latent_list[-1]
    #     model=self.model
    #     if use_direct_inversion:
    #         model2 = self.model2
            
    #     batch_size = len(prompt)
    #     height = width = 512
    #     ptp_utils.register_attention_control(model, controller)
    #     text_input = model.tokenizer(
    #         prompt,
    #         padding="max_length",
    #         max_length=model.tokenizer.model_max_length,
    #         truncation=True,
    #         return_tensors="pt",
    #     )
    #     text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    #     max_length = text_input.input_ids.shape[-1]
    #     if use_direct_inversion:
    #         delta_weights = get_delta_weights(time_step=num_inference_steps, 
    #                                             inclination=inclination, mode=mode,
    #                                             coef_ref=delta_coef_ref,
    #                                             coef_src=delta_coef_src).to(model.device)
    #     if uncond_embeddings is None:
    #         uncond_input = model.tokenizer(
    #             [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    #         )
    #         uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    #     else:
    #         uncond_embeddings_ = None
    #     latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    #     model.scheduler.set_timesteps(num_inference_steps)
    #     for _i in range(len(latent_list)):
    #         _, latent_list[_i] = ptp_utils.init_latent(latent_list[_i], model, height, width, generator, batch_size)
        
    #     if return_all:
    #         all_latents = [latents]
    #     if use_direct_inversion:
    #         latents_compute = latent_list[-1].to(model.device)
    #         init_difference = (latent_list[-1] - latents).to(model.device)
    #     for i, t in enumerate(tqdm(model.scheduler.timesteps[-num_inference_steps:])):
    #         if use_direct_inversion:
    #             delta_weight = delta_weights[i]
    #         if uncond_embeddings_ is None:
    #             context = torch.cat([uncond_embeddings[i//(num_inference_steps // 50)].expand(*text_embeddings.shape).to(model.device), text_embeddings])
    #         else:
    #             context = torch.cat([uncond_embeddings_, text_embeddings])
    #         if use_direct_inversion and i != num_inference_steps - 1:
    #             difference_src_ref = latent_list[(num_inference_steps - i - 2)//(num_inference_steps // 50)] - ptp_utils.diffusion_step(model2, None, latents_compute, context, t, guidance_scale, low_resource=False)
    #             difference_src_ref = difference_src_ref * delta_weight[:1]
    #             delta_difference = (init_difference - difference_src_ref) * delta_weight[1:]
    #             latents_compute = ptp_utils.diffusion_step(model2, None, latents_compute, context, t, guidance_scale, low_resource=False)
    #             latents_compute = torch.concat((latents_compute[:1] + difference_src_ref[:1], latents_compute[1:]))
    #         latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
    #         if use_direct_inversion and i != num_inference_steps - 1:
    #             latents = torch.concat((latents[:1] + difference_src_ref[:1], latents[1:] + delta_difference[1:]))
    #             init_difference = difference_src_ref
    #         if return_all:
    #             all_latents.append(latents)
            
    #     if return_all:
    #         return all_latents
    #     else:
    #         return latents
    
    def my_add_noise(
        self,
        latents,
        time_step: int,
        noise=None
    ):
        # 实现 DDCM 的 scheduler 支持一步加噪到任何时间步
        if noise is None:
            noise = torch.randn_like(latents)
        alpha = self.model.scheduler.alphas_cumprod[time_step]
        noised_latents = alpha ** 0.5 * latents + (1 - alpha) ** 0.5 * noise
        return noised_latents

    def my_get_delta(
        self,
        z_0,
        z_t,
        t,
    ):
        alpha = self.model.scheduler.alphas_cumprod[t]
        beta = 1 - alpha
        delta = (z_t - alpha ** 0.5 * z_0) / beta ** 0.5
        return delta
    
    def my_denoise(
        self,
        z_t,
        t,
        noise_pred,
    ):
        alpha = self.model.scheduler.alphas_cumprod[t]
        beta = 1 - alpha
        z_0 = (z_t - beta ** 0.5 * noise_pred) / alpha ** 0.5
        return z_0
    
    @torch.no_grad()
    def backward_diffusion_ptp(self,
        src_latent,
        tar_latent,
        src_embeddings,
        ref_embeddings,
        null_embeddings,
        num_inference_steps,
        cfg_guidence=None,
        tar_weight=None,
        src_weight=None,
        return_all=False,
        lora=False,
    ):
        model = self.model
        if lora is True:
            model.unet.load_lora_weights("./lora-res/", weight_name="cat2.safetensors", adapter_name="hat")
            model.unet.load_lora_weights("./lora-res/", weight_name="cat1.safetensors", adapter_name="cartoon")
            model.unet.set_adapters(["hat", "cartoon"], adapter_weights=[0.7, 0.3])

        tar_latent_predict = src_latent.clone().detach()
        model.scheduler.set_timesteps(num_inference_steps)
        if return_all:
            all_latents = [tar_latent_predict]
            
        if src_weight is None:
            src_weight = get_cos_weight(num_inference_steps, model.device)
            src_weight = torch.tensor([0] * num_inference_steps).to(model.device)
            # tar_weight = 1 - src_weight
        else :
            src_weight = torch.tensor([src_weight] * num_inference_steps).to(model.device)
        if tar_weight is None:
            tar_weight = get_cos_weight(num_inference_steps, model.device)
        else:
            tar_weight = torch.tensor([tar_weight] * num_inference_steps).to(model.device)
            
            
        if cfg_guidence is None:
            cfg_guidence = get_cos_weight(num_inference_steps, model.device)
        else :
            cfg_guidence = np.full((num_inference_steps, 1), cfg_guidence) 
            cfg_guidence = torch.from_numpy(cfg_guidence).float().to('cuda:0')
        
        noise_pred = torch.randn_like(tar_latent)   
        for i, t in enumerate(tqdm(model.scheduler.timesteps[-num_inference_steps:])):
            src_latent_t = self.my_add_noise(src_latent, t, noise_pred)
            tar_latent_predict_t = self.my_add_noise(tar_latent_predict, t, noise_pred)
            src_noise_pred = self.model.unet(src_latent_t, t, encoder_hidden_states=src_embeddings)["sample"]
            tar_noise_pred = self.model.unet(tar_latent_predict_t, t, encoder_hidden_states=ref_embeddings)["sample"]
            null_src_noise_pred = self.model.unet(src_latent_t, t, encoder_hidden_states=null_embeddings)["sample"]
            null_tar_noise_pred = self.model.unet(tar_latent_predict_t, t, encoder_hidden_states=null_embeddings)["sample"]
            
            src_noise = cfg_guidence[i] * src_noise_pred + (1 - cfg_guidence[i]) * null_src_noise_pred
            tar_noise = cfg_guidence[i] * tar_noise_pred + (1 - cfg_guidence[i]) * null_tar_noise_pred
            delta_src = self.my_get_delta(src_latent, src_latent_t, t)
            delta_tar = self.my_get_delta(tar_latent, tar_latent_predict_t, t)
           
            #print("cfg_guidance = ",cfg_guidence[t].item())
            # noise_pred = tar_noise_pred - src_noise_pred + delta
            # noise_pred = tar_noise_pred  - weight[i] * (src_noise_pred - delta_src) 
            noise_pred = tar_noise_pred  - tar_weight[i] * (tar_noise_pred - delta_tar) - src_weight[i] * (src_noise_pred - delta_src)
            # noise_pred = tar_noise_pred  - weight[i] * ((t - 1) / (len(model.scheduler.timesteps[-num_inference_steps + 1:]) - 1)) * (src_noise_pred - delta) 
            # noise_pred = tar_noise  - weight[i] * (src_noise - delta)
            # noise_pred = tar_noise  - weight[i] * ((t - 1) / (len(model.scheduler.timesteps[-num_inference_steps + 1:]) - 1)) * (src_noise - delta) 
       
            # noise_free = guidance_scale * (noise_pred - tmp)
            tar_latent_predict = self.my_denoise(tar_latent_predict_t, t, noise_pred)
            if return_all:
                all_latents.append(tar_latent_predict)
        if return_all:
            return all_latents
        else:
            return torch.cat([src_latent, tar_latent_predict])
    
    @torch.no_grad()
    def backward_diffusion_delta(self,
        latent_list, # ?? src latent 里面 有没有 分一半给ref ？？？ 
        src_embeddings,
        ref_embeddings,
        num_inference_steps: int = 20,
        cfg_guidence=7.50,
        return_all=False, 
        controller=None,
        generator=None,
    ):
        cfg_guidence = np.linspace(1.0, 0.01, num_inference_steps).reshape(-1, 1)
        #src_embeddings = uncond_embeddings
        #ref_embeddings = cond_embeddings
        
        #for _i in range(len(latent_list)):
        #    latent_list[_i] = torch.cat((latent_list[_i], latent_list[_i]), dim=0)
        #    _, latent_list[_i] = ptp_utils.init_latent(latent_list[_i], model, 512, 512, generator, batch_size=2)
        
        model = self.model
        src_latent = latent_list[-1]
        tar_latent = src_latent.clone().detach()
        model.scheduler.set_timesteps(num_inference_steps)
        # ptp_utils.register_attention_control(model, controller)
        init_noised_tar_latent = init_noise_src_latent = torch.randn_like(tar_latent)
        delta_1 = self.my_get_delta(src_latent, init_noise_src_latent, model.scheduler.timesteps[-num_inference_steps])
        init_src_pred = self.model.unet(src_latent, model.scheduler.timesteps[-num_inference_steps], encoder_hidden_states=src_embeddings)["sample"]
        init_tar_pred = self.model.unet(tar_latent, model.scheduler.timesteps[-num_inference_steps], encoder_hidden_states=ref_embeddings)["sample"]
        tar_latent = self.my_denoise(init_noised_tar_latent, model.scheduler.timesteps[-num_inference_steps], init_tar_pred - init_src_pred + delta_1)
        if return_all:
            all_latents = [tar_latent]
        
        latents_compute = torch.cat((src_latent, tar_latent)).to(model.device)
        #latents_compute = src_latent
        print("latents_compute ", latents_compute.shape)
        for i, t in enumerate(tqdm(model.scheduler.timesteps[-num_inference_steps + 1:])):
            noise = torch.randn_like(tar_latent)
            noised_src_latent = self.my_add_noise(src_latent, t, noise)
            noised_tar_latent = self.my_add_noise(tar_latent, t, noise)
            src_noise_pred = self.model.unet(noised_src_latent, t, encoder_hidden_states=src_embeddings)["sample"]
            tar_noise_pred = self.model.unet(noised_tar_latent, t, encoder_hidden_states=ref_embeddings)["sample"]
            delta = self.my_get_delta(src_latent, noised_src_latent, t)
            # noise_pred = (tar_noise_pred - src_noise_pred) * cfg_guidence
            noise_pred = tar_noise_pred  - cfg_guidence * (src_noise_pred - delta) 
            print("noise_pred ", noise_pred.shape)
            # noise_free = guidance_scale * (noise_pred - tmp)
            tar_latent = self.my_denoise(noised_tar_latent, t, noise_pred)
            src_latent = self.my_denoise(noised_src_latent, t, delta)
            
            if return_all:
                all_latents.append(tar_latent)
        if return_all:
            return all_latents
        else:
            return torch.cat([src_latent, tar_latent])
    
    
    @torch.no_grad()
    def forward_diffusion_raw(self, 
                          latents,
                          text_embeddings,
                          guidance_scale=1,
                          num_inference_steps=50
                          ):
        
        do_classifier_free_guidance = guidance_scale > 1.0
        self.scheduler.set_timesteps(num_inference_steps)
        all_latent=[]

        for i in range(num_inference_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            noise_pred = self.get_noise_pred_single(latent_model_input, t, text_embeddings)
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            latents = self.next_step(noise_pred, t, latents)
            all_latent.append(latents)
        
        return all_latent
    
    @torch.no_grad()
    def backward_diffusion_raw(
        self,
        latents,
        text_embeddings,
        guidance_scale=1,
        num_inference_steps=50,
    ):
        do_classifier_free_guidance = guidance_scale > 1.0
        self.scheduler.set_timesteps(num_inference_steps)
        all_latent=[]

        for i in range(num_inference_steps):
            t = self.model.scheduler.timesteps[i]
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            noise_pred = self.get_noise_pred_single(latent_model_input, t, text_embeddings)
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            latents = self.model.scheduler.step(noise_pred, t, latents)["prev_sample"]
            all_latent.append(latents)

        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent,num_ddim_steps=50)
        return image_rec, ddim_latents
     
    
    def null_optimization_path(self, latent_list,cond_embeddings,num_inner_steps, epsilon, guidance_scale=7.5,num_ddim_steps=50):
        uncond_embeddings = self.get_text_embedding("")
        uncond_embeddings_list = []
        latent_cur = latent_list[-1]
        bar = tqdm(total=num_inner_steps * num_ddim_steps)
        for i in range(num_ddim_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latent_list[len(latent_list) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
                
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                
                logs = {"loss": loss_item}
                bar.set_postfix(**logs)
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        bar.close()
        return uncond_embeddings_list,latent_cur

    def prompt_optimization_cond(self, latent_list, cond_embeddings, uncond_embeddings, num_inner_steps, epsilon, guidance_scale=7.5,
                               num_ddim_steps=50):
        cond_embeddings_list = []
        latent_cur = latent_list[-1]
        bar = tqdm(total=num_inner_steps * num_ddim_steps)
        for i in range(num_ddim_steps):
            cond_embeddings = cond_embeddings.clone().detach()
            cond_embeddings.requires_grad = True
            optimizer = Adam([cond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latent_list[len(latent_list) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)

            for j in range(num_inner_steps):
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()

                logs = {"loss": loss_item}
                bar.set_postfix(**logs)
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            cond_embeddings_list.append(cond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        bar.close()
        return cond_embeddings_list, latent_cur
    
    def get_original(
        self,
        model_output, 
        sample: torch.FloatTensor,
        timestep: int
    ):
        t = timestep
        scheduler = self.model.scheduler
        alpha_prod_t = scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        if scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {scheduler.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )
        return pred_original_sample
    
    def null_optimization_target(self, latent_cur,latent_target,cond_embeddings,num_inner_steps, epsilon,guidance_scale=7.5,num_ddim_steps=50):
        uncond_embeddings_list = []
        
        bar = tqdm(total=num_inner_steps * num_ddim_steps)
        for i in range(num_ddim_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            # latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
                
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
                latents_prev_rec = self.model.scheduler.get_original(noise_pred, latent_cur, t)
                
                loss = nnf.mse_loss(latents_prev_rec, latent_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
                    
            with torch.no_grad():
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                # x_t_min_1_loss = nnf.mse_loss(latents_prev_rec, latent_prev)        
                print(loss_item)
                
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        bar.close()
        return uncond_embeddings_list,latent_cur
    
    def invert(self, image_path: str, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt,ddim_inversion=50)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings
        
    
    def __init__(self, model, num_ddim_steps=50):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.model2 = copy.deepcopy(model)
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(num_ddim_steps)
        self.model2.scheduler.set_timesteps(num_ddim_steps)
        self.prompt = None
        self.context = None
        
    @torch.inference_mode()
    def decode_image(self, latents: torch.FloatTensor, **kwargs) -> List["PIL_IMAGE"]:
        scaled_latents = 1 / 0.18215 * latents
        image = [
            self.model.vae.decode(scaled_latents[i : i + 1]).sample for i in range(len(latents))
        ]
        image = torch.cat(image, dim=0)
        return image

    @torch.inference_mode()
    def torch_to_numpy(self, image) -> List["PIL_IMAGE"]:
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return image

    @torch.inference_mode()
    def latents_to_imgs(self,latents):
        x = self.decode_image(latents)
        x = self.torch_to_numpy(x)
        x = self.numpy_to_pil(x)
        return x

@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image'
):
    batch_size = len(prompt)
    ptp_utils.register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
        
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent
