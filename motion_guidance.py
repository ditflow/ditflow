import os
import argparse
from pathlib import Path
import shutil
import numpy as np
import math
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.amp import GradScaler
from tqdm import tqdm
from transformers import logging
import torch.nn.functional as F
import imageio
from torchvision.io import read_video
from torchvision.transforms import ToPILImage
from PIL import Image
from torchvision.io import write_video
import gc
from typing import Union, List

from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.utils import export_to_video

from guidance_utils.custom_transformer import ControlledTransformer
from guidance_utils.custom_embeddings import prepare_rotary_positional_embeddings
from guidance_utils.custom_modules import ModuleWithGuidance, InjectionProcessor
from guidance_utils.motion_flow_utils import compute_motion_flow

# suppress partial model loading warning
logging.set_verbosity_error()

def isinstance_str(x: object, cls_name: Union[str, List[str]]):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.

    Useful for patching!
    """
    if type(cls_name) == str:
        for _cls in x.__class__.__mro__:
            if _cls.__name__ == cls_name:
                return True
    else:
        for _cls in x.__class__.__mro__:
            if _cls.__name__ in cls_name:
                return True
    return False

def clean_memory():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

def save_video(video, path):
    video_codec = "libx264"
    video_options = {
        "crf": "17",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
        "preset": "slow",  # Encoding preset (e.g., ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
    }   
    write_video(
        path,
        video,
        fps=10,
        video_codec=video_codec,
        options=video_options,
    )

def get_timesteps(timesteps, guidance_timestep_range, skip_timesteps=1):
    max_guidance_timestep, min_guidance_timestep = guidance_timestep_range
    num_inference_steps = len(timesteps)
    init_timestep = min(max_guidance_timestep, num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    t_end = min_guidance_timestep
    if t_end > 0:
        guidance_schedule = timesteps[t_start : -t_end : skip_timesteps]
    else:
        guidance_schedule = timesteps[t_start::skip_timesteps]
    return guidance_schedule

class Guidance(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device(config["device"])
        self.eta=0
        self.batch_size = 1
        self.num_inference_steps = config["num_inference_steps"]
        self._guidance_scale = self.config.guidance_scale

        print("Loading video model")
        if config.model_key=="THUDM/CogVideoX-2b":
            self.dtype = torch.float16
            self.pipe = CogVideoXPipeline.from_pretrained(config.model_key, torch_dtype=self.dtype).to("cuda")
            self.pipe.scheduler = CogVideoXDDIMScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")
            self.use_dynamic_cfg = False
        else:
            self.dtype = torch.bfloat16
            self.pipe = CogVideoXPipeline.from_pretrained(config.model_key, torch_dtype=self.dtype).to("cuda")
            self.pipe.scheduler = CogVideoXDPMScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")
            self.use_dynamic_cfg = True

        # Controlled transformer
        controlled_transformer = ControlledTransformer(**self.pipe.transformer.config)
        controlled_transformer.load_state_dict(self.pipe.transformer.state_dict())
        self.pipe.transformer = controlled_transformer.to(device=self.device, dtype=self.dtype)
        self.pipe.transformer.init_pos_embedding = self.pipe.transformer.init_pos_embedding.to(self.device)
        
        if self.config.enable_gradient_checkpointing:
            self.pipe.transformer.enable_gradient_checkpointing()
        
        self.pipe.scheduler.set_timesteps(self.num_inference_steps, device="cuda")
        self.timesteps = self.pipe.scheduler.timesteps
        self.guidance_schedule = get_timesteps(self.timesteps, self.config.guidance_timestep_range)

        ## Optimizations
        self.pipe.enable_model_cpu_offload()
        # self.pipe.enable_sequential_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

        self.vae = self.pipe.vae
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.transformer = self.pipe.transformer
        self.scheduler = self.pipe.scheduler
        print("video model loaded")

        self.generator = torch.Generator(device='cuda').manual_seed(config.seed)

        #### Pipeline setup - simplified from CogVideoX pipeline code ####
        height = config.height or self.transformer.config.sample_size * self.vae_scale_factor_spatial
        width = config.width or self.transformer.config.sample_size * self.vae_scale_factor_spatial
        num_videos_per_prompt = 1
        assert (height % 16 == 0) and (width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
        self.resolution = (width, height)
        self.config.text_seq_length = 226 # TODO: extract from pipeline
        self.video_length = config.video_length
        self.latent_num_frames = (self.video_length - 1) // self.pipe.vae_scale_factor_temporal + 1
        self.latent_height = height // self.pipe.vae_scale_factor_spatial
        self.latent_width = width // self.pipe.vae_scale_factor_spatial
        self.patch_size = self.pipe.transformer.config.patch_size
        self.patches_height = self.latent_height // self.patch_size
        self.patches_width = self.latent_width // self.patch_size

        self.pipe.check_inputs(
            config["target_prompt"],
            height,
            width,
            config["negative_prompt"],
            callback_on_step_end_tensor_inputs=None,
        )

        with torch.no_grad():
            self.source_embeds, _ = self.pipe.encode_prompt(
                config["source_prompt"],
                device=self.device,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=True,
                negative_prompt=config["negative_prompt"],
            )

            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                config["target_prompt"],
                device=self.device,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=True,
                negative_prompt=config["negative_prompt"],
            )
            self.guidance_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        latent_channels = self.transformer.config.in_channels
        self.init_latents = self.pipe.prepare_latents(
            self.batch_size * num_videos_per_prompt,
            latent_channels,
            self.video_length,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            self.generator,
        )

        self.extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(self.generator, self.eta)

        init_rope = (
            prepare_rotary_positional_embeddings(
                height, 
                width, 
                self.init_latents.size(1), 
                self.pipe.vae_scale_factor_spatial, 
                self.transformer.config.patch_size, 
                self.transformer.config.attention_head_dim,
                self.device
            )
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        self.transformer.init_rope = init_rope
        self.transformer.guidance_blocks = self.config.guidance_blocks

        # Path verification
        self.output_path = self.config['output_path']
        os.makedirs(self.output_path, exist_ok=True)

        embeds_path = os.path.join(self.output_path, "embeds")
        if self.config.inject_embeds:
            if not os.path.exists(embeds_path):
                raise FileNotFoundError(f"Embeds folder not found at {embeds_path}. Make sure to first run motion guidance without inject_embeds=True so that the trained embeddings can be stored. These can then be injected with a new prompt by running again with inject_embeds=True.")
        else:
            if os.path.exists(embeds_path):
                shutil.rmtree(embeds_path)
            os.makedirs(embeds_path, exist_ok=True)

        ## GUIDANCE SETUP
        self.motion_timestep = torch.tensor([0], device='cuda')

        self.register_guidance(block_idxs=self.config.guidance_blocks)
        self.register_attention_processor(block_idxs=list(range(len(self.transformer.transformer_blocks))))
        
        num_guidance_steps = self.config.guidance_timestep_range[0] - self.config.guidance_timestep_range[1] + 1
        self.lr_range = np.linspace(self.config.lr[0], self.config.lr[1], num_guidance_steps)
        
        print("Loading features from motion video")
        self.motion_latent = self.load_latent()
        if self.config.loss_type =='flow':
            self.motion_attn_features = self.load_attn_features()
        elif self.config.loss_type=='smm':
            self.motion_orig_features = self.load_features()
        elif self.config.loss_type=='moft':
            self.motion_orig_features, self.motion_channels = self.load_features(moft=True)

    def register_guidance(self, block_idxs):
        """Register guidance blocks to be able to save features in forward pass"""
        for out_i in block_idxs:
            block_name = f"block_{out_i}"
            self.transformer.transformer_blocks[out_i] = ModuleWithGuidance(
                self.transformer.transformer_blocks[out_i],
                self.latent_height,
                self.latent_width,
                self.pipe.transformer.config.patch_size,
                block_name=block_name,
                num_frames=self.latent_num_frames
            )
    
    def register_attention_processor(self, block_idxs):
        """Register attention processors"""
        for out_i in block_idxs:
            block_name = f"block_{out_i}_attn1_processor"
            processor = InjectionProcessor(block_name=block_name)
            self.transformer.transformer_blocks[out_i].attn1.set_processor(processor)
    
    @property
    def guidance_scale(self):
        return self._guidance_scale
    
    @torch.no_grad()
    def load_latent(self):
        """ Load video and pass through VAE encoder"""
        data_path = self.config.video_path

        if data_path.endswith(".mp4"):
            video = read_video(data_path, pts_unit="sec")[0].permute(0, 3, 1, 2).cuda() / 255
            video = [ToPILImage()(video[i]).resize(self.resolution) for i in range(video.shape[0])]
        else:
            images = list(Path(data_path).glob("*.png")) + list(Path(data_path).glob("*.jpg"))
            images = sorted(images, key=lambda x: int(x.stem.split('f')[-1]))
            video = [Image.open(img).resize(self.resolution).convert('RGB') for img in images]

        video = video[: self.config.video_length]
        save_video([np.array(img) for img in video], str(Path(self.config.output_path) / f"original.mp4"))

        video = self.pipe.video_processor.preprocess_video(video)
        video = video.to(self.dtype).to("cuda")
        latents = self.vae.config.scaling_factor * self.vae.encode(video)[0].sample()
        latents = latents.permute(0,2,1,3,4)
        
        return latents

    @torch.no_grad()
    def load_features(self, moft=False):
        """Load saved features for motion video
        moft: Whether to compute motion channels for MOFT method
        """
        motion_features = {}
        motion_channels = {}
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            self.transformer(
                hidden_states=self.motion_latent.to('cuda'),
                encoder_hidden_states=self.source_embeds,
                timestep=self.motion_timestep,
                return_dict=False,
            )
        for block_id in self.config.guidance_blocks:
            module = self.transformer.transformer_blocks[block_id]
            orig_features = module.saved_features
            motion_features[module.block_name] = orig_features
            
            if moft:
                orig_norm = orig_features - torch.mean(orig_features, axis=0)[None]
                num_frames, c, h, w = orig_norm.shape
                channels = orig_norm.permute(0,2,3,1).reshape(-1, c)
                _, _, Vt = torch.linalg.svd(channels.to(torch.float32), full_matrices=False)
                top_n = list(torch.argsort(torch.abs(Vt[0]), descending=True)[:int(self.config.prop_motion*c)])
                motion_channels[module.block_name] = top_n
        if moft:
            return motion_features, motion_channels
        return motion_features
    
    @torch.no_grad()
    def load_attn_features(self):
        """ 🔍 AMF Extraction """
        for block_id in self.config.guidance_blocks:
            self.transformer.transformer_blocks[block_id].attn1.processor.inject_kv = False
            self.transformer.transformer_blocks[block_id].attn1.processor.copy_kv = True
        
        attn_features = {}
        # Store keys and queries for all attention blocks
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            self.transformer(
                hidden_states=self.motion_latent,
                encoder_hidden_states=self.source_embeds,
                timestep=self.motion_timestep,
                return_dict=False,
            )
        for block_id in self.config.guidance_blocks:
            module = self.transformer.transformer_blocks[block_id].attn1.processor
            attn_features[module.block_name] = compute_motion_flow(module.query, module.key, 
                                                    h=self.patches_height, 
                                                    w=self.patches_width, 
                                                    temp=self.config.motion_temp, 
                                                    argmax=self.config.argmax_motion_flow)
        
            self.transformer.transformer_blocks[block_id].attn1.processor.copy_kv = False
            self.transformer.transformer_blocks[block_id].attn1.processor.key = None
            self.transformer.transformer_blocks[block_id].attn1.processor.query = None
            self.transformer.transformer_blocks[block_id].attn1.processor.value = None
        
        return attn_features
    
    def change_mode(self, train=True):
        """During guidance training, pass through later output blocks to reduce unnecessary computation"""
        @staticmethod
        def dummy_pass(*args, **kwargs):
            if len(args) == 0:
                return kwargs["hidden_states"], kwargs['encoder_hidden_states']
            elif len(args)<2:
                return args[0]
            else:
                return args[0], args[1]
        
        def set_forward_mode(module, pass_through=True):
            if pass_through:
                module.original_forward = module.forward
                module.forward = dummy_pass
            else:
                try:
                    module.forward = module.original_forward
                except AttributeError:
                    pass
        
        # Switch mode
        if len(self.config.guidance_blocks) != 0:
            index = max(self.config.guidance_blocks)
            for i, block in enumerate(self.transformer.transformer_blocks):
                if i > index:
                    set_forward_mode(block, pass_through=train)
        for block in [self.transformer.norm_out, self.transformer.norm_final]:
            set_forward_mode(block, pass_through=train)
    
    ############################## GUIDANCE LOSS FUNCTIONS ##############################
    def compute_motion_flow_loss(self, x, ts, rope=None, pos_emb=None):
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            self.transformer(
                hidden_states=x,
                encoder_hidden_states=self.guidance_embeds[1:2],
                timestep=ts.expand(x.shape[0]).to('cuda'),
                rope=rope,
                pos_embedding=pos_emb,
                return_dict=False,
            )

        # Attention guidance
        total_loss = 0
        for block_id in self.config.guidance_blocks:
            module = self.transformer.transformer_blocks[block_id].attn1.processor
            motion_flow = compute_motion_flow(module.query, module.key, 
                                                h=self.patches_height, 
                                                w=self.patches_width, 
                                                temp=self.config.motion_temp, 
                                                nframes=self.latent_num_frames)
            
            ref_motion_flow = self.motion_attn_features[module.block_name].detach()

            # Threshold loss on motion flow (d x 1350 x 2) for d displacement maps
            if self.config.threshloss:
                flow_norms = torch.norm(ref_motion_flow, dim=-1)
                idxs = flow_norms > 0
                attn_loss = F.mse_loss(ref_motion_flow[idxs], motion_flow[idxs])
            else:
                attn_loss = F.mse_loss(ref_motion_flow, motion_flow)

            total_loss += attn_loss
        if len(self.config.guidance_blocks) > 0:
            total_loss /= len(self.config.guidance_blocks)
        
        for block_id in self.config.guidance_blocks:
            self.transformer.transformer_blocks[block_id].attn1.processor.query = None
            self.transformer.transformer_blocks[block_id].attn1.processor.key = None
            self.transformer.transformer_blocks[block_id].attn1.processor.value = None
        return total_loss

    def compute_moft_loss(self, x, ts, rope=None, pos_emb=None):
        """Motion Feature (MOFT) Loss"""
        def compute_MOFT(orig, 
                 target, 
                 motion_channels,
                 ):
            # Compute motion channels from current video only (T x C x H x W) and extract top prop_motion% channels
            orig_norm = orig - torch.mean(orig, axis=0)[None]
            target_norm = target - torch.mean(target, axis=0)[None]

            features_diff_loss = 0
            for f in range(orig_norm.shape[0]):
                top_n = motion_channels
                orig_moft_f = orig_norm[f, top_n]
                target_moft_f = target_norm[f, top_n]
                features_diff_loss += 1 - F.cosine_similarity(target_moft_f, orig_moft_f.detach(), dim=0).mean()

            features_diff_loss /= orig_norm.shape[0]
            return features_diff_loss
        
        target_features = {}
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            self.transformer(
                hidden_states=x,
                encoder_hidden_states=self.guidance_embeds[1:2],
                timestep=ts.expand(x.shape[0]).to('cuda'),
                rope=rope,
                pos_embedding=pos_emb,
                return_dict=False,
            )

        total_loss = 0
        for block_id in self.config.guidance_blocks:
            module = self.transformer.transformer_blocks[block_id]
            target_name = module.block_name
            target_features = module.saved_features

            orig_features = self.motion_orig_features[target_name]
            motion_channels = self.motion_channels[target_name]

            loss = compute_MOFT(
                orig_features.detach(), 
                target_features,
                motion_channels,
            )
            total_loss += loss
        if len(self.config.guidance_blocks) > 0:
            total_loss /= len(self.config.guidance_blocks)
        
        return total_loss

    def compute_smm_loss(self, x, ts, rope=None, pos_emb=None):
        """Spatial Marginal Mean Loss"""
        def compute_SMM(orig, 
                        target,
                        ):
            # Take spatial mean
            orig_smm = orig.mean(dim=(-1, -2), keepdim=True)
            target_smm = target.mean(dim=(-1, -2), keepdim=True)

            features_diff_loss = 0
            for f in range(orig_smm.shape[0]):
                orig_anchor = orig_smm[f]
                target_anchor = orig_smm[f]
                orig_diffs = orig_smm - orig_anchor  # t d 1 1
                target_diffs = target_smm - target_anchor  # t d 1 1
                features_diff_loss += 1 - F.cosine_similarity(target_diffs, orig_diffs.detach(), dim=0).mean()
            features_diff_loss /= orig_smm.shape[0]
            return features_diff_loss
    
        target_features = {}
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            self.transformer(
                hidden_states=x,
                encoder_hidden_states=self.guidance_embeds[1:2],
                timestep=ts.expand(x.shape[0]).to('cuda'),
                rope=rope,
                pos_embedding=pos_emb,
                return_dict=False,
            )
        total_loss = 0
        for block_id in self.config.guidance_blocks:
            module = self.transformer.transformer_blocks[block_id]
            target_name = module.block_name
            target_features = module.saved_features

            orig_features = self.motion_orig_features[target_name]

            loss = compute_SMM(
                orig_features.detach(), 
                target_features,
            )
            total_loss += loss
        if len(self.config.guidance_blocks) > 0:
            total_loss /= len(self.config.guidance_blocks)
        
        return total_loss

    ############################## GUIDED DENOISING METHODS ##############################
    def guidance_step(self, x, i, t, mode, loss_type):
        """⚙️ Motion Optimization
        Optimisation at single denoising step for number of steps
        mode: rope, posemb, latent (object being optimized)
        loss_type: flow, moft, smm (loss computation)
        """
        for block_id in self.config.guidance_blocks:
            self.transformer.transformer_blocks[block_id].attn1.processor.inject_kv = False
            self.transformer.transformer_blocks[block_id].attn1.processor.copy_kv = True
        
        lr = self.lr_range[i]
        optimized_emb = None
        optimized_rope = None
        self.change_mode(train=True)
        
        scaler = GradScaler()

        if loss_type == "flow":
            loss_method = self.compute_motion_flow_loss
        elif loss_type == "moft":
            loss_method = self.compute_moft_loss
        elif loss_type == "smm":
            loss_method = self.compute_smm_loss
        else:
            print("Invalid loss type")
        
        if mode=="rope":
            if self.transformer.trainable_rope is None:
                optimized_rope = torch.stack([self.transformer.init_rope, self.transformer.init_rope], dim=0)
            else:
                optimized_rope = self.transformer.trainable_rope
            
            optimized_rope = optimized_rope.clone().detach().to(dtype=torch.float32, device=self.device).requires_grad_(True)
            optimizer = torch.optim.Adam([optimized_rope], lr=lr)

            for step_i in tqdm(range(self.config.optimization_steps)):
                optimizer.zero_grad()

                total_loss = loss_method(x, t, rope=optimized_rope)
                
                if self.config.verbose:
                    print(f"Loss t={t}: {total_loss.item()}")
                scaler.scale(total_loss).backward()

                scaler.step(optimizer)
                scaler.update()
                clean_memory()
            
            self.transformer.trainable_rope = optimized_rope.detach()
            if self.config.save_embeds:
                os.makedirs(os.path.join(self.output_path, 'embeds'), exist_ok=True)
                torch.save(optimized_rope.detach(), os.path.join(self.output_path, 'embeds', f"rope_{t}.pt"))
            optimized_x = x
        elif mode == "posemb":
            if self.transformer.trainable_pos_embedding is None:
                text_seq_length = self.config.text_seq_length
                seq_length = self.patches_height * self.patches_width * self.latent_num_frames
                optimized_emb = self.transformer.init_pos_embedding[:, text_seq_length:(text_seq_length+seq_length)].clone().detach().to(dtype=torch.float32, device=self.device).requires_grad_(True)
            else:
                optimized_emb = self.transformer.trainable_pos_embedding.clone().detach().to(dtype=torch.float32, device=self.device).requires_grad_(True)

            optimizer = torch.optim.Adam([optimized_emb], lr=lr)

            for step_i in tqdm(range(self.config.optimization_steps)):
                optimizer.zero_grad()

                total_loss = loss_method(x, t, pos_emb=optimized_emb)

                if self.config.verbose:
                    print(f"Loss t={t}: {total_loss.item()}")
                scaler.scale(total_loss).backward()

                scaler.step(optimizer)
                scaler.update()
                clean_memory()
            self.transformer.trainable_pos_embedding = optimized_emb.detach()
            if self.config.save_embeds:
                os.makedirs(os.path.join(self.output_path, 'embeds'), exist_ok=True)
                torch.save(optimized_emb.detach(), os.path.join(self.output_path, 'embeds', f"posemb_{t}.pt"))
            optimized_x = x
        elif mode=="latent":
            optimized_x = x.clone().detach().to(dtype=torch.float32).requires_grad_(True)
            optimizer = torch.optim.Adam([optimized_x], lr=lr)

            for step_i in tqdm(range(self.config.optimization_steps)):
                optimizer.zero_grad()

                total_loss = loss_method(optimized_x, t)
                
                if self.config.verbose:
                    print(f"Loss t={t}: {total_loss.item()}")
                scaler.scale(total_loss).backward()

                scaler.step(optimizer)
                scaler.update()
            
            if self.config.save_embeds:
                os.makedirs(os.path.join(self.output_path, 'embeds'), exist_ok=True)
                torch.save(optimized_x, os.path.join(self.output_path, 'embeds', f"latent_{t}.pt"))
                
        self.change_mode(train=False)
        return optimized_x.detach(), optimized_emb, optimized_rope

    @torch.no_grad()
    def denoise_step(self, latents, i, prompt_embeds, old_pred_original_sample, pos_emb=None, rope=None):

        t = self.timesteps[i]

        latent_model_input = latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        ts = t.expand(latent_model_input.shape[0]).to('cuda')

        noise_pred_text = self.transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds[1:2],
            timestep=ts,
            return_dict=False,
            pos_embedding=pos_emb,
            rope=rope,
        )[0]
        noise_pred_text = noise_pred_text.float()

        noise_pred_uncond = self.transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds[:1],
            timestep=ts,
            return_dict=False,
            pos_embedding=pos_emb,
            rope=rope,
        )[0]
        noise_pred_uncond = noise_pred_uncond.float()


        if self.use_dynamic_cfg:
            self._guidance_scale = 1 + self.config.guidance_scale * (
                (1 - math.cos(math.pi * ((self.num_inference_steps - t.item()) / self.num_inference_steps) ** 5.0)) / 2
            )

        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        if not isinstance(self.scheduler, CogVideoXDPMScheduler):
            # CogVideo-2B
            latents = self.scheduler.step(noise_pred, t, latents, **self.extra_step_kwargs, return_dict=False)[0]
        else:
            latents, old_pred_original_sample = self.scheduler.step(
                noise_pred,
                old_pred_original_sample,
                t,
                self.timesteps[i - 1] if i > 0 else None,
                latents,
                **self.extra_step_kwargs,
                return_dict=False,
            )
        latents = latents.to(prompt_embeds.dtype)

        return latents, old_pred_original_sample

    @torch.no_grad()
    @torch.autocast(device_type="cuda")
    def run(self, pos_emb=None, rope=None):
        clean_memory()
        latents = self.init_latents
        x0_prev = None # for DPM-solver++
        
        for i, t in enumerate(tqdm(self.timesteps, desc="Sampling")):
            is_guidance_step = t in self.guidance_schedule
            # Clear embeddings after guidance phase
            if not is_guidance_step:
                pos_emb = rope = None
            
            # KV Injection
            for block_id in self.config.injection_blocks:
                processor = self.transformer.transformer_blocks[block_id].attn1.processor
                processor.inject_kv = False
                processor.copy_kv = True

            if is_guidance_step:
                # Store KV from motion video in injection_blocks
                noise = self.init_latents
                noisy_latent = self.scheduler.add_noise(self.motion_latent, noise, t)
                noisy_latent = self.scheduler.scale_model_input(noisy_latent, t)
                
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    self.transformer(
                        hidden_states=noisy_latent,
                        encoder_hidden_states=self.guidance_embeds[1:2],
                        timestep=t.expand(noisy_latent.shape[0]).to('cuda'),
                        return_dict=False,
                    )
                
                for block_id in self.config.injection_blocks:
                    processor = self.transformer.transformer_blocks[block_id].attn1.processor
                    processor.inject_kv = True
                    processor.copy_kv = False
            
            # Apply guidance if needed
            with torch.enable_grad():
                if is_guidance_step and self.config.guidance_blocks:
                    if not self.config.inject_embeds:
                        latents, pos_emb, rope = self.guidance_step(latents, i, t, 
                                                                    mode=self.config.guidance_mode, loss_type=self.config.loss_type)
                    else:
                        # 🔄 Zero-shot Motion Injection - Load pre-computed embeddings
                        embeds_path = os.path.join(self.output_path, "embeds")
                        if self.config.guidance_mode == "rope":
                            rope = torch.load(os.path.join(embeds_path, f"rope_{t}.pt")).to(dtype=self.dtype, device=self.device)
                        elif self.config.guidance_mode == "posemb":
                            pos_emb = torch.load(os.path.join(embeds_path, f"posemb_{t}.pt")).to(dtype=self.dtype, device=self.device)
                        elif self.config.guidance_mode == "latent":
                            latents = torch.load(os.path.join(embeds_path, f"latent_{t}.pt")).to(dtype=self.dtype, device=self.device)
            
            # Perform denoising step
            latents, x0_prev = self.denoise_step(
                latents, 
                i, 
                self.guidance_embeds,
                x0_prev,
                pos_emb=pos_emb,
                rope=rope,
            )
        
        # Decode and save results
        with torch.no_grad():
            decoded_frames = self.pipe.decode_latents(latents)
        video = self.pipe.video_processor.postprocess_video(video=decoded_frames, output_type='pil')[0]

        result_name = f"results"
        if self.config.inject_embeds:
            result_name += '_inject_embeds'
        if self.config.save_format=="frames":
            Path(self.config["output_path"], result_name).mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(video):
                frame.save(Path(self.config["output_path"], result_name, f"{i:04d}.png"))
        elif self.config.save_format=="gif":
            imageio.mimsave(str(Path(self.config["output_path"]) / f"{result_name}.gif"), video)
        elif self.config.save_format=="mp4":
            export_to_video(video, str(Path(self.config["output_path"]) / f"{result_name}.mp4"), fps=8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video_path", type=str, required=True, help="Motion video path to transfer motion from (.mp4 or directory of .png)")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Prompt for new generation")

    parser.add_argument("--model", type=str, default="5b", choices=['5b','2b'])
    parser.add_argument("-n", "--video_length", type=int, default=24, help="Load the first n frames of the video in video_path")
    parser.add_argument("--negative_prompt", type=str, default="bad quality, distortions, unrealistic, distorted image, watermark, signature", help="Negative prompt for new generation")
    parser.add_argument("--loss_type", type=str, default="flow", choices=["flow", "moft", "smm"], help="Use MOFT or SMM for guidance")
    parser.add_argument("--opt_mode", type=str, default="latent", choices=["latent", "emb"])
    parser.add_argument("--no_guidance", action="store_true", help="Disable guidance")
    parser.add_argument("--no_injection", action="store_true", help="Disable KV injection")
    parser.add_argument("--inject_embeds", action="store_true", help="Inject previously trained embeddings in embeds/ into the new generation specified by the prompt argument")
    parser.add_argument("--output_path", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save_format", type=str, default="mp4", choices=["mp4", "gif", "frames"])
    parser.add_argument("--verbose", action="store_true", help="Print loss values") 
    opt = parser.parse_args()

    config = OmegaConf.load(f"configs/guidance_config.yaml")

    if opt.no_injection:
        config.injection_blocks = []

    cli_config = {
        'model_key': f"THUDM/CogVideoX-{opt.model}",
        'video_path': opt.video_path,
        'target_prompt': opt.prompt,
        'negative_prompt': opt.negative_prompt,
        'video_length': opt.video_length,
        'output_path': opt.output_path,
        'seed': opt.seed,
        'opt_mode': opt.opt_mode,
        'loss_type': opt.loss_type,
        'save_format': opt.save_format,
        'save_embeds': True,
        'inject_embeds': opt.inject_embeds,
        'verbose': opt.verbose,
    }
    config = OmegaConf.merge(config, cli_config)

    # Model-specific arguments
    config['guidance_blocks'] = config[f'guidance_blocks_{opt.model}']
    if opt.no_guidance:
        config['guidance_blocks'] = []
    
    if config.opt_mode == "latent":
        config.guidance_mode = "latent"
    elif config.opt_mode == "emb":
        if opt.model == "5b":
            config.guidance_mode = "rope"
        elif opt.model == "2b":
            config.guidance_mode = "posemb"

    Path(config["output_path"]).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, Path(config["output_path"]) / "config.yaml")

    guidance = Guidance(config)
    guidance.run()