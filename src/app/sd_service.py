import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import random
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler
from diffusers import PNDMScheduler, LMSDiscreteScheduler
from diffusers import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler
from diffusers import UniPCMultistepScheduler

import torch
from PIL import Image
import gc

model_dir = "src/Models/SD/BaseModels/qchanAnimeMix_v40" 
vae_path = ""

def create_sd_image():
    pipe = StableDiffusionPipeline.from_pretrained(
    model_dir, 
    torch_dtype=torch.float32,
    safety_checker=None)  # 禁用 safety checker

    # sampler_factory = get_scheduler_factory(target_sampler_name)
    # pipe.scheduler = sampler_factory(pipe.scheduler.config)
