import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import random
from core.config import settings
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

def create_sd_image(model_dir = model_dir, 
                    torch_dtype = torch.float32, 
                    safety_checker=None,
                    target_sampler_name="euler_a"):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_dir = model_dir,
        torch_dtype=torch_dtype,
        safety_checker=safety_checker)  # 禁用 safety checker

    sampler_factory = _get_scheduler_factory(target_sampler_name)
    pipe.scheduler = sampler_factory(pipe.scheduler.config)

def _get_scheduler_factory(sampler_name: str):
    sampler_name = sampler_name.lower()
    sampler_info = settings.SD_SAMPLERS.get(sampler_name)

    if not sampler_info:
        raise ValueError(
            f"未知的採樣器 '{sampler_name}'。\n"
            f"可用選項：{', '.join(settings.SD_SAMPLERS.keys())}"
        )

    SchedulerCls = sampler_info["cls"]
    extra_kwargs = sampler_info.get("kwargs", {})

    return lambda config: SchedulerCls.from_config(config, **extra_kwargs)
