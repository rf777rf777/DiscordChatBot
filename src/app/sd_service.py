import os, io, asyncio
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import random
from core.config import settings
from diffusers import StableDiffusionPipeline, AutoencoderKL

import torch
import gc

# model_dir = "src/Models/SD/BaseModels/qchanAnimeMix_v40" 
# vae_path = ""

async def create_sd_image(mode_name, 
                    vae_name,
                    prompt,
                    sampler_name="euler_a",
                    seed=None,
                    steps=40,
                    torch_dtype = torch.float32, 
                    # 禁用 safety checker
                    safety_checker=None):
    await asyncio.sleep(0.2)

    pipe = StableDiffusionPipeline.from_pretrained(
        f'src/Models/SD/BaseModels/{mode_name}',
        torch_dtype=torch_dtype,
        safety_checker=safety_checker).to("mps")
    
    if vae_name:
        pipe.vae = AutoencoderKL.from_pretrained(f'src/Models/SD/VAE/{vae_name}', 
                                                 torch_dtype=torch_dtype).to("mps")

    sampler_factory = _get_scheduler_factory(sampler_name)
    pipe.scheduler = sampler_factory(pipe.scheduler.config)

    embedding_dirs = [
        ("src/Models/SD/Embeddings/easynegative.safetensors", "easynegative"),
        ("src/Models/SD/Embeddings/badhandv4.pt", "badhandv4"),
        ("src/Models/SD/Embeddings/HDA_Ahegao.pt", "HDA_Ahegao")
    ]
    
    for path, trigger_word in embedding_dirs:
        pipe.load_textual_inversion(path, token=trigger_word, mean_resizing=False)

    pipe.enable_attention_slicing()
    
    lora_weights = { "animetarotV51": 0.9 
                    # ,"hyouka_offset": 0.2
                    }
    pipe.load_lora_weights("src/Models/SD/Loras/animetarotV51.safetensors", 
                            adapter_name="animetarotV51")
    
    # pipe.load_lora_weights("src/Models/SD/Loras/hyouka_offset.safetensors",
    #                      adapter_name="hyouka_offset")

    pipe.set_adapters(list(lora_weights.keys()), list(lora_weights.values()))
    
    if not seed:
        seed =  random.randint(0, 2**32 - 1)
    
    #要產生的圖片數量
    num_images = 1
    #設定隨機種子
    generator = [torch.manual_seed(seed + i) for i in range(num_images)]
    guidance_scale = 7
    steps = steps #40
    width = 576
    height = 1024

    imageList = []
    for i in range(1):
        generator = torch.manual_seed(seed + i)
        image = pipe(
            prompt=prompt,
            negative_prompt= "easynegative, badhandv4, watermark",
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=height,
            width=width,
            num_images_per_prompt=num_images
        ).images[0]
        imageList.append(image)
        torch.mps.empty_cache()
        gc.collect()
        
    # 儲存影像至記憶體(bytes)
    img_byte_arr = io.BytesIO()
    imageList[0].save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr

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
