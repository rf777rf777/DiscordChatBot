import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import random
from diffusers import StableDiffusionPipeline, AutoencoderKL, DPMSolverMultistepScheduler
import torch
from PIL import Image

# --- 參數設定 ---
model_dir = "src/Models/Anything-v3-0-better-vae"  # 你的主模型資料夾（已轉成 diffusers 格式）
#model_dir = "src/meinaunreal_v5"
vae_path = "src/temp/orangemixvaeReupload_v10.pt"  # VAE 模型（需要轉換格式）
#vae_path = "src/meinaunreal_v5/kl-f8-anime2.safetensors"  # VAE 模型

#seed = 311620705 
seed =  random.randint(0, 2**32 - 1)
guidance_scale = 7
steps = 20
width = 576
height = 1024


pipe = StableDiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.float32, safety_checker=None)  # ✅ 禁用 safety checker

# 使用 DPM++ 2M Karras 採樣器
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    use_karras_sigmas=True  # ✅ 開啟 Karras 分布
)

pipe = pipe.to("mps")

pipe.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32).to("mps")

pipe.load_textual_inversion("src/temp/easynegative.safetensors", token="easynegative", mean_resizing=False)
pipe.load_textual_inversion("src/temp/badhandv4.pt", token="badhandv4", mean_resizing=False)

pipe.load_lora_weights("src/temp/Lora/animetarotV51/animetarotV51.safetensors", adapter_name="animetarotV51")
pipe.set_adapters(["animetarotV51"])

pipe.enable_attention_slicing()

#positive_prompt = "masterpiece, best quality, 1girl, solo, light smile, fire, red theme, alternate costume, <lora:animetarotV51:1>"
positive_prompt = "((masterpiece, best quality, thighhighs, long legs, beautiful legs)), 1girl, solo, light smile, electricity, purple theme, alternate costume, <lora:animetarotV51:1>"

negative_prompt = "easynegative, badhandv4, watermark"

#設定隨機種子
generator = torch.manual_seed(seed)

image = pipe(
    prompt=positive_prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=20,
    guidance_scale=7,
    generator=generator,
    height=height,
    width=width
).images[0]

result_path = f"src/results/tarotV51-{seed}.png"
image.save(result_path)
Image.open(result_path).show()