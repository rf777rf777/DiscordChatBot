from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
import torch
from PIL import Image

# --- 參數設定 ---
model_dir = "src/Models/Anything-v3-0-better-vae"  # 你的主模型資料夾（已轉成 diffusers 格式）
#model_dir = "src/meinaunreal_v5"
vae_path = "src/temp/orangemixvaeReupload_v10.pt"  # VAE 模型（需要轉換格式）
#vae_path = "src/meinaunreal_v5/kl-f8-anime2.safetensors"  # VAE 模型

negative_prompt = "easynegative, badhandv4"
prompt = "masterpiece, best quality, 1girl, solo, light smile, fire, red theme, alternate costume, <lora:animetarotV51:1>"
seed = 374149656
guidance_scale = 7
steps = 20
width = 576
height = 1024


pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_dir, torch_dtype=torch.float32, safety_checker=None)  # ✅ 禁用 safety checker
pipe = pipe.to("mps")
pipe.enable_attention_slicing()

pipe.load_lora_weights("src/temp/animeoutlineV4_16.safetensors", adapter_name="animeoutline")
pipe.set_adapters(["animeoutline"])
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
generator = torch.manual_seed(seed)


init_image = Image.open("src/temp/dog.jpg").convert("RGB").resize((width, height))

# image = pipe(
#     prompt="",
#     image=init_image,
#     strength=0.7,         # 改變程度（0.0 = 幾乎不變，1.0 = 完全再生成）
#     guidance_scale=0.8,   # prompt 影響程度
#     num_inference_steps=30
# ).images[0]

image = pipe(
    image=init_image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=steps,
    guidance_scale=guidance_scale,
    generator=generator,
    #num_images_per_prompt=1,
    strength=0.7
).images[0]


image.save("outline.png")