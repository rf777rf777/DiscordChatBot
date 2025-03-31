from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
import torch
from PIL import Image

model_dir = "src/Anything-v3-0-better-vae"
model_path = "path/to/meinaunreal_v3.safetensors"     # 你需要轉換成 Diffusers 格式或使用 Checkpoint Loader
vae_path = "src/temp/orangemixvaeReupload_v10.pt"                 # 可以透過 pipeline.to(torch_dtype=torch.float16).vae.load_state_dict(...) 加載
negative_prompt = "easynegative"
prompt = "sketch,masterpiece,best quality"
seed = 2245550560
guidance_scale = 7
steps = 20
width = 512
height = 768


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