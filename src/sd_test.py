from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
from diffusers import StableDiffusionPipeline
import torch

#ckpt路徑
ckpt_path = "src/Anything-v3-0-better-vae.ckpt"
yaml_path = "src/v1-inference.yaml"
output_path = "src/Anything-v3-0-better-vae"

pipe = download_from_original_stable_diffusion_ckpt( 
    checkpoint_path_or_dict=ckpt_path,
    original_config_file=yaml_path,
    from_safetensors=False  # 若是 .safetensors 則設 True
)

pipe.save_pretrained(output_path)

pipe = StableDiffusionPipeline.from_pretrained(output_path, torch_dtype=torch.float16, safety_checker=None)  # ✅ 禁用 safety checker
pipe = pipe.to("mps")
pipe.enable_attention_slicing()

# positive_prompt = "masterpiece, best quality, 1girl, silver hair, glowing armor, futuristic background"
# negative_prompt = "lowres, bad anatomy, worst quality, blurry, extra hands, ugly, watermark"

positive_prompt = "masterpiece, best quality, "
negative_prompt = "lowres, bad anatomy, worst quality, blurry, extra hands, ugly, watermark"

#設定隨機種子
generator = torch.manual_seed(42)

image = pipe(
    prompt=positive_prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=8.5,
    generator=generator,
    height=512,
    width=512
).images[0]

image.save("result2.png")