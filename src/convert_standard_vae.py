import torch
from diffusers import AutoencoderKL
from safetensors.torch import load_file
import os
# # 1. 載入 .safetensors 權重
# vae_path = "DiscordChatBot/src/Models/ClearVAE_V2.3_fp16.safetensors"
# vae_state_dict = load_file(vae_path)

# # 2. 建立 AutoencoderKL 模型架構
# # 使用 Stable Diffusion 1.5 的預設 config（這會自動抓模型架構）
# vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")

# # 3. 套用權重
# missing, unexpected = vae.load_state_dict(vae_state_dict, strict=False)
# print("✅ Missing:", missing)
# print("✅ Unexpected:", unexpected)

# # 4. 儲存成 diffusers 格式
# vae.save_pretrained("DiscordChatBot/src/Models/VAE/ClearVAE_V2.3_fp16")
# print("🎉 已儲存為 diffusers 格式！")


root_dir = 'src/AI/VAE'
for root, dirs, files in os.walk(root_dir):
    for file in files:
        full_path = os.path.join(root, file)
        filename = full_path.split("/")[-1].split('.')[0]
        vae_state_dict = load_file(full_path)
        vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
        missing, unexpected = vae.load_state_dict(vae_state_dict, strict=False)
        print("✅ Missing:", missing)
        print("✅ Unexpected:", unexpected)
        vae.save_pretrained(f"src/Models/VAE/{filename}")
print(f"🎉 {filename} 已儲存為 diffusers 格式！")