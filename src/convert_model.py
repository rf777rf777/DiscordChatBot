from diffusers.pipelines.stable_diffusion.convert_from_ckpt import create_vae_diffusers_config, convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint, convert_ldm_clip_checkpoint, download_from_original_stable_diffusion_ckpt
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL
import torch
import os
from safetensors.torch import load_file

root_dir = 'src/AI/Models'
yaml_path = "src/AI/v1-inference.yaml"
for root, dirs, files in os.walk(root_dir):
    for file in files:
        full_path = os.path.join(root, file)
        save_folder_name = file.split('.')[0]
        save_folder_path = f'src/Models/SD/BaseModels/{save_folder_name}'
        pipeline = download_from_original_stable_diffusion_ckpt(
            checkpoint_path_or_dict=full_path,
            original_config_file=yaml_path,  # 配套的 yaml
            from_safetensors=True
        )
        pipeline.save_pretrained(save_folder_path)
  
# pipeline = download_from_original_stable_diffusion_ckpt(
#     checkpoint_path_or_dict="DiscordChatBot/src/Models/malkmilfmix_v20.safetensors",
#     original_config_file="DiscordChatBot/src/Models/v1-inference.yaml",  # 配套的 yaml
#     from_safetensors=True
# )
# pipeline.save_pretrained("DiscordChatBot/src/Models/malkmilfmix_v20")

#v1-inference
# pipe = download_from_original_stable_diffusion_ckpt(
#      checkpoint_path_or_dict="DiscordChatBot/src/Models/anypastelAnythingV45_anypastelAnythingV45.safetensors",
#      original_config_file="DiscordChatBot/src/Models/v1-inference.yaml",  # 配套的 yaml
#      from_safetensors=True
# )

# pipe.save_pretrained("DiscordChatBot/src/Models/anypastelAnythingV45_anypastelAnythingV45")

#Convert VAE(.pt)

# import torch
# from diffusers import AutoencoderKL
# from pathlib import Path


# var_path = "DiscordChatBot/src/Models/orangemixvaeReupload_v10.pt"
# # 載入 .pt
# ckpt = torch.load(var_path, map_location="cpu")

# # 如果是 lightning 格式，就解包出 state_dict
# if "state_dict" in ckpt:
#     state_dict = ckpt["state_dict"]
# else:
#     state_dict = ckpt

# # 去掉前綴 "first_stage_model." 或 "vae."
# state_dict = {k.replace("first_stage_model.", "").replace("vae.", ""): v for k, v in state_dict.items()}

# # 初始化 diffusers 的 AutoencoderKL
# vae = AutoencoderKL()
# vae.load_state_dict(state_dict, strict=False)

# # 儲存為 diffusers-compatible 格式
# vae.save_pretrained("DiscordChatBot/src/Models/converted-orangemixvaeReupload_v10")

#Convert VAE (Large)

# import torch
# from diffusers.models import AutoencoderKL
# from pathlib import Path

# ckpt = torch.load("DiscordChatBot/src/Models/orangemixvaeReupload_v10.pt", map_location="cpu")
# state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
# state_dict = {k.replace("first_stage_model.", "").replace("vae.", ""): v for k, v in state_dict.items()}

# # 🧠 自訂較大 VAE 結構
# vae = AutoencoderKL(
#     in_channels=3,
#     out_channels=3,
#     down_block_types=(
#         "DownEncoderBlock2D",
#         "DownEncoderBlock2D",
#         "DownEncoderBlock2D",
#         "DownEncoderBlock2D",
#     ),
#     up_block_types=(
#         "UpDecoderBlock2D",
#         "UpDecoderBlock2D",
#         "UpDecoderBlock2D",
#         "UpDecoderBlock2D",
#     ),
#     block_out_channels=(128, 256, 512, 512),  # 🟡 這是關鍵，對應你 orangemix 的架構
#     latent_channels=4,
#     sample_size=512
# )

# vae.load_state_dict(state_dict, strict=False)
# vae.save_pretrained("DiscordChatBot/src/Models/converted-orangemixvaeReupload_v10")
