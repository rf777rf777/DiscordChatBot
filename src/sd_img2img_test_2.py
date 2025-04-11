import os
# os.environ["PYTORCH_ENABLE_CUDA_FALLBACK"] = "1"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# 建一個你有權限的 temp 資料夾
# os.environ["TMPDIR"] = "/home/rd/workspace/LLM_test/DiscordChatBot/tmp"
# os.makedirs(os.environ["TMPDIR"], exist_ok=True)

import random
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionControlNetImg2ImgPipeline, ControlNetModel,DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, AutoencoderKL

# --- 參數設定 ---
#model_dir = "src/Models/SD/BaseModels/anypastelAnythingV45_anypastelAnythingV45"  # 你的主模型資料夾（已轉成 diffusers 格式）
#model_dir = "src/Models/SD/BaseModels/meinaunreal_v5"
#model_dir = "src/Models/SD/BaseModels/hadrianDelice_deliceV20"
#model_dir = "src/Models/SD/BaseModels/meinamix_v12Final"
model_dir = "src/Models/SD/BaseModels/qchanAnimeMix_v40"
#vae_path = "DiscordChatBot/src/Models/converted-orangemixvaeReupload_v10"  # VAE 模型（需要轉換格式）
#vae_path = "src/Models/SD/VAE/klF8Anime2VAE_klF8Anime2VAE"
vae_path = ""

def get_image_resized_info(image_path: str, bigger:bool=False, smaller:bool=False) -> tuple[str , Image.Image]:
    img = Image.open(image_path)
    w, h = img.size
    if bigger:
        resized_img = img.resize((w*2,h*2), Image.BICUBIC)
        return img.filename.split('/')[-1], resized_img
    if smaller:
        resized_img = img.resize((w//2,h//2), Image.LANCZOS)
        return img.filename.split('/')[-1], resized_img
    # --- 計算新的大小 ---
    if w == h:
        new_size = (512, 512)
    elif w > h:
        new_w = 768
        new_h = int(h * (768 / w))
        new_size = (new_w, new_h)
    else:
        new_h = 768
        new_w = int(w * (768 / h))
        if new_w > 512:
            new_w = 512
        new_size = (new_w, new_h)
    
    # BICUBIC適合放大圖像
    # LANCZOS適合縮小圖像
    resized_img = img.resize(new_size, Image.LANCZOS)

    return img.filename.split('/')[-1], resized_img

# --- 載入原圖進行縮放 ---
original_image = Image.open("src/inputs/cosplay.jpg")
info = get_image_resized_info("src/inputs/cosplay.jpg", True)
init_image = info[1].convert("RGB")
init_image_name = info[0]

def get_canny_img(image_name: str):
    canny_path = f'src/inputs/{image_name}-canny.png'
    if os.path.exists(canny_path):
        return canny_path, Image.open(canny_path)
    else: 
        # --- 利用 OpenCV 生成 Canny 邊緣圖 ---
        # 轉換成 numpy 陣列，並轉成灰階
        np_image = np.array(init_image)
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        # 使用 Canny 偵測邊緣，這裡參數可依需求調整
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        # 將邊緣圖轉回 RGB PIL Image
        control_image = Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
        control_image.save(canny_path)
        return canny_path, control_image

canny_path, control_image = get_canny_img(init_image_name)

_ , control_image = get_image_resized_info(canny_path)

# --- 載入 ControlNet 模型（以 Canny 為例） ---
# 請確保已下載 "lllyasviel/sd-controlnet-canny"
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float32
)

# --- 建立 ControlNet + img2img 管線 ---
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    # "runwayml/stable-diffusion-v1-5",
    model_dir,
    controlnet=controlnet,
    torch_dtype=torch.float32,
    safety_checker=None
)


# # ✅ 載入 VAE（用 from_pretrained）
# vae = AutoencoderKL.from_pretrained(
#     "DiscordChatBot/src/Models/converted-orangemixvaeReupload_v10",
#     torch_dtype=torch.float32
# )

# # 🔗 指派給管線
# pipe.vae = vae

#pipe.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32).to("mps")
if vae_path:
    pipe.vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float32).to("mps")

pipe = pipe.to("mps")  # 或 "mps" 根據你的設備
pipe.enable_attention_slicing()

# --- 載入 LoRA 模型 ---
# pipe.load_lora_weights("src/temp/animeoutlineV4_16.safetensors", adapter_name="animeoutline")
#pipe.load_lora_weights("src/temp/OutlineSuperV9.safetensors", adapter_name="OutlineSuperV9")

# pipe.load_lora_weights("DiscordChatBot/src/Models/Lora/best/adapter_model.safetensors", adapter_name="ghibli")

# pipe.set_adapters(["ghibli"])

#pipe.load_lora_weights("DiscordChatBot/src/Models/Lora/best/adapter_model.safetensors", adapter_name="ghibli")
# pipe.load_lora_weights("DiscordChatBot/src/Models/Lora/best/adapter_model.safetensors")

# pipe.load_lora_weights("DiscordChatBot/src/Models/animetarotV51.safetensors", 
#                         adapter_name="animetarotV51")
# lora_weights = { "animetarotV51": 1 }
# pipe.set_adapters(list(lora_weights.keys()), list(lora_weights.values()))

# pipe.load_lora_weights("src/Models/SD/Loras/ghibli_style_offset.safetensors", 
#                         adapter_name="ghibli_style")
# lora_weights = { "ghibli_style": 1 }
# pipe.set_adapters(list(lora_weights.keys()), list(lora_weights.values()))

pipe.load_textual_inversion("src/Models/SD/Embeddings/easynegative.safetensors", token="easynegative", mean_resizing=False)
pipe.load_textual_inversion("src/Models/SD/Embeddings/badhandv4.pt", token="badhandv4", mean_resizing=False)

# --- 設定採樣器 ---
# pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
# 使用 DPM++ 2M Karras 採樣器
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    use_karras_sigmas=True  # ✅ 開啟 Karras 分布
)
negative_prompt = "easynegative, badhandv4"
prompt = "(((masterpiece))),(((bestquality))),1girl,beautiful eyes"
#seed = 2245550560
seed =  random.randint(0, 2**32 - 1)
guidance_scale = 7
steps = 25
generator = torch.manual_seed(seed)

# --- 生成圖片 ---
# 注意這裡除了原圖 (image) 參數外，多了一個 control_image 參數
result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=get_image_resized_info("src/inputs/cosplay.jpg", smaller=True)[1],
    control_image=get_image_resized_info(canny_path, smaller=True)[1],
    strength=0.5,         # 調整風格轉換程度，建議保留原圖結構可調低
    guidance_scale=guidance_scale,
    num_inference_steps=steps,
    generator=generator
).images[0]

result_path = f"src/results-new/{init_image_name}-result-{seed}.png"
result.save(result_path)
print(result_path)
#Image.open(result_path).show()