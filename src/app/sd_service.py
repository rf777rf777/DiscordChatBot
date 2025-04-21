import os, io, asyncio
os.environ["PYTORCH_ENABLE_cuda_FALLBACK"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from datetime import datetime
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from datetime import datetime
import random
from core.config import settings
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers import StableDiffusionControlNetImg2ImgPipeline
from diffusers import ControlNetModel, DPMSolverMultistepScheduler
import torch
import gc
import cv2
import numpy as np
from PIL import Image
# model_dir = "src/Models/SD/BaseModels/qchanAnimeMix_v40" 
# vae_path = ""

async def create_sd_image(mode_name, 
                    vae_name,
                    prompt,
                    sampler_name="euler_a",
                    seed=None,
                    steps=40,
                    torch_dtype = torch.float16, 
                    # 禁用 safety checker
                    safety_checker=None):
    await asyncio.sleep(0.2)

    pipe = StableDiffusionPipeline.from_pretrained(
        f'src/Models/SD/BaseModels/{mode_name}',
        torch_dtype=torch_dtype,
        safety_checker=safety_checker).to("cuda")
    
    if vae_name:
        pipe.vae = AutoencoderKL.from_pretrained(f'src/Models/SD/VAE/{vae_name}', 
                                                 torch_dtype=torch_dtype).to("cuda")

    sampler_factory = _get_scheduler_factory(sampler_name)
    pipe.scheduler = sampler_factory(pipe.scheduler.config)

    embedding_dirs = [
        ("src/Models/SD/Embeddings/easynegative.safetensors", "easynegative"),
        ("src/Models/SD/Embeddings/badhandv4.pt", "badhandv4"),
        ("src/Models/SD/Embeddings/HDA_Ahegao.pt", "HDA_Ahegao"),
        #("src/Models/SD/Embeddings/FastNegativeV2.pt", "FastNegativeV2"),
        #("src/Models/SD/Embeddings/bad-hands-5.pt", "bad-hands-5"),
    ]
    
    for path, trigger_word in embedding_dirs:
        pipe.load_textual_inversion(path, token=trigger_word, mean_resizing=False)

    pipe.enable_attention_slicing()
    
    lora_weights = { "animetarotV51": 1,
                    #   "beautifulDetailedEyes": 0.5
                    }
    pipe.load_lora_weights("src/Models/SD/Loras/animetarotV51.safetensors", 
                            adapter_name="animetarotV51")
    # pipe.load_lora_weights("src/Models/SD/Loras/beautifulDetailedEyes.safetensors", 
    #                         adapter_name="beautifulDetailedEyes")
    # pipe.load_lora_weights("src/Models/SD/Loras/hyouka_offset.safetensors",
    #                      adapter_name="hyouka_offset")

    pipe.set_adapters(list(lora_weights.keys()), list(lora_weights.values()))
    
    if not seed:
        seed =  random.randint(0, 2**32 - 1)
    
    #一次要產生的圖片數量
    num_images = 1
    #設定隨機種子
    #generator = [torch.manual_seed(seed + i) for i in range(num_images)]
    guidance_scale = 7
    steps = steps #40
    width = 576
    height = 1024

    prompt_items = prompt.split(",")
    filtered_items = [x.strip() for x in prompt_items if x.strip() not in ["1girl", "masterpiece", "best quality"]]
    filtered_prompt = ""
    if len(filtered_items) > 0:
        filtered_prompt = f"{','.join(filtered_items)},"
    prompt = f"1girl,masterpiece,best quality,animetarotV51,{filtered_prompt}8k,highres"
    
    imageList = []
    for i in range(1):
        target_seed = seed + i 
        print(f"Seed: {target_seed}")
        generator = [torch.manual_seed(seed + i) for i in range(num_images)]
        # generator = torch.manual_seed(target_seed)
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                negative_prompt= "easynegative,badhandv4,watermark", #"FastNegativeV2, badhandv4, watermark",
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                height=height,
                width=width,
                num_images_per_prompt=num_images
            ).images[0]
            imageList.append(image)
        torch.cuda.empty_cache()
        gc.collect()
        
    # 儲存影像至記憶體(bytes)
    img_byte_arr = io.BytesIO()
    imageList[0].save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return img_byte_arr

async def image_to_image(file, mode_name, 
                    vae_name,
                    prompt,
                    sampler_name="euler_a",
                    seed=None,
                    steps=40,
                    torch_dtype = torch.float16, 
                    # 禁用 safety checker
                    safety_checker=None):
    contents = await file.read()
    original_image = Image.open(io.BytesIO(contents)).convert("RGB")    
    resized_image = get_resized_image(original_image, True)
    resized_image_RGB = resized_image.convert("RGB")
    #init_image_name = info[0]
    control_image = get_canny_img(resized_image_RGB)
    control_image = get_resized_image(control_image)

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=torch_dtype
    )
    
    # --- 建立 ControlNet + img2img 管線 ---
    pipe = StableDiffusionControlNetImg2ImgPipeline .from_pretrained(
        # "runwayml/stable-diffusion-v1-5",
        f'src/Models/SD/BaseModels/{mode_name}',
        controlnet=controlnet,
        torch_dtype=torch_dtype,
        safety_checker=None
    )
    
    if vae_name:
        pipe.vae = AutoencoderKL.from_pretrained(f'src/Models/SD/VAE/{vae_name}', 
                                                 torch_dtype=torch_dtype).to("cuda")

    pipe = pipe.to("cuda")  # 或 "cuda" 根據你的設備
    pipe.enable_attention_slicing()
    pipe.load_textual_inversion("src/Models/SD/Embeddings/easynegative.safetensors", token="easynegative", mean_resizing=False)
    pipe.load_textual_inversion("src/Models/SD/Embeddings/badhandv4.pt", token="badhandv4", mean_resizing=False)

    sampler_factory = _get_scheduler_factory(sampler_name)
    pipe.scheduler = sampler_factory(pipe.scheduler.config)

    negative_prompt = "easynegative, badhandv4"
    prompt = "(((masterpiece))),(((bestquality))),1girl"
    #seed = 2245550560
    seed =  random.randint(0, 2**32 - 1)
    guidance_scale = 7
    generator = torch.manual_seed(seed)
    original_image = get_resized_image(original_image)
    # --- 生成圖片 ---
    # 注意這裡除了原圖 (image) 參數外，多了一個 control_image 參數
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=get_resized_image(original_image),
        control_image=control_image,
        strength=0.5,         # 調整風格轉換程度，建議保留原圖結構可調低
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=generator
    ).images[0]
    
    #torch.mps.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()
        
    # 儲存影像至記憶體(bytes)
    img_byte_arr = io.BytesIO()
    result.save(img_byte_arr, format='PNG')
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

def get_canny_img(input_image):
    # --- 利用 OpenCV 生成 Canny 邊緣圖 ---
    # 轉換成 numpy 陣列，並轉成灰階
    np_image = np.array(input_image)
    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    # 使用 Canny 偵測邊緣，這裡參數可依需求調整
    edges = cv2.Canny(gray, threshold1=120, threshold2=250)
    # 將邊緣圖轉回 RGB PIL Image
    control_image = Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

    nowtime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    control_image.save(f"temp_{nowtime}.jpg")
    return control_image

def get_resized_image(img, bigger:bool=False, smaller:bool=False) -> Image.Image:
    img = img
    w, h = img.size
    if bigger:
        resized_img = img.resize((w*2,h*2), Image.BICUBIC)
        return resized_img
    if smaller:
        resized_img = img.resize((w//2,h//2), Image.LANCZOS)
        return resized_img
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

    return resized_img

def get_canny_img(input_image):
    # --- 利用 OpenCV 生成 Canny 邊緣圖 ---
    # 轉換成 numpy 陣列，並轉成灰階
    np_image = np.array(input_image)
    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    # 使用 Canny 偵測邊緣，這裡參數可依需求調整
    edges = cv2.Canny(gray, threshold1=120, threshold2=250)
    # 將邊緣圖轉回 RGB PIL Image
    control_image = Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

    nowtime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    control_image.save(f"temp_{nowtime}.jpg")
    return control_image

def get_resized_image(img, bigger:bool=False, smaller:bool=False) -> Image.Image:
    img = img
    w, h = img.size
    if bigger:
        resized_img = img.resize((w*2,h*2), Image.BICUBIC)
        return resized_img
    if smaller:
        resized_img = img.resize((w//2,h//2), Image.LANCZOS)
        return resized_img
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

    return resized_img