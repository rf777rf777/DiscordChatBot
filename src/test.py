import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from openai import OpenAI
import json
from diffusers import StableDiffusionPipeline
import torch

# 設定 OpenAI API 金鑰
openAI_apikey = "[apikey]"
img_prompt_type = "英文 Stable Diffusion:Anything-v3-0-better-vae" #"英文"

def create_chat(client: OpenAI, prompt):
    # 設計提示詞
    sys_prompt = f'''
    你是一個智慧助手，請判斷以下使用者輸入的意圖，並以 JSON 格式回應。格式如下：
    {{
      "intent": "<意圖類型>",
      "content": "<回應內容>",
      "negative-content": "<負向回應內容>",
      "style": "<圖片風格>"
    }}
    意圖類型可以是：
    - "image"：如果使用者要求生成圖片，此時的 <回應內容> 必須是詳細、豐富且具體的{img_prompt_type} prompt。請盡可能具象化內容、補足細節與背景。 <負向回應內容> 應為使用者敘述不想要的部分，一樣是{img_prompt_type} prompt，若沒有則一樣為空字串。
    - "chat"：如果使用者進行一般對話，將回應寫在 <回應內容> ，並且 <負向回應內容> 保持為空字串。    
    圖片風格可以是：
    - "realistic"
    - "anime"
    - "illustration"
    - "pixel_art"
    - "cyberpunk"
    - "japanese"
    - "fantasy"
    - "steampunk"
    - "":若使用者進行一般對話，或無法判斷為何種圖片風格時，保持空字串。
    以上圖片風格可以複選，使用逗點分隔。
    
    請僅回傳 JSON，無需其他說明。
    '''

    response = client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}],
        temperature=1.7
    )

    # 解析回應
    response_text = response.choices[0].message.content.strip()
    try:
        response_json = json.loads(response_text)
        return response_json
    except json.JSONDecodeError:
        return {"intent": "error", "content": "無法解析的回應"}

def create_dall_e_image(client: OpenAI, prompt):
    response = client.images.generate(
        model="dall-e-2",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return response.data[0].url

def create_sd_image(pos_prompt, neg_prompt, style):
    pipe = StableDiffusionPipeline.from_pretrained("src/Anything-v3-0-better-vae", torch_dtype=torch.float32, safety_checker=None)  # ✅ 禁用 safety checker
    pipe = pipe.to("mps")
    
    pipe.load_textual_inversion("src/temp/easynegative.safetensors", token="easynegative", mean_resizing=False)
    
    pipe.enable_attention_slicing()
    style_prompt = get_style_prompt(style)
    
    positive_prompt = f"masterpiece, best quality, {style_prompt}, {pos_prompt}"
    negative_prompt = f"easynegative, lowres, bad anatomy, worst quality, blurry, extra hands, ugly, watermark, {neg_prompt}"
    #設定隨機種子
    #generator = torch.manual_seed(42)

    image = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=8.5,
        #generator=generator,
        height=512,
        width=512
    ).images[0]

    image.save("result3.png")

def get_style_prompt(style_str):
    if not style_str:
        return ""
    
    style_prompt_dict = {
        "realistic": "realistic, photorealistic, ultra detailed",
        "anime": "anime, anime style, 2d, vibrant colors",
        "illustration": "illustration, concept art, soft shading",
        "pixel_art": "pixel art, 8-bit, retro game style",
        "cyberpunk": "cyberpunk, neon lights, futuristic, sci-fi",
        "japanese": "japanese style, kimono, sakura, ukiyo-e",
        "fantasy": "fantasy, magical, epic lighting, ethereal glow",
        "steampunk": "steampunk, brass gears, goggles, Victorian era"
    }    
    styles = [s.strip() for s in style_str.split(",")]
    prompts = [style_prompt_dict[s] for s in styles if s in style_prompt_dict]
    return ", ".join(prompts)

client = OpenAI(api_key=openAI_apikey)

#url = create_image(client, "A cute cartoon-style cat on Mars drinking bubble tea, in the art style of The Powerpuff Girls. The cat has large expressive eyes, a small round body, and floats slightly above the red rocky Martian surface with a space-themed background. The bubble tea cup is oversized with a straw, and there are craters and distant stars in the scene. The overall aesthetic is bright, colorful, and bold, with clean lines and minimal shading, closely mimicking the style of The Powerpuff Girls animation.")
#print(url)

user_prpmpt = "我想畫真實感鄉村風背景，不要都市風、不想要高樓建築，也不要太現代的風格"
#user_prpmpt = "什麼是大型語言模型"
result = create_chat(client, user_prpmpt)
print(result)

if result["intent"] == "image":
    #await message.channel.send(f"正在為你生成圖片：`{content}`")
    try:
        # img_url = create_dall_e_image(client, result["content"])
        # print(img_url)
        create_sd_image(result["content"], result["negative-content"], result["style"])
    except Exception as e:
        print(e)
        #await message.channel.send(f"生成圖片時出錯：{str(e)}")
else:
    print(result["content"])        

