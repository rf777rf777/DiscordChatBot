import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from openai import OpenAI
import json
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

# 設定 OpenAI API 金鑰
openAI_apikey = "[apikey]"
img_prompt_type = "英文 Stable Diffusion" #"英文"
model_name = "Anything-v3-0-better-vae"

def create_chat_function_call(client: OpenAI, prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一個智慧助手，用 Function Calling 方式判斷使用者是否想生成圖片、描述圖片或風格轉換。"},
            {"role": "user", "content": prompt}
        ],
        functions=[
            {
                "name": "create_image",
                "description": "根據提示詞產生圖片",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": f"正向提示詞，從原本的輸入prompt中找出使用者想要的元素，做成{img_prompt_type} prompt，使用逗點分隔，並盡可能補足細節與背景元素，轉換成 Stable Diffusion 正向提示詞。"},
                        "negative": {"type": "string", "description": f"負向提示詞，從原本的prompt中找出使用者不想要的元素，一樣是{img_prompt_type} prompt，若沒有則為空字串，最後轉換成 Stable Diffusion 負向提示詞。"},
                        "style": {"type": "array", "items": {"type": "string"}, "description": "圖片風格，可複選，選項如下: realistic、anime、illustration、pixel_art、cyberpunk、japanese、fantasy、steampunk"}
                    },
                    "required": ["prompt"]
                }
            },
            {
                "name": "describe_image",
                "description": "用自然語言描述圖片中的內容",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string", "description": "圖片檔案路徑"}
                    },
                    "required": ["image_path"]
                }
            },
            {
                "name": "stylize_image",
                "description": "將圖片轉換為特定風格（如手繪、插畫等）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string", "description": "圖片檔案路徑"},
                        "style": {"type": "array", "items": {"type": "string"}, "description": "要套用的風格，如 anime、illustration，可複選"}
                    },
                    "required": ["image_path", "style"]
                }
            },
            {
                "name": "chat",
                "description": "一般對話，非圖像生成",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"}
                    },
                    "required": ["content"]
                }
            }
        ],
        function_call="auto",
        temperature=1.2
    )

    choice = response.choices[0]
    if choice.finish_reason == "function_call":
        func_call = choice.message.function_call
        name = func_call.name
        args = json.loads(func_call.arguments)
        return {"function": name, "args": args}
    else:
        return {"function": "chat", "args": {"content": choice.message.content.strip()}}


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
    - "image"：如果使用者要求生成圖片，此時的 <回應內容> 必須是詳細、豐富且具體的{img_prompt_type} prompt，模型使用：{model_name}。請盡可能具象化內容、補足細節與背景。 <負向回應內容> 應為使用者敘述不想要的部分，一樣是{img_prompt_type} prompt，若沒有則一樣為空字串。
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
        temperature=1
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
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to("mps")
    
    pipe.load_textual_inversion("src/temp/easynegative.safetensors", token="easynegative", mean_resizing=False)
    pipe.load_textual_inversion("src/temp/badhandv4.pt", token="badhandv4", mean_resizing=False)

    pipe.enable_attention_slicing()
    style_prompt = get_style_prompt(style)
    
    positive_prompt = f"masterpiece, best quality, (white stocking), {style_prompt}, {pos_prompt}"
    negative_prompt = f"easynegative, (badhandv4), lowres, bad anatomy, worst quality, blurry, extra hands, ugly, watermark, {neg_prompt}"
    #設定隨機種子
    #generator = torch.manual_seed(42)

    image = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=6.5,
        #generator=generator,
        height=512,
        width=512
    ).images[0]

    image.save("result_test1.png")

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
    #styles = [s.strip() for s in style_str.split(",")]
    styles = [s.strip() for s in style_str]

    prompts = [style_prompt_dict[s] for s in styles if s in style_prompt_dict]
    return ", ".join(prompts)

def describe_image(text):
    print(text)
    
def stylize_image(text1, text2):
    print(text1)
    print(text2)

def chat_type1():
    result = create_chat(client, user_prompt)
    print(result)

    if result["intent"] == "image":
        #await message.channel.send(f"正在為你生成圖片：`{content}`")
        try:
            # img_url = create_dall_e_image(client, result["content"])
            create_sd_image(result["prompt"], result["negative"], result["style"])
        except Exception as e:
            print(e)
            #await message.channel.send(f"生成圖片時出錯：{str(e)}")
    else:
        print(result["content"])  

def chat_type2():
    result = create_chat_function_call(client, user_prompt)
    args = result["args"]
    if result["function"] == "create_image":
        create_sd_image(args["prompt"], args.get("negative", ""), args.get("style", []))
    elif result["function"] == "describe_image":
        describe_image(args["image_path"])
    elif result["function"] == "stylize_image":
        stylize_image(args["image_path"], args.get("style", []))
    else:
        print("💬", args.get("content", "無法回應"))

client = OpenAI(api_key=openAI_apikey)

#url = create_image(client, "A cute cartoon-style cat on Mars drinking bubble tea, in the art style of The Powerpuff Girls. The cat has large expressive eyes, a small round body, and floats slightly above the red rocky Martian surface with a space-themed background. The bubble tea cup is oversized with a straw, and there are craters and distant stars in the scene. The overall aesthetic is bright, colorful, and bold, with clean lines and minimal shading, closely mimicking the style of The Powerpuff Girls animation.")
#print(url)

user_prompt = "我想畫日式奇幻風背景，不要都市風、不想要高樓建築，也不要太現代的風格"
#user_prompt = "什麼是大型語言模型"
user_prompt = '''
二次元風格插畫，一位穿著華麗中式改良服飾的年輕女性角色。她具有以下特徵：

    - 髮型與髮色：長髮呈深紫色或接近黑色，髮尾略帶些微粉紫色漸層。髮型略顯波浪，柔順地垂至身體兩側。

    - 服飾：她穿著融合中華與日系風格的華麗服裝，上身是白色與藍色交織的設計，繡有金色花紋，下身是藍色百褶短裙，腰間綁有寬大的藍紫色腰封，上面也有精緻的金色圖騰。袖子為寬大的和風樣式，帶有透明的淡藍紗與花朵圖案。

    - 配件：頭上配戴華麗的髮飾，包含金色髮簪與紫色、藍色的花朵裝飾，搭配金色葉片與紅色絲帶。

    - 神情與姿勢：她的表情略帶羞澀或內斂，眼神清澈，注視著前方。她一手自然垂下，另一手扶著腰帶，看起來柔美而沉靜。

    - 背景：背景為木質窗戶，窗外有明亮的自然光灑入，營造出溫暖與寧靜的氛圍。

整體畫面色調柔和，色彩搭配細緻優雅，給人一種古典與夢幻融合的美感。

'''      

#chat_type1()
chat_type2()

