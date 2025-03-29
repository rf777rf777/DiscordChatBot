from openai import OpenAI
import json

# 設定 OpenAI API 金鑰
openAI_apikey = "[apikey]"

def create_chat(client: OpenAI, prompt):
    # 設計提示詞
    sys_prompt = f'''
    你是一個智慧助手，請判斷以下使用者輸入的意圖，並以 JSON 格式回應。格式如下：
    {{
      "intent": "<意圖類型>",
      "content": "<回應內容>"
    }}
    意圖類型可以是：
    - "image"：如果使用者要求生成圖片，此時的<回應內容>必須是要丟給DALL-E的prompt。
    - "chat"：如果使用者進行一般對話。

    請僅回傳 JSON，無需其他說明。
    '''

    response = client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}],
        temperature=0.7
    )

    # 解析回應
    response_text = response.choices[0].message.content.strip()
    try:
        response_json = json.loads(response_text)
        return response_json
    except json.JSONDecodeError:
        return {"intent": "error", "content": "無法解析的回應"}

def create_image(client: OpenAI, prompt):
    response = client.images.generate(
        model="dall-e-2",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return response.data[0].url

client = OpenAI(api_key=openAI_apikey)

#url = create_image(client, "A cute cartoon-style cat on Mars drinking bubble tea, in the art style of The Powerpuff Girls. The cat has large expressive eyes, a small round body, and floats slightly above the red rocky Martian surface with a space-themed background. The bubble tea cup is oversized with a straw, and there are craters and distant stars in the scene. The overall aesthetic is bright, colorful, and bold, with clean lines and minimal shading, closely mimicking the style of The Powerpuff Girls animation.")
#print(url)

user_prpmpt = "請用宮崎駿大師的風個畫出一張瑪爾濟斯在火星上喝奶茶"
result = create_chat(client, user_prpmpt)
#print(result)

if result["intent"] == "image":
    #await message.channel.send(f"正在為你生成圖片：`{content}`")
    try:
        img_url = create_image(client, result["content"])
        print(img_url)
    except Exception as e:
        print(e)
        #await message.channel.send(f"生成圖片時出錯：{str(e)}")
else:
    print(result["content"])        

