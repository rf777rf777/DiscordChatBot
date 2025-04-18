import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from typing import Dict, List, Any
import uuid
import requests

# 設置環境變數
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 初始化 FastAPI
app = FastAPI(
    title="MCP Server",
    description="一個支援 JSON-RPC 的 Multi-step Tool Server，可供 LLM 調用 create_image、describe_image、stylize_image 等工具。",
    version="1.0.0"
)

# ✅ system_prompt：提供給 LLM 的系統提示詞
MCP_SYSTEM_PROMPT = """
你是一個智慧助手，能透過 MCP JSON-RPC 協議與伺服器互動，執行圖片相關任務。請遵循以下流程：

1. 使用 `tools/list` 查詢目前伺服器支援的工具清單與參數格式：
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "id": 1
}

2. 接著根據使用者的需求，呼叫 `tools/call` 並指定要使用的工具名稱與參數內容，例如：
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "create_image",
    "arguments": {
      "prompt": "a fantasy girl with sword",
      "negative": "blurry, ugly",
      "style": ["anime", "fantasy"]
    }
  },
  "id": 2
}

3. 工具執行結果會以 result 回傳給你，例如：
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "image_id": "abc123",
    "path": "images/abc123.png"
  }
}

4. 若要對圖片進行描述或風格化處理，可以使用：
- describe_image：提供 image_id，伺服器會回傳圖片描述
- stylize_image：提供 image_id 與 style 陣列，進行圖片風格轉換

請在與使用者互動過程中，根據意圖選擇適當工具，並正確構造 JSON-RPC 請求。
"""

# Stable Diffusion 模型配置
model_name = "Anything-v3-0-better-vae"
pipe = None

def init_stable_diffusion():
    global pipe
    pipe = StableDiffusionPipeline.from_pretrained(
        f"src/{model_name}", torch_dtype=torch.float32, safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("mps")
    pipe.load_textual_inversion("src/temp/easynegative.safetensors", token="easynegative")
    pipe.load_textual_inversion("src/temp/badhandv4.pt", token="badhandv4")
    pipe.enable_attention_slicing()

# 儲存生成的圖片（模擬 MCP 的 Resources）
image_storage: Dict[str, str] = {}  # {resource_id: image_path}

# 工具定義
tools = [
    {
        "name": "create_image",
        "description": "根據提示詞產生圖片",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "正向提示詞，英文 Stable Diffusion prompt，使用逗點分隔，補足細節與背景"},
                "negative": {"type": "string", "description": "負向提示詞，若無則為空字串"},
                "style": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "圖片風格，可複選：realistic, anime, illustration, pixel_art, cyberpunk, japanese, fantasy, steampunk"
                }
            },
            "required": ["prompt"]
        }
    },
    {
        "name": "describe_image",
        "description": "描述圖片內容",
        "parameters": {
            "type": "object",
            "properties": {
                "image_id": {"type": "string", "description": "圖片的資源 ID"}
            },
            "required": ["image_id"]
        }
    },
    {
        "name": "stylize_image",
        "description": "將圖片轉換為特定風格",
        "parameters": {
            "type": "object",
            "properties": {
                "image_id": {"type": "string", "description": "圖片的資源 ID"},
                "style": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要套用的風格，如 anime, illustration"
                }
            },
            "required": ["image_id", "style"]
        }
    }
]

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

def get_style_prompt(styles: List[str]) -> str:
    prompts = [style_prompt_dict[s] for s in styles if s in style_prompt_dict]
    return ", ".join(prompts)

async def create_image(args: Dict[str, Any]) -> Dict[str, Any]:
    prompt = args["prompt"]
    negative = args.get("negative", "")
    styles = args.get("style", [])
    style_prompt = get_style_prompt(styles)
    positive_prompt = f"masterpiece, best quality, {style_prompt}, {prompt}"
    negative_prompt = f"easynegative, (badhandv4), lowres, bad anatomy, worst quality, blurry, extra hands, ugly, watermark, {negative}"

    image = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=6.5,
        height=512,
        width=512
    ).images[0]

    image_id = str(uuid.uuid4())
    os.makedirs("images", exist_ok=True)
    path = f"images/{image_id}.png"
    image.save(path)
    image_storage[image_id] = path
    return {"image_id": image_id, "path": path}

async def describe_image(args: Dict[str, Any]) -> Dict[str, Any]:
    image_id = args["image_id"]
    path = image_storage.get(image_id)
    if not path:
        raise HTTPException(status_code=404, detail="Image not found")
    return {"description": f"Description of image {image_id} (mock)"}

async def stylize_image(args: Dict[str, Any]) -> Dict[str, Any]:
    image_id = args["image_id"]
    styles = args["style"]
    path = image_storage.get(image_id)
    if not path:
        raise HTTPException(status_code=404, detail="Image not found")
    return {"result": f"Stylized {image_id} with styles {', '.join(styles)} (mock)"}

class JsonRpcRequest(BaseModel):
    jsonrpc: str
    id: Any
    method: str
    params: Dict[str, Any] = {}

@app.post("/rpc")
async def handle_rpc(request: JsonRpcRequest):
    if request.jsonrpc != "2.0":
        return {"jsonrpc": "2.0", "id": request.id, "error": {"code": -32600, "message": "Invalid JSON-RPC version"}}

    method = request.method
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request.id, "result": tools}
    elif method == "tools/call":
        tool_name = request.params.get("name")
        arguments = request.params.get("arguments", {})
        if tool_name == "create_image":
            result = await create_image(arguments)
        elif tool_name == "describe_image":
            result = await describe_image(arguments)
        elif tool_name == "stylize_image":
            result = await stylize_image(arguments)
        else:
            return {"jsonrpc": "2.0", "id": request.id, "error": {"code": -32601, "message": "Tool not found"}}
        return {"jsonrpc": "2.0", "id": request.id, "result": result}
    else:
        return {"jsonrpc": "2.0", "id": request.id, "error": {"code": -32601, "message": "Method not found"}}

@app.on_event("startup")
async def startup():
    init_stable_diffusion()

if __name__ == "__main__":
    import uvicorn
    from time import sleep

    # 啟動伺服器（僅開發測試用）
    # uvicorn.run(app, host="0.0.0.0", port=8000)

    # ✅ MCP Client 調用測試
    def call_mcp(method: str, params: dict = None, id: int = 1):
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": id
        }
        response = requests.post("http://localhost:8000/rpc", json=payload)
        return response.json()

    print("🧪 呼叫 tools/list：")
    print(json.dumps(call_mcp("tools/list"), indent=2, ensure_ascii=False))

    print("\n🧪 呼叫 tools/call -> create_image：")
    result = call_mcp("tools/call", {
        "name": "create_image",
        "arguments": {
            "prompt": "a girl flying in a cyberpunk city",
            "negative": "blurry, low quality",
            "style": ["cyberpunk", "anime"]
        }
    }, id=2)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
from openai import OpenAI

openai_api_key = "your-openai-api-key"
client = OpenAI(api_key=openai_api_key)

# 模擬使用者輸入
user_input = "我想畫一張蒸氣龐克風格的機器少女，背景有齒輪與霧氣"

# 設定系統提示詞
system_prompt = MCP_SYSTEM_PROMPT

# 發送聊天請求給 GPT 模型（包含系統提示詞與使用者輸入）
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ],
    temperature=0.7
)

# 擷取 GPT 回傳的 function call 結果
content = response.choices[0].message.content.strip()
print("🔧 GPT 回傳的工具呼叫 JSON:")
print(content)

# 將工具呼叫轉成 JSON-RPC 格式，並呼叫 MCP API
try:
    tool_call = json.loads(content)
    method = "tools/call"
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": {
            "name": tool_call["tool"],
            "arguments": tool_call["arguments"]
        },
        "id": 99
    }
    rpc_response = requests.post("http://localhost:8000/rpc", json=payload)
    print("✅ MCP 回應：")
    print(json.dumps(rpc_response.json(), indent=2, ensure_ascii=False))
except Exception as e:
    print("❌ 錯誤：無法解析 GPT 回傳或調用 MCP API")
    print(str(e))

