from PIL import Image

def load_sd_parameters(image_path: str) -> str:
    """
    從 SD 生成的圖片中讀取 `parameters` 欄位（包含 prompt、seed 等）
    """
    try:
        img = Image.open(image_path)
        return img.text.get("parameters", "")
    except Exception as e:
        print(f"讀取失敗: {e}")
        return ""

def parse_sd_parameters(param_text: str) -> dict:
    """
    將 SD metadata 中的 parameters 欄位字串解析成 dict
    """
    lines = param_text.strip().split("\n")
    if not lines:
        return {}

    prompt = lines[0]
    negative_prompt = ""
    params = {}

    # 處理 Negative prompt
    if len(lines) > 1 and lines[1].startswith("Negative prompt:"):
        negative_prompt = lines[1][len("Negative prompt:"):].strip()
        param_line = lines[2] if len(lines) > 2 else ""
    else:
        param_line = lines[1] if len(lines) > 1 else ""

    # 解析參數（如 Steps: 20, Seed: 12345）
    for part in param_line.split(","):
        if ":" in part:
            k, v = part.strip().split(":", 1)
            params[k.strip()] = v.strip()

    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        **params
    }

param_text = load_sd_parameters("src/temp/Lora/animetarotV51/322483.jpeg")
info = parse_sd_parameters(param_text)

print(info)
# ➜ {
#   'prompt': 'masterpiece, best quality, a fantasy castle',
#   'negative_prompt': 'low quality, blurry, bad anatomy',
#   'Steps': '20',
#   'Sampler': 'Euler a',
#   'CFG scale': '7',
#   'Seed': '12345678',
#   'Size': '512x512',
#   'Model hash': 'abcdef1234',
#   'Model': 'dreamlike-v2',
#   'Denoising strength': '0.75'
# }
