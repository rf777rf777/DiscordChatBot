from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from app.sd_service import create_sd_image

from core.config import settings
import asyncio

# router初始化
router = APIRouter(prefix=f'/{settings.ROUTER_NAME_SDImage}', 
                   tags=[settings.ROUTER_NAME_SDImage])
@router.post('', summary = 'Stable Diffusion 圖片生成塔羅牌', 
            description = '使用Stable Diffusion生成塔羅牌圖片',
            response_class = StreamingResponse,
            responses = {200: {"content": {"image/png": {}}}})
async def use_stable_diffusion(
    model_name: str = Query(description='模型名稱', default = 'qchanAnimeMix_v40'),
    vae_name: str = Query(description='VAE名稱', default = ''),
    prompt: str = Query(description='提示詞', default = 'masterpiece, best quality, 1girl'),
    sampler: str = Query(description='取樣器名稱', default = 'euler_a'),
    seed: int = Query(description='隨機種子', default = None),
    steps: int = Query(description='取樣步驟', default = 40)): 
    
    # 使用 create_sd_image 函數生成圖片
    img_byte_arr = await create_sd_image(
        mode_name = model_name,
        vae_name = vae_name,
        prompt = prompt,
        sampler_name = sampler,
        seed = seed,
        steps = steps
    )    

    
    # 使用 StreamingResponse 返回處理後的圖片，並包含 filename
    # headers = {
    #     'Content-Disposition': f'attachment; filename="test.png"'
    # }
    
    return StreamingResponse(img_byte_arr, 
                             media_type = "image/png"
                             #, headers = headers
                             )
