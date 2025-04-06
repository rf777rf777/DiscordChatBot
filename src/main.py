from fastapi import FastAPI
# from api.v1.real_esgran_routes import router as RealESRGAN_router
from starlette.routing import Route
from core.config import settings
import uvicorn, re

app = FastAPI(
    title=settings.PROJECT_NAME_zh_TW,
    description=settings.PROJECT_NAME_EN_US,
    version=settings.VERSION,
    openapi_tags=[
        {
            "name": f'{settings.ROUTER_NAME_RealESRGAN}',
            "description": f'{settings.ROUTER_Description_RealESRGAN}'
        }
    ])

# 註冊路由
# app.include_router(RealESRGAN_router)

# 讓路由大小寫不敏感(case-insensitive)
for route in app.router.routes:
    if isinstance(route, Route):
        route.path_regex = re.compile(route.path_regex.pattern, re.IGNORECASE)

# 啟動API
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)