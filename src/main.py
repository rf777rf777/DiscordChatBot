from fastapi import FastAPI
from api.v1.sd_routes import router as SD_Routes
from starlette.routing import Route
from core.config import settings
import uvicorn, re

app = FastAPI(
    title=settings.PROJECT_NAME_zh_TW,
    description=settings.PROJECT_NAME_EN_US,
    version=settings.VERSION,
    openapi_tags=[
        {
            "name": f'{settings.ROUTER_NAME_SDImage}',
            "description": f'{settings.ROUTER_Description_SDImage}'
        }
    ])

app.include_router(SD_Routes)

#讓路由大小寫不敏感(case-insensitive)
for route in app.router.routes:
    if isinstance(route, Route):
        route.path_regex = re.compile(route.path_regex.pattern, re.IGNORECASE)

#啟動API
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)