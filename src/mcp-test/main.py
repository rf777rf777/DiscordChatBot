"""
簡化版 MCP 伺服器 - 無需任何驗證
用於測試 CLINE 客戶端的配置和使用
"""

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
import asyncio
import uvicorn
import json
import uuid
import time
import logging
from datetime import datetime

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp-server")

# 建立應用
app = FastAPI(
    title="簡化版 MCP 伺服器",
    description="無需驗證的 MCP 服務，用於測試 CLINE 設定和使用",
    version="1.0.0"
)

# 添加 CORS 支援
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- 資料儲存 -----
# 全部使用記憶體儲存，用於測試目的

# 預設 client 設定
client_id = "test_client"

# 事件佇列
event_queue = asyncio.Queue()

# 工作階段資訊
sessions = {}

# CLINE 配置
cline_config = {
    "model_name": "default-model",
    "max_tokens": 1024,
    "temperature": 0.7,
    "streaming": True,
    "tools_enabled": True
}

# 工具註冊表
tools_registry = {
    "add": {
        "description": "將兩個數字相加",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "第一個數字"},
                "b": {"type": "number", "description": "第二個數字"}
            },
            "required": ["a", "b"]
        },
        "handler": lambda params: params["a"] + params["b"]
    },
    "subtract": {
        "description": "將兩個數字相減",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "被減數"},
                "b": {"type": "number", "description": "減數"}
            },
            "required": ["a", "b"]
        },
        "handler": lambda params: params["a"] - params["b"]
    },
    "multiply": {
        "description": "將兩個數字相乘",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "第一個數字"},
                "b": {"type": "number", "description": "第二個數字"}
            },
            "required": ["a", "b"]
        },
        "handler": lambda params: params["a"] * params["b"]
    },
    "get_current_time": {
        "description": "獲取當前的時間",
        "parameters": {
            "type": "object",
            "properties": {
                "format": {"type": "string", "description": "時間格式，例如 'YYYY-MM-DD HH:MM:SS'"}
            },
            "required": []
        },
        "handler": lambda params: datetime.now().strftime(params.get("format", "%Y-%m-%d %H:%M:%S"))
    }
}

# ----- 資料模型 -----

class ClientConfig(BaseModel):
    model_name: str = Field(..., description="欲使用的模型名稱")
    max_tokens: int = Field(1024, description="生成的最大 token 數")
    temperature: float = Field(0.7, description="溫度參數，控制生成的隨機性", ge=0.0, le=1.0)
    top_p: Optional[float] = Field(None, description="核採樣參數")
    streaming: bool = Field(True, description="是否使用串流回應")
    tools_enabled: bool = Field(True, description="是否啟用工具")

class ToolCallRequest(BaseModel):
    tool_name: str = Field(..., description="要呼叫的工具名稱")
    parameters: Dict[str, Any] = Field(..., description="工具參數")
    request_id: Optional[str] = Field(None, description="請求 ID")

class MessageRequest(BaseModel):
    session_id: str = Field(..., description="工作階段 ID")
    messages: List[Dict[str, Any]] = Field(..., description="對話訊息列表")
    config: Optional[ClientConfig] = None

# ----- 輔助函數 -----

async def add_event(event_type: str, data: Any):
    """將事件添加到事件佇列"""
    event = {
        "type": event_type,
        "timestamp": time.time(),
        "data": data
    }
    
    await event_queue.put(event)
    logger.info(f"已添加事件 {event_type}")

# ----- API 端點 -----

@app.get("/")
def root():
    """健康檢查端點"""
    return {
        "status": "online",
        "service": "Simplified MCP Server",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/events")
async def events_stream():
    """SSE 事件流，用於即時通知"""
    async def event_generator():
        try:
            ping_interval = 15  # 秒
            last_ping = time.time()
            
            while True:
                # 檢查是否需要發送心跳
                current_time = time.time()
                if current_time - last_ping >= ping_interval:
                    yield f"event: ping\ndata: {json.dumps({'timestamp': current_time})}\n\n"
                    last_ping = current_time
                
                # 嘗試獲取事件，但不阻塞太久
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                    yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # 如果沒有事件，繼續下一次循環
                    pass
        except asyncio.CancelledError:
            logger.info("事件流已中斷")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/cline/configure")
async def configure_cline(config: ClientConfig, background_tasks: BackgroundTasks):
    """配置 CLINE 客戶端 - 無需驗證"""
    global cline_config
    
    # 更新配置
    cline_config = config.dict()
    
    # 添加配置成功事件
    background_tasks.add_task(
        add_event,
        "cline_configured",
        {"config": config.dict()}
    )
    
    logger.info("CLINE 已配置完成")
    
    return {
        "status": "success",
        "message": "CLINE 配置成功",
        "client_id": client_id
    }

@app.post("/sessions/create")
async def create_session(background_tasks: BackgroundTasks):
    """建立新的工作階段 - 無需驗證"""
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    
    sessions[session_id] = {
        "created_at": datetime.now().isoformat(),
        "messages": [],
        "active": True
    }
    
    # 添加工作階段建立事件
    background_tasks.add_task(
        add_event,
        "session_created",
        {"session_id": session_id}
    )
    
    return {
        "status": "success",
        "session_id": session_id
    }

@app.get("/tools/list")
def list_tools():
    """列出可用的工具 - 無需驗證"""
    # 檢查是否已啟用工具
    if not cline_config.get("tools_enabled", True):
        return []
    
    tools = []
    for name, info in tools_registry.items():
        tools.append({
            "name": name,
            "description": info["description"],
            "parameters": info["parameters"]
        })
    
    return tools

@app.post("/tools/call")
async def call_tool(req: ToolCallRequest, background_tasks: BackgroundTasks):
    """呼叫工具並執行 - 無需驗證"""
    # 生成請求 ID (如果未提供)
    request_id = req.request_id or f"req_{uuid.uuid4().hex[:8]}"
    
    # 檢查工具是否存在
    if req.tool_name not in tools_registry:
        error_response = {
            "status": "error",
            "request_id": request_id,
            "error": f"找不到工具: {req.tool_name}"
        }
        
        # 添加工具呼叫失敗事件
        background_tasks.add_task(
            add_event,
            "tool_call_failed",
            {
                "request_id": request_id,
                "tool_name": req.tool_name,
                "error": f"找不到工具: {req.tool_name}"
            }
        )
        
        return error_response
    
    # 獲取工具處理函數
    tool = tools_registry[req.tool_name]
    
    try:
        # 執行工具
        result = tool["handler"](req.parameters)
        
        # 記錄工具呼叫
        tool_call_record = {
            "request_id": request_id,
            "tool_name": req.tool_name,
            "parameters": req.parameters,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        # 添加工具呼叫完成事件
        background_tasks.add_task(
            add_event,
            "tool_call_completed",
            tool_call_record
        )
        
        return {
            "status": "success",
            "request_id": request_id,
            "result": result
        }
    except Exception as e:
        error_message = str(e)
        
        # 添加工具呼叫失敗事件
        background_tasks.add_task(
            add_event,
            "tool_call_failed",
            {
                "request_id": request_id,
                "tool_name": req.tool_name,
                "error": error_message
            }
        )
        
        return {
            "status": "error",
            "request_id": request_id,
            "error": error_message
        }

@app.post("/chat/completions")
async def chat_completions(req: MessageRequest, request: Request, background_tasks: BackgroundTasks):
    """處理聊天完成請求 - 無需驗證"""
    # 檢查工作階段是否存在
    if req.session_id not in sessions:
        logger.warning(f"工作階段 {req.session_id} 不存在，自動創建")
        sessions[req.session_id] = {
            "created_at": datetime.now().isoformat(),
            "messages": [],
            "active": True
        }
    
    # 更新工作階段的訊息歷史
    sessions[req.session_id]["messages"].extend(req.messages)
    
    # 獲取配置
    config = req.config or ClientConfig(**cline_config)
    
    # 模擬生成回應
    response_id = f"resp_{uuid.uuid4().hex[:8]}"
    
    if config.streaming:
        async def generate_stream():
            # 取得最後一條用戶訊息
            last_message = "你好"
            for msg in req.messages:
                if msg.get("role") == "user":
                    last_message = msg.get("content", "")
            
            # 模擬串流回應
            response_text = f"這是對「{last_message}」的模擬回應。MCP 伺服器正在處理您的請求。"
            chunks = [response_text[i:i+10] for i in range(0, len(response_text), 10)]
            
            for i, chunk in enumerate(chunks):
                yield f"data: {json.dumps({'id': response_id, 'choices': [{'delta': {'content': chunk}}], 'chunk_index': i})}\n\n"
                await asyncio.sleep(0.3)  # 模擬延遲
            
            # 發送完成事件
            yield f"data: [DONE]\n\n"
            
            # 記錄回應已完成
            sessions[req.session_id]["messages"].append({
                "role": "assistant",
                "content": response_text
            })
            
            # 添加生成完成事件
            background_tasks.add_task(
                add_event,
                "generation_completed",
                {"session_id": req.session_id, "response_id": response_id}
            )
        
        # 添加生成開始事件
        background_tasks.add_task(
            add_event,
            "generation_started",
            {"session_id": req.session_id, "response_id": response_id}
        )
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )
    else:
        # 取得最後一條用戶訊息
        last_message = "你好"
        for msg in req.messages:
            if msg.get("role") == "user":
                last_message = msg.get("content", "")
        
        # 非串流回應
        response_text = f"這是對「{last_message}」的完整模擬回應。MCP 伺服器已處理您的請求。"
        
        # 記錄回應
        sessions[req.session_id]["messages"].append({
            "role": "assistant",
            "content": response_text
        })
        
        # 添加生成完成事件
        background_tasks.add_task(
            add_event,
            "generation_completed",
            {"session_id": req.session_id, "response_id": response_id}
        )
        
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": config.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ]
        }

@app.get("/status")
def get_status():
    """獲取伺服器狀態 - 無需驗證"""
    return {
        "status": "online",
        "version": "1.0.0",
        "client": {
            "id": client_id,
            "type": "CLINE",
        },
        "active_sessions": sum(1 for session in sessions.values() if session["active"]),
        "tools_available": len(tools_registry),
        "server_time": datetime.now().isoformat()
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, background_tasks: BackgroundTasks):
    """刪除工作階段 - 無需驗證"""
    if session_id not in sessions:
        return {"status": "error", "message": "工作階段不存在"}
    
    # 標記為非活動
    sessions[session_id]["active"] = False
    
    # 添加工作階段已刪除事件
    background_tasks.add_task(
        add_event,
        "session_deleted",
        {"session_id": session_id}
    )
    
    return {"status": "success", "message": "工作階段已刪除"}

# 啟動伺服器
if __name__ == "__main__":
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=8000, reload=True)