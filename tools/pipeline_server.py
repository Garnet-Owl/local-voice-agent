import contextlib
import os
import warnings

import uvicorn
from fastapi import FastAPI, WebSocket

from agent.connections.websocket_handler import WebSocketHandler
from agent.service import VoiceAgentService
from shared.logging import setup_logging

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

logger = setup_logging("server")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global ws_handler
    service = VoiceAgentService()
    orchestrator = service.initialize()
    ws_handler = WebSocketHandler(orchestrator)
    yield


app = FastAPI(title="Local Voice Agent Pipeline Server", lifespan=lifespan)
ws_handler = None


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if ws_handler:
        await ws_handler.handle_connection(websocket)
    else:
        await websocket.close(code=1001)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
