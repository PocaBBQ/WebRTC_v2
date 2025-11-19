# app.py

import json
import logging
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("signaling")

app = FastAPI()
# mount static relative to project root: server/static
app.mount("/static", StaticFiles(directory="server/static"), name="static")

clients: Dict[str, WebSocket] = {}
streamer_caps: Dict[str, Any] = {}

async def safe_send(ws: WebSocket, payload: dict):
    try:
        await ws.send_text(json.dumps(payload))
    except Exception:
        logger.exception("Failed to send to client")

@app.get("/")
async def index():
    try:
        return FileResponse("server/static/index.html")
    except Exception:
        return HTMLResponse("<html><body><h2>Index not found</h2><p>Place your index.html in server/static.</p></body></html>", status_code=200)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = None
    try:
        while True:
            text = await websocket.receive_text()
            try:
                data = json.loads(text)
            except Exception:
                await safe_send(websocket, {"type":"error","message":"invalid json"})
                continue

            typ = data.get("type")
            # register
            if typ == "register":
                client_id = data.get("id")
                if not client_id:
                    await safe_send(websocket, {"type":"error","message":"missing id in register"})
                    continue
                clients[client_id] = websocket
                logger.info("Registered client: %s", client_id)
                await safe_send(websocket, {"type":"registered", "id": client_id})
                continue

            # caps from streamer
            if typ == "caps":
                if client_id:
                    streamer_caps[client_id] = data.get("caps")
                    logger.info("Stored caps for %s", client_id)
                    await safe_send(websocket, {"type":"caps_ack", "id": client_id})
                else:
                    logger.info("caps from unregistered ws")
                continue

            # list_streamers
            if typ == "list_streamers":
                items = []
                for sid, caps in streamer_caps.items():
                    items.append({"id": sid, "caps": caps})
                await safe_send(websocket, {"type":"streamer_list", "items": items})
                continue

            # forward to specific client if 'to' present
            to = data.get("to")
            # derive from: prefer explicit 'from' in message else use registered client_id
            frm_field = data.get("from") or client_id
            if to:
                dest_ws = clients.get(to)
                if dest_ws:
                    # ensure forwarded payload includes 'from' so recipient knows origin
                    forwarded = dict(data)
                    if 'from' not in forwarded or forwarded.get('from') != frm_field:
                        forwarded['from'] = frm_field
                    await safe_send(dest_ws, forwarded)
                    logger.info("Forwarded %s from %s -> %s", typ, frm_field, to)
                else:
                    logger.warning("Destination %s not connected; notify sender", to)
                    await safe_send(websocket, {"type":"error","message":f"destination {to} not connected"})
                continue

            # no 'to': handle some known types locally or broadcast
            if typ in ("offer","answer","candidate","request","update_ack"):
                # broadcast to all except sender, and include 'from' info
                for cid, cws in list(clients.items()):
                    if cws is websocket:
                        continue
                    forwarded = dict(data)
                    if 'from' not in forwarded or forwarded.get('from') != frm_field:
                        forwarded['from'] = frm_field
                    await safe_send(cws, forwarded)
                logger.info("Broadcasted %s from %s", typ, frm_field)
                continue

            # unknown
            await safe_send(websocket, {"type":"error","message": f"unknown type {typ}"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnect: %s", client_id)
        if client_id and client_id in clients:
            del clients[client_id]
        if client_id and client_id in streamer_caps:
            del streamer_caps[client_id]
    except Exception:
        logger.exception("WS handler exception")
        if client_id and client_id in clients:
            del clients[client_id]
        if client_id and client_id in streamer_caps:
            del streamer_caps[client_id]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
