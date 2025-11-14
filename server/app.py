# server/app.py
import os
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# === static client dir ===
BASE_DIR = os.path.dirname(__file__)
CLIENT_DIR = os.path.join(BASE_DIR, "client")
os.makedirs(CLIENT_DIR, exist_ok=True)
app.mount("/client", StaticFiles(directory=CLIENT_DIR), name="client")
INDEX_PATH = os.path.join(CLIENT_DIR, "index.html")

@app.get("/", response_class=HTMLResponse)
async def root():
    if os.path.exists(INDEX_PATH):
        return FileResponse(INDEX_PATH)
    return {"status": "server running - place index.html in server/client/"}

# --- runtime state (shared) ---
clients = {"streamer": {}, "viewer": {}}
# specs reported by streamers: { streamer_id: [ {width, height, fps_list}, ... ] }
streamer_specs = {}
lock = asyncio.Lock()
# -----------------------------

@app.get("/streamers")
async def streamers():
    # return list of currently connected streamer ids (async-safe)
    async with lock:
        lst = list(clients["streamer"].keys())
    return JSONResponse({"streamers": lst})

@app.get("/streamer_specs")
async def streamer_specs_api():
    # return the dynamic specs reported by connected streamers (async-safe)
    async with lock:
        specs_copy = dict(streamer_specs)
    return JSONResponse({"specs": specs_copy})

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    role = None
    identity = None
    try:
        # first message must be register
        msg_text = await ws.receive_text()
        reg = json.loads(msg_text)
        if reg.get("type") != "register" or "role" not in reg or "id" not in reg:
            await ws.send_text(json.dumps({"type":"error","message":"first message must be register with role and id"}))
            await ws.close()
            return

        role = reg["role"]
        identity = reg["id"]

        async with lock:
            clients[role][identity] = ws
        print(f"Registered {role}:{identity}")

        # If streamer sends capabilities inside the register message (optional), accept it
        if role == "streamer" and "caps" in reg:
            async with lock:
                streamer_specs[identity] = reg["caps"]
            print(f"Received caps from {identity} during register")

        # message loop
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            # handle announce from streamer: {"type":"announce","caps": [...]}
            if msg.get("type") == "announce" and role == "streamer":
                caps = msg.get("caps")
                async with lock:
                    if caps:
                        streamer_specs[identity] = caps
                        print(f"Announced caps from {identity}: {caps}")
                        await ws.send_text(json.dumps({"type":"announce_ack","status":"ok"}))
                    else:
                        # remove if empty caps passed
                        if identity in streamer_specs:
                            del streamer_specs[identity]
                        await ws.send_text(json.dumps({"type":"announce_ack","status":"removed"}))
                continue

            # forwarding (signaling) messages
            if msg.get("type") == "forward":
                target_role = msg.get("target_role")
                target_id = msg.get("target_id")
                payload = msg.get("payload")
                async with lock:
                    target_ws = clients.get(target_role, {}).get(target_id)
                if target_ws:
                    await target_ws.send_text(json.dumps({
                        "type": "forward",
                        "from_role": role,
                        "from_id": identity,
                        "payload": payload
                    }))
                else:
                    await ws.send_text(json.dumps({"type":"error","message":"target not connected"}))
                continue

            # viewer asking for current connected streamers (optional)
            if msg.get("type") == "list_streamers":
                async with lock:
                    keys = list(clients["streamer"].keys())
                await ws.send_text(json.dumps({"type":"list_streamers","streamers": keys}))
                continue

            # heartbeat
            if msg.get("type") == "heartbeat":
                await ws.send_text(json.dumps({"type":"heartbeat_ack"}))
                continue

            await ws.send_text(json.dumps({"type":"error","message":"unknown type"}))

    except WebSocketDisconnect:
        print("WebSocket disconnected", role, identity)
    except Exception as e:
        print("WebSocket error:", e)
    finally:
        async with lock:
            if role and identity and identity in clients.get(role, {}):
                del clients[role][identity]
                # remove specs if a streamer disconnected
                if role == "streamer" and identity in streamer_specs:
                    del streamer_specs[identity]
                    print(f"Removed specs for disconnected streamer {identity}")
                print(f"Unregistered {role}:{identity}")
