#!/usr/bin/env python3
"""
pc_stream.py - Streamer (full)

Features included:
- Auto-detect camera device path (v4l2-ctl or /dev)
- Parse capabilities via v4l2-ctl (only keep fps >= 15)
- Choose reasonable default camera mode (prefer 1280x720@30)
- Single camera capture handle, background reader keeps only latest frame
- Throttle capture thread to target FPS to reduce CPU
- MediaRelay to serve multiple viewers with low-latency frames
- Robust normalization of ICE candidates before calling pc.addIceCandidate()
- Announce caps to signaling server during register

Environment:
  SIGNALING_SERVER (optional) - default ws://localhost:8000/ws
  STREAMER_ID (optional) - default streamer-01
  DEVICE_PATH (optional) - e.g. /dev/video0
  FORCE_WIDTH / FORCE_HEIGHT / FORCE_FPS (optional) - force default capture mode

Usage:
  export FORCE_WIDTH=640 FORCE_HEIGHT=360 FORCE_FPS=15
  python3 pc_stream.py
"""
import asyncio
import json
import os
import re
import subprocess
import time
import threading
import cv2
import numpy as np
from fractions import Fraction

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame

import websockets

# ========== CONFIG ==========
SIGNALING_SERVER = os.getenv("SIGNALING_SERVER", "ws://localhost:8000/ws")
STREAMER_ID = os.getenv("STREAMER_ID", "streamer-01")
DEVICE_PATH = os.getenv("DEVICE_PATH")  # if provided, use it
# ============================

# -------------------------
# Device detection helpers
# -------------------------
def find_camera_device_by_v4l2():
    """Try to parse `v4l2-ctl --list-devices` and return first /dev/video* (prefer Logitech)."""
    try:
        res = subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True, check=True)
        out = res.stdout
    except Exception:
        return None

    blocks = []
    cur_name = None
    cur_devs = []
    for line in out.splitlines():
        if line.strip() == "":
            if cur_name:
                blocks.append((cur_name, cur_devs))
                cur_name = None
                cur_devs = []
            continue
        if not line.startswith("\t") and not line.startswith(" "):
            if cur_name:
                blocks.append((cur_name, cur_devs))
            cur_name = line.strip()
            cur_devs = []
        else:
            dev = line.strip()
            if os.path.exists(dev):
                cur_devs.append(dev)
    if cur_name:
        blocks.append((cur_name, cur_devs))

    # prefer Logitech / C930 etc
    for name, devs in blocks:
        ln = name.lower()
        if "logitech" in ln or "c930" in ln or "c930e" in ln:
            if devs:
                return devs[0]
    # fallback first found
    for _, devs in blocks:
        if devs:
            return devs[0]
    return None

def find_camera_device_fallback():
    """Check /dev/video0..7 for existing device."""
    for i in range(0, 8):
        p = f"/dev/video{i}"
        if os.path.exists(p):
            return p
    return None

def detect_camera_device(preferred=None):
    """Return device path string like /dev/video0 or None."""
    if preferred and os.path.exists(preferred):
        return preferred
    dev = find_camera_device_by_v4l2()
    if dev:
        print("Auto-detected camera device via v4l2-ctl:", dev)
        return dev
    dev2 = find_camera_device_fallback()
    if dev2:
        print("Auto-detected camera device via /dev scan:", dev2)
        return dev2
    return None

# -------------------------
# v4l2 caps parsing (keep fps >= 15 only)
# -------------------------
def detect_v4l2_caps(device="/dev/video0"):
    """
    Use `v4l2-ctl --device device --list-formats-ext` to parse supported resolutions and fps.
    Only keep resolutions that support fps >= 15.
    Returns list of dicts: [{width, height, fps_list}, ...]
    """
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--device", device, "--list-formats-ext"],
            capture_output=True, text=True, check=True
        )
    except Exception as e:
        print("v4l2-ctl failed:", e)
        return []

    lines = result.stdout.splitlines()
    caps = []
    cur_w = cur_h = None
    fps_list = []
    size_re = re.compile(r"Size:\s+Discrete\s+(\d+)x(\d+)")
    fps_re = re.compile(r"Interval:\s+Discrete\s+[0-9.]+s\s+\(([0-9.]+)\s+fps\)")

    for line in lines:
        sm = size_re.search(line)
        fm = fps_re.search(line)
        if sm:
            # flush previous block
            if cur_w and cur_h and fps_list:
                filtered = [f for f in set(fps_list) if f >= 15]
                if filtered:
                    caps.append({
                        "width": int(cur_w),
                        "height": int(cur_h),
                        "fps_list": sorted(filtered, reverse=True)
                    })
                fps_list = []
            cur_w, cur_h = sm.groups()
            continue
        if fm and cur_w and cur_h:
            try:
                fps_val = float(fm.group(1))
                fps_list.append(round(fps_val))
            except Exception:
                pass

    # last block
    if cur_w and cur_h and fps_list:
        filtered = [f for f in set(fps_list) if f >= 15]
        if filtered:
            caps.append({
                "width": int(cur_w),
                "height": int(cur_h),
                "fps_list": sorted(filtered, reverse=True)
            })

    # normalize fps lists
    for c in caps:
        c["fps_list"] = sorted(list(set(int(x) for x in c.get("fps_list", []))), reverse=True)

    print("v4l2-ctl caps (fps>=15 only):", caps)
    return caps

# -------------------------
# Fallback OpenCV detection
# -------------------------
def detect_opencv_caps(device_index=0):
    print("Fallback OpenCV detection...")
    candidates = [(1280,720), (1920,1080), (640,480)]
    caps = []
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        print("OpenCV cannot open device", device_index)
        return caps
    for w,h in candidates:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        time.sleep(0.1)
        ret, frame = cap.read()
        if ret and frame is not None:
            caps.append({"width": w, "height": h, "fps_list": [30,15]})
    cap.release()
    print("OpenCV fallback caps:", caps)
    return caps

# -------------------------
# Utilities: sort + choose default
# -------------------------
def sort_caps_desc(caps):
    """Sort caps by area descending and fps lists descending."""
    if not caps:
        return caps
    for c in caps:
        c["fps_list"] = sorted(list(set(int(x) for x in c.get("fps_list", []))), reverse=True)
    caps_sorted = sorted(caps, key=lambda x: x["width"] * x["height"], reverse=True)
    return caps_sorted

def choose_default_mode(caps):
    """
    Preference logic:
    - If env FORCE_WIDTH & FORCE_HEIGHT specified -> use them (and FORCE_FPS if given; else 15)
    - Prefer 1280x720 with fps >=30 (pick highest fps >=30 on that res)
    - Else pick largest resolution that has fps >=15 (choose highest fps >=15)
    - Else fallback to caps[0] highest fps
    Returns (width, height, fps)
    """
    fw = os.getenv("FORCE_WIDTH")
    fh = os.getenv("FORCE_HEIGHT")
    ff = os.getenv("FORCE_FPS")
    if fw and fh:
        try:
            w = int(fw); h = int(fh); fps = int(ff) if ff else 15
            print(f"[config] Default mode forced by env: {w}x{h}@{fps}")
            return (w, h, fps)
        except Exception:
            pass

    if not caps:
        return (1280, 720, 30)

    # search for 1280x720 first
    for c in caps:
        if c["width"] == 1280 and c["height"] == 720:
            fpss = sorted(c.get("fps_list", []), reverse=True)
            for f in fpss:
                if f >= 30:
                    return (1280, 720, f)
            if fpss:
                return (1280, 720, fpss[0])

    # else choose largest with fps >=15
    for c in caps:
        fpss = sorted(c.get("fps_list", []), reverse=True)
        if any(f >= 15 for f in fpss):
            return (c["width"], c["height"], max(f for f in fpss if f >= 15))

    # fallback to caps[0]
    c = caps[0]
    fpss = sorted(c.get("fps_list", []), reverse=True)
    return (c["width"], c["height"], fpss[0] if fpss else 30)

# -------------------------
# Low-latency camera track with background reader
# -------------------------
class OpenCVCameraTrack(VideoStreamTrack):
    """
    Background reader keeps only latest_frame to avoid backlog and reduce latency.
    Throttles capture to target fps to reduce CPU usage.
    """
    def __init__(self, device_index=0, width=640, height=360, fps=15):
        super().__init__()
        self.device_index = int(device_index)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        # open capture
        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera device index {self.device_index}")

        # set properties (driver may ignore)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        try:
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        except Exception:
            pass

        self._running = True
        self._latest_frame = None
        self._lock = threading.Lock()

        # start background thread
        self._thread = threading.Thread(target=self._reader_thread, daemon=True)
        self._thread.start()

        print(f"[camera] opened device_index={self.device_index} {self.width}x{self.height}@{self.fps}")

    def _reader_thread(self):
        target_interval = 1.0 / max(1, self.fps)
        next_time = time.time()
        while self._running:
            try:
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    time.sleep(0.005)
                    continue
                # resize if necessary
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                # atomic swap latest_frame (simple reference swap is fine)
                with self._lock:
                    self._latest_frame = frame
            except Exception as e:
                print("[camera reader] exception:", e)
                time.sleep(0.02)

            # throttle to target fps
            next_time += target_interval
            sleep_for = next_time - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # behind schedule, reset next_time
                next_time = time.time()

        try:
            self._cap.release()
        except Exception:
            pass

    async def recv(self):
        # pts in milliseconds and time_base = 1/1000
        pts = int(time.time() * 1000)
        time_base = Fraction(1, 1000)

        # wait briefly until frame available
        wait_until = time.time() + 1.0
        frame = None
        while True:
            with self._lock:
                if self._latest_frame is not None:
                    # copy reference to avoid modification while converting
                    frame = self._latest_frame.copy()
            if frame is not None:
                break
            if time.time() > wait_until:
                # fallback black frame
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                break
            await asyncio.sleep(0.005)

        # convert BGR->RGB
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb = frame[..., ::-1]

        vframe = VideoFrame.from_ndarray(rgb, format="rgb24")
        vframe.pts = pts
        vframe.time_base = time_base
        return vframe

    def stop(self):
        self._running = False
        try:
            self._thread.join(timeout=0.5)
        except Exception:
            pass
        try:
            self._cap.release()
        except Exception:
            pass

# -------------------------
# Robust candidate normalization
# -------------------------
def make_candidate_init_in(cand):
    """
    Normalize ICE candidate provided by viewer into RTCIceCandidateInit-like dict.
    Accepts multiple shapes:
      - {'candidate': 'candidate:...','sdpMid':'0','sdpMLineIndex':0}
      - {'candidate': {'candidate': 'candidate:...', 'sdpMid':..., ...}}
      - sometimes nested or missing fields (we fill defaults)
    Returns dict or None.
    """
    if not cand or not isinstance(cand, dict):
        return None

    # If the payload itself uses key 'candidate' and that is dict -> unwrap
    inner = cand.get("candidate")
    if isinstance(inner, dict):
        candidate_str = inner.get("candidate")
        sdpMid = inner.get("sdpMid", cand.get("sdpMid"))
        sdpMLineIndex = inner.get("sdpMLineIndex", cand.get("sdpMLineIndex"))
    else:
        candidate_str = cand.get("candidate")
        sdpMid = cand.get("sdpMid")
        sdpMLineIndex = cand.get("sdpMLineIndex")

    # On some clients the whole payload might be under another key
    # Try common fallbacks
    if not isinstance(candidate_str, str):
        # attempt other keys
        for k in ("candidateStr", "cand", "candidate_string"):
            if isinstance(cand.get(k), str):
                candidate_str = cand.get(k)
                break

    if not isinstance(candidate_str, str):
        # invalid candidate format
        # print for debugging
        print("[candidate] invalid or missing candidate string:", cand)
        return None

    # ensure candidate string looks like "candidate:..."
    if not candidate_str.strip().startswith("candidate:"):
        # sometimes browsers include "a=" prefix; accept common variants
        if "candidate" not in candidate_str:
            print("[candidate] candidate string doesn't contain 'candidate':", candidate_str)
            return None

    # default sdpMid/sdpMLineIndex when missing
    if sdpMid is None:
        sdpMid = "0"
    if sdpMLineIndex is None:
        # attempt to convert if it's stringable
        try:
            sdpMLineIndex = int(cand.get("sdpMLineIndex", 0))
        except Exception:
            sdpMLineIndex = 0

    return {
        "candidate": candidate_str,
        "sdpMid": str(sdpMid),
        "sdpMLineIndex": int(sdpMLineIndex)
    }

# -------------------------
# WebRTC offer handling
# -------------------------
async def handle_offer(pc, offer_sdp, ws, viewer_id, relay, camera_source, video_settings=None):
    """
    Set remote description, subscribe to relay (shared camera source), create answer,
    and send ICE candidates via ws.
    """
    if not offer_sdp:
        print("[handle_offer] empty offer_sdp")
        return None
    offer = RTCSessionDescription(sdp=offer_sdp, type="offer")
    await pc.setRemoteDescription(offer)

    # ensure we send video
    if not pc.getTransceivers():
        pc.addTransceiver('video', direction='sendonly')

    # try to subscribe to relay; fallback to new track if necessary
    try:
        track = relay.subscribe(camera_source)
        pc.addTrack(track)
    except Exception as e:
        print("[handle_offer] relay.subscribe failed:", e)
        # fallback: create a temporary track with requested settings (less optimal)
        width = int(video_settings.get("width", 1280)) if video_settings else 1280
        height = int(video_settings.get("height", 720)) if video_settings else 720
        fps = int(video_settings.get("fps", 30)) if video_settings else 30
        print("[handle_offer] fallback creating local track", width, height, fps)
        pc.addTrack(OpenCVCameraTrack(device_index=camera_source.device_index if hasattr(camera_source, 'device_index') else 0,
                                      width=width, height=height, fps=fps))

    @pc.on("icecandidate")
    def on_icecandidate(candidate):
        if candidate is None:
            return
        cand = {
            "candidate": getattr(candidate, "candidate", None),
            "sdpMid": getattr(candidate, "sdpMid", None),
            "sdpMLineIndex": getattr(candidate, "sdpMLineIndex", None)
        }
        # forward to viewer via signaling server
        asyncio.ensure_future(ws.send(json.dumps({
            "type": "forward",
            "target_role": "viewer",
            "target_id": viewer_id,
            "payload": {"action": "candidate", "candidate": cand}
        })))

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return pc.localDescription.sdp

# -------------------------
# Main run loop
# -------------------------
async def run():
    # detect device path
    device_path = detect_camera_device(DEVICE_PATH)
    if not device_path:
        print("No camera device detected. Exiting.")
        return

    # map to index for OpenCV
    m = re.search(r"/dev/video(\d+)$", device_path)
    cam_index = 0
    if m:
        cam_index = int(m.group(1))
    print("[main] using device:", device_path, "OpenCV index:", cam_index)

    # detect caps
    caps = detect_v4l2_caps(device_path)
    if not caps:
        caps = detect_opencv_caps(cam_index)
    caps = sort_caps_desc(caps)
    print("[main] detected caps (sorted):", caps)

    # choose default mode (prefer 1280x720@30)
    default_w, default_h, default_fps = choose_default_mode(caps)
    print("[main] default camera mode chosen:", default_w, default_h, default_fps)

    # create camera source and relay
    try:
        camera_source = OpenCVCameraTrack(device_index=cam_index, width=default_w, height=default_h, fps=default_fps)
    except Exception as e:
        print("[main] failed to open camera track:", e)
        return
    relay = MediaRelay()

    print("[main] connecting to signaling server:", SIGNALING_SERVER)
    async with websockets.connect(SIGNALING_SERVER) as ws:
        # register and include caps if available
        reg_msg = {"type": "register", "role": "streamer", "id": STREAMER_ID}
        if caps:
            reg_msg["caps"] = caps
        await ws.send(json.dumps(reg_msg))
        print("[main] registered as", STREAMER_ID)

        pcs = {}  # viewer_id -> pc

        async for message in ws:
            try:
                msg = json.loads(message)
            except Exception:
                print("[main] received non-json message:", message)
                continue

            if msg.get("type") != "forward":
                # ignore other types
                continue

            from_role = msg.get("from_role")
            from_id = msg.get("from_id")
            payload = msg.get("payload", {})

            # ignore messages from self
            if from_id == STREAMER_ID:
                continue

            # handle offer
            if payload.get("action") == "offer":
                viewer_id = from_id
                sdp_offer = payload.get("sdp")
                video_settings = payload.get("video_settings") or {}
                if not sdp_offer:
                    print("[main] offer from", viewer_id, "missing sdp")
                    continue

                pc = RTCPeerConnection()
                pcs[viewer_id] = pc

                @pc.on("iceconnectionstatechange")
                def on_ice_state_change():
                    print(f"[pc:{viewer_id}] ICE state ->", pc.iceConnectionState)
                    if pc.iceConnectionState in ("failed", "disconnected"):
                        asyncio.ensure_future(pc.close())
                        pcs.pop(viewer_id, None)

                print("[main] handling offer from", viewer_id, "settings:", video_settings)
                answer_sdp = await handle_offer(pc, sdp_offer, ws, viewer_id, relay, camera_source, video_settings)
                if answer_sdp:
                    await ws.send(json.dumps({
                        "type": "forward",
                        "target_role": "viewer",
                        "target_id": viewer_id,
                        "payload": {"action": "answer", "sdp": answer_sdp}
                    }))
                    print("[main] sent answer to", viewer_id)
                else:
                    print("[main] failed to create answer for", viewer_id)

            # handle candidate
            elif payload.get("action") == "candidate":
                viewer_id = from_id
                candidate_raw = payload.get("candidate")
                pc = pcs.get(viewer_id)
                if not pc:
                    print("[main] no pc for viewer", viewer_id, "- ignoring candidate")
                    continue
                cand_init = make_candidate_init_in(candidate_raw)
                if not cand_init:
                    print("[main] invalid candidate from", viewer_id, candidate_raw)
                    continue
                try:
                    await pc.addIceCandidate(cand_init)
                    # debug log
                    print(f"[main] addIceCandidate OK for {viewer_id}")
                except Exception as e:
                    print("[main] addIceCandidate error for", viewer_id, ":", e)

    # cleanup (if ws loop ends)
    try:
        camera_source.stop()
    except Exception:
        pass

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Interrupted, exiting.")
