"""
Streamer with aspect-ratio-preserving scaling + padding and dynamic quality switching.
Optimized: reuses ScaledVideoTrack instances per (width x height @ fps) to reduce memory/CPU when multiple viewers
request the same quality, and additionally pools MediaRelay subscriptions to reduce memory usage on long runs / many reconnects.

Features:
- Device auto-detect (keywords: logitech, c930e)
- Parse v4l2 caps and keep resolutions with FPS >= FPS_MIN (default 30)
- Pool ScaledVideoTrack per settings key (e.g., "1280x720@30")
- Pool MediaRelay.subscribe(player.video) per settings key to avoid creating too many underlying sources
- Provide per-viewer RTCPeerConnection; reuse pooled tracks when possible
- Support dynamic update: viewer sends a 'request' message with new settings; streamer replaces sender's track
- Robust ICE candidate handling and compatibility workarounds for aiortc versions
- Designed for LAN use (no STUN required on client side)
Requirements:
- v4l2-ctl present on system for accurate device caps (fallback provided if not found)
- numpy and av (PyAV) installed
- aiortc, aiohttp installed
"""

import asyncio
import inspect
import json
import logging
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import aiohttp
import av
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRelay

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pc_stream")

# ---------- Config ----------
SIGNALING_URL = "ws://localhost:8000/ws"
STREAMER_ID = "streamer-01"

DEVICE_KEYWORDS = ("logitech", "c930e")
FPS_MIN = 30  # keep only fps >= FPS_MIN in caps
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
DEVICE = "/dev/video0"
# ----------------------------

# Globals
player: Optional[MediaPlayer] = None
relay: Optional[MediaRelay] = None
pcs: Dict[str, RTCPeerConnection] = {}
senders: Dict[str, Any] = {}  # store RTCRtpSender for each viewer id
DEVICE_CAPS: List[Dict[str, Any]] = []

# Pool for ScaledVideoTrack instances keyed by "WxH@FPS"
track_pool: Dict[str, VideoStreamTrack] = {}
track_refcount: Dict[str, int] = {}
viewer_track_key: Dict[str, str] = {}

# Pool for MediaRelay subscriptions keyed by "WxH@FPS"
# (this limits number of underlying sources and helps memory stability)
video_source_pool: Dict[str, Any] = {}


# ---------- helpers ----------
def _inc_track_ref(key: str) -> None:
    """Increase refcount for a pooled track key and log it."""
    current = track_refcount.get(key, 0) + 1
    track_refcount[key] = current
    logger.debug("Track %s refcount -> %d", key, current)


def _dec_track_ref(key: str) -> None:
    """
    Decrease refcount, and drop from pools when refcount reaches zero.
    Also stops tracks and underlying sources to free memory.
    """
    global track_pool, video_source_pool

    if key not in track_refcount:
        return
    current = track_refcount[key] - 1
    if current <= 0:
        logger.info("No more users for track %s, removing from pool", key)
        track_refcount.pop(key, None)

        track = track_pool.pop(key, None)
        if track is not None and hasattr(track, "stop"):
            try:
                track.stop()
            except Exception:
                logger.exception("Error stopping track %s", key)

        source = video_source_pool.pop(key, None)
        if source is not None and hasattr(source, "stop"):
            try:
                source.stop()
            except Exception:
                logger.exception("Error stopping video source %s", key)
    else:
        track_refcount[key] = current
        logger.debug("Track %s refcount -> %d", key, current)


def run_cmd(cmd: str) -> str:
    try:
        out = subprocess.check_output(
            shlex.split(cmd), stderr=subprocess.STDOUT
        ).decode(errors="ignore")
        return out
    except subprocess.CalledProcessError as e:
        logger.debug(
            "Command failed: %s -> %s",
            cmd,
            e.output.decode(errors="ignore"),
        )
        return ""
    except FileNotFoundError:
        logger.debug("Command missing: %s", cmd)
        return ""


def list_v4l2_devices() -> List[Dict[str, Any]]:
    if not shutil.which("v4l2-ctl"):
        logger.warning("v4l2-ctl not found")
        return []
    out = run_cmd("v4l2-ctl --list-devices")
    devices = []
    if not out:
        return devices
    lines = out.splitlines()
    i = 0
    while i < len(lines):
        name = lines[i].strip()
        if not name.endswith(":"):
            i += 1
            continue
        name = name[:-1].strip()
        i += 1
        nodes = []
        while i < len(lines) and lines[i].startswith("\t"):
            nodes.append(lines[i].strip())
            i += 1
        devices.append({"name": name, "nodes": nodes})
    return devices


def choose_device_by_keyword(
    devices: List[Dict[str, Any]], keywords=DEVICE_KEYWORDS
) -> Optional[str]:
    for dev in devices:
        lname = dev["name"].lower()
        for kw in keywords:
            if kw in lname and dev["nodes"]:
                return dev["nodes"][0]
    if devices and devices[0]["nodes"]:
        return devices[0]["nodes"][0]
    return None


def parse_v4l2_formats(output: str) -> List[Dict[str, Any]]:
    entries = []
    width = height = None
    pixel_format = None
    fpss: List[int] = []
    for line in output.splitlines():
        line = line.strip()
        m_pf = re.match(r"^\[\d+\]:\s+'?([A-Za-z0-9_]+)'?", line)
        if m_pf:
            pixel_format = m_pf.group(1)
            continue
        m_size = (
            re.match(r"^Size:\s*Discrete\s*([0-9]+)x([0-9]+)", line)
            or re.match(r"^Size:\s*([0-9]+)x([0-9]+)", line)
            or re.match(r"^([0-9]+)x([0-9]+)", line)
        )
        if m_size:
            if width and height and fpss:
                fpss_filtered = sorted(
                    [f for f in set(fpss) if f >= FPS_MIN], reverse=True
                )
                if fpss_filtered:
                    entries.append(
                        {
                            "width": width,
                            "height": height,
                            "fps_list": fpss_filtered,
                            "pixel_format": pixel_format,
                        }
                    )
            width = int(m_size.group(1))
            height = int(m_size.group(2))
            fpss = []
            continue
        m_interval = re.search(r"Interval:.*\(([\d\.]+)\s*fps\)", line)
        if m_interval:
            try:
                fps_val = float(m_interval.group(1))
                fpss.append(int(round(fps_val)))
            except Exception:
                pass
            continue
        m_paren = re.search(r"\(([\d\.]+)\s*fps\)", line)
        if m_paren:
            try:
                fps_val = float(m_paren.group(1))
                fpss.append(int(round(fps_val)))
            except Exception:
                pass
            continue
    if width and height and fpss:
        fpss_filtered = sorted(
            [f for f in set(fpss) if f >= FPS_MIN], reverse=True
        )
        if fpss_filtered:
            entries.append(
                {
                    "width": width,
                    "height": height,
                    "fps_list": fpss_filtered,
                    "pixel_format": pixel_format,
                }
            )
    uniq: Dict[tuple, Dict[str, Any]] = {}
    for e in entries:
        k = (e["width"], e["height"])
        if k not in uniq:
            uniq[k] = {
                "width": e["width"],
                "height": e["height"],
                "fps_list": list(e["fps_list"]),
                "pixel_format": e.get("pixel_format"),
            }
        else:
            uniq[k]["fps_list"] = sorted(
                list(set(uniq[k]["fps_list"] + e["fps_list"])), reverse=True
            )
    return list(uniq.values())


def get_device_caps(dev_node: str) -> List[Dict[str, Any]]:
    if not shutil.which("v4l2-ctl"):
        logger.warning("v4l2-ctl not found, returning fallback caps")
        return [
            {
                "width": 1280,
                "height": 720,
                "fps_list": [DEFAULT_FPS],
            },
            {
                "width": 640,
                "height": 480,
                "fps_list": [DEFAULT_FPS],
            },
        ]
    out = run_cmd(f"v4l2-ctl --list-formats-ext -d {dev_node}")
    caps = parse_v4l2_formats(out)
    if not caps:
        return [
            {
                "width": 1280,
                "height": 720,
                "fps_list": [DEFAULT_FPS],
            },
            {
                "width": 640,
                "height": 480,
                "fps_list": [DEFAULT_FPS],
            },
        ]
    return caps


class ParsedIceCandidate:
    def __init__(self, cand_dict: dict):
        self.sdpMid = cand_dict.get("sdpMid")
        self.sdpMLineIndex = cand_dict.get("sdpMLineIndex")
        self.candidate = cand_dict.get("candidate")

        self.foundation = None
        self.component = None
        self.protocol = None
        self.priority = None
        self.ip = None
        self.port = None
        self.typ = None
        self.type = None
        self.relatedAddress = None
        self.relatedPort = None
        self.related_address = None
        self.related_port = None
        self.tcptype = None
        self.tcpType = None

        if isinstance(self.candidate, str) and self.candidate.startswith(
            "candidate:"
        ):
            try:
                parts = self.candidate[len("candidate:") :].split()
                if len(parts) >= 8:
                    self.foundation = parts[0]
                    try:
                        self.component = int(parts[1])
                    except Exception:
                        self.component = parts[1]
                    self.protocol = parts[2].lower()
                    try:
                        self.priority = int(parts[3])
                    except Exception:
                        self.priority = parts[3]
                    self.ip = parts[4]
                    try:
                        self.port = int(parts[5])
                    except Exception:
                        self.port = parts[5]
                    if parts[6] == "typ" and len(parts) >= 8:
                        self.typ = parts[7]
                        self.type = self.typ
                    idx = 8
                    while idx < len(parts):
                        token = parts[idx]
                        if token == "raddr" and idx + 1 < len(parts):
                            self.relatedAddress = parts[idx + 1]
                            self.related_address = parts[idx + 1]
                            idx += 2
                            continue
                        if token == "rport" and idx + 1 < len(parts):
                            try:
                                self.relatedPort = int(parts[idx + 1])
                                self.related_port = self.relatedPort
                            except Exception:
                                self.relatedPort = parts[idx + 1]
                                self.related_port = parts[idx + 1]
                            idx += 2
                            continue
                        if token == "tcptype" and idx + 1 < len(parts):
                            self.tcptype = parts[idx + 1]
                            self.tcpType = parts[idx + 1]
                            idx += 2
                            continue
                        if idx + 1 < len(parts):
                            idx += 2
                        else:
                            idx += 1
            except Exception:
                logger.debug(
                    "Failed parsing candidate string", exc_info=True
                )


def scale_and_pad_frame(
    src_frame: av.VideoFrame, target_w: int, target_h: int
) -> av.VideoFrame:
    src_w = src_frame.width
    src_h = src_frame.height
    if src_w == 0 or src_h == 0:
        return src_frame

    src_ar = src_w / src_h
    target_ar = target_w / target_h

    if src_ar > target_ar:
        new_w = target_w
        new_h = int(round(target_w / src_ar))
    else:
        new_h = target_h
        new_w = int(round(target_h * src_ar))

    if new_w <= 0:
        new_w = 1
    if new_h <= 0:
        new_h = 1

    try:
        scaled = src_frame.reformat(
            width=new_w, height=new_h, format="rgb24"
        )
    except Exception:
        scaled = src_frame.reformat(
            width=new_w, height=new_h, format="rgb24"
        )

    arr = scaled.to_ndarray(format="rgb24")
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w, :] = arr
    out_frame = av.VideoFrame.from_ndarray(canvas, format="rgb24")
    try:
        out_frame.pts = src_frame.pts
        out_frame.time_base = (
            src_frame.time_base
            if src_frame.time_base is not None
            else av.time_base
        )
    except Exception:
        pass
    return out_frame


class ScaledVideoTrack(VideoStreamTrack):
    def __init__(
        self, source_track: VideoStreamTrack, width: int, height: int, fps: int
    ):
        super().__init__()
        self.source = source_track
        self.target_w = int(width)
        self.target_h = int(height)
        try:
            self.fps = int(fps)
            if self.fps <= 0:
                self.fps = DEFAULT_FPS
        except Exception:
            self.fps = DEFAULT_FPS
        self.frame_time = 1.0 / self.fps
        self._last_send = None
        logger.info(
            "ScaledVideoTrack init: %dx%d@%d (preserve aspect + pad)",
            self.target_w,
            self.target_h,
            self.fps,
        )

    async def recv(self):
        frame = await self.source.recv()
        try:
            out_frame = scale_and_pad_frame(
                frame, self.target_w, self.target_h
            )
        except Exception:
            logger.exception(
                "scale_and_pad_frame failed, falling back to reformat"
            )
            try:
                out_frame = frame.reformat(
                    width=self.target_w,
                    height=self.target_h,
                    format="rgb24",
                )
            except Exception:
                out_frame = frame

        now = time.time()
        if self._last_send is None:
            self._last_send = now
        else:
            elapsed = now - self._last_send
            to_wait = self.frame_time - elapsed
            if to_wait > 0:
                await asyncio.sleep(to_wait)
                self._last_send = time.time()
            else:
                self._last_send = now
        return out_frame


async def create_player(
    dev_node: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
):
    global player, relay
    if player is not None and relay is not None:
        return
    options = {"framerate": str(fps), "video_size": f"{width}x{height}"}
    logger.info("Opening device %s with options %s", dev_node, options)
    try:
        player = MediaPlayer(dev_node, format="v4l2", options=options)
    except Exception:
        logger.exception("Failed to open MediaPlayer for %s", dev_node)
        raise
    relay = MediaRelay()


def pool_key(width: int, height: int, fps: int) -> str:
    return f"{int(width)}x{int(height)}@{int(fps)}"


async def handle_offer(ws, peer_from: str, data: dict, dev_node: str):
    global video_source_pool

    viewer_id = peer_from
    logger.info("Handling offer from %s settings: %s", viewer_id, data.get("settings"))

    # If this viewer already has an old PC, close & cleanup before creating a new one
    old_pc = pcs.get(viewer_id)
    if old_pc is not None:
        logger.info("Closing old pc for viewer %s before new offer", viewer_id)
        try:
            await old_pc.close()
        except Exception:
            logger.exception("Error closing old pc for %s", viewer_id)
        pcs.pop(viewer_id, None)
        senders.pop(viewer_id, None)
        old_key = viewer_track_key.pop(viewer_id, None)
        if old_key:
            _dec_track_ref(old_key)

    settings = data.get("settings") or {}
    width = settings.get("width", DEFAULT_WIDTH)
    height = settings.get("height", DEFAULT_HEIGHT)
    fps = settings.get("fps", DEFAULT_FPS)

    if player is None or relay is None:
        await create_player(
            dev_node,
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            fps=DEFAULT_FPS,
        )

    pc = RTCPeerConnection()
    pcs[viewer_id] = pc

    @pc.on("iceconnectionstatechange")
    def on_ice_state_change():
        logger.info(
            "[pc:%s] ICE -> %s", viewer_id, pc.iceConnectionState
        )
        if pc.iceConnectionState in ("closed", "failed", "disconnected"):
            async def _close_and_cleanup():
                try:
                    await pc.close()
                except Exception:
                    pass
                pcs.pop(viewer_id, None)
                senders.pop(viewer_id, None)
                old_key = viewer_track_key.pop(viewer_id, None)
                if old_key:
                    _dec_track_ref(old_key)
            try:
                asyncio.create_task(_close_and_cleanup())
            except Exception:
                pass

    try:
        offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
        await pc.setRemoteDescription(offer)
    except Exception:
        logger.exception("Failed to set remote desc for %s", viewer_id)
        try:
            await ws.send_json(
                {
                    "type": "error",
                    "to": viewer_id,
                    "message": "server failed setRemoteDescription",
                }
            )
        except Exception:
            pass
        return

    # Acquire or create pooled scaled track + pooled relay source
    key = pool_key(width, height, fps)
    try:
        if key in track_pool:
            scaled = track_pool[key]
            logger.info(
                "Reusing ScaledVideoTrack from pool for %s -> %s",
                viewer_id,
                key,
            )
        else:
            if key not in video_source_pool:
                video_source_pool[key] = relay.subscribe(player.video)
            source = video_source_pool[key]
            scaled = ScaledVideoTrack(
                source, width=width, height=height, fps=fps
            )
            track_pool[key] = scaled
            logger.info(
                "Created ScaledVideoTrack and stored in pool: %s", key
            )
        _inc_track_ref(key)
        viewer_track_key[viewer_id] = key
    except Exception:
        logger.exception(
            "Failed to create/subscribe scaled track for %s", viewer_id
        )
        return

    try:
        sender = pc.addTrack(scaled)
        senders[viewer_id] = sender
        logger.info(
            "Added track for %s -> %dx%d@%d (sender stored)",
            viewer_id,
            width,
            height,
            fps,
        )
    except Exception:
        logger.exception("Error adding track for %s", viewer_id)

    try:
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
    except Exception:
        logger.exception("Failed create/set answer for %s", viewer_id)
        try:
            await ws.send_json(
                {
                    "type": "error",
                    "to": viewer_id,
                    "message": "server failed createAnswer",
                }
            )
        except Exception:
            pass
        return

    try:
        payload = {
            "type": "answer",
            "to": viewer_id,
            "from": STREAMER_ID,
            "sdp": pc.localDescription.sdp,
            "type_sdp": pc.localDescription.type,
        }
        await ws.send_json(payload)
        logger.info("sent answer to %s", viewer_id)
    except Exception:
        logger.exception("Failed to send answer to %s", viewer_id)


async def handle_update_request(ws, peer_from: str, data: dict):
    """
    Robust update handling:
    - Prefer exact match by peer_from (viewer id)
    - Fallback 1: if no peer_from match and exactly one active pc+sender, apply update to that one
    - Fallback 2: if multiple active pcs and no match, log and optionally reply error ack
    """
    global video_source_pool

    viewer_id = peer_from
    settings = data.get("settings") or {}
    width = settings.get("width", DEFAULT_WIDTH)
    height = settings.get("height", DEFAULT_HEIGHT)
    fps = settings.get("fps", DEFAULT_FPS)

    pc = pcs.get(viewer_id)
    sender = senders.get(viewer_id)
    if pc is None or sender is None:
        # fallback strategy
        active_pairs = [(k, s) for k, s in senders.items() if s is not None]
        if len(active_pairs) == 1:
            # single viewer connected -> assume update is for that viewer
            fallback_viewer, fallback_sender = active_pairs[0]
            logger.warning(
                "No existing pc/sender for %s; falling back to single active viewer %s",
                viewer_id,
                fallback_viewer,
            )
            pc = pcs.get(fallback_viewer)
            sender = fallback_sender
            viewer_id = fallback_viewer  # reflect actual target
        else:
            logger.warning(
                "No existing pc/sender for %s when handling update; active viewers: %s; ignoring",
                viewer_id,
                list(senders.keys()),
            )
            try:
                await ws.send_json(
                    {
                        "type": "error",
                        "to": data.get("from"),
                        "from": STREAMER_ID,
                        "message": "no active pc/sender for requested viewer",
                    }
                )
            except Exception:
                pass
            return

    try:
        if player is None or relay is None:
            logger.warning("player/relay not ready for update request")
            return

        key = pool_key(width, height, fps)
        if key in track_pool:
            new_track = track_pool[key]
            logger.info("Using pooled track for update -> %s", key)
        else:
            if key not in video_source_pool:
                video_source_pool[key] = relay.subscribe(player.video)
            source = video_source_pool[key]
            new_track = ScaledVideoTrack(
                source, width=width, height=height, fps=fps
            )
            track_pool[key] = new_track
            logger.info("Created new pooled track for update -> %s", key)

        # update refcounts: decrease old, increase new
        old_key = viewer_track_key.get(viewer_id)
        if old_key and old_key != key:
            _dec_track_ref(old_key)
        _inc_track_ref(key)
        viewer_track_key[viewer_id] = key

        # Try replaceTrack (may return None or coroutine depending on aiortc version)
        try:
            res = None
            try:
                res = sender.replaceTrack(new_track)
            except AttributeError:
                res = None

            if inspect.isawaitable(res):
                await res
                logger.info(
                    "Replaced track for %s using awaitable replaceTrack -> %dx%d@%d",
                    viewer_id,
                    width,
                    height,
                    fps,
                )
            elif res is None:
                if hasattr(sender, "replaceTrack"):
                    logger.info(
                        "Called replaceTrack (sync) for %s -> %dx%d@%d",
                        viewer_id,
                        width,
                        height,
                        fps,
                    )
                else:
                    logger.info(
                        "sender.replaceTrack not available, using fallback remove/add for %s",
                        viewer_id,
                    )
                    try:
                        pc.removeTrack(sender)
                    except Exception:
                        logger.debug(
                            "pc.removeTrack failed or not supported, continuing"
                        )
                    new_sender = pc.addTrack(new_track)
                    senders[viewer_id] = new_sender
                    logger.info(
                        "Fallback replaced by remove/add for %s", viewer_id
                    )
            else:
                logger.info(
                    "replaceTrack returned non-awaitable result for %s: %r",
                    viewer_id,
                    res,
                )
        except Exception:
            logger.exception(
                "Exception while attempting replaceTrack for %s, falling back",
                viewer_id,
            )
            try:
                pc.removeTrack(sender)
            except Exception:
                logger.debug(
                    "pc.removeTrack failed or not supported in fallback"
                )
            new_sender = pc.addTrack(new_track)
            senders[viewer_id] = new_sender
            logger.info(
                "Fallback replaced by remove/add for %s", viewer_id
            )

        try:
            await ws.send_json(
                {
                    "type": "update_ack",
                    "to": data.get("from") or viewer_id,
                    "from": STREAMER_ID,
                    "settings": {
                        "width": width,
                        "height": height,
                        "fps": fps,
                    },
                }
            )
        except Exception:
            pass
    except Exception:
        logger.exception("Failed to update track for %s", viewer_id)


async def add_candidate_to_pc(pc: RTCPeerConnection, candidate_dict: dict):
    if candidate_dict is None:
        await pc.addIceCandidate(None)
        return
    try:
        from aiortc import RTCIceCandidate  # type: ignore

        try:
            cobj = RTCIceCandidate(
                sdpMid=candidate_dict.get("sdpMid"),
                sdpMLineIndex=candidate_dict.get("sdpMLineIndex"),
                candidate=candidate_dict.get("candidate"),
            )
            await pc.addIceCandidate(cobj)
            return
        except TypeError:
            try:
                cobj = RTCIceCandidate(
                    candidate_dict.get("sdpMid"),
                    candidate_dict.get("sdpMLineIndex"),
                    candidate_dict.get("candidate"),
                )
                await pc.addIceCandidate(cobj)
                return
            except Exception:
                pass
    except Exception:
        pass
    parsed = ParsedIceCandidate(candidate_dict)
    await pc.addIceCandidate(parsed)


async def main():
    global DEVICE_CAPS
    devices = list_v4l2_devices()
    chosen = choose_device_by_keyword(devices, DEVICE_KEYWORDS)
    if chosen is None:
        chosen = DEVICE
    logger.info("Selected device: %s", chosen)

    DEVICE_CAPS = get_device_caps(chosen)
    logger.info(
        "Device caps (filtered FPS_MIN=%d): %s", FPS_MIN, DEVICE_CAPS
    )

    session = aiohttp.ClientSession()
    try:
        ws = await session.ws_connect(SIGNALING_URL)
    except Exception:
        logger.exception("WS connect failed")
        await session.close()
        return

    # register & send caps
    await ws.send_json({"type": "register", "id": STREAMER_ID})
    await asyncio.sleep(0.05)
    await ws.send_json(
        {
            "type": "caps",
            "caps": {"device": chosen, "caps": DEVICE_CAPS},
        }
    )
    logger.info("registered and sent caps")

    # open camera once
    try:
        await create_player(
            chosen,
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            fps=DEFAULT_FPS,
        )
    except Exception:
        logger.error("Failed to open device")
        await session.close()
        return

    async for msg in ws:
        if msg.type != aiohttp.WSMsgType.TEXT:
            continue
        try:
            data = json.loads(msg.data)
        except Exception:
            logger.warning("Invalid JSON: %s", msg.data)
            continue

        typ = data.get("type")
        if typ == "offer":
            peer_from = data.get("from")
            await handle_offer(ws, peer_from, data, chosen)
            continue
        if typ == "candidate":
            peer_from = data.get("from")
            candidate = data.get("candidate")
            if peer_from not in pcs:
                logger.warning(
                    "No pc for %s when receiving candidate", peer_from
                )
                continue
            pc = pcs[peer_from]
            try:
                await add_candidate_to_pc(pc, candidate)
            except Exception:
                logger.exception("addIceCandidate error for %s", peer_from)
            continue
        if typ == "request":
            action = data.get("action")
            peer_from = data.get("from")
            if action == "update":
                await handle_update_request(ws, peer_from, data)
            else:
                logger.warning("Unknown request action: %s", action)
            continue
        if typ == "update_ack":
            logger.info("Received update_ack: %s", data.get("settings"))
            continue
        if typ == "registered":
            logger.info("Registered: %s", data.get("id"))
            continue
        if typ == "error":
            logger.error("Signaling error: %s", data.get("message"))
            continue

    await session.close()


def _signal(sig, frame):
    logger.info("signal received, exiting")
    try:
        if player:
            player.stop()
    except Exception:
        pass
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal)
    signal.signal(signal.SIGTERM, _signal)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down")
