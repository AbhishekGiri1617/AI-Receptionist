"""
Piper Live Voice Bridge Server
Watches live conversation turns from `python -m voice_agent.main console`
and broadcasts them instantly to the Avatar UI for real-time lip-sync and dialogue display.
"""

import sys
import os
import json
import asyncio
from pathlib import Path
from aiohttp import web, WSMsgType

# Root paths
ROOT_DIR = Path(__file__).resolve().parent.parent
LIVE_TURN_FILE = ROOT_DIR / "data" / "transcripts" / "live_turn.json"
UI_DIR = Path(__file__).resolve().parent

# Ensure transcripts directory exists
LIVE_TURN_FILE.parent.mkdir(parents=True, exist_ok=True)

# Connected UI clients
connected_clients = set()
last_emitted_time_ms = 0


async def broadcast_event(data: dict):
    """Broadcast an event to all connected UI clients."""
    if not connected_clients:
        return
    message = json.dumps(data, ensure_ascii=False)
    for ws in list(connected_clients):
        if not ws.closed:
            try:
                await ws.send_str(message)
            except Exception:
                connected_clients.discard(ws)


async def live_turn_watcher():
    """Asynchronously watches data/transcripts/live_turn.json for new turns from voice_agent.main console."""
    global last_emitted_time_ms
    while True:
        try:
            if LIVE_TURN_FILE.exists():
                text = LIVE_TURN_FILE.read_text(encoding="utf-8").strip()
                if text:
                    data = json.loads(text)
                    time_ms = data.get("time_ms", 0)
                    if time_ms > last_emitted_time_ms:
                        last_emitted_time_ms = time_ms
                        # Format as live turn event
                        event = {
                            "type": "live_turn",
                            "role": data.get("role", "assistant"),
                            "speaker": data.get("speaker", "Piper"),
                            "text": data.get("text", ""),
                            "timestamp": data.get("timestamp", ""),
                            "time_ms": time_ms
                        }
                        speaker_label = "🤖 Piper" if event["role"] == "assistant" else "👤 Customer"
                        print(f"[{event['timestamp']}] {speaker_label}: {event['text']}")
                        await broadcast_event(event)
        except Exception as e:
            pass
        await asyncio.sleep(0.12)  # fast, low-overhead check (120ms)


async def websocket_handler(request):
    """Handles WebSocket connections from the Piper Avatar UI."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    connected_clients.add(ws)
    print(f"[UI] Avatar client connected ({len(connected_clients)} active)")

    # Send current agent info
    await ws.send_json({
        "type": "agent_info",
        "agent_name": "Piper",
        "company_name": "Salesforce / Mobius"
    })

    # If there's an existing live turn, send it immediately
    if LIVE_TURN_FILE.exists():
        try:
            latest = json.loads(LIVE_TURN_FILE.read_text(encoding="utf-8"))
            await ws.send_json({
                "type": "live_turn",
                **latest
            })
        except Exception:
            pass

    try:
        async for msg in ws:
            if msg.type == WSMsgType.ERROR:
                print(f"[UI] WebSocket error: {ws.exception()}")
    finally:
        connected_clients.discard(ws)
        print("[UI] Avatar client disconnected")

    return ws


async def index_handler(request):
    """Serves the Avatar UI index.html."""
    return web.FileResponse(UI_DIR / "index.html")


def create_app():
    """Create aiohttp web application."""
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_static("/", UI_DIR, show_index=True)
    return app


async def start_background_watcher(app):
    app["watcher_task"] = asyncio.create_task(live_turn_watcher())
    yield
    app["watcher_task"].cancel()
    await asyncio.gather(app["watcher_task"], return_exceptions=True)


def main():
    app = create_app()
    app.cleanup_ctx.append(start_background_watcher)
    
    port = int(os.environ.get("PORT", 8000))
    print("=" * 65)
    print("  Piper Live Voice Avatar Bridge Server")
    print(f"  • UI URL: http://localhost:{port}")
    print("  • Watching: python -m voice_agent.main console")
    print("=" * 65)

    web.run_app(app, host="localhost", port=port, print=None)


if __name__ == "__main__":
    main()
