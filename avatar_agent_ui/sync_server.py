"""
Avatar UI Live Synchronization Server
Runs alongside `python -m voice_agent.main console`.
Starts a WebSocket broadcast server and serves the Avatar UI so the browser
shows customer/AI dialogue in real-time and animates Piper's lip sync synchronously.
"""

import os
import sys
import json
import asyncio
import threading
import http.server
import socketserver
from pathlib import Path

# Paths
UI_DIR = Path(__file__).resolve().parent
ROOT_DIR = UI_DIR.parent
LIVE_FILE = ROOT_DIR / "data" / "transcripts" / "live_turn.json"

WS_PORT = 8765
HTTP_PORT = 8000


class AvatarSyncServer:
    def __init__(self):
        self._started = False
        self._loop = None
        self._clients = set()
        self._lock = threading.Lock()

    def ensure_started(self):
        """Starts WebSocket and HTTP servers in background daemon threads."""
        with self._lock:
            if self._started:
                return
            self._started = True

        # Start HTTP static server in background thread
        http_thread = threading.Thread(target=self._run_http_server, daemon=True)
        http_thread.start()

        # Start WebSocket broadcast server in background thread
        ws_thread = threading.Thread(target=self._run_ws_server, daemon=True)
        ws_thread.start()

        print("\n" + "=" * 60)
        print("  [AVATAR UI] Piper Avatar UI Synced with LiveKit Console")
        print(f"  * Open in your browser: http://localhost:{HTTP_PORT}")
        print("=" * 60 + "\n")

    def _run_http_server(self):
        """Serves avatar_agent_ui files on port 8000."""
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(UI_DIR), **kwargs)

            def log_message(self, format, *args):
                pass  # suppress noisy static HTTP access logs

        try:
            socketserver.TCPServer.allow_reuse_address = True
            with socketserver.TCPServer(("0.0.0.0", HTTP_PORT), Handler) as httpd:
                httpd.serve_forever()
        except OSError:
            pass  # Port already in use

    def _run_ws_server(self):
        """Runs the WebSocket server on port 8765."""
        import websockets

        async def handler(websocket):
            self._clients.add(websocket)
            try:
                # Send welcome event
                await websocket.send(json.dumps({
                    "type": "connected",
                    "status": "synced"
                }))
                # If there's an existing live turn, send it
                if LIVE_FILE.exists():
                    try:
                        latest = json.loads(LIVE_FILE.read_text(encoding="utf-8"))
                        await websocket.send(json.dumps({
                            "type": "conversation_item",
                            "role": latest.get("role", "assistant"),
                            "text": latest.get("text", "")
                        }))
                    except Exception:
                        pass
                await websocket.wait_closed()
            finally:
                self._clients.discard(websocket)

        async def main():
            self._loop = asyncio.get_running_loop()
            try:
                async with websockets.serve(handler, "0.0.0.0", WS_PORT):
                    await asyncio.Future()  # run forever
            except OSError:
                pass

        asyncio.run(main())

    def broadcast(self, payload: dict):
        """Thread-safe event broadcast to all connected web clients."""
        if not self._clients or not self._loop:
            return

        msg = json.dumps(payload, ensure_ascii=False)

        def _send():
            dead_clients = set()
            for ws in list(self._clients):
                try:
                    asyncio.create_task(ws.send(msg))
                except Exception:
                    dead_clients.add(ws)
            self._clients.difference_update(dead_clients)

        try:
            self._loop.call_soon_threadsafe(_send)
        except Exception:
            pass

    def notify_agent_state(self, state: str):
        """Called when LiveKit AgentStateChangedEvent triggers (speaking, listening, idle)."""
        self.broadcast({
            "type": "agent_state",
            "state": state
        })

    def notify_user_speech(self, transcript: str, is_final: bool = True):
        """Called when LiveKit UserInputTranscribedEvent triggers."""
        self.broadcast({
            "type": "user_speech",
            "text": transcript,
            "is_final": is_final
        })

    def notify_conversation_item(self, role: str, text: str):
        """Called when conversation item is committed."""
        self.broadcast({
            "type": "conversation_item",
            "role": role,
            "text": text
        })
        # Save to disk as well
        try:
            LIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
            LIVE_FILE.write_text(json.dumps({
                "role": role,
                "text": text
            }, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass


# Global singleton instance
avatar_sync = AvatarSyncServer()
