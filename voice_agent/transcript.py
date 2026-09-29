import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config import business_config


TRANSCRIPTS_DIR = Path("data") / "transcripts"
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
MASTER_LOG_FILE = TRANSCRIPTS_DIR / "conversations.jsonl"


class TranscriptRecorder:
    """Records voice conversation turns into human-readable Markdown and structured JSONL."""

    def __init__(self, room_name: str = "console-room"):
        self.room_name = room_name
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.company_name = business_config.get("company_name", "Our Company")
        self.agent_name = business_config.get("agent_name", "Assistant")
        self.turns: List[dict] = []
        self._saved = False

    def add_message(self, role: str, text: str):
        """Add a conversation message turn."""
        text = text.strip() if text else ""
        if not text:
            return

        # Skip internal system prompt messages like "Greet the user warmly as..."
        if role == "system" or text.startswith("Greet the caller") or text.startswith("Greet the user"):
            return

        speaker = self.agent_name if role == "assistant" else "Caller"
        timestamp = datetime.now().strftime("%H:%M:%S")

        turn_data = {
            "timestamp": timestamp,
            "role": role,
            "speaker": speaker,
            "text": text,
            "time_ms": int(datetime.now().timestamp() * 1000)
        }
        self.turns.append(turn_data)

        # Broadcast live turn for real-time Avatar UI lip-sync and dialogue display
        try:
            live_turn_path = TRANSCRIPTS_DIR / "live_turn.json"
            live_turn_path.write_text(json.dumps(turn_data, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def save(self) -> Optional[Path]:
        """Save the conversation transcript to both a Markdown file and master JSONL log."""
        if self._saved or not self.turns:
            return None

        self._saved = True
        self.end_time = datetime.now()
        duration_sec = int((self.end_time - self.start_time).total_seconds())
        duration_str = f"{duration_sec // 60}m {duration_sec % 60}s" if duration_sec >= 60 else f"{duration_sec}s"

        timestamp_str = self.start_time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"call_{timestamp_str}.md"
        filepath = TRANSCRIPTS_DIR / filename

        # 1. Format Markdown
        lines = [
            f"# 📞 Call Transcript — {self.company_name}",
            f"- **Date**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **Agent**: {self.agent_name}",
            f"- **Caller**: Caller",
            f"- **Room ID**: `{self.room_name}`",
            f"- **Call Duration**: {duration_str}",
            f"- **Total Turns**: {len(self.turns)}",
            "",
            "---",
            "",
            "### 💬 Conversation Dialogue:",
            "",
        ]

        for turn in self.turns:
            time = turn["timestamp"]
            speaker = turn["speaker"]
            text = turn["text"]

            if turn["role"] == "assistant":
                lines.append(f"**[{time}] 🤖 {speaker}:**")
                lines.append(f"> {text}\n")
            else:
                lines.append(f"**[{time}] 👤 {speaker}:**")
                lines.append(f"> *\"{text}\"*\n")

        lines.extend([
            "---",
            f"*Transcript automatically recorded by {self.agent_name} Voice Agent.*",
            "",
        ])

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"[TRANSCRIPT SAVED] {filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to save markdown transcript: {e}")

        # 2. Append to Master JSONL
        master_entry = {
            "call_id": f"call_{timestamp_str}",
            "room_name": self.room_name,
            "company_name": self.company_name,
            "agent_name": self.agent_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": duration_sec,
            "total_turns": len(self.turns),
            "turns": self.turns,
            "transcript_file": str(filepath),
        }

        try:
            with open(MASTER_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(master_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[ERROR] Failed to append to master transcript log: {e}")

        return filepath
