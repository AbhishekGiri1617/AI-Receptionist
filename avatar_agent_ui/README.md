# Piper AI Voice Avatar — Live Console Sync & Lip-Sync

A focused, lightweight Avatar interface designed to synchronize live with:
```bash
python -m voice_agent.main console
```

---

## What It Does
1. **Avatar-Focused Viewport**:
   - 3D Animated Piper avatar (with a toggle for Photorealistic mode).
   - Audio-driven **real-time lip syncing** whenever the AI speaks.
   - Natural eye blinking, breathing, and speaking wave indicators.
   - Live status badge indicating whether Customer or Piper is speaking.

2. **Full Dialogue Feed (No Type Fields)**:
   - Displays all spoken conversation turns:
     - 👤 **Customer**: Exactly what the caller said.
     - 🤖 **Piper**: Exactly what the assistant said.
   - Automatic turn counter and clean timestamps.

3. **Zero Impact on Original Code**:
   - Lives completely inside `avatar_agent_ui/`.
   - Your core voice agent runs exactly as before.

---

## How to Use

### Step 1: Start the Avatar UI Server (once)
In a terminal, run:
```powershell
python avatar_agent_ui/piper_bridge.py
```
Open **`http://localhost:8000`** in Chrome or Edge.

---

### Step 2: Run your Voice Agent in Console Mode
In your main terminal, run:
```powershell
python -m voice_agent.main console
```

1. Speak into your microphone as the customer.
2. The UI immediately displays `👤 Customer: [what you said]`.
3. Piper responds in voice through your speakers/console.
4. The UI displays `🤖 Piper: [what AI said]` and **Piper's lips sync directly with her speech**!
