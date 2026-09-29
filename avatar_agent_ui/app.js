/**
 * Piper Live Voice Avatar & Real-Time Sync Controller
 * Automatically synchronizes with `python -m voice_agent.main console` via WebSocket (port 8765).
 * Real-time lip-sync triggered on agent speaking states and live dialogue feed.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const connectionStatus = document.getElementById('connectionStatus');
  const statusText = document.getElementById('statusText');
  const voiceStatusPill = document.getElementById('voiceStatusPill');
  const pillIcon = document.getElementById('pillIcon');
  const pillLabel = document.getElementById('pillLabel');
  const speakingWave = document.getElementById('speakingWave');
  const conversationStream = document.getElementById('conversationStream');
  const emptyState = document.getElementById('emptyState');
  const turnCountEl = document.getElementById('turnCount');
  const avatarModeToggle = document.getElementById('avatarModeToggle');

  // Initialize Avatar Lip-Sync Engine
  const avatarEngine = new AvatarEngine('avatarCanvas');

  // State
  let ws = null;
  let turnCount = 0;
  let currentCustomerTurnEl = null;

  // 1. Connect to sync server on 127.0.0.1:8765
  function connectSyncSocket() {
    const wsUrl = "ws://127.0.0.1:8765";

    try {
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log("[Avatar UI] Connected to LiveKit Console bridge:", wsUrl);
        connectionStatus.classList.add('connected');
        statusText.textContent = "LiveKit Console Synced";
        updateVoiceStatus('ready', 'Voice Agent Ready');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'agent_state') {
            handleAgentState(data.state);
          } else if (data.type === 'user_speech') {
            handleUserSpeech(data.text, data.is_final);
          } else if (data.type === 'conversation_item') {
            handleConversationItem(data.role, data.text);
          }
        } catch (e) {
          console.error("Parse error:", e);
        }
      };

      ws.onclose = () => {
        connectionStatus.classList.remove('connected');
        statusText.textContent = "Waiting for console...";
        updateVoiceStatus('waiting', 'Run: python -m voice_agent.main console');
        setTimeout(connectSyncSocket, 1200);
      };

      ws.onerror = () => {
        // Handled in onclose
      };
    } catch (e) {
      setTimeout(connectSyncSocket, 2000);
    }
  }

  // 2. Handle Agent Speaking / Listening State (Drives Real-Time Lip Sync)
  function handleAgentState(state) {
    console.log("[Agent State]:", state);

    if (state === 'speaking') {
      // Agent is speaking: animate lips and mouth in real time!
      avatarEngine.setSpeaking(true);
      speakingWave.classList.add('active');
      updateVoiceStatus('ai-speaking', 'Piper Speaking (Lip-Syncing)...');
    } else {
      // Agent finished speaking or is listening
      avatarEngine.setSpeaking(false);
      speakingWave.classList.remove('active');
      if (state === 'listening') {
        updateVoiceStatus('customer-speaking', 'Listening to Customer...');
      } else {
        updateVoiceStatus('ready', 'Voice Agent Ready');
      }
    }
  }

  // 3. Handle Live Customer Speech Transcriptions
  function handleUserSpeech(transcript, isFinal) {
    if (!transcript || !transcript.trim()) return;

    if (emptyState) emptyState.style.display = 'none';

    if (!currentCustomerTurnEl) {
      // Create new customer turn bubble
      turnCount++;
      turnCountEl.textContent = `${turnCount} turn${turnCount === 1 ? '' : 's'}`;

      currentCustomerTurnEl = document.createElement('div');
      currentCustomerTurnEl.className = 'dialogue-turn customer-turn';
      currentCustomerTurnEl.innerHTML = `
        <div class="turn-header">
          <span class="speaker-badge"><span>👤</span> <span>Customer</span></span>
          <span class="turn-time">${new Date().toLocaleTimeString()}</span>
        </div>
        <div class="turn-content">${transcript}</div>
      `;
      conversationStream.appendChild(currentCustomerTurnEl);
    } else {
      // Update text in-place
      const contentEl = currentCustomerTurnEl.querySelector('.turn-content');
      if (contentEl) contentEl.textContent = transcript;
    }

    conversationStream.scrollTop = conversationStream.scrollHeight;

    if (isFinal) {
      currentCustomerTurnEl = null; // Commit turn
    }
  }

  // 4. Handle Conversation Turns (AI or Customer)
  function handleConversationItem(role, text) {
    if (!text || !text.trim()) return;

    if (emptyState) emptyState.style.display = 'none';

    // If it's an assistant message, append clean AI card
    if (role === 'assistant') {
      currentCustomerTurnEl = null;
      turnCount++;
      turnCountEl.textContent = `${turnCount} turn${turnCount === 1 ? '' : 's'}`;

      const aiCard = document.createElement('div');
      aiCard.className = 'dialogue-turn ai-turn speaking-now';
      aiCard.innerHTML = `
        <div class="turn-header">
          <span class="speaker-badge"><span>🤖</span> <span>Piper</span></span>
          <span class="turn-time">${new Date().toLocaleTimeString()}</span>
        </div>
        <div class="turn-content">${text}</div>
      `;
      conversationStream.appendChild(aiCard);
      conversationStream.scrollTop = conversationStream.scrollHeight;

      // Start phonetic text-to-viseme lip sync
      avatarEngine.speakText(text);
      speakingWave.classList.add('active');

      const wordCount = text.split(/\s+/).length;
      const durationMs = Math.max(2200, Math.min(16000, wordCount * 360));
      setTimeout(() => {
        aiCard.classList.remove('speaking-now');
        speakingWave.classList.remove('active');
        updateVoiceStatus('ready', 'Listening to Customer...');
      }, durationMs);
    }
  }

  // 5. Update Status Pill
  function updateVoiceStatus(state, label) {
    voiceStatusPill.className = 'voice-status-pill';
    pillLabel.textContent = label;

    if (state === 'ai-speaking') {
      voiceStatusPill.classList.add('speaking');
      pillIcon.textContent = '🗣️';
    } else if (state === 'customer-speaking') {
      voiceStatusPill.classList.add('customer-talking');
      pillIcon.textContent = '🎙️';
    } else if (state === 'waiting') {
      pillIcon.textContent = '⏳';
    } else {
      pillIcon.textContent = '🎙️';
    }
  }

  if (avatarModeToggle) {
    avatarModeToggle.addEventListener('click', () => {
      // Piper 3D Active
    });
  }

  const btnToggleHud = document.getElementById('btnToggleHud');
  const dialogueHud = document.getElementById('dialogueHud');
  if (btnToggleHud && dialogueHud) {
    btnToggleHud.addEventListener('click', () => {
      dialogueHud.classList.toggle('minimized');
    });
  }

  // Start connection
  connectSyncSocket();
});
