/**
 * Phoneme-Accurate 3D Avatar Lip-Sync Engine
 * Maps actual spoken words and letters to real phonetic visemes:
 * - 'round' (O, U, W, OO): "Hello", "Bloom", "how", "you", "to"
 * - 'wide' (EE, I, S, Y): "this", "is", "see", "meet", "assist"
 * - 'open' (AA, AH, O): "Abhishek", "can", "ask", "have", "customer"
 * - 'closed' (M, B, P, pause): Bilabial closure on "Mobius", "Bloom", "from"
 * - 'mid' (E, R, L, N, T): Neutral conversational transition
 */

class AvatarEngine {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');

    // Base Frame (Closed mouth, natural smile)
    this.baseImage = new Image();
    this.baseImage.src = 'assets/piper_idle.jpg';

    // Real Viseme Patches (Smooth alpha-feathered, zero body jitter)
    this.visemes = {
      round: new Image(),
      wide: new Image(),
      open: new Image(),
      mid: new Image(),
      blink: new Image()
    };

    this.visemes.round.src = 'assets/patch_mouth_round.png';
    this.visemes.wide.src = 'assets/patch_mouth_wide.png';
    this.visemes.open.src = 'assets/patch_mouth_open.png';
    this.visemes.mid.src = 'assets/patch_mouth_mid.png';
    this.visemes.blink.src = 'assets/patch_eyes_blink.png';

    // State
    this.isSpeaking = false;
    this.currentViseme = 'closed'; // 'closed', 'round', 'wide', 'open', 'mid'
    this.visemeWeights = {
      round: 0,
      wide: 0,
      open: 0,
      mid: 0
    };

    // Phoneme Timeline Queue
    this.timeline = [];
    this.timelineStartTime = 0;
    this.totalDuration = 0;

    // Blinking
    this.isBlinking = false;
    this.blinkAlpha = 0;
    this.nextBlinkTime = performance.now() + 2500;
    this.blinkDuration = 120; // ms

    // Head micro-motion & breathing
    this.breathingTime = 0;
    this.headNod = 0;

    this.resizeCanvas();
    window.addEventListener('resize', () => this.resizeCanvas());
    requestAnimationFrame((t) => this.renderLoop(t));
  }

  resizeCanvas() {
    const w = window.innerWidth || document.documentElement.clientWidth;
    const h = window.innerHeight || document.documentElement.clientHeight;
    if (w > 0 && h > 0) {
      this.canvas.width = w * (window.devicePixelRatio || 1);
      this.canvas.height = h * (window.devicePixelRatio || 1);
    }
  }

  /**
   * Generates a phonetically accurate viseme timeline from spoken text.
   * Recognizes bilabials (M, B, P), rounded vowels (O, U, W), wide vowels (EE, I),
   * open vowels (AA, AH), and natural word/sentence breath pauses.
   */
  speakText(text) {
    if (!text || !text.trim()) return;

    this.isSpeaking = true;
    this.timeline = [];
    let currentTime = 0;

    // Split text into words while keeping punctuation
    const words = text.trim().split(/\s+/);

    for (let wIndex = 0; wIndex < words.length; wIndex++) {
      let word = words[wIndex].toLowerCase();
      const hasComma = word.includes(',');
      const hasPeriod = word.includes('.') || word.includes('?') || word.includes('!');
      word = word.replace(/[^a-z]/g, '');

      if (!word) continue;

      // Phonetic parsing for the word
      let i = 0;
      while (i < word.length) {
        const char = word[i];
        const nextChar = word[i + 1] || '';
        const combo = char + nextChar;

        let viseme = 'mid';
        let dur = 75; // ms

        // 1. Bilabials (M, B, P) -> MUST CLOSE LIPS!
        if (char === 'm' || char === 'b' || char === 'p') {
          viseme = 'closed';
          dur = 85;
          i += 1;
        }
        // 2. Rounded Vowels (OO, OU, O, U, W) -> Pucker round O lips
        else if (combo === 'oo' || combo === 'ou' || combo === 'ow') {
          viseme = 'round';
          dur = 130;
          i += 2;
        } else if (char === 'o' || char === 'u' || char === 'w') {
          viseme = 'round';
          dur = 100;
          i += 1;
        }
        // 3. Wide Spread Vowels & Sibilants (EE, EA, I, E, Y, S, Z)
        else if (combo === 'ee' || combo === 'ea' || combo === 'ie') {
          viseme = 'wide';
          dur = 120;
          i += 2;
        } else if (char === 'i' || char === 'e' || char === 'y' || char === 's') {
          viseme = 'wide';
          dur = 85;
          i += 1;
        }
        // 4. Open Vowels (AA, AH, A, O in 'mobius')
        else if (char === 'a') {
          viseme = 'open';
          dur = 95;
          i += 1;
        }
        // 5. Consonants & Dentals (T, D, N, L, R, K, G, F, V) -> Mid
        else {
          viseme = 'mid';
          dur = 65;
          i += 1;
        }

        this.timeline.push({
          start: currentTime,
          end: currentTime + dur,
          viseme: viseme
        });
        currentTime += dur;
      }

      // Inter-word micro pause (35ms - 50ms)
      currentTime += 35;

      // Punctuation breath pauses
      if (hasComma) {
        this.timeline.push({ start: currentTime, end: currentTime + 160, viseme: 'closed' });
        currentTime += 160;
      } else if (hasPeriod) {
        this.timeline.push({ start: currentTime, end: currentTime + 240, viseme: 'closed' });
        currentTime += 240;
      }
    }

    // Trailing closure
    this.timeline.push({ start: currentTime, end: currentTime + 100, viseme: 'closed' });
    currentTime += 100;

    this.totalDuration = currentTime;
    this.timelineStartTime = performance.now();
  }

  setSpeaking(speaking) {
    this.isSpeaking = speaking;
    if (!speaking) {
      this.timeline = [];
      this.targetWeights = { round: 0, wide: 0, open: 0, mid: 0 };
    }
  }

  renderLoop(time) {
    this.updateAnimation(time);
    this.draw(time);
    requestAnimationFrame((t) => this.renderLoop(t));
  }

  updateAnimation(time) {
    // 1. Natural Eye Blinking
    if (time > this.nextBlinkTime && !this.isBlinking) {
      this.isBlinking = true;
      this.blinkStartTime = time;
    }

    if (this.isBlinking) {
      const elapsed = time - this.blinkStartTime;
      const progress = elapsed / this.blinkDuration;
      if (progress < 0.5) {
        this.blinkAlpha = progress * 2;
      } else if (progress <= 1.0) {
        this.blinkAlpha = (1.0 - progress) * 2;
      } else {
        this.isBlinking = false;
        this.blinkAlpha = 0;
        this.nextBlinkTime = time + 2800 + Math.random() * 2000;
      }
    }

    // 2. Process Phoneme Timeline
    let activeViseme = 'closed';
    let nodTarget = 0;

    if (this.isSpeaking && this.timeline.length > 0) {
      const elapsed = time - this.timelineStartTime;

      if (elapsed > this.totalDuration) {
        // Speech finished
        this.isSpeaking = false;
        this.timeline = [];
        activeViseme = 'closed';
      } else {
        // Find current viseme in timeline
        for (let j = 0; j < this.timeline.length; j++) {
          const item = this.timeline[j];
          if (elapsed >= item.start && elapsed <= item.end) {
            activeViseme = item.viseme;
            break;
          }
        }
      }

      // Add gentle head nod on open vowels and syllable emphasis
      if (activeViseme === 'open' || activeViseme === 'round') {
        nodTarget = 1.4;
      } else if (activeViseme !== 'closed') {
        nodTarget = 0.6;
      }
    } else if (this.isSpeaking) {
      // Fallback cadence if no text was provided
      const t = time * 0.01;
      const cycle = Math.sin(t * 1.8);
      if (cycle > 0.4) activeViseme = 'round';
      else if (cycle > -0.1) activeViseme = 'wide';
      else if (cycle > -0.6) activeViseme = 'mid';
      else activeViseme = 'closed';
    }

    this.currentViseme = activeViseme;
    this.headNod += (nodTarget - this.headNod) * 0.25;

    // Smooth viseme weight interpolation (zero flickering, smooth fluid mouth articulation)
    const blendSpeed = 0.38;
    for (const key of ['round', 'wide', 'open', 'mid']) {
      const target = (key === activeViseme) ? 1.0 : 0.0;
      this.visemeWeights[key] += (target - this.visemeWeights[key]) * blendSpeed;
      if (this.visemeWeights[key] < 0.01) this.visemeWeights[key] = 0;
    }
  }

  draw(time) {
    const { ctx, canvas } = this;
    const w = canvas.width;
    const h = canvas.height;
    if (w === 0 || h === 0) return;

    ctx.clearRect(0, 0, w, h);

    const base = this.baseImage;
    if (!base.complete) return;

    // Breathing offset + speech head nod
    this.breathingTime += 0.03;
    const breathingY = Math.sin(this.breathingTime) * 1.4 + this.headNod;

    // Aspect-ratio cover
    const imgRatio = base.width / base.height;
    const canvasRatio = w / h;
    let renderW, renderH, offsetX, offsetY;

    if (canvasRatio > imgRatio) {
      renderW = w;
      renderH = w / imgRatio;
      offsetX = 0;
      offsetY = (h - renderH) / 2 + breathingY;
    } else {
      renderH = h;
      renderW = h * imgRatio;
      offsetX = (w - renderW) / 2;
      offsetY = breathingY;
    }

    ctx.save();

    // 1. Draw Master Base Frame (Completely solid, closed mouth, sharp 3D render)
    ctx.drawImage(base, offsetX, offsetY, renderW, renderH);

    // 2. Draw Active Phonetic Visemes with Smooth Alpha Blending
    for (const [key, weight] of Object.entries(this.visemeWeights)) {
      if (weight > 0.02) {
        const patch = this.visemes[key];
        if (patch && patch.complete) {
          ctx.globalAlpha = Math.min(1.0, weight);
          ctx.drawImage(patch, offsetX, offsetY, renderW, renderH);
        }
      }
    }

    // 3. Draw Natural Eye Blink Patch
    if (this.blinkAlpha > 0.02) {
      const blinkPatch = this.visemes.blink;
      if (blinkPatch && blinkPatch.complete) {
        ctx.globalAlpha = Math.min(1.0, this.blinkAlpha);
        ctx.drawImage(blinkPatch, offsetX, offsetY, renderW, renderH);
      }
    }

    ctx.restore();
  }
}

window.AvatarEngine = AvatarEngine;
