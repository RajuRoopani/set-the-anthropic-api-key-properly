/**
 * AI Interview Platform — app.js
 * Handles WebSocket communication, UI state management, and AI streaming.
 * Supports text mode interviews powered by Claude AI.
 *
 * VOICE MODE: Full duplex v2.0 — mic captures and streams continuously while AI
 * audio plays back simultaneously. Like a phone call, not a walkie-talkie.
 * No push-to-talk, no turn-gating, no isAISpeaking mic gate.
 *
 * Vanilla JS, no frameworks.
 */

'use strict';

(function () {
  // ─── Constants ───────────────────────────────────────────────────────────────
  const MAX_CHARS = 2000;
  const CHAR_WARN_THRESHOLD = 1800;

  /**
   * Audio capture settings for the voice mode mic pipeline.
   * AI service requires: 16-bit PCM, 16 kHz, mono.
   */
  const CAPTURE_SAMPLE_RATE = 16000;
  const CAPTURE_CHANNELS    = 1;

  /**
   * Default audio playback sample rate.
   * Overridden at runtime by session_ready data.output_sample_rate.
   */
  const DEFAULT_PLAYBACK_SAMPLE_RATE = 24000;

  // ─── State ───────────────────────────────────────────────────────────────────
  let ws              = null;
  let currentAIBubble = null;   // Active streaming AI message element (text mode)
  let isAIResponding  = false;
  let interviewMode   = 'text'; // 'text' | 'voice'

  // ─── Voice State ─────────────────────────────────────────────────────────────
  //
  // Full-duplex v2.0 model:
  //   - audioContext: created on session_ready with server-provided sample rate
  //   - micStream: always open while in voice mode; NEVER paused for AI playback
  //   - nextPlayTime: gapless playback clock — reset on ai_interrupted to flush queue
  //   - No isAISpeaking mic gate — both directions always open simultaneously
  //
  let audioContext       = null;  // AudioContext (created on session_ready)
  let micStream          = null;  // MediaStream from getUserMedia
  let audioWorkletNode   = null;  // AudioWorkletNode (mic-capture-processor)
  let micSourceNode      = null;  // MediaStreamAudioSourceNode
  let micAnalyserNode    = null;  // AnalyserNode for mic visualizer
  let aiAnalyserNode     = null;  // AnalyserNode for AI output visualizer
  let aiGainNode         = null;  // GainNode for AI output (enables speaker mute)
  let isMicMuted         = false; // True = chunks not sent but pipeline stays active
  let isSpeakerMuted     = false; // True = AI audio gain=0
  let isVoiceActive      = false; // True = voice pipeline is fully running
  let micVizFrameId      = null;  // rAF ID for mic waveform loop
  let aiVizFrameId       = null;  // rAF ID for AI waveform loop
  let nextPlayTime       = 0.0;   // AudioContext time when next chunk should be scheduled
  let currentSourceNodes = [];    // Active AudioBufferSourceNodes (for cleanup)
  let recTimerInterval   = null;  // setInterval ID for recording elapsed timer
  let recStartTime       = 0;     // Date.now() when voice session started
  // Tracks whether AI audio is currently arriving/playing (for UI indicators only —
  // does NOT gate the mic in any way).
  let aiAudioActive      = false;

  // ─── DOM References (populated in init) ──────────────────────────────────────
  let setupScreen, chatScreen, userNameInput, roleSelect, startBtn;
  let chatMessages, messageInput, sendBtn, endBtn;
  let typingIndicator, connectionStatus, currentRoleDisplay, charCount;

  // Voice UI elements (added by UX engineer; may be null in text-only builds)
  let interviewModeSelect, micBtn, speakerBtn, endBtnVoice;
  let audioVisualizer, recordingIndicator, aiSpeakingIndicator;
  let voiceInputBar, textInputBar, voiceModeBadge, voiceModeTip;
  let startBtnIcon, startBtnLabel, voiceControls;
  let callTimerEl, userSpeakingIndicator; // v2 full-duplex additions

  // ─── Initialization ───────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    // --- Core DOM ---
    setupScreen        = document.getElementById('setup-screen');
    chatScreen         = document.getElementById('chat-screen');
    userNameInput      = document.getElementById('user-name');
    roleSelect         = document.getElementById('role-select');
    startBtn           = document.getElementById('start-btn');
    chatMessages       = document.getElementById('chat-messages');
    messageInput       = document.getElementById('message-input');
    sendBtn            = document.getElementById('send-btn');
    endBtn             = document.getElementById('end-btn');
    typingIndicator    = document.getElementById('typing-indicator');
    connectionStatus   = document.getElementById('connection-status');
    currentRoleDisplay = document.getElementById('current-role');
    charCount          = document.getElementById('char-count');

    // --- Voice UI DOM (optional — only present when UX engineer's HTML is deployed) ---
    interviewModeSelect = document.getElementById('interview-mode');
    micBtn              = document.getElementById('mic-btn');
    speakerBtn          = document.getElementById('speaker-btn');
    endBtnVoice         = document.getElementById('end-btn-voice');
    audioVisualizer     = document.getElementById('audio-visualizer');
    recordingIndicator  = document.getElementById('recording-indicator');
    aiSpeakingIndicator = document.getElementById('ai-speaking-indicator');
    voiceInputBar       = document.getElementById('voice-input-bar');
    textInputBar        = document.getElementById('text-input-bar');
    voiceModeBadge        = document.getElementById('voice-mode-toggle');
    voiceModeTip          = document.getElementById('voice-mode-tip');
    startBtnIcon          = document.getElementById('start-btn-icon');
    startBtnLabel         = document.getElementById('start-btn-label');
    voiceControls         = document.getElementById('voice-controls');
    callTimerEl           = document.getElementById('call-timer');
    userSpeakingIndicator = document.getElementById('user-speaking-indicator');

    setConnectionStatus('disconnected');

    // Start button disabled until roles are fetched
    startBtn.disabled = true;
    fetchRoles().then(attachEventListeners);
  });

  // ─── Fetch Roles ─────────────────────────────────────────────────────────────
  /**
   * Fetches available interview roles from the API and populates the dropdown.
   * Clears ALL existing options to avoid hardcoded stale values from HTML.
   */
  async function fetchRoles() {
    try {
      const response = await fetch('/api/config/roles');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      const roles = Array.isArray(data) ? data : (data.roles || []);
      _populateRoleSelect(roles);
    } catch (err) {
      console.error('Failed to fetch roles:', err);
      _populateRoleSelect(['Software Engineer', 'Data Scientist', 'Product Manager', 'System Design']);
    }
  }

  /** Clears and repopulates the role <select> with the given list. */
  function _populateRoleSelect(roles) {
    while (roleSelect.options.length > 0) roleSelect.remove(0);

    const placeholder = document.createElement('option');
    placeholder.value       = '';
    placeholder.disabled    = true;
    placeholder.selected    = true;
    placeholder.textContent = 'Select a role…';
    roleSelect.appendChild(placeholder);

    roles.forEach((role) => {
      const option = document.createElement('option');
      option.value       = typeof role === 'string' ? role : (role.id || role.name);
      option.textContent = typeof role === 'string' ? role : role.name;
      roleSelect.appendChild(option);
    });
  }

  // ─── Event Listeners ─────────────────────────────────────────────────────────
  function attachEventListeners() {
    roleSelect.addEventListener('change', () => {
      startBtn.disabled = !roleSelect.value;
    });

    if (interviewModeSelect) {
      // #interview-mode is a <div role="radiogroup"> with .mode-btn children,
      // NOT a <select>. Listen for clicks on the buttons; read dataset.mode.
      const modeBtns = interviewModeSelect.querySelectorAll('.mode-btn');
      modeBtns.forEach((btn) => {
        btn.addEventListener('click', () => {
          const newMode = btn.dataset.mode || 'text';
          modeBtns.forEach((b) => {
            const isActive = b === btn;
            b.classList.toggle('mode-btn--active', isActive);
            b.setAttribute('aria-checked', String(isActive));
          });
          interviewModeSelect.setAttribute('data-mode', newMode);
          interviewMode = newMode;
          _updateModeUI();
        });
      });
      // Read initial mode from the radiogroup's data-mode attribute (set in HTML)
      interviewMode = interviewModeSelect.dataset.mode || 'text';
      _updateModeUI();
    }

    startBtn.addEventListener('click', startInterview);
    sendBtn.addEventListener('click', sendMessage);

    messageInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    messageInput.addEventListener('input', () => {
      sendBtn.disabled = !messageInput.value.trim() || isAIResponding;
      const len = messageInput.value.length;
      if (charCount) {
        charCount.textContent = `${len}/${MAX_CHARS}`;
        charCount.classList.toggle('warn', len > CHAR_WARN_THRESHOLD);
      }
      messageInput.style.height = 'auto';
      messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
    });

    endBtn.addEventListener('click', endInterview);
    if (endBtnVoice) endBtnVoice.addEventListener('click', endInterview);

    // Full-duplex: mic button = mic MUTE toggle, not record/stop.
    if (micBtn)     micBtn.addEventListener('click', toggleMicMute);
    if (speakerBtn) speakerBtn.addEventListener('click', toggleSpeaker);

    // Header badge: mode indicator (informational during a call)
    if (voiceModeBadge) {
      voiceModeBadge.addEventListener('click', () => {
        const newMode = interviewMode === 'voice' ? 'text' : 'voice';
        interviewMode = newMode;
        if (interviewModeSelect) {
          interviewModeSelect.setAttribute('data-mode', newMode);
          const modeBtns = interviewModeSelect.querySelectorAll('.mode-btn');
          modeBtns.forEach((b) => {
            const isActive = b.dataset.mode === newMode;
            b.classList.toggle('mode-btn--active', isActive);
            b.setAttribute('aria-checked', String(isActive));
          });
        }
        _updateModeUI();
      });
    }

    window.addEventListener('beforeunload', _cleanupVoice);
  }

  /**
   * Updates body data-attribute and button disabled states to reflect mode.
   * Also toggles the voice/text input bars, updates the header badge,
   * the voice-mode-tip hint, and the start button icon/label.
   */
  function _updateModeUI() {
    document.body.setAttribute('data-interview-mode', interviewMode);
    const isVoice = interviewMode === 'voice';

    if (voiceInputBar) voiceInputBar.hidden = !isVoice;
    if (textInputBar)  textInputBar.hidden  = isVoice;

    if (voiceModeBadge) {
      voiceModeBadge.setAttribute('data-mode', interviewMode);
      voiceModeBadge.setAttribute(
        'aria-label',
        isVoice ? 'Currently in voice mode' : 'Currently in text mode'
      );
      const iconEl  = voiceModeBadge.querySelector('.voice-mode-badge-icon');
      const labelEl = voiceModeBadge.querySelector('.voice-mode-badge-label');
      if (iconEl)  iconEl.textContent  = isVoice ? '🎙' : '💬';
      if (labelEl) labelEl.textContent = isVoice ? 'Voice' : 'Text';
    }

    if (voiceModeTip) {
      voiceModeTip.hidden = !isVoice;
      if (isVoice) {
        voiceModeTip.textContent = 'Voice mode is not available with the current AI provider. Please use text mode.';
      }
    }

    if (startBtnIcon)  startBtnIcon.textContent  = isVoice ? '🎙' : '💬';
    if (startBtnLabel) startBtnLabel.textContent  = isVoice ? 'Start Voice Interview' : 'Start Interview';

    if (micBtn)     micBtn.disabled     = true;
    if (speakerBtn) speakerBtn.disabled = !isVoice;
  }


  // ─── Start Interview (branches on mode) ──────────────────────────────────────
  function startInterview() {
    const selectedRole = roleSelect.value;
    if (!selectedRole) {
      alert('Please select an interview role to continue.');
      return;
    }
    const userName = (userNameInput && userNameInput.value.trim()) || 'Candidate';

    startBtn.disabled = true;
    setupScreen.style.display = 'none';
    chatScreen.style.display  = 'flex';
    chatScreen.classList.add('fade-in');
    chatMessages.innerHTML = '';

    if (currentRoleDisplay) currentRoleDisplay.textContent = selectedRole;

    if (interviewMode === 'voice') {
      _startVoiceInterview(selectedRole, userName);
    } else {
      _startTextInterview(selectedRole, userName);
    }
  }

  // ─── Text Mode: WebSocket Setup ───────────────────────────────────────────────
  function _startTextInterview(selectedRole, userName) {
    const wsUrl = _buildWsUrl('/ws/interview');
    ws = new WebSocket(wsUrl);
    setConnectionStatus('connecting');

    ws.addEventListener('open', () => {
      setConnectionStatus('connected');
      wsSend({ type: 'start', data: { role: selectedRole, candidate_name: userName } });
      showTypingIndicator();
    });

    ws.addEventListener('message', handleWSMessage);

    ws.addEventListener('close', (event) => {
      setConnectionStatus('disconnected');
      isAIResponding = false;
      disableInput(true);
      if (chatScreen.style.display !== 'none' && !event.wasClean) {
        showReconnectPrompt();
      }
    });

    ws.addEventListener('error', () => {
      setConnectionStatus('error');
      console.error('WebSocket error encountered.');
    });
  }

  // ─── Voice Mode: WebSocket Setup ──────────────────────────────────────────────
  /**
   * Opens the voice WebSocket and sends the start frame.
   * AudioContext and mic capture are NOT started here — they are deferred to
   * session_ready, which confirms the server-side AI session is ready
   * and provides the authoritative output_sample_rate for the AudioContext.
   *
   * Session lifecycle:
   *   WS open → send start{mode:"voice"} → await session_ready
   *   session_ready → initAudioContext(output_sample_rate) → startMicCapture()
   *   [fully duplex from here: mic audio flows to server, AI audio plays back]
   *   user clicks End → stopMicCapture() → send end → ws.close()
   */
  function _startVoiceInterview(selectedRole, userName) {
    addSpecialMessage(
      'Voice mode is not available with the current AI provider. Please use text mode instead.',
      'error'
    );
    // Return to setup screen after a moment
    setTimeout(resetToSetup, 3000);
  }

  /** Returns a fully-qualified ws:// or wss:// URL for the given path. */
  function _buildWsUrl(path) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}${path}`;
  }


  // ─── Text WebSocket Message Handler ──────────────────────────────────────────
  /**
   * Routes incoming text-mode WebSocket messages.
   * Server sends: chunk, response, error.
   */
  function handleWSMessage(event) {
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch (err) {
      console.error('Failed to parse WS message:', event.data, err);
      return;
    }

    const { type, data } = msg;

    switch (type) {
      case 'chunk':
        hideTypingIndicator();
        if (!currentAIBubble) currentAIBubble = createAIMessageBubble();
        appendChunkToAIBubble(currentAIBubble, data.content || '');
        scrollToBottom();
        break;

      case 'response':
        hideTypingIndicator();
        if (!currentAIBubble) currentAIBubble = createAIMessageBubble();
        if (data.content && currentAIBubble.textContent === '') {
          setAIBubbleText(currentAIBubble, data.content);
        }
        finalizeAIBubble(currentAIBubble);
        currentAIBubble = null;
        isAIResponding  = false;
        disableInput(false);
        scrollToBottom();
        break;

      case 'error':
        hideTypingIndicator();
        addSpecialMessage(data.content || 'An error occurred.', 'error');
        isAIResponding = false;
        disableInput(false);
        scrollToBottom();
        break;

      default:
        console.warn('Unknown message type from server:', type, msg);
    }
  }

  // ─── Voice WebSocket Message Handler ─────────────────────────────────────────
  /**
   * Routes incoming voice-mode WebSocket frames.
   *
   * Binary frames  → raw PCM audio (Int16, 24 kHz, mono) → enqueue for playback.
   * JSON frames    → session_ready | transcript | ai_turn_complete |
   *                  ai_interrupted | error
   *
   * FULL DUPLEX: audio playback is completely independent of mic capture.
   * We never pause/mute the mic when AI audio arrives, and we never pause
   * playback while the user is speaking. Both run simultaneously.
   *
   * v2.0 REMOVED: audio_start / audio_end handlers (no longer emitted by server).
   */
  function handleVoiceWSMessage(event) {
    // Binary frame → enqueue for immediate playback (does NOT affect mic capture)
    if (event.data instanceof ArrayBuffer) {
      _enqueueAudioChunk(event.data);
      return;
    }

    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch (err) {
      console.error('Failed to parse voice WS message:', event.data, err);
      return;
    }

    const { type, data } = msg;

    switch (type) {

      case 'session_ready':
        // Server has established the AI voice session.
        // data: { mode: "voice", output_sample_rate: 24000 }
        // NOW is the correct time to:
        //   1. Create AudioContext using the server-provided sample rate
        //   2. Start mic capture — streams continuously until session end
        //   3. Show voice controls UI
        _onSessionReady(data || {});
        break;

      case 'transcript':
        // Text transcript of spoken content.
        // data: { speaker: "ai"|"user", content: "...", final: true }
        // Both speakers are displayed; user transcripts shown with user styling.
        if (data && data.content) {
          const isAI = !data.speaker || data.speaker === 'ai';
          if (isAI) {
            if (!currentAIBubble) currentAIBubble = createAIMessageBubble();
            if (currentAIBubble.textContent === '') {
              setAIBubbleText(currentAIBubble, data.content);
            } else {
              appendChunkToAIBubble(currentAIBubble, data.content);
            }
            // Finalize on final transcript (or when final flag absent/true)
            if (data.final !== false) {
              finalizeAIBubble(currentAIBubble);
              currentAIBubble = null;
            }
          } else {
            // User transcript — show in chat log with user styling
            addMessage(data.content, 'user');
          }
          scrollToBottom();
        }
        break;

      case 'ai_turn_complete':
        // AI has finished its response turn.
        // data: {} — purely decorative; update UI indicators only.
        // Does NOT affect audio flow or mic state.
        aiAudioActive = false;
        _scheduleAISpeakingOff();
        break;

      case 'ai_interrupted':
        // Barge-in detected: user spoke while AI was responding.
        // AI service stopped generating — flush our playback queue so the
        // user hears the cutoff immediately.
        // Reset nextPlayTime to "now" so no further chunks are scheduled
        // ahead of the current audio engine time.
        if (audioContext) {
          nextPlayTime = audioContext.currentTime;
        }
        aiAudioActive = false;
        setAISpeakingIndicator(false);
        _stopAIVisualizer();
        _updateDuplexState();
        break;

      case 'error':
        addSpecialMessage((data && data.content) || 'A voice error occurred.', 'error');
        setAISpeakingIndicator(false);
        _stopAIVisualizer();
        _updateDuplexState();
        scrollToBottom();
        break;

      default:
        console.warn('Unknown voice message type:', type, msg);
    }
  }

  // ─── session_ready Handler ────────────────────────────────────────────────────
  /**
   * Called when the server confirms the AI voice session is ready.
   * Initializes AudioContext using the server-provided output sample rate,
   * then starts mic capture immediately (fully duplex from this point).
   *
   * @param {{ mode?: string, output_sample_rate?: number }} sessionData
   */
  function _onSessionReady(sessionData) {
    const outputSampleRate = sessionData.output_sample_rate || DEFAULT_PLAYBACK_SAMPLE_RATE;

    // (Re-)initialize the AudioContext with the authoritative server sample rate.
    // Must be called in the context of a user gesture (which this is — the user
    // clicked Start Interview earlier in this call stack).
    _initAudioContext(outputSampleRate);

    // Start mic capture — from this moment both directions are open simultaneously.
    _requestMicAndStartStreaming();

    // Show voice controls UI (hidden until session is live)
    if (voiceControls) voiceControls.style.display = 'flex';

    // Start the call timer from this moment
    recStartTime = Date.now();
    _updateCallTimer();
    recTimerInterval = setInterval(_updateCallTimer, 1000);

    setConnectionStatus('connected', 'Session ready');
    _updateDuplexState();
  }

  // ─── Audio: Context Initialization ───────────────────────────────────────────
  /**
   * Creates (or resumes) the shared AudioContext at the given sample rate.
   * MUST be called inside a user-gesture handler to satisfy browser autoplay policy.
   *
   * The context is created at the server's output sample rate so that AI audio
   * chunks can be decoded and played without resampling artifacts.
   * Mic capture runs at 16 kHz via getUserMedia constraints — the browser handles
   * resampling from the mic's native rate to 16 kHz independently.
   *
   * @param {number} sampleRate  Playback sample rate provided by session_ready
   */
  function _initAudioContext(sampleRate) {
    // Close any stale context from a previous session
    if (audioContext && audioContext.state !== 'closed') {
      audioContext.close().catch(() => {});
    }

    audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: sampleRate,
    });

    // Reset playback clock for the new session
    nextPlayTime = 0;

    // Build the AI output signal chain:
    //   AI source nodes → aiGainNode → aiAnalyserNode → destination
    aiGainNode     = audioContext.createGain();
    aiAnalyserNode = audioContext.createAnalyser();
    aiAnalyserNode.fftSize = 256;

    aiGainNode.connect(aiAnalyserNode);
    aiAnalyserNode.connect(audioContext.destination);

    // Apply current speaker-mute state to the new context
    aiGainNode.gain.value = isSpeakerMuted ? 0 : 1;
  }

  // ─── Audio: Microphone Capture ────────────────────────────────────────────────
  /**
   * Requests microphone access. On grant, wires the capture pipeline using
   * AudioWorkletNode (mic-worklet.js) and begins streaming immediately.
   *
   * getUserMedia constraints:
   *   - echoCancellation: true  — browser AEC prevents feedback loop when
   *     speakers are open (essential for always-on mic / full-duplex)
   *   - sampleRate: 16000       — AI service requires 16 kHz mono PCM16
   *   - channelCount: 1 (mono)
   *
   * Falls back gracefully to ScriptProcessorNode if AudioWorklet is unavailable.
   */
  async function _requestMicAndStartStreaming() {
    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount:     CAPTURE_CHANNELS,
          sampleRate:       CAPTURE_SAMPLE_RATE,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl:  true,
        },
        video: false,
      });

      await _wireMicPipeline(micStream);
      isVoiceActive = true;

      // Enable mic-mute button now that we have permission
      if (micBtn) {
        micBtn.disabled = false;
        micBtn.classList.add('active');
        micBtn.setAttribute('aria-pressed', 'false'); // not muted = mic is live
        micBtn.title = 'Mute microphone';
      }

      _setLiveIndicator(true);
      _startMicVisualizer();

    } catch (err) {
      console.error('Microphone access denied or unavailable:', err);
      const banner = document.getElementById('voice-error-banner');
      const bannerText = document.getElementById('voice-error-text');
      if (banner && bannerText) {
        bannerText.textContent =
          'Microphone access was denied. Voice mode requires microphone permission. ' +
          'Please allow access in your browser settings and restart the interview.';
        banner.hidden = false;
      } else {
        addSpecialMessage(
          '⚠️ Microphone access was denied. Voice mode requires microphone permission. ' +
          'Please allow access in your browser settings and restart the interview.',
          'error'
        );
      }
      if (micBtn) {
        micBtn.disabled = true;
        micBtn.title    = 'Microphone permission denied';
      }
    }
  }

  /**
   * Wires the mic MediaStream into the AudioContext capture pipeline using
   * AudioWorkletNode (mic-capture-processor from mic-worklet.js).
   *
   * Pipeline:
   *   MediaStreamSource → micAnalyserNode → AudioWorkletNode → destination
   *
   * The AudioWorkletNode runs on a dedicated audio thread, accumulates samples
   * in 4096-sample chunks, converts Float32→Int16 PCM16, and posts each chunk
   * to the main thread via port.postMessage. The main thread sends it as a
   * binary WebSocket frame.
   *
   * Falls back to ScriptProcessorNode if AudioWorklet is not supported.
   *
   * @param {MediaStream} stream
   */
  async function _wireMicPipeline(stream) {
    micSourceNode   = audioContext.createMediaStreamSource(stream);
    micAnalyserNode = audioContext.createAnalyser();
    micAnalyserNode.fftSize = 256;

    if (audioContext.audioWorklet) {
      // ── AudioWorklet path (preferred) ──────────────────────────────────────
      try {
        await audioContext.audioWorklet.addModule('/static/mic-worklet.js');
        audioWorkletNode = new AudioWorkletNode(audioContext, 'mic-capture-processor');

        audioWorkletNode.port.onmessage = (evt) => {
          // evt.data is an Int16Array (zero-copy transfer from worklet thread)
          if (isMicMuted) return;
          if (!ws || ws.readyState !== WebSocket.OPEN) return;
          // Send the underlying ArrayBuffer as a binary WS frame
          ws.send(evt.data.buffer);
        };

        micSourceNode.connect(micAnalyserNode);
        micAnalyserNode.connect(audioWorkletNode);
        // Connect to destination to keep worklet node alive in the graph
        audioWorkletNode.connect(audioContext.destination);
        return;
      } catch (workletErr) {
        console.warn('AudioWorklet failed, falling back to ScriptProcessorNode:', workletErr);
      }
    }

    // ── ScriptProcessorNode fallback ────────────────────────────────────────
    // Deprecated but universally supported. Used when AudioWorklet is blocked
    // (e.g. missing HTTPS, older browser, or worklet file load failure).
    const scriptProcessor = audioContext.createScriptProcessor(4096, CAPTURE_CHANNELS, CAPTURE_CHANNELS);

    scriptProcessor.onaudioprocess = (evt) => {
      if (isMicMuted) return;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;

      const channelData = evt.inputBuffer.getChannelData(0);
      const sourceSR    = evt.inputBuffer.sampleRate;
      const pcm16       = _floatTo16BitPCM(channelData, sourceSR, CAPTURE_SAMPLE_RATE);
      ws.send(pcm16.buffer);
    };

    micSourceNode.connect(micAnalyserNode);
    micAnalyserNode.connect(scriptProcessor);
    scriptProcessor.connect(audioContext.destination);
  }

  /**
   * Converts a Float32Array of audio samples to a downsampled Int16Array (PCM 16-bit).
   * Uses linear interpolation for the downsample step (e.g. 24→16 kHz).
   *
   * @param {Float32Array} float32Array     Input samples at sourceSampleRate
   * @param {number}       sourceSampleRate Input rate (AudioContext rate)
   * @param {number}       targetSampleRate Target rate required by AI service (16000)
   * @returns {Int16Array}
   */
  function _floatTo16BitPCM(float32Array, sourceSampleRate, targetSampleRate) {
    const ratio        = sourceSampleRate / targetSampleRate;
    const outputLength = Math.round(float32Array.length / ratio);
    const output       = new Int16Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      const srcIndex = i * ratio;
      const lower    = Math.floor(srcIndex);
      const upper    = Math.min(lower + 1, float32Array.length - 1);
      const fraction = srcIndex - lower;

      const sample  = float32Array[lower] + fraction * (float32Array[upper] - float32Array[lower]);
      const clamped = Math.max(-1.0, Math.min(1.0, sample));
      output[i]     = clamped < 0
        ? Math.round(clamped * 32768)
        : Math.round(clamped * 32767);
    }
    return output;
  }

  // ─── Voice Controls: Mic Mute Toggle ─────────────────────────────────────────
  /**
   * Toggles mic mute state. Does NOT stop the capture pipeline — the
   * AudioWorkletNode keeps running so AEC remains active and the visualizer
   * continues to animate. Only the outbound WS send is gated.
   *
   * This is the full-duplex equivalent of a phone's mute button.
   */
  function toggleMicMute() {
    if (!isVoiceActive) return;

    isMicMuted = !isMicMuted;

    if (micBtn) {
      micBtn.classList.toggle('muted', isMicMuted);
      micBtn.setAttribute('aria-pressed', String(isMicMuted));
      micBtn.title = isMicMuted ? 'Unmute microphone' : 'Mute microphone';
    }

    if (recordingIndicator) {
      recordingIndicator.classList.toggle('mic-muted', isMicMuted);
    }

    // Update the user mic panel active state and the rec-label text
    if (userSpeakingIndicator) {
      userSpeakingIndicator.classList.toggle('speaking-active', !isMicMuted);
    }
    _updateRecLabel();

    _updateDuplexState();
  }

  // ─── Voice Controls: Speaker Mute Toggle ─────────────────────────────────────
  /**
   * Toggles AI audio output mute. Uses GainNode to silence output in real-time
   * without stopping the playback pipeline or losing sync.
   */
  function toggleSpeaker() {
    isSpeakerMuted = !isSpeakerMuted;

    if (speakerBtn) {
      speakerBtn.classList.toggle('muted', isSpeakerMuted);
      speakerBtn.setAttribute('aria-pressed', String(isSpeakerMuted));
      speakerBtn.title = isSpeakerMuted ? 'Unmute AI audio' : 'Mute AI audio';
    }

    // Smooth ramp to avoid audio clicks (5 ms)
    if (aiGainNode && audioContext) {
      aiGainNode.gain.linearRampToValueAtTime(
        isSpeakerMuted ? 0 : 1,
        audioContext.currentTime + 0.005
      );
    }

    if (isSpeakerMuted) {
      setAISpeakingIndicator(false);
      _stopAIVisualizer();
    }
  }

  // ─── Voice Pipeline Stop ──────────────────────────────────────────────────────
  /**
   * Gracefully stops the voice pipeline (called on interview end or WS close).
   * Does NOT close the WS — caller handles that.
   */
  function _stopVoicePipeline() {
    _cleanupVoice();
  }


  // ─── Audio: Playback ──────────────────────────────────────────────────────────
  /**
   * Receives a raw PCM ArrayBuffer from the server (Int16, 24 kHz, mono),
   * converts to Float32, and schedules it for seamless sequential playback
   * using the nextPlayTime accumulator pattern.
   *
   * nextPlayTime guarantees gapless scheduling:
   *   - Each chunk starts exactly where the previous one ended.
   *   - If the queue empties (network hiccup), nextPlayTime is snapped to
   *     audioContext.currentTime so the next chunk plays immediately.
   *   - On ai_interrupted, nextPlayTime is reset to currentTime, which means
   *     new chunks are not scheduled ahead — perceived near-instant cutoff.
   *
   * Full-duplex: this runs completely independently of mic capture.
   *
   * @param {ArrayBuffer} buffer  Raw Int16 PCM data from server
   */
  function _enqueueAudioChunk(buffer) {
    if (!audioContext || !aiGainNode) return;

    const int16Array  = new Int16Array(buffer);
    const float32Data = _int16ToFloat32(int16Array);
    const sampleRate  = audioContext.sampleRate;

    const audioBuffer = audioContext.createBuffer(1, float32Data.length, sampleRate);
    audioBuffer.getChannelData(0).set(float32Data);

    const sourceNode = audioContext.createBufferSource();
    sourceNode.buffer = audioBuffer;
    // Route through gain (speaker mute) and analyser (AI visualizer)
    sourceNode.connect(aiGainNode);

    // Snap nextPlayTime to now if the queue has drained (avoids scheduling
    // chunks in the past, which would cause them to play immediately/stacked)
    if (nextPlayTime < audioContext.currentTime) {
      nextPlayTime = audioContext.currentTime;
    }
    sourceNode.start(nextPlayTime);
    nextPlayTime += audioBuffer.duration;

    // Mark AI audio as active for UI indicators (decorative only — no mic gating)
    aiAudioActive = true;
    setAISpeakingIndicator(true);
    _startAIVisualizer();

    currentSourceNodes.push(sourceNode);
    sourceNode.onended = () => {
      currentSourceNodes = currentSourceNodes.filter((n) => n !== sourceNode);
      // When the last node finishes and the turn is complete, clear indicator
      if (currentSourceNodes.length === 0 && !aiAudioActive) {
        setAISpeakingIndicator(false);
        _stopAIVisualizer();
        _updateDuplexState();
      }
    };
  }

  /**
   * Converts Int16Array PCM samples to Float32Array in [-1.0, 1.0] range.
   * @param {Int16Array} int16Array
   * @returns {Float32Array}
   */
  function _int16ToFloat32(int16Array) {
    const float32 = new Float32Array(int16Array.length);
    for (let i = 0; i < int16Array.length; i++) {
      float32[i] = int16Array[i] / (int16Array[i] < 0 ? 32768 : 32767);
    }
    return float32;
  }

  /**
   * Schedules the AI speaking indicator off after the playback queue drains.
   * Called on ai_turn_complete — chunks may still be playing at that point.
   */
  function _scheduleAISpeakingOff() {
    if (!audioContext) {
      setAISpeakingIndicator(false);
      _stopAIVisualizer();
      _updateDuplexState();
      return;
    }
    const remaining = nextPlayTime - audioContext.currentTime;
    if (remaining > 0) {
      setTimeout(() => {
        if (!aiAudioActive) {
          setAISpeakingIndicator(false);
          _stopAIVisualizer();
          _updateDuplexState();
        }
      }, Math.ceil(remaining * 1000) + 100); // +100 ms buffer
    } else {
      setAISpeakingIndicator(false);
      _stopAIVisualizer();
      _updateDuplexState();
    }
  }

  // ─── Duplex State Indicator ───────────────────────────────────────────────────
  /**
   * Sets a data attribute on the chat screen when both sides are simultaneously
   * active (user not muted AND AI speaking). CSS uses this for visual feedback.
   */
  function _updateDuplexState() {
    if (!chatScreen) return;
    const bothActive = isVoiceActive && !isMicMuted && aiAudioActive;
    chatScreen.setAttribute('data-duplex-active', String(bothActive));
  }

  // ─── Audio: Mic Visualizer ────────────────────────────────────────────────────
  /**
   * Starts the mic waveform animation loop on #audio-visualizer canvas.
   * Driven by micAnalyserNode — reflects user's mic input in real time.
   * Runs continuously while voice mode is active (even when mic-muted,
   * so the user can see whether they are actually producing sound).
   */
  function _startMicVisualizer() {
    if (!audioVisualizer || !micAnalyserNode) return;
    if (micVizFrameId) return; // Already running

    // Mark the user mic panel as active
    if (userSpeakingIndicator) userSpeakingIndicator.classList.add('speaking-active');

    const canvas  = audioVisualizer;
    const ctx     = canvas.getContext('2d');
    const bufLen  = micAnalyserNode.frequencyBinCount; // fftSize/2 = 128
    const dataArr = new Uint8Array(bufLen);

    function draw() {
      micVizFrameId = requestAnimationFrame(draw);
      micAnalyserNode.getByteTimeDomainData(dataArr);

      const W = canvas.width;
      const H = canvas.height;

      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = 'rgba(0,0,0,0.05)';
      ctx.fillRect(0, 0, W, H);

      ctx.lineWidth   = 2;
      ctx.strokeStyle = isMicMuted ? 'rgba(100,100,100,0.4)' : '#4f46e5';
      ctx.beginPath();

      const sliceWidth = W / bufLen;
      let x = 0;

      for (let i = 0; i < bufLen; i++) {
        const v = dataArr[i] / 128.0;
        const y = (v * H) / 2;
        if (i === 0) ctx.moveTo(x, y);
        else         ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(W, H / 2);
      ctx.stroke();
    }

    draw();
  }

  /** Stops the mic waveform animation and clears the canvas. */
  function _stopMicVisualizer() {
    if (micVizFrameId) {
      cancelAnimationFrame(micVizFrameId);
      micVizFrameId = null;
    }
    if (userSpeakingIndicator) userSpeakingIndicator.classList.remove('speaking-active');
    if (audioVisualizer) {
      const ctx = audioVisualizer.getContext('2d');
      ctx.clearRect(0, 0, audioVisualizer.width, audioVisualizer.height);
    }
  }

  // ─── Audio: AI Output Visualizer ─────────────────────────────────────────────
  /**
   * Starts a second waveform animation driven by aiAnalyserNode.
   * Rendered on the same #audio-visualizer canvas as an overlay in a
   * contrasting colour so both user and AI levels are visible simultaneously.
   */
  function _startAIVisualizer() {
    if (!audioVisualizer || !aiAnalyserNode) return;
    if (aiVizFrameId) return; // Already running

    const canvas  = audioVisualizer;
    const ctx     = canvas.getContext('2d');
    const bufLen  = aiAnalyserNode.frequencyBinCount;
    const dataArr = new Uint8Array(bufLen);

    function draw() {
      aiVizFrameId = requestAnimationFrame(draw);
      aiAnalyserNode.getByteTimeDomainData(dataArr);

      const W = canvas.width;
      const H = canvas.height;

      ctx.lineWidth   = 2;
      ctx.strokeStyle = 'rgba(34, 197, 94, 0.75)'; // green, semi-transparent
      ctx.beginPath();

      const sliceWidth = W / bufLen;
      let x = 0;

      for (let i = 0; i < bufLen; i++) {
        const v = dataArr[i] / 128.0;
        const y = (v * H) / 2;
        if (i === 0) ctx.moveTo(x, y);
        else         ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(W, H / 2);
      ctx.stroke();
    }

    draw();
  }

  /** Stops the AI output waveform animation loop. */
  function _stopAIVisualizer() {
    if (aiVizFrameId) {
      cancelAnimationFrame(aiVizFrameId);
      aiVizFrameId = null;
    }
  }


  // ─── Voice: Live Indicator ────────────────────────────────────────────────────
  /**
   * Shows/hides the #recording-indicator and sets the .rec-label text.
   * In full-duplex mode this reflects that the mic pipeline is active,
   * not that a specific recording is in progress.
   *
   * @param {boolean} active
   */
  function _setLiveIndicator(active) {
    if (!recordingIndicator) return;

    if (active) {
      recordingIndicator.hidden = false;
      _updateRecLabel();
    } else {
      recordingIndicator.hidden = true;
    }
  }

  /**
   * Updates the .rec-label span inside #recording-indicator.
   * Shows "Mic Live" when mic is active, "Muted" when muted.
   * Does NOT write elapsed time — that goes to #call-timer via _updateCallTimer().
   */
  function _updateRecLabel() {
    if (!recordingIndicator) return;
    const labelEl = recordingIndicator.querySelector('.rec-label');
    if (labelEl) {
      labelEl.textContent = isMicMuted ? 'Muted' : 'Mic Live';
    }
  }

  /**
   * Updates the elapsed call time displayed in #call-timer.
   * Format: M:SS for durations under 1 hour (e.g. 0:00, 1:23, 10:05),
   *         H:MM:SS for 1 hour or more (e.g. 1:00:00).
   * Called every second via recTimerInterval (started in _onSessionReady).
   */
  function _updateCallTimer() {
    if (!callTimerEl) return;
    const elapsed = Math.floor((Date.now() - recStartTime) / 1000);
    const hours   = Math.floor(elapsed / 3600);
    const minutes = Math.floor((elapsed % 3600) / 60);
    const seconds = elapsed % 60;
    const ss      = String(seconds).padStart(2, '0');

    if (hours > 0) {
      const mm = String(minutes).padStart(2, '0');
      callTimerEl.textContent = `${hours}:${mm}:${ss}`;
    } else {
      callTimerEl.textContent = `${minutes}:${ss}`;
    }
  }

  /**
   * @deprecated Kept for backward compatibility only — delegates to _updateRecLabel.
   * The old implementation wrote elapsed time to #recording-indicator; that is now
   * handled by _updateCallTimer() writing to #call-timer instead.
   */
  function _updateRecTimer() {
    _updateRecLabel();
  }

  // ─── Voice Cleanup ────────────────────────────────────────────────────────────
  /**
   * Fully tears down all voice resources:
   *   - Stops all active playback source nodes
   *   - Disconnects AudioWorkletNode / analyser nodes
   *   - Stops mic stream tracks
   *   - Closes AudioContext
   *   - Cancels animation frames and timers
   *   - Resets all voice state flags
   *
   * Safe to call multiple times.
   */
  function _cleanupVoice() {
    _stopMicVisualizer();
    _stopAIVisualizer();

    // Stop all active playback nodes
    currentSourceNodes.forEach((node) => {
      try { node.stop(); } catch (_) { /* already stopped */ }
    });
    currentSourceNodes = [];
    nextPlayTime = 0;

    if (recTimerInterval) {
      clearInterval(recTimerInterval);
      recTimerInterval = null;
    }

    // Reset call timer display
    if (callTimerEl) callTimerEl.textContent = '0:00';

    // Disconnect AudioWorklet capture graph
    if (audioWorkletNode) {
      try { audioWorkletNode.disconnect(); } catch (_) {}
      audioWorkletNode = null;
    }
    if (micAnalyserNode) {
      try { micAnalyserNode.disconnect(); } catch (_) {}
      micAnalyserNode = null;
    }
    if (micSourceNode) {
      try { micSourceNode.disconnect(); } catch (_) {}
      micSourceNode = null;
    }

    // Disconnect AI output graph
    if (aiGainNode) {
      try { aiGainNode.disconnect(); } catch (_) {}
      aiGainNode = null;
    }
    if (aiAnalyserNode) {
      try { aiAnalyserNode.disconnect(); } catch (_) {}
      aiAnalyserNode = null;
    }

    // Release mic hardware
    if (micStream) {
      micStream.getTracks().forEach((track) => track.stop());
      micStream = null;
    }

    // Close AudioContext
    if (audioContext && audioContext.state !== 'closed') {
      audioContext.close().catch(() => {});
      audioContext = null;
    }

    // Reset state flags
    isVoiceActive = false;
    isMicMuted    = false;
    aiAudioActive = false;

    // Hide voice controls panel
    if (voiceControls) voiceControls.style.display = 'none';

    // Reset UI indicators
    _setLiveIndicator(false);
    setAISpeakingIndicator(false);
    if (chatScreen) chatScreen.setAttribute('data-duplex-active', 'false');

    if (micBtn) {
      micBtn.disabled = true;
      micBtn.classList.remove('active', 'muted');
      micBtn.setAttribute('aria-pressed', 'false');
      micBtn.title = 'Microphone';
    }
  }


  // ─── Send Message (text mode only) ───────────────────────────────────────────
  function sendMessage() {
    if (!messageInput) return;
    const text = messageInput.value.trim();
    if (!text) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      addSpecialMessage('Connection lost. Please refresh the page.', 'error');
      return;
    }

    addMessage(text, 'user');

    messageInput.value        = '';
    messageInput.style.height = 'auto';
    sendBtn.disabled          = true;

    if (charCount) {
      charCount.textContent = `0/${MAX_CHARS}`;
      charCount.classList.remove('warn');
    }

    isAIResponding = true;
    disableInput(true);
    showTypingIndicator();

    wsSend({ type: 'message', data: { content: text } });
  }

  // ─── End Interview ────────────────────────────────────────────────────────────
  /**
   * Sends end signal to server and tears down voice resources if in voice mode.
   * Voice session lifecycle on end:
   *   1. stopMicCapture (via _cleanupVoice)
   *   2. send { type: "end" }
   *   3. ws.close()
   */
  function endInterview() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      resetToSetup();
      return;
    }

    if (!confirm('Are you sure you want to end the interview?')) return;

    disableInput(true);
    endBtn.disabled = true;

    if (interviewMode === 'text') showTypingIndicator();

    // Voice: stop mic capture first, then signal server
    if (interviewMode === 'voice') {
      _cleanupVoice();
    }

    // Send interview-end control frame
    wsSend({ type: 'end', data: {} });
  }

  // ─── WebSocket Send Helper ────────────────────────────────────────────────────
  function wsSend(payload) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      console.warn('wsSend called but WS is not open:', payload);
      return;
    }
    ws.send(JSON.stringify(payload));
  }

  // ─── UI: Add Message Bubble ───────────────────────────────────────────────────
  function addMessage(content, sender) {
    const wrapper = document.createElement('div');
    wrapper.classList.add('message', `message-${sender}`);

    const bubble = document.createElement('div');
    bubble.classList.add('bubble');
    bubble.textContent = content;

    const meta = document.createElement('div');
    meta.classList.add('message-meta');
    meta.textContent = formatTime(new Date());

    wrapper.appendChild(bubble);
    wrapper.appendChild(meta);
    chatMessages.appendChild(wrapper);

    scrollToBottom();
    return wrapper;
  }

  // ─── UI: Create Empty AI Bubble (for streaming) ───────────────────────────────
  function createAIMessageBubble() {
    const wrapper = document.createElement('div');
    wrapper.classList.add('message', 'message-ai', 'streaming');

    const bubble = document.createElement('div');
    bubble.classList.add('bubble');
    bubble.textContent = '';

    const meta = document.createElement('div');
    meta.classList.add('message-meta');
    meta.textContent = formatTime(new Date());

    wrapper.appendChild(bubble);
    wrapper.appendChild(meta);
    chatMessages.appendChild(wrapper);

    scrollToBottom();
    return bubble;
  }

  function appendChunkToAIBubble(bubbleEl, chunk) { bubbleEl.textContent += chunk; }
  function setAIBubbleText(bubbleEl, text)         { bubbleEl.textContent = text; }

  function finalizeAIBubble(bubbleEl) {
    if (bubbleEl && bubbleEl.parentElement) {
      bubbleEl.parentElement.classList.remove('streaming');
    }
  }

  // ─── UI: Special Messages ─────────────────────────────────────────────────────
  function addSpecialMessage(content, type) {
    const el = document.createElement('div');
    el.classList.add('special-message', `special-message-${type}`);
    el.textContent = content;
    chatMessages.appendChild(el);
    scrollToBottom();
  }

  // ─── UI: Typing Indicator ─────────────────────────────────────────────────────
  function showTypingIndicator() {
    if (typingIndicator) {
      typingIndicator.style.display = 'flex';
      scrollToBottom();
    }
  }

  function hideTypingIndicator() {
    if (typingIndicator) typingIndicator.style.display = 'none';
  }

  // ─── UI: Connection Status ────────────────────────────────────────────────────
  function setConnectionStatus(status, label) {
    if (!connectionStatus) return;

    connectionStatus.classList.remove('connected', 'disconnected', 'connecting', 'reconnecting', 'error');
    connectionStatus.classList.add(status);

    const statusLabels = {
      connected:    'Connected',
      disconnected: 'Disconnected',
      connecting:   'Connecting…',
      reconnecting: 'Reconnecting…',
      error:        'Connection Error',
    };

    const labelEl = connectionStatus.querySelector('.status-label');
    if (labelEl) {
      labelEl.textContent = label || statusLabels[status] || status;
    } else {
      connectionStatus.textContent = label || statusLabels[status] || status;
    }
  }

  // ─── UI: Recording / Live Indicator ──────────────────────────────────────────
  /**
   * Shows or hides the #recording-indicator element.
   * Kept for compatibility with test_voice.py which calls this by name.
   * @param {boolean} active
   */
  function setRecordingIndicator(active) {
    _setLiveIndicator(active);
  }

  // ─── UI: AI Speaking Indicator ────────────────────────────────────────────────
  /**
   * Shows or hides the #ai-speaking-indicator element.
   * Purely decorative — does NOT gate the mic.
   * @param {boolean} active
   */
  function setAISpeakingIndicator(active) {
    if (!aiSpeakingIndicator) return;
    aiSpeakingIndicator.classList.toggle('speaking', active);
    aiSpeakingIndicator.classList.toggle('speaking-active', active);
    aiSpeakingIndicator.setAttribute('aria-hidden', String(!active));
    _updateDuplexState();
  }

  // ─── UI: Input Enable / Disable ──────────────────────────────────────────────
  /**
   * Enables or disables the message input and send button.
   * In voice mode the textarea is hidden — this is a no-op for voice.
   * @param {boolean} disabled
   */
  function disableInput(disabled) {
    if (messageInput) messageInput.disabled = disabled;
    if (sendBtn) {
      sendBtn.disabled = disabled ||
        !(messageInput && messageInput.value && messageInput.value.trim());
    }
  }

  // ─── UI: Auto-scroll ──────────────────────────────────────────────────────────
  function scrollToBottom() {
    if (chatMessages) chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // ─── UI: Reconnect Prompt ─────────────────────────────────────────────────────
  function showReconnectPrompt() {
    const el = document.createElement('div');
    el.classList.add('special-message', 'special-message-error');

    const text = document.createTextNode('⚠️ Connection lost. ');
    el.appendChild(text);

    const btn = document.createElement('button');
    btn.id          = 'reconnect-btn';
    btn.className   = 'reconnect-btn';
    btn.textContent = 'Return to Setup';
    btn.addEventListener('click', resetToSetup);
    el.appendChild(btn);

    chatMessages.appendChild(el);
    scrollToBottom();
  }

  // ─── Reset to Setup Screen ────────────────────────────────────────────────────
  function resetToSetup() {
    if (ws && ws.readyState === WebSocket.OPEN) ws.close(1000, 'User reset');
    ws = null;
    currentAIBubble = null;
    isAIResponding  = false;

    _cleanupVoice();

    if (chatMessages) chatMessages.innerHTML = '';
    hideTypingIndicator();

    if (messageInput) messageInput.style.height = 'auto';
    if (charCount) {
      charCount.textContent = '';
      charCount.classList.remove('warn');
    }

    disableInput(false);
    if (endBtn)   endBtn.disabled   = false;
    if (startBtn) startBtn.disabled = !(roleSelect && roleSelect.value);

    if (speakerBtn) {
      isSpeakerMuted = false;
      speakerBtn.classList.remove('muted');
      speakerBtn.setAttribute('aria-pressed', 'false');
      speakerBtn.title = 'Mute AI audio';
    }

    chatScreen.style.display  = 'none';
    setupScreen.style.display = '';
    setConnectionStatus('disconnected');
  }

  // ─── Utility: Format Time ─────────────────────────────────────────────────────
  function formatTime(date) {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

})(); // End IIFE
