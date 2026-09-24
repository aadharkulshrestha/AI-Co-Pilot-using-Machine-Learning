/**
 * Conversational Voice AI & FCOM RAG Copilot - Cockpit Frontend Logic
 * Manages Web Speech STT/TTS, Cockpit State Machine, Waveform Canvas, and API interaction.
 */

document.addEventListener('DOMContentLoaded', () => {

  // DOM Elements
  const pttBtn = document.getElementById('btn-ptt');
  const pttLabel = document.getElementById('ptt-label');
  const stateBox = document.getElementById('voice-state-box');
  const stateIcon = document.getElementById('state-icon');
  const stateHeadline = document.getElementById('state-headline');
  const queryForm = document.getElementById('query-form');
  const textInput = document.getElementById('text-query-input');
  const logStream = document.getElementById('log-stream');
  const citationsContainer = document.getElementById('citations-container');
  const aircraftSelect = document.getElementById('aircraft-select');
  const docCounter = document.getElementById('doc-counter');
  const demoModeTag = document.getElementById('demo-mode-tag');

  const btnSpeakLast = document.getElementById('btn-speak-last');
  const btnStopSpeech = document.getElementById('btn-stop-speech');
  const btnClearChat = document.getElementById('btn-clear-chat');

  const manualsDrawer = document.getElementById('manuals-drawer');
  const btnOpenManuals = document.getElementById('btn-open-manuals');
  const btnCloseDrawer = document.getElementById('btn-close-drawer');
  const btnTriggerReindex = document.getElementById('btn-trigger-reindex');
  const btnLoadDemo = document.getElementById('btn-load-demo');
  const pdfUploadInput = document.getElementById('pdf-upload-input');
  const uploadStatus = document.getElementById('upload-status');
  const manualsList = document.getElementById('manuals-list');

  const passageModal = document.getElementById('passage-modal');
  const modalPassageTitle = document.getElementById('modal-passage-title');
  const modalPassageBody = document.getElementById('modal-passage-body');
  const btnCloseModal = document.getElementById('btn-close-modal');

  // Waveform Canvas
  const canvas = document.getElementById('waveform-canvas');
  const ctx = canvas.getContext('2d');

  // Application State
  const CONVERSATION_ID = 'session_' + Math.random().toString(36).substring(2, 9);
  let lastAIResponseText = "";
  let isListening = false;
  let recognition = null;
  let synth = window.speechSynthesis;
  let currentUtterance = null;
  let activeAudioLevel = 0;
  let animationFrameId = null;

  // Voice State Definitions
  const STATES = {
    IDLE: { icon: '🟢', text: 'SYSTEM READY', class: '' },
    LISTENING: { icon: '🎙️', text: 'LISTENING (PTT ACTIVE)...', class: 'state-listening' },
    TRANSCRIBING: { icon: '📝', text: 'TRANSCRIBING AUDIO...', class: 'state-listening' },
    SEARCHING_MANUALS: { icon: '🔎', text: 'SEARCHING FLIGHT MANUAL...', class: 'state-searching' },
    GENERATING_RESPONSE: { icon: '🤖', text: 'GENERATING RESPONSE...', class: 'state-searching' },
    SPEAKING: { icon: '🔊', text: 'TRANSMITTING VOICE...', class: 'state-speaking' },
    ERROR: { icon: '⚠️', text: 'SYSTEM ADVISORY', class: 'state-error' }
  };

  function setVoiceState(stateKey, customMsg = null) {
    const s = STATES[stateKey] || STATES.IDLE;
    stateIcon.textContent = s.icon;
    stateHeadline.textContent = customMsg || s.text;

    // Reset classes
    stateBox.className = 'voice-state-display ' + (s.class || '');

    if (stateKey === 'LISTENING') {
      pttBtn.classList.add('listening');
      pttLabel.textContent = 'TRANSMITTING...';
    } else {
      pttBtn.classList.remove('listening');
      pttLabel.textContent = 'PRESS & SPEAK';
    }
  }

  // ==============================================================================
  // SPEECH RECOGNITION (STT)
  // ==============================================================================
  function setupSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.warn("Browser Web Speech API not supported. Direct text transmission active.");
      return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      isListening = true;
      setVoiceState('LISTENING');
      stopAudioPlayback();
    };

    recognition.onresult = (event) => {
      let interimTranscript = '';
      let finalTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        } else {
          interimTranscript += event.results[i][0].transcript;
        }
      }

      if (interimTranscript) {
        textInput.value = interimTranscript;
        setVoiceState('TRANSCRIBING', `HEARD: "${interimTranscript}"`);
      }

      if (finalTranscript) {
        textInput.value = finalTranscript;
        recognition.stop();
        submitQuery(finalTranscript);
      }
    };

    recognition.onerror = (event) => {
      console.error("Speech Recognition Error:", event.error);
      isListening = false;
      if (event.error === 'not-allowed') {
        setVoiceState('ERROR', 'MICROPHONE ACCESS DENIED');
      } else if (event.error === 'no-speech') {
        setVoiceState('IDLE', 'NO SPEECH DETECTED');
      } else {
        setVoiceState('ERROR', `VOICE ERROR: ${event.error.toUpperCase()}`);
      }
    };

    recognition.onend = () => {
      isListening = false;
      if (stateHeadline.textContent.includes('LISTENING')) {
        setVoiceState('IDLE');
      }
    };
  }

  function toggleListening() {
    if (!recognition) {
      alert("Browser speech recognition is not supported in this browser. Please type your query in the transmission box.");
      return;
    }

    if (isListening) {
      recognition.stop();
      isListening = false;
      setVoiceState('IDLE');
    } else {
      try {
        recognition.start();
      } catch (err) {
        console.error("Recognition start error:", err);
      }
    }
  }

  // ==============================================================================
  // TEXT-TO-SPEECH (TTS)
  // ==============================================================================
  function speakResponse(text) {
    if (!synth) return;
    stopAudioPlayback();

    // Clean text for aviation radio readout
    let cleanText = text
      .replace(/[*_#`~]/g, '')
      .replace(/\[.*?\]/g, '') // strip citation codes
      .replace(/⚠️/g, 'Caution: ')
      .replace(/•/g, ', ');

    currentUtterance = new SpeechSynthesisUtterance(cleanText);
    currentUtterance.rate = 1.05;
    currentUtterance.pitch = 0.95;

    // Pick crisp English voice
    const voices = synth.getVoices();
    const aeroVoice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('David') || v.name.includes('Samantha')));
    if (aeroVoice) currentUtterance.voice = aeroVoice;

    currentUtterance.onstart = () => {
      setVoiceState('SPEAKING');
      activeAudioLevel = 0.8;
    };

    currentUtterance.onend = () => {
      setVoiceState('IDLE');
      activeAudioLevel = 0;
    };

    currentUtterance.onerror = () => {
      setVoiceState('IDLE');
      activeAudioLevel = 0;
    };

    synth.speak(currentUtterance);
  }

  function stopAudioPlayback() {
    if (synth && synth.speaking) {
      synth.cancel();
      setVoiceState('IDLE');
      activeAudioLevel = 0;
    }
  }

  // ==============================================================================
  // RADAR / WAVEFORM CANVAS VISUALIZER
  // ==============================================================================
  function renderWaveform() {
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    ctx.lineWidth = 1.8;
    ctx.strokeStyle = isListening ? '#00e5ff' : (synth && synth.speaking ? '#00e676' : 'rgba(0, 229, 255, 0.4)');
    ctx.beginPath();

    const time = Date.now() * 0.005;
    const midY = height / 2;
    const amplitude = isListening ? 22 : (synth && synth.speaking ? 18 : 3);

    for (let x = 0; x < width; x += 3) {
      const freq = isListening ? 0.04 : 0.02;
      const y = midY + Math.sin(x * freq + time) * amplitude * Math.cos(x * 0.01 + time * 0.5);
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    animationFrameId = requestAnimationFrame(renderWaveform);
  }

  // ==============================================================================
  // API QUERY PIPELINE
  // ==============================================================================
  async function submitQuery(queryText) {
    if (!queryText || !queryText.trim()) return;
    const query = queryText.trim();
    textInput.value = '';

    // Append Pilot Query to Log
    appendLogEntry('PILOT', query);
    setVoiceState('SEARCHING_MANUALS');

    const aircraft = aircraftSelect.value;

    try {
      setVoiceState('GENERATING_RESPONSE');
      const response = await fetch('/api/copilot/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          aircraft: aircraft,
          conversation_id: CONVERSATION_ID,
          include_risk: true
        })
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const data = await response.json();
      lastAIResponseText = data.answer;

      // Append Copilot Response
      appendLogEntry('COPILOT', data.answer, data.sources, data.risk_analysis);

      // Render Citation Cards
      renderCitations(data.sources);

      // Automatic Voice Readout
      speakResponse(data.answer);

      // Handle 3D SVS Actions
      if (data.svs_action && window.svsInstance) {
        if (data.svs_action === 'HIGHLIGHT_TERRAIN') {
          window.svsInstance.setCameraPreset('TOP');
          const cfitBanner = document.getElementById('cfit-banner');
          if (cfitBanner) {
            cfitBanner.classList.remove('hidden');
            setTimeout(() => cfitBanner.classList.add('hidden'), 7000);
          }
        } else if (data.svs_action === 'TOGGLE_WEATHER') {
          window.svsInstance.setCameraPreset('APPROACH');
        } else if (data.svs_action === 'SHOW_SVS') {
          switchViewMode('svs');
        }
      }

    } catch (err) {
      console.error("Query Error:", err);
      setVoiceState('ERROR', 'COMMUNICATION LINK FAILURE');
      appendLogEntry('SYSTEM', `Transmission failed: ${err.message}. Ensure backend is active.`);
    }
  }

  // ==============================================================================
  // LOG STREAM RENDERING
  // ==============================================================================
  function appendLogEntry(role, text, sources = [], riskData = null) {
    const entry = document.createElement('div');
    entry.className = `log-entry ${role.toLowerCase()}-entry`;

    const now = new Date();
    const utcTime = now.toTimeString().substring(0, 8) + ' UTC';

    let roleBadge = role === 'PILOT' ? 'PILOT (TRANSMIT)' : (role === 'COPILOT' ? 'AI CO-PILOT (DECISION SUPPORT)' : 'SYSTEM');

    let html = `
      <div class="entry-meta">
        <span class="badge-role">${roleBadge}</span>
        <span class="timestamp">${utcTime}</span>
      </div>
      <div class="entry-content">
        ${formatMarkdown(text)}
      </div>
    `;

    // If Risk Data present, add mini telemetry chip
    if (riskData && riskData.status === 'success' && riskData.total_asrs_reports > 0) {
      html += `
        <div class="risk-chip-banner">
          <span>📊 ASRS MATCHES: <strong>${riskData.total_asrs_reports}</strong></span>
          <span>TIER: <strong>${riskData.composite_risk_level}</strong></span>
          <span>DIVERSION RATE: <strong>${riskData.diversion_rate_pct}%</strong></span>
        </div>
      `;
    }

    entry.innerHTML = html;
    logStream.appendChild(entry);
    logStream.scrollTop = logStream.scrollHeight;
  }

  function formatMarkdown(str) {
    if (!str) return '';
    let escaped = str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // Bold
    escaped = escaped.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Italics
    escaped = escaped.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // Bullet lines
    escaped = escaped.replace(/^• (.*$)/gim, '<li>$1</li>');
    // Numbered lines
    escaped = escaped.replace(/^(\d+\.) (.*$)/gim, '<li><strong>$1</strong> $2</li>');
    // Newlines to breaks
    escaped = escaped.replace(/\n\n/g, '<br><br>').replace(/\n/g, '<br>');

    return escaped;
  }

  // ==============================================================================
  // CITATIONS INSPECTOR
  // ==============================================================================
  function renderCitations(sources) {
    citationsContainer.innerHTML = '';

    if (!sources || sources.length === 0) {
      citationsContainer.innerHTML = '<div class="empty-citations-hint">No specific manual sections cited for this query.</div>';
      return;
    }

    sources.forEach((src) => {
      const card = document.createElement('div');
      card.className = 'citation-card';
      card.innerHTML = `
        <div class="cite-doc">📘 ${src.document}</div>
        <div class="cite-meta">PAGE ${src.page} • FLEET: ${src.aircraft || 'FLEET'} • RELEVANCE: ${(src.score * 100).toFixed(0)}%</div>
        <div class="cite-sec">${src.section}</div>
      `;

      card.addEventListener('click', () => {
        modalPassageTitle.textContent = `${src.document} — Page ${src.page} [${src.section}]`;
        modalPassageBody.textContent = src.text || "No passage preview available.";
        passageModal.classList.remove('hidden');
      });

      citationsContainer.appendChild(card);
    });
  }

  // ==============================================================================
  // MANUALS & FLEET DOCUMENTATION
  // ==============================================================================
  async function loadManualsStatus() {
    try {
      const res = await fetch('/api/manuals');
      const data = await res.json();

      docCounter.textContent = `INDEXED: ${data.total_indexed} MANUALS (${data.total_files} DETECTED)`;

      manualsList.innerHTML = '';
      if (data.files.length === 0) {
        manualsList.innerHTML = '<li class="manual-item">No manuals found in data/manuals/pdf/</li>';
      } else {
        data.files.forEach(f => {
          const li = document.createElement('li');
          li.className = 'manual-item';
          li.innerHTML = `
            <span>📄 ${f.filename} (${f.size_kb} KB)</span>
            <span class="status-badge">${f.is_indexed ? '✓ INDEXED' : 'UNINDEXED'}</span>
          `;
          manualsList.appendChild(li);
        });
      }

      // Check health
      const healthRes = await fetch('/api/copilot/health');
      const healthData = await healthRes.json();
      if (healthData.is_demo_mode) {
        demoModeTag.classList.remove('hidden');
      } else {
        demoModeTag.classList.add('hidden');
      }

    } catch (err) {
      console.warn("Could not retrieve manuals status:", err);
    }
  }

  // ==============================================================================
  // EVENT LISTENERS
  // ==============================================================================

  // PTT Microphone Button
  pttBtn.addEventListener('click', toggleListening);

  // Text Form Submission
  queryForm.addEventListener('submit', (e) => {
    e.preventDefault();
    submitQuery(textInput.value);
  });

  // Aux Controls
  btnSpeakLast.addEventListener('click', () => {
    if (lastAIResponseText) speakResponse(lastAIResponseText);
  });

  btnStopSpeech.addEventListener('click', stopAudioPlayback);

  btnClearChat.addEventListener('click', async () => {
    stopAudioPlayback();
    logStream.innerHTML = `
      <div class="log-entry system-entry">
        <div class="entry-meta">
          <span class="badge-role">COPILOT SYSTEM</span>
          <span class="timestamp">UTC ACTIVE</span>
        </div>
        <div class="entry-content">
          <p>Flight deck log cleared. Conversational memory reset.</p>
        </div>
      </div>
    `;
    citationsContainer.innerHTML = '<div class="empty-citations-hint">No citations currently loaded.</div>';
    await fetch('/api/copilot/clear', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ conversation_id: CONVERSATION_ID })
    });
  });

  // Quick Preset Queries
  document.querySelectorAll('.btn-quick-query').forEach(btn => {
    btn.addEventListener('click', () => {
      const q = btn.getAttribute('data-query');
      submitQuery(q);
    });
  });

  // Manuals Drawer Toggle
  btnOpenManuals.addEventListener('click', () => {
    manualsDrawer.classList.toggle('hidden');
    loadManualsStatus();
  });

  btnCloseDrawer.addEventListener('click', () => {
    manualsDrawer.classList.add('hidden');
  });

  // Re-index Button
  btnTriggerReindex.addEventListener('click', async () => {
    uploadStatus.textContent = 'Re-indexing manuals in ChromaDB...';
    try {
      const res = await fetch('/api/manuals/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ force: true })
      });
      const data = await res.json();
      uploadStatus.textContent = `Ingestion complete! ${data.chunks_added} chunks processed.`;
      loadManualsStatus();
    } catch (e) {
      uploadStatus.textContent = `Ingestion error: ${e.message}`;
    }
  });

  // Load Demo Manual Button
  btnLoadDemo.addEventListener('click', async () => {
    uploadStatus.textContent = 'Generating synthetic flight manual...';
    try {
      const res = await fetch('/api/copilot/demo');
      const data = await res.json();
      uploadStatus.textContent = 'Synthetic Demo Manual loaded & indexed!';
      loadManualsStatus();
    } catch (e) {
      uploadStatus.textContent = `Demo setup error: ${e.message}`;
    }
  });

  // PDF File Upload Handler
  pdfUploadInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    uploadStatus.textContent = `Uploading ${file.name}...`;
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/manuals/upload', {
        method: 'POST',
        body: formData
      });
      if (!res.ok) throw new Error("Upload failed.");
      const data = await res.json();
      uploadStatus.textContent = `Uploaded and indexed ${file.name}!`;
      loadManualsStatus();
    } catch (err) {
      uploadStatus.textContent = `Upload error: ${err.message}`;
    }
  });

  // ==============================================================================
  // SVS 3D CONTROLS & VIEW SWITCHER
  // ==============================================================================
  function switchViewMode(mode) {
    document.body.className = `cockpit-theme view-${mode}`;
    document.querySelectorAll('.btn-view-tab').forEach(b => b.classList.remove('active'));
    const activeTab = document.getElementById(`tab-${mode}`);
    if (activeTab) activeTab.classList.add('active');

    // Trigger canvas resize
    if (window.svsInstance) {
      setTimeout(() => window.svsInstance.onResize(), 150);
    }
  }

  document.getElementById('tab-voice').addEventListener('click', () => switchViewMode('voice-full'));
  document.getElementById('tab-split').addEventListener('click', () => switchViewMode('split'));
  document.getElementById('tab-svs').addEventListener('click', () => switchViewMode('svs-full'));

  // SVS Layer Toggles
  function setupLayerToggle(btnId, toggleFn) {
    const btn = document.getElementById(btnId);
    if (!btn) return;
    btn.addEventListener('click', () => {
      btn.classList.toggle('active');
      const isActive = btn.classList.contains('active');
      if (window.svsInstance) toggleFn(isActive);
    });
  }

  setupLayerToggle('toggle-terrain', (v) => window.svsInstance.toggleTerrain(v));
  setupLayerToggle('toggle-weather', (v) => window.svsInstance.toggleWeather(v));
  setupLayerToggle('toggle-obs', (v) => window.svsInstance.toggleObstacles(v));
  setupLayerToggle('toggle-path', (v) => window.svsInstance.toggleFlightPath(v));
  setupLayerToggle('toggle-glide', (v) => window.svsInstance.toggleGlidePath(v));

  // Camera Presets
  document.querySelectorAll('.btn-hud-camera').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.btn-hud-camera').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const cam = btn.getAttribute('data-cam');
      if (window.svsInstance) window.svsInstance.setCameraPreset(cam);
    });
  });

  // OpenSky Flight Selection
  const openSkySelect = document.getElementById('opensky-flight-select');
  async function loadOpenSkyFlights() {
    try {
      const res = await fetch('/api/svs/status');
      if (!res.ok) return;
      const data = await res.json();
      if (data.opensky_flights && data.opensky_flights.length > 0) {
        openSkySelect.innerHTML = '<option value="SIM_APPROACH_07L">SIM APPROACH 07L (B787)</option>';
        data.opensky_flights.forEach(f => {
          const opt = document.createElement('option');
          opt.value = f.flight_id;
          opt.textContent = `${f.callsign} (${f.typecode}) — ${f.origin}→${f.destination} [${f.problem_category}]`;
          openSkySelect.appendChild(opt);
        });
      }
    } catch (err) {
      console.warn("Could not load OpenSky flights:", err);
    }
  }

  openSkySelect.addEventListener('change', async () => {
    const selectedFlight = openSkySelect.value;
    try {
      const res = await fetch('/api/svs/simulation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ flight_id: selectedFlight })
      });
      if (res.ok && window.svsInstance) {
        const state = await res.json();
        window.svsInstance.svsState = state;
        window.svsInstance.aircraft.updateState(state.aircraft);
        window.svsInstance.aircraft.updateFlightPath(state.flight_path);
        appendLogEntry('SYSTEM', `SVS telemetry switched to OpenSky flight: ${state.aircraft.callsign} (${state.aircraft.typecode}).`);
      }
    } catch (e) {
      console.error("Failed to select OpenSky flight:", e);
    }
  });

  // Modal Close
  btnCloseModal.addEventListener('click', () => {
    passageModal.classList.add('hidden');
  });

  // Initialize
  setupSpeechRecognition();
  renderWaveform();
  loadManualsStatus();
  loadOpenSkyFlights();
  setVoiceState('IDLE');

  // Handle Query Params for View Mode and hiding tabs
  const urlParams = new URLSearchParams(window.location.search);
  const viewMode = urlParams.get('view');
  if (viewMode) {
    switchViewMode(viewMode);
  }
  if (urlParams.get('hideTabs') === 'true') {
    const switcher = document.querySelector('.cockpit-view-switcher');
    if (switcher) switcher.style.display = 'none';
  }

  // Launch 3D Synthetic Vision System
  try {
    if (typeof THREE !== 'undefined') {
      window.svsInstance = new SyntheticVisionSystem('svs-canvas-container');
      window.svsInstance.init();
    } else {
      console.warn("Three.js not loaded. Operating in 2D Telemetry mode.");
    }
  } catch (svsErr) {
    console.error("SVS Initialization error:", svsErr);
  }
});
