/**
 * H.E.A.R.T. AI — Emotion-Aware Companion for Older Adults
 * Reliability-Aware Multimodal Demonstrator Controller (ES6)
 *
 * Primary Workflow: Single MP4 Upload -> Automatic Speech ASR (Whisper) -> Text Emotion (RoBERTa)
 *                                      -> Speech Emotion (Wav2Vec2)
 *                                      -> Facial Emotion (ViT)
 *                                      -> Dynamic Fusion -> D1/D2/D3 Disagreement -> Reliability Gate -> Gemini LLM Companion
 *
 * Secondary Workflow: Research & Validation Mode (MELD 2,610 Test Split, Controlled Scenarios, Manual Text)
 */

let currentTab = 'companion';
let currentResearchSubMode = 'meld';
let uiState = 'idle';
let meldSamples = [];
let selectedMeldSample = null;
let lastAnalysisAction = null;
let conversationHistory = [];

// Primary Single Video File State
let selectedPrimaryVideoFile = null;

// Initialize Application on Page Load
document.addEventListener('DOMContentLoaded', async () => {
  // Set initial state to IDLE (do NOT auto-run analysis or show preloaded results)
  setUIState('idle');

  // Fetch initial LLM provider status
  await fetchProviderStatus();

  // Populate MELD test dataset samples for Research Mode
  await fetchMeldSamples();

  // Setup drag-and-drop file dropzone
  setupDropzones();
});

/**
 * Switch top-level tabs (Companion, Research, Pipeline, Evidence, Ethics)
 */
function switchTab(tabId) {
  currentTab = tabId;
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

  const tabBtn = document.getElementById(`tab-btn-${tabId}`);
  if (tabBtn) tabBtn.classList.add('active');

  const tabContent = document.getElementById(`tab-${tabId}`);
  if (tabContent) tabContent.classList.add('active');
}

/**
 * Switch submodes in Research / Validation tab (meld / demo / text)
 */
function setResearchSubMode(submode) {
  currentResearchSubMode = submode;
  document.querySelectorAll('.research-sub-nav .mode-pill-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.res-subpanel').forEach(p => p.classList.remove('active'));

  const btn = document.getElementById(`btn-res-${submode}`);
  if (btn) btn.classList.add('active');

  const panel = document.getElementById(`res-panel-${submode}`);
  if (panel) panel.classList.add('active');
}

/**
 * Set UI State: 'idle' | 'loading' | 'error' | 'success'
 */
function setUIState(state, message = '', loadingText = '') {
  uiState = state;

  const states = ['idle', 'loading', 'error', 'success'];
  states.forEach(s => {
    const el = document.getElementById(`results-${s}`);
    if (el) {
      if (s === state) {
        el.classList.add('active');
      } else {
        el.classList.remove('active');
      }
    }
  });

  if (state === 'loading') {
    const statusTextEl = document.getElementById('loading-status-text');
    if (statusTextEl) {
      statusTextEl.textContent = loadingText || 'Processing conversation video through H.E.A.R.T. AI multimodal pipeline...';
    }
  }

  if (state === 'error') {
    const errorMsgEl = document.getElementById('results-error-msg');
    if (errorMsgEl) {
      errorMsgEl.textContent = message || 'An unexpected error occurred during analysis.';
    }
  }
}

/**
 * Animate the 7-stage visible pipeline telemetry tracker
 */
let pipelineProgressInterval = null;

function startPipelineProgressAnimation() {
  const container = document.getElementById('pipeline-progress-container');
  if (container) container.style.display = 'block';

  // Reset all steps
  for (let i = 1; i <= 7; i++) {
    const stepEl = document.getElementById(`step-${i}`);
    if (stepEl) {
      stepEl.className = 'pipeline-step-pill';
    }
  }

  const stages = [
    { step: 1, label: 'Step 1 of 7: Extracting Media Tracks (Video & Audio)...' },
    { step: 2, label: 'Step 2 of 7: Transcribing Speech via OpenAI Whisper ASR...' },
    { step: 3, label: 'Step 3 of 7: Analysing Acoustic Speech Emotion (Wav2Vec2 SER)...' },
    { step: 4, label: 'Step 4 of 7: Detecting Faces & Analysing Facial Emotion (ViT FER)...' },
    { step: 5, label: 'Step 5 of 7: Fusing Multimodal Evidence & Calibrating Logits...' },
    { step: 6, label: 'Step 6 of 7: Assessing Reliability (D1, D2, D3, Entropy)...' },
    { step: 7, label: 'Step 7 of 7: Generating Companion Attuned Response (Gemini LLM)...' },
  ];

  let currentStageIndex = 0;
  const stageLabelEl = document.getElementById('progress-stage-label');

  // Activate first step immediately
  const firstStep = document.getElementById('step-1');
  if (firstStep) firstStep.classList.add('active');
  if (stageLabelEl) stageLabelEl.textContent = stages[0].label;

  clearInterval(pipelineProgressInterval);
  pipelineProgressInterval = setInterval(() => {
    if (currentStageIndex < stages.length - 1) {
      const prevStep = document.getElementById(`step-${stages[currentStageIndex].step}`);
      if (prevStep) {
        prevStep.classList.remove('active');
        prevStep.classList.add('completed');
      }

      currentStageIndex++;
      const nextStep = document.getElementById(`step-${stages[currentStageIndex].step}`);
      if (nextStep) nextStep.classList.add('active');
      if (stageLabelEl) stageLabelEl.textContent = stages[currentStageIndex].label;
    }
  }, 1200);
}

function stopPipelineProgressAnimation(success = true) {
  clearInterval(pipelineProgressInterval);
  const stageLabelEl = document.getElementById('progress-stage-label');
  if (success) {
    for (let i = 1; i <= 7; i++) {
      const stepEl = document.getElementById(`step-${i}`);
      if (stepEl) {
        stepEl.className = 'pipeline-step-pill completed';
      }
    }
    if (stageLabelEl) stageLabelEl.textContent = 'All 7 Multimodal Pipeline Stages Executed Successfully!';
  }
}

/**
 * Clear full conversation history and reset chat stream
 */
function clearConversationHistory() {
  conversationHistory = [];
  const streamEl = document.getElementById('chat-history-stream');
  if (streamEl) {
    streamEl.innerHTML = `
      <div class="chat-turn assistant-turn">
        <div class="avatar bot-avatar">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>
          </svg>
        </div>
        <div class="bubble-wrap">
          <div class="assistant-head-row">
            <span class="speaker-label">H.E.A.R.T. AI Companion</span>
            <span class="llm-generation-tag">Session Reset</span>
          </div>
          <div class="chat-bubble assistant-bubble">"Conversation reset. I am here with you. Please upload a conversation video whenever you are ready."</div>
        </div>
      </div>
    `;
  }
}

/**
 * Setup Drag-and-Drop file listeners on primary single-MP4 dropzone
 */
function setupDropzones() {
  const dropzone = document.getElementById('primary-mp4-dropzone');
  if (dropzone) {
    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('dragover');
      });
    });
    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('dragover');
      });
    });
    dropzone.addEventListener('drop', (e) => {
      const files = e.dataTransfer?.files;
      if (files && files.length > 0) {
        handlePrimaryVideoSelection(files[0]);
      }
    });
  }
}

/**
 * Primary Video File Selection Handlers
 */
function onPrimaryVideoSelected(input) {
  if (input.files && input.files[0]) {
    handlePrimaryVideoSelection(input.files[0]);
  }
}

function handlePrimaryVideoSelection(file) {
  selectedPrimaryVideoFile = file;
  const nameDisplay = document.getElementById('primary-video-name-display');
  const bar = document.getElementById('primary-video-selected-bar');
  const label = document.getElementById('primary-video-file-label');
  const sizeTag = document.getElementById('primary-video-size-tag');

  const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
  if (nameDisplay) nameDisplay.textContent = `${file.name} (${sizeMb} MB)`;
  if (label) label.textContent = file.name;
  if (sizeTag) sizeTag.textContent = `${sizeMb} MB`;
  if (bar) bar.style.display = 'flex';
}

function clearPrimaryVideo(event) {
  if (event) event.stopPropagation();
  selectedPrimaryVideoFile = null;
  const input = document.getElementById('primary-video-input');
  if (input) input.value = '';

  const nameDisplay = document.getElementById('primary-video-name-display');
  const bar = document.getElementById('primary-video-selected-bar');

  if (nameDisplay) nameDisplay.textContent = 'Choose Conversation Video or Drag & Drop';
  if (bar) bar.style.display = 'none';

  const progressContainer = document.getElementById('pipeline-progress-container');
  if (progressContainer) progressContainer.style.display = 'none';

  setUIState('idle');
}

/**
 * PRIMARY WORKFLOW: Analyze Single Uploaded Conversation Video (MP4)
 */
async function analyzePrimaryVideo() {
  if (!selectedPrimaryVideoFile) {
    alert('Please select or drag-and-drop an MP4 conversation video file to analyze.');
    return;
  }

  setUIState('loading', '', 'Decomposing MP4 container: extracting speech audio track and video frames...');
  startPipelineProgressAnimation();

  lastAnalysisAction = () => analyzePrimaryVideo();

  const formData = new FormData();
  formData.append('mode', 'upload');
  formData.append('video', selectedPrimaryVideoFile);
  formData.append('history', JSON.stringify(conversationHistory));

  try {
    const response = await fetch('/api/demo/analyze', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errJson = await response.json().catch(() => ({}));
      throw new Error(errJson.detail || `Server returned error ${response.status}`);
    }

    const data = await response.json();
    stopPipelineProgressAnimation(true);
    renderAnalysisResults(data);
    setUIState('success');

  } catch (err) {
    console.error('Error analyzing conversation video:', err);
    stopPipelineProgressAnimation(false);
    setUIState('error', `Analysis failed: ${err.message}`);
  }
}

/**
 * RESEARCH MODE A: Analyze Selected MELD Utterance
 */
async function analyzeMeldSelection() {
  const select = document.getElementById('meld-sample-select');
  const sampleId = select?.value;

  if (!sampleId) {
    alert('Please select a MELD sample from the dropdown first.');
    return;
  }

  // Switch to main tab to view results
  switchTab('companion');
  setUIState('loading', '', `Retrieving canonical multimodal prediction distributions for MELD sample [${sampleId}]...`);
  lastAnalysisAction = () => analyzeMeldSelection();

  try {
    const response = await fetch('/api/demo/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mode: 'meld',
        sample_id: sampleId,
        history: conversationHistory,
      }),
    });

    if (!response.ok) {
      const errJson = await response.json().catch(() => ({}));
      throw new Error(errJson.detail || `Server returned error ${response.status}`);
    }

    const data = await response.json();
    renderAnalysisResults(data);
    setUIState('success');

  } catch (err) {
    console.error('Error analyzing MELD sample:', err);
    setUIState('error', `MELD analysis failed: ${err.message}`);
  }
}

/**
 * RESEARCH MODE B: Run Controlled Demonstration Scenario
 */
async function runDemoScenario(scenarioKey) {
  // Switch to main tab to view results
  switchTab('companion');
  setUIState('loading', '', `Loading controlled scenario [${scenarioKey}]...`);
  lastAnalysisAction = () => runDemoScenario(scenarioKey);

  try {
    const response = await fetch('/api/demo/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mode: 'demo',
        scenario_id: scenarioKey,
        history: conversationHistory,
      }),
    });

    if (!response.ok) {
      const errJson = await response.json().catch(() => ({}));
      throw new Error(errJson.detail || `Server returned error ${response.status}`);
    }

    const data = await response.json();
    renderAnalysisResults(data);
    setUIState('success');

  } catch (err) {
    console.error('Error running scenario:', err);
    setUIState('error', `Scenario execution failed: ${err.message}`);
  }
}

/**
 * RESEARCH MODE C: Analyze Arbitrary Custom Text
 */
async function analyzeCustomInput() {
  const inputEl = document.getElementById('user-utterance-input');
  const text = inputEl ? inputEl.value.trim() : '';

  if (!text) {
    alert('Please enter a conversational utterance to analyze.');
    return;
  }

  // Switch to main tab to view results
  switchTab('companion');
  setUIState('loading', '', 'Processing custom text through RoBERTa text emotion classifier...');
  lastAnalysisAction = () => analyzeCustomInput();

  try {
    const response = await fetch('/api/demo/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mode: 'custom',
        text: text,
        history: conversationHistory,
      }),
    });

    if (!response.ok) {
      const errJson = await response.json().catch(() => ({}));
      throw new Error(errJson.detail || `Server returned error ${response.status}`);
    }

    const data = await response.json();
    renderAnalysisResults(data);
    setUIState('success');

  } catch (err) {
    console.error('Error analyzing custom text:', err);
    setUIState('error', `Text analysis failed: ${err.message}`);
  }
}

function clearCustomInput() {
  const inputEl = document.getElementById('user-utterance-input');
  if (inputEl) inputEl.value = '';
}

function retryLastAction() {
  if (typeof lastAnalysisAction === 'function') {
    lastAnalysisAction();
  } else {
    setUIState('idle');
  }
}

/**
 * Core Renderer: Populate all UI cards from analysis response JSON
 */
function renderAnalysisResults(data) {
  if (!data) return;

  // 1. Analyzed Utterance Banner & Transcript Source
  const utteranceEl = document.getElementById('analyzed-utterance-display');
  const transcriptSourcePill = document.getElementById('transcript-source-pill');
  const modeBadge = document.getElementById('analysis-mode-badge');

  const rawUtterance = data.analyzed_utterance || data.input_text || 'Conversation Input';
  if (utteranceEl) utteranceEl.textContent = `"${rawUtterance}"`;

  if (transcriptSourcePill) {
    const tMod = data.modalities?.text;
    if (tMod?.transcript_source) {
      transcriptSourcePill.textContent = tMod.transcript_source;
      transcriptSourcePill.className = 'transcript-source-pill active-asr';
    } else if (data.transcript_status) {
      transcriptSourcePill.textContent = data.transcript_status;
      transcriptSourcePill.className = 'transcript-source-pill';
    } else {
      transcriptSourcePill.textContent = 'User provided';
      transcriptSourcePill.className = 'transcript-source-pill';
    }
  }

  if (modeBadge) {
    modeBadge.textContent = data.mode_label || 'H.E.A.R.T. AI Live Pipeline';
  }

  // 2. Media Decomposition Status Strip
  const decompStrip = document.getElementById('media-decomposition-strip');
  const vSub = document.getElementById('decomp-video-sub');
  const aSub = document.getElementById('decomp-audio-sub');
  const asrSub = document.getElementById('decomp-asr-sub');

  const vMod = data.modalities?.visual;
  const aMod = data.modalities?.audio;
  const tMod = data.modalities?.text;

  if (vMod?.video_metadata) {
    const vm = vMod.video_metadata;
    if (vSub) vSub.textContent = `${vm.resolution || '--'} • ${vm.fps ? vm.fps.toFixed(1) + ' fps' : '--'} • ${vm.duration_seconds ? vm.duration_seconds.toFixed(1) + 's' : '--'} (${vm.num_frames_sampled || 0} frames)`;
  } else {
    if (vSub) vSub.textContent = vMod?.is_available ? 'Video frames processed' : 'Not provided / No video';
  }

  if (aMod?.audio_metadata) {
    const am = aMod.audio_metadata;
    if (aSub) aSub.textContent = `${am.sample_rate || 16000}Hz mono • ${am.duration_seconds ? am.duration_seconds.toFixed(1) + 's' : '--'} • RMS=${am.rms_energy ? am.rms_energy.toFixed(3) : '--'}`;
  } else {
    if (aSub) aSub.textContent = aMod?.is_available ? 'Audio waveform processed' : 'Not provided / No audio';
  }

  if (tMod?.asr_telemetry) {
    const asr = tMod.asr_telemetry;
    if (asrSub) asrSub.textContent = `${asr.model_name || 'OpenAI Whisper Tiny'} • ${asr.word_count || 0} words • ${asr.inference_time_seconds ? asr.inference_time_seconds.toFixed(2) + 's' : '--'}`;
  } else {
    if (asrSub) asrSub.textContent = tMod?.transcript ? 'Transcript available' : (tMod?.status || 'No speech transcribed');
  }

  // 3. Modality Cards (Text, Audio, Visual)
  renderModalityCard('text', tMod);
  renderModalityCard('audio', aMod);
  renderModalityCard('visual', vMod);

  // 4. Multimodal Fusion Card
  const fusion = data.fusion || {};
  const fArch = document.getElementById('fusion-arch-tag');
  const fEmo = document.getElementById('fused-emotion-name');
  const fConf = document.getElementById('fused-conf-val');
  const fRawConf = document.getElementById('fused-raw-conf');
  const fEntropy = document.getElementById('fused-entropy-val');

  if (fArch) fArch.textContent = fusion.architecture || 'Multimodal Fusion & Calibration';
  if (fEmo) {
    fEmo.textContent = (fusion.emotion || fusion.predicted_emotion || '--').toUpperCase();
    fEmo.className = `fused-emotion-name emotion-${(fusion.emotion || '').toLowerCase()}`;
  }
  if (fConf) fConf.textContent = `${((fusion.confidence || 0) * 100).toFixed(1)}%`;
  if (fRawConf) fRawConf.textContent = `${((fusion.confidence_raw || fusion.confidence || 0) * 100).toFixed(1)}%`;
  if (fEntropy) {
    const h = data.uncertainty?.predictive_entropy ?? 0;
    fEntropy.textContent = `${Number(h).toFixed(2)} nats`;
  }

  // Flow nodes
  const flowT = document.getElementById('flow-t-emo');
  const flowA = document.getElementById('flow-a-emo');
  const flowV = document.getElementById('flow-v-emo');
  if (flowT) flowT.textContent = tMod?.is_available ? (tMod.emotion || tMod.predicted_emotion || '--') : 'N/A';
  if (flowA) flowA.textContent = aMod?.is_available ? (aMod.emotion || aMod.predicted_emotion || '--') : 'N/A';
  if (flowV) flowV.textContent = vMod?.is_available ? (vMod.emotion || vMod.predicted_emotion || '--') : 'N/A';

  // Fused probability bars
  renderDistributionBars('fused-dist-bars', fusion.probabilities, fusion.distribution);

  // 5. Cross-Modal Disagreement Engine
  const dis = data.disagreement || {};
  const d1Val = document.getElementById('d1-val');
  const d2Val = document.getElementById('d2-val');
  const d3Val = document.getElementById('d3-val');
  const dBadge = document.getElementById('disagree-level-badge');
  const dBanner = document.getElementById('disagree-status-banner');
  const dStatusText = document.getElementById('disagree-status-text');
  const dExpl = document.getElementById('disagree-explanation');

  if (d1Val) d1Val.textContent = dis.d1_display || (dis.is_applicable ? Number(dis.d1).toFixed(3) : 'N/A');
  if (d2Val) d2Val.textContent = dis.d2_display || (dis.is_applicable ? Number(dis.d2).toFixed(3) : 'N/A');
  if (d3Val) d3Val.textContent = dis.d3_display || (dis.is_applicable ? String(dis.d3) : 'N/A');

  const tier = dis.status_tier || 'low';
  if (dBadge) {
    dBadge.textContent = dis.is_applicable ? `${tier.toUpperCase()} DISAGREEMENT` : 'N/A (SINGLE MODALITY)';
    dBadge.className = `disagree-badge-pill ${tier}`;
  }

  if (dBanner) {
    dBanner.className = `disagree-status-banner ${tier}`;
  }
  if (dStatusText) {
    dStatusText.textContent = dis.status || (dis.is_applicable ? `${tier.toUpperCase()} DISAGREEMENT` : 'SINGLE MODALITY');
  }
  if (dExpl) {
    dExpl.textContent = dis.notice || (
      dis.is_applicable
        ? `Cross-modal metrics evaluated: D1 (Generalized JSD) = ${Number(dis.d1).toFixed(3)}, D3 (Categorical Conflict) = ${dis.d3}.`
        : 'Cross-modal disagreement requires at least two genuine active modalities.'
    );
  }

  // 6. Reliability Gate & Safety Response Strategy
  const rel = data.reliability || {};
  const policy = data.response_policy || {};
  const relPill = document.getElementById('reliability-level-pill');
  const policyPill = document.getElementById('adaptation-policy-pill');
  const relReason = document.getElementById('rel-reason-text');
  const relSafety = document.getElementById('rel-safety-guidance');

  const relLevel = (rel.level || 'MEDIUM').toUpperCase();
  if (relPill) {
    relPill.textContent = `Reliability: ${relLevel}`;
    relPill.className = `rel-pill rel-${relLevel.toLowerCase()}`;
  }

  const policyName = policy.adaptation_badge || policy.adaptation_name || 'Cautious Adaptation';
  if (policyPill) {
    policyPill.textContent = `Strategy: ${policyName}`;
    policyPill.className = `policy-pill policy-${(policy.adaptation || 'cautious').toLowerCase()}`;
  }

  if (relReason) {
    relReason.textContent = rel.explanation || rel.reason || 'Reliability evaluated based on confidence and cross-modal agreement.';
  }
  if (relSafety) {
    relSafety.textContent = policy.safety_guidance || 'Gated response adaptation active.';
  }

  // Update Gate Tiers Visual
  ['strong', 'cautious', 'neutral'].forEach(t => {
    const el = document.getElementById(`gate-${t}`);
    if (el) el.classList.remove('active');
  });
  if (relLevel === 'HIGH' && (policy.adaptation || '').includes('strong')) {
    const g = document.getElementById('gate-strong');
    if (g) g.classList.add('active');
  } else if (relLevel === 'MEDIUM' || (policy.adaptation || '').includes('cautious')) {
    const g = document.getElementById('gate-cautious');
    if (g) g.classList.add('active');
  } else {
    const g = document.getElementById('gate-neutral');
    if (g) g.classList.add('active');
  }

  // 7. Conversational Manager & H.E.A.R.T. AI Companion Response
  const comp = data.companion_response || {};
  const userMsgEl = document.getElementById('chat-user-msg');
  const botReplyEl = document.getElementById('chat-bot-reply');
  const explainEl = document.getElementById('explain-body-text');

  if (userMsgEl) userMsgEl.textContent = `"${rawUtterance}"`;
  if (botReplyEl) botReplyEl.textContent = `"${comp.reply || data.response || 'I am here listening to you.'}"`;
  if (explainEl) explainEl.textContent = comp.why_explanation || data.why_response || 'Response generated with reliability awareness.';

  // Telemetry bar
  const teleEmo = document.getElementById('tele-emo');
  const teleConf = document.getElementById('tele-conf');
  const teleDis = document.getElementById('tele-dis');
  const teleRel = document.getElementById('tele-rel');
  const telePolicy = document.getElementById('tele-policy');

  if (teleEmo) teleEmo.textContent = (fusion.emotion || '--').toUpperCase();
  if (teleConf) teleConf.textContent = `${((fusion.confidence || 0) * 100).toFixed(1)}%`;
  if (teleDis) teleDis.textContent = dis.is_applicable ? `D1=${Number(dis.d1).toFixed(2)}, D3=${dis.d3}` : 'N/A';
  if (teleRel) teleRel.textContent = relLevel;
  if (telePolicy) telePolicy.textContent = policyName;

  // Append to conversation history for multi-turn context
  conversationHistory.push({
    role: 'user',
    content: rawUtterance,
  });
  conversationHistory.push({
    role: 'assistant',
    content: comp.reply || data.response || '',
    emotion: fusion.emotion,
    reliability: relLevel,
  });
}

/**
 * Render unimodal card details
 */
function renderModalityCard(modalityKey, modData) {
  const card = document.getElementById(`card-${modalityKey}`);
  if (!card) return;

  const encTag = document.getElementById(`${modalityKey}-encoder-tag`);
  const predEmo = document.getElementById(`${modalityKey}-pred-emotion`);
  const statusBadge = document.getElementById(`${modalityKey}-status-badge`);
  const confPill = document.getElementById(`${modalityKey}-conf-pill`);
  const distBars = document.getElementById(`${modalityKey}-dist-bars`);
  const metaBox = document.getElementById(`${modalityKey}-meta-box`);

  if (!modData || !modData.is_available) {
    card.classList.add('unavailable');
    if (predEmo) predEmo.textContent = modData?.state === 'NO_FACES_DETECTED' ? 'No Faces Detected' : (modData?.predicted_emotion || 'Not Provided');
    if (statusBadge) statusBadge.textContent = modData?.status || 'Not Available';
    if (confPill) {
      confPill.textContent = '0.0% Conf';
      confPill.className = 'conf-pill inactive';
    }
    if (distBars) distBars.innerHTML = '<div class="dist-empty-notice">Modality evidence not available in fusion.</div>';
    if (metaBox) metaBox.style.display = 'none';
    return;
  }

  card.classList.remove('unavailable');
  if (encTag && modData.encoder) encTag.textContent = modData.encoder;
  if (predEmo) {
    predEmo.textContent = (modData.emotion || modData.predicted_emotion || '--').toUpperCase();
  }
  if (statusBadge) {
    statusBadge.textContent = modData.status || 'Analyzed &bull; Emotion Inferred';
  }
  if (confPill) {
    const confVal = (modData.confidence || 0) * 100;
    confPill.textContent = `${confVal.toFixed(1)}% Conf`;
    confPill.className = 'conf-pill active';
  }

  // Transcript preview for Text modality
  if (modalityKey === 'text') {
    const tBox = document.getElementById('text-transcript-preview');
    const tText = document.getElementById('text-transcript-text');
    if (tBox && tText) {
      if (modData.transcript) {
        tBox.style.display = 'block';
        tText.textContent = `"${modData.transcript}"`;
      } else {
        tBox.style.display = 'none';
      }
    }
  }

  // Distribution bars
  renderDistributionBars(`${modalityKey}-dist-bars`, modData.probabilities, modData.distribution);
}

/**
 * Render emotion probability bar chart inside a container
 */
const CANONICAL_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'];

function renderDistributionBars(containerId, probabilities, distribution) {
  const container = document.getElementById(containerId);
  if (!container) return;

  container.innerHTML = '';
  const probs = probabilities || (distribution ? CANONICAL_EMOTIONS.map(e => distribution[e] || 0) : []);

  if (!probs || probs.length === 0) {
    container.innerHTML = '<div class="dist-empty-notice">No probability data.</div>';
    return;
  }

  CANONICAL_EMOTIONS.forEach((emo, idx) => {
    const pVal = probs[idx] ?? 0;
    const pPercent = (pVal * 100).toFixed(1);

    const row = document.createElement('div');
    row.className = 'dist-bar-row';
    row.innerHTML = `
      <span class="dist-emo-label">${emo}</span>
      <div class="dist-bar-track">
        <div class="dist-bar-fill emo-fill-${emo}" style="width: ${Math.max(pPercent, 2)}%;"></div>
      </div>
      <span class="dist-pct-label">${pPercent}%</span>
    `;
    container.appendChild(row);
  });
}

let activeMeldFilter = 'all';

/**
 * Fetch MELD test dataset samples for research mode
 */
async function fetchMeldSamples() {
  try {
    const response = await fetch('/api/demo/meld_samples');
    if (response.ok) {
      meldSamples = await response.json();
      renderMeldSelectOptions(meldSamples);
    }
  } catch (e) {
    console.warn('Could not fetch MELD samples:', e);
  }
}

function renderMeldSelectOptions(samples) {
  const select = document.getElementById('meld-sample-select');
  const countSpan = document.getElementById('meld-sample-count');
  if (countSpan) countSpan.textContent = `${samples.length} loaded`;

  if (select) {
    select.innerHTML = '<option value="">-- Choose a MELD Utterance --</option>';
    samples.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s.sample_id;
      opt.textContent = `[${s.sample_id}] ${s.speaker}: "${s.utterance.substring(0, 50)}..." (${s.ground_truth_emotion})`;
      select.appendChild(opt);
    });

    if (samples.length > 0) {
      select.value = samples[0].sample_id;
      onMeldSampleSelected();
    }
  }
}

/**
 * Filter MELD samples by ground-truth emotion tag
 */
function filterMeldSamples(emotion) {
  activeMeldFilter = emotion;

  // Update active chip UI
  document.querySelectorAll('.meld-filter-strip .filter-chip').forEach(chip => {
    chip.classList.remove('active');
  });
  const activeChip = document.getElementById(`filter-chip-${emotion}`);
  if (activeChip) activeChip.classList.add('active');

  const filtered = emotion === 'all'
    ? meldSamples
    : meldSamples.filter(s => (s.ground_truth_emotion || '').toLowerCase() === emotion.toLowerCase());

  renderMeldSelectOptions(filtered);
}

/**
 * Pick a random sample from the currently active filter
 */
function pickRandomMeldSample() {
  const filtered = activeMeldFilter === 'all'
    ? meldSamples
    : meldSamples.filter(s => (s.ground_truth_emotion || '').toLowerCase() === activeMeldFilter.toLowerCase());

  if (!filtered || filtered.length === 0) return;

  const randomSample = filtered[Math.floor(Math.random() * filtered.length)];
  const select = document.getElementById('meld-sample-select');
  if (select) {
    select.value = randomSample.sample_id;
    onMeldSampleSelected();
  }
}

/**
 * Set custom prompt in manual text test input
 */
function setCustomPrompt(text) {
  const inputEl = document.getElementById('user-utterance-input');
  if (inputEl) {
    inputEl.value = text;
    inputEl.focus();
  }
}

function onMeldSampleSelected() {
  const select = document.getElementById('meld-sample-select');
  const sampleId = select?.value;
  if (!sampleId) return;

  const sample = meldSamples.find(s => s.sample_id === sampleId);
  if (!sample) return;

  selectedMeldSample = sample;

  const speakerTag = document.getElementById('meld-speaker-tag');
  const emoTag = document.getElementById('meld-emotion-tag');
  const sentTag = document.getElementById('meld-sentiment-tag');
  const displayText = document.getElementById('meld-sample-display-text');

  if (speakerTag) speakerTag.textContent = `Speaker: ${sample.speaker}`;
  if (emoTag) emoTag.textContent = `Ground Truth: ${sample.ground_truth_emotion.toUpperCase()}`;
  if (sentTag) sentTag.textContent = `Sentiment: ${sample.sentiment.toUpperCase()}`;
  if (displayText) displayText.textContent = `"${sample.utterance}"`;
}

/**
 * Fetch LLM Provider Status
 */
async function fetchProviderStatus() {
  try {
    const response = await fetch('/api/demo/provider_status');
    if (response.ok) {
      const data = await response.json();
      const headBadge = document.getElementById('header-llm-status');
      if (headBadge) {
        if (data.is_available && data.provider && data.provider.includes('Gemini')) {
          headBadge.innerHTML = `<span class="status-dot llm-dot"></span> Gemini Live (${data.model || 'gemini-flash-lite'})`;
          headBadge.className = 'status-pill llm-pill connected';
        } else {
          headBadge.innerHTML = `<span class="status-dot"></span> Provider: ${data.provider || 'Rule-Based'}`;
          headBadge.className = 'status-pill';
        }
      }
    }
  } catch (e) {
    console.warn('Could not fetch provider status:', e);
  }
}
