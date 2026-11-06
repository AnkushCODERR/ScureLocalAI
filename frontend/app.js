// ============================================================================
// ScureLocalAI - Multimodal Knowledge Base & Voice Assistant Client
// ============================================================================

// State
const state = {
  activeView: "chat",
  documents: [],
  figures: [],
  selectedFigure: null,
  attachedImageBase64: null,
  attachedImageFile: null,
  isRecording: false,
  isSpeaking: false,
  currentUtteranceQueue: [],
  speechSettings: {
    voiceURI: null,
    rate: 1.0,
    pitch: 1.0,
    autoSpeak: false, // Default off so browser autoplay restrictions don't block
  },
};

// DOM References
const navTabs = document.querySelectorAll(".tab-btn");
const viewPanels = document.querySelectorAll(".view-panel");
const documentList = document.querySelector("#document-list");
const docCountBadge = document.querySelector("#doc-count-badge");
const navFiguresCount = document.querySelector("#nav-figures-count");
const fileInput = document.querySelector("#file-input");
const dropZone = document.querySelector("#drop-zone");
const uploadStatus = document.querySelector("#upload-status");
const refreshButton = document.querySelector("#refresh-documents");
const docSearchInput = document.querySelector("#doc-search");

// Chat DOM
const messagesContainer = document.querySelector("#messages");
const chatForm = document.querySelector("#chat-form");
const questionInput = document.querySelector("#question-input");
const sendButton = document.querySelector("#send-button");
const imageInput = document.querySelector("#image-input");
const voiceInputBtn = document.querySelector("#voice-input-btn");
const attachmentPreview = document.querySelector("#attachment-preview");
const attachmentImg = document.querySelector("#attachment-img");
const attachmentName = document.querySelector("#attachment-name");
const removeAttachmentBtn = document.querySelector("#remove-attachment-btn");

// Voice Header Controls
const autoSpeakToggle = document.querySelector("#auto-speak-toggle");
const speechIndicator = document.querySelector("#speech-indicator");
const stopSpeakingBtn = document.querySelector("#stop-speaking-btn");
const voiceSettingsBtn = document.querySelector("#voice-settings-btn");
const voiceModal = document.querySelector("#voice-modal");
const voiceModalClose = document.querySelector("#voice-modal-close");
const voiceSelect = document.querySelector("#voice-select");
const voiceRateSlider = document.querySelector("#voice-rate");
const voiceRateVal = document.querySelector("#voice-rate-val");
const voicePitchSlider = document.querySelector("#voice-pitch");
const voicePitchVal = document.querySelector("#voice-pitch-val");
const voiceTestBtn = document.querySelector("#voice-test-btn");
const voiceSaveBtn = document.querySelector("#voice-save-btn");

// Figures Gallery DOM
const figuresGrid = document.querySelector("#figures-grid");
const figuresDocFilter = document.querySelector("#figures-doc-filter");
const figuresKindFilter = document.querySelector("#figures-kind-filter");

// Relevance Checker DOM
const relevanceDropzone = document.querySelector("#relevance-dropzone");
const relevanceFileInput = document.querySelector("#relevance-file-input");
const relevanceEmptyView = document.querySelector("#relevance-empty-view");
const relevanceImagePreview = document.querySelector("#relevance-image-preview");
const relevanceImg = document.querySelector("#relevance-img");
const relevanceFocusQuery = document.querySelector("#relevance-focus-query");
const checkRelevanceBtn = document.querySelector("#check-relevance-btn");
const relevanceResults = document.querySelector("#relevance-results");

// Figure Modal DOM
const figureModal = document.querySelector("#figure-modal");
const modalCloseBtn = document.querySelector("#modal-close-btn");
const modalImg = document.querySelector("#modal-img");
const modalKindBadge = document.querySelector("#modal-kind-badge");
const modalPageBadge = document.querySelector("#modal-page-badge");
const modalDocBadge = document.querySelector("#modal-doc-badge");
const modalCaption = document.querySelector("#modal-caption");
const modalExplanation = document.querySelector("#modal-explanation");
const modalContext = document.querySelector("#modal-context");
const modalAskBtn = document.querySelector("#modal-ask-btn");
const modalSpeakDescBtn = document.querySelector("#modal-speak-desc-btn");

// ============================================================================
// Initialization & LocalStorage
// ============================================================================
function initSettings() {
  const saved = localStorage.getItem("scure_voice_settings");
  if (saved) {
    try {
      const parsed = JSON.parse(saved);
      Object.assign(state.speechSettings, parsed);
    } catch (_) {}
  }
  if (autoSpeakToggle) {
    autoSpeakToggle.checked = Boolean(state.speechSettings.autoSpeak);
  }
  if (voiceRateSlider) {
    voiceRateSlider.value = state.speechSettings.rate;
    voiceRateVal.textContent = `${state.speechSettings.rate}x`;
  }
  if (voicePitchSlider) {
    voicePitchSlider.value = state.speechSettings.pitch;
    voicePitchVal.textContent = `${state.speechSettings.pitch}`;
  }
}

// ============================================================================
// Navigation Tabs
// ============================================================================
function switchView(viewName) {
  state.activeView = viewName;
  navTabs.forEach((tab) => {
    const isTarget = tab.dataset.view === viewName;
    tab.classList.toggle("active", isTarget);
    tab.setAttribute("aria-selected", isTarget ? "true" : "false");
  });
  viewPanels.forEach((panel) => {
    panel.classList.toggle("active", panel.id === `view-${viewName}`);
  });

  if (viewName === "figures") {
    loadAllFigures();
  }
}

navTabs.forEach((tab) => {
  tab.addEventListener("click", () => switchView(tab.dataset.view));
});

// ============================================================================
// Voice Agent Engine (TTS & STT)
// ============================================================================
const synth = window.speechSynthesis;
let availableVoices = [];

function loadVoices() {
  if (!synth) return;
  try {
    availableVoices = synth.getVoices().filter((v) => v.lang.startsWith("en") || v.lang.startsWith("en-"));
    if (availableVoices.length === 0) {
      availableVoices = synth.getVoices();
    }

    if (voiceSelect) {
      voiceSelect.innerHTML = "";
      for (const voice of availableVoices) {
        const opt = document.createElement("option");
        opt.value = voice.voiceURI;
        opt.textContent = `${voice.name} (${voice.lang})`;
        if (state.speechSettings.voiceURI && voice.voiceURI === state.speechSettings.voiceURI) {
          opt.selected = true;
        }
        voiceSelect.appendChild(opt);
      }
    }
  } catch (err) {
    console.warn("Could not load voices:", err);
  }
}

if (synth) {
  synth.onvoiceschanged = loadVoices;
  loadVoices();
}

function cleanMarkdownForSpeech(text) {
  if (!text) return "";
  return text
    .replace(/```[\s\S]*?```/g, " Code block omitted. ")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/^#+\s+/gm, "")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\*([^*]+)\*/g, "$1")
    .replace(/__([^_]+)__/g, "$1")
    .replace(/_([^_]+)_/g, "$1")
    .replace(/https?:\/\/\S+/g, "")
    .replace(/^\s*[-*•]\s+/gm, "")
    .replace(/Sources:[\s\S]*$/i, "")
    .replace(/\s+/g, " ")
    .trim();
}

function splitIntoSpeechChunks(text, maxLength = 160) {
  const sentences = text.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [text];
  const chunks = [];
  let current = "";

  for (const sentence of sentences) {
    if ((current + " " + sentence).length > maxLength && current.trim()) {
      chunks.push(current.trim());
      current = sentence;
    } else {
      current += (current ? " " : "") + sentence;
    }
  }
  if (current.trim()) chunks.push(current.trim());
  return chunks;
}

function stopSpeaking() {
  if (!synth) return;
  try {
    synth.cancel();
  } catch (_) {}
  state.isSpeaking = false;
  state.currentUtteranceQueue = [];
  if (speechIndicator) speechIndicator.classList.add("hidden");
  if (stopSpeakingBtn) stopSpeakingBtn.classList.add("hidden");
  document.querySelectorAll(".action-tiny-btn.speaking").forEach((btn) => {
    btn.classList.remove("speaking");
    btn.innerHTML = `🔊 Read`;
  });
}

function speakText(rawText, triggerButton = null) {
  if (!synth) {
    console.warn("Speech synthesis is not supported in this browser.");
    return;
  }

  stopSpeaking();
  const cleaned = cleanMarkdownForSpeech(rawText);
  if (!cleaned) return;

  const chunks = splitIntoSpeechChunks(cleaned);
  state.currentUtteranceQueue = chunks;
  state.isSpeaking = true;
  if (speechIndicator) speechIndicator.classList.remove("hidden");
  if (stopSpeakingBtn) stopSpeakingBtn.classList.remove("hidden");

  if (triggerButton) {
    triggerButton.classList.add("speaking");
    triggerButton.innerHTML = `⏹️ Stop`;
  }

  function playNextChunk(index) {
    if (index >= chunks.length || !state.isSpeaking) {
      stopSpeaking();
      return;
    }

    try {
      const utterance = new SpeechSynthesisUtterance(chunks[index]);
      utterance.rate = state.speechSettings.rate;
      utterance.pitch = state.speechSettings.pitch;

      if (state.speechSettings.voiceURI) {
        const selected = availableVoices.find((v) => v.voiceURI === state.speechSettings.voiceURI);
        if (selected) utterance.voice = selected;
      }

      utterance.onend = () => {
        playNextChunk(index + 1);
      };

      utterance.onerror = () => {
        stopSpeaking();
      };

      synth.speak(utterance);
    } catch (err) {
      console.warn("Speech synthesis error:", err);
      stopSpeaking();
    }
  }

  playNextChunk(0);
}

// Speech Recognition (Dictation)
let recognition = null;
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (SpeechRecognition) {
  try {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onstart = () => {
      state.isRecording = true;
      voiceInputBtn.classList.add("recording");
      voiceInputBtn.title = "Listening... Speak your question";
    };

    recognition.onresult = (event) => {
      let transcript = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      questionInput.value = transcript;
      autoResizeTextarea(questionInput);
    };

    recognition.onerror = () => {
      stopDictation();
    };

    recognition.onend = () => {
      stopDictation();
    };
  } catch (err) {
    console.warn("SpeechRecognition init error:", err);
  }
}

function toggleDictation() {
  if (!recognition) {
    alert("Speech recognition is not supported in this browser. Please use Chrome, Edge, or Safari.");
    return;
  }

  if (state.isRecording) {
    recognition.stop();
  } else {
    try {
      recognition.start();
    } catch (_) {}
  }
}

function stopDictation() {
  state.isRecording = false;
  if (voiceInputBtn) {
    voiceInputBtn.classList.remove("recording");
    voiceInputBtn.title = "Dictate question by voice";
  }
}

if (voiceInputBtn) voiceInputBtn.addEventListener("click", toggleDictation);
if (stopSpeakingBtn) stopSpeakingBtn.addEventListener("click", stopSpeaking);

if (autoSpeakToggle) {
  autoSpeakToggle.addEventListener("change", (e) => {
    state.speechSettings.autoSpeak = e.target.checked;
    localStorage.setItem("scure_voice_settings", JSON.stringify(state.speechSettings));
  });
}

// Voice Settings Modal Handlers
if (voiceSettingsBtn) {
  voiceSettingsBtn.addEventListener("click", () => {
    loadVoices();
    voiceModal.classList.remove("hidden");
  });
}

if (voiceModalClose) {
  voiceModalClose.addEventListener("click", () => {
    voiceModal.classList.add("hidden");
  });
}

if (voiceRateSlider) {
  voiceRateSlider.addEventListener("input", (e) => {
    voiceRateVal.textContent = `${e.target.value}x`;
  });
}

if (voicePitchSlider) {
  voicePitchSlider.addEventListener("input", (e) => {
    voicePitchVal.textContent = `${e.target.value}`;
  });
}

if (voiceTestBtn) {
  voiceTestBtn.addEventListener("click", () => {
    speakText("Hello! ScureLocalAI is ready to read answers and documents aloud.");
  });
}

if (voiceSaveBtn) {
  voiceSaveBtn.addEventListener("click", () => {
    state.speechSettings.voiceURI = voiceSelect.value;
    state.speechSettings.rate = parseFloat(voiceRateSlider.value);
    state.speechSettings.pitch = parseFloat(voicePitchSlider.value);
    localStorage.setItem("scure_voice_settings", JSON.stringify(state.speechSettings));
    voiceModal.classList.add("hidden");
  });
}

// ============================================================================
// Document Management
// ============================================================================
function setStatus(message, stateClass = "busy") {
  if (!uploadStatus) return;
  uploadStatus.textContent = message;
  uploadStatus.className = `status-line ${stateClass}`;
  if (stateClass !== "busy") {
    setTimeout(() => {
      uploadStatus.textContent = "";
      uploadStatus.className = "status-line";
    }, 6000);
  }
}

function renderDocuments(docs) {
  state.documents = docs;
  if (docCountBadge) docCountBadge.textContent = docs.length;
  if (!documentList) return;
  documentList.innerHTML = "";

  // Update doc filter in gallery
  if (figuresDocFilter) {
    figuresDocFilter.innerHTML = '<option value="">All Documents</option>';
    docs.forEach((doc) => {
      const opt = document.createElement("option");
      opt.value = doc.filename;
      opt.textContent = doc.filename;
      figuresDocFilter.appendChild(opt);
    });
  }

  const query = (docSearchInput?.value || "").toLowerCase().trim();
  const filtered = query ? docs.filter((d) => d.filename.toLowerCase().includes(query)) : docs;

  if (filtered.length === 0) {
    const item = document.createElement("li");
    item.className = "document-card";
    item.innerHTML = `<span class="doc-badge">${docs.length === 0 ? "No documents indexed yet." : "No matching documents."}</span>`;
    documentList.appendChild(item);
    return;
  }

  for (const doc of filtered) {
    const item = document.createElement("li");
    item.className = "document-card";
    const figuresBadge = doc.figures > 0
      ? `<span class="doc-badge-figures" data-doc="${escapeHtml(doc.filename)}" title="View figures in this document">📸 ${doc.figures} figure${doc.figures > 1 ? "s" : ""}</span>`
      : "";

    item.innerHTML = `
      <div class="doc-main-row">
        <span class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
      </div>
      <div class="doc-badges">
        <span class="doc-badge">${doc.chunks} chunks</span>
        ${figuresBadge}
      </div>
      <div class="doc-actions">
        <button class="doc-delete-btn" type="button" data-id="${escapeHtml(doc.id)}">Delete</button>
      </div>
    `;
    documentList.appendChild(item);
  }
}

async function loadDocuments() {
  try {
    const response = await fetch("/api/documents");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const docs = await response.json();
    renderDocuments(docs);
  } catch (error) {
    setStatus(`Failed to load documents: ${error.message}`, "error");
  }
}

async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  setStatus(`Indexing ${file.name}... (analyzing text & figures)`, "busy");
  const response = await fetch("/api/documents", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Upload failed with status ${response.status}`);
  }

  const result = await response.json();
  const figText = result.figures > 0 ? ` & ${result.figures} visual figure(s)` : "";
  setStatus(`Successfully indexed ${result.filename} (${result.chunks} chunks${figText}).`, "success");
}

async function handleFileUploads(files) {
  for (const file of files) {
    await uploadFile(file);
  }
  await loadDocuments();
  await loadAllFigures();
}

async function deleteDocument(docId) {
  if (!confirm("Are you sure you want to delete this document and all its indexed figures?")) return;
  const response = await fetch(`/api/documents/${encodeURIComponent(docId)}`, {
    method: "DELETE",
  });
  if (!response.ok) throw new Error("Delete failed");
  setStatus("Document deleted.", "success");
  await loadDocuments();
  await loadAllFigures();
}

if (fileInput) {
  fileInput.addEventListener("change", async () => {
    try {
      await handleFileUploads([...fileInput.files]);
    } catch (err) {
      setStatus(err.message, "error");
    } finally {
      fileInput.value = "";
    }
  });
}

if (dropZone) {
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragging");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragging");
  });

  dropZone.addEventListener("drop", async (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragging");
    try {
      await handleFileUploads([...e.dataTransfer.files]);
    } catch (err) {
      setStatus(err.message, "error");
    }
  });
}

if (refreshButton) refreshButton.addEventListener("click", loadDocuments);
if (docSearchInput) docSearchInput.addEventListener("input", () => renderDocuments(state.documents));

if (documentList) {
  documentList.addEventListener("click", async (e) => {
    const delBtn = e.target.closest(".doc-delete-btn");
    if (delBtn) {
      try {
        await deleteDocument(delBtn.dataset.id);
      } catch (err) {
        setStatus(err.message, "error");
      }
      return;
    }

    const figBadge = e.target.closest(".doc-badge-figures");
    if (figBadge) {
      switchView("figures");
      if (figuresDocFilter) {
        figuresDocFilter.value = figBadge.dataset.doc;
        filterFigures();
      }
    }
  });
}

// ============================================================================
// Figures & Diagrams Gallery
// ============================================================================
async function loadAllFigures() {
  try {
    const response = await fetch("/api/documents/figures");
    if (!response.ok) throw new Error("Failed to load figures");
    state.figures = await response.json();
    if (navFiguresCount) navFiguresCount.textContent = state.figures.length;
    renderFiguresGallery(state.figures);
  } catch (err) {
    console.warn("Could not load figures:", err);
  }
}

function filterFigures() {
  if (!figuresDocFilter || !figuresKindFilter) return;
  const docVal = figuresDocFilter.value;
  const kindVal = figuresKindFilter.value;

  const filtered = state.figures.filter((fig) => {
    const matchDoc = !docVal || fig.filename === docVal;
    const matchKind = !kindVal || fig.kind === kindVal;
    return matchDoc && matchKind;
  });

  renderFiguresGallery(filtered);
}

if (figuresDocFilter) figuresDocFilter.addEventListener("change", filterFigures);
if (figuresKindFilter) figuresKindFilter.addEventListener("change", filterFigures);

function renderFiguresGallery(figures) {
  if (!figuresGrid) return;
  figuresGrid.innerHTML = "";
  if (figures.length === 0) {
    figuresGrid.innerHTML = `
      <div style="grid-column: 1/-1; text-align: center; padding: 48px; color: var(--text-muted);">
        <p>No figures found matching the current filters.</p>
        <span style="font-size: 12px; color: var(--text-dim);">Upload a PDF or presentation containing charts, photos, or diagrams.</span>
      </div>
    `;
    return;
  }

  for (const fig of figures) {
    const card = document.createElement("div");
    card.className = "figure-gallery-card";

    const kindLabel = formatKind(fig.kind);
    const caption = fig.caption || "No caption in document";
    const explanation = fig.vlm_description || "Visual figure extracted from document.";

    card.innerHTML = `
      <div class="gallery-card-thumb-wrap">
        <img src="${escapeHtml(fig.image_url)}" alt="${escapeHtml(caption)}" loading="lazy" />
      </div>
      <div class="gallery-card-body">
        <div class="badge-row">
          <span class="badge-kind">${escapeHtml(kindLabel)}</span>
          <span class="badge-page">Page ${fig.page}</span>
          <span class="badge-doc" title="${escapeHtml(fig.filename)}">${escapeHtml(fig.filename)}</span>
        </div>
        <div class="card-caption">${escapeHtml(caption)}</div>
        <div class="card-explanation-snippet">${escapeHtml(explanation)}</div>
        <div class="gallery-card-actions">
          <button type="button" class="action-btn-sm inspect-btn">🔍 Inspect & Explain</button>
          <button type="button" class="action-btn-sm ask-btn">💬 Ask in Chat</button>
        </div>
      </div>
    `;

    card.querySelector(".gallery-card-thumb-wrap").addEventListener("click", () => openFigureModal(fig));
    card.querySelector(".inspect-btn").addEventListener("click", () => openFigureModal(fig));
    card.querySelector(".ask-btn").addEventListener("click", () => {
      switchView("chat");
      questionInput.value = `Explain the ${kindLabel.toLowerCase()} on page ${fig.page} of ${fig.filename} in detail.`;
      autoResizeTextarea(questionInput);
      questionInput.focus();
    });

    figuresGrid.appendChild(card);
  }
}

function formatKind(kind) {
  if (!kind) return "Visual";
  if (kind === "chart") return "Chart / Graph";
  if (kind === "diagram_or_illustration") return "Diagram / Flowchart";
  if (kind === "photo") return "Photograph";
  return kind.replace(/_/g, " ");
}

// Lightbox Modal
function openFigureModal(fig) {
  state.selectedFigure = fig;
  modalImg.src = fig.image_url;
  modalKindBadge.textContent = formatKind(fig.kind);
  modalPageBadge.textContent = `Page ${fig.page}`;
  modalDocBadge.textContent = fig.filename;
  modalCaption.textContent = fig.caption || "No caption provided in document.";
  modalExplanation.textContent = fig.vlm_description || "AI visual explanation is ready.";
  modalContext.textContent = fig.surrounding_text || "No surrounding text available for this figure.";
  figureModal.classList.remove("hidden");
}

if (modalCloseBtn) modalCloseBtn.addEventListener("click", () => figureModal.classList.add("hidden"));
if (figureModal) {
  const backdrop = figureModal.querySelector(".modal-backdrop");
  if (backdrop) backdrop.addEventListener("click", () => figureModal.classList.add("hidden"));
}

if (modalSpeakDescBtn) {
  modalSpeakDescBtn.addEventListener("click", () => {
    if (state.selectedFigure && state.selectedFigure.vlm_description) {
      speakText(state.selectedFigure.vlm_description, modalSpeakDescBtn);
    }
  });
}

if (modalAskBtn) {
  modalAskBtn.addEventListener("click", () => {
    if (!state.selectedFigure) return;
    figureModal.classList.add("hidden");
    switchView("chat");
    questionInput.value = `Explain the ${formatKind(state.selectedFigure.kind).toLowerCase()} on page ${state.selectedFigure.page} of ${state.selectedFigure.filename}: what data and conclusions are presented?`;
    autoResizeTextarea(questionInput);
    questionInput.focus();
  });
}

// ============================================================================
// External Image Relevance Checker
// ============================================================================
let externalRelevanceFile = null;

if (relevanceFileInput) {
  relevanceFileInput.addEventListener("change", () => {
    const file = relevanceFileInput.files[0];
    if (file) handleRelevanceFile(file);
  });
}

if (relevanceDropzone) {
  relevanceDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    relevanceDropzone.classList.add("dragging");
  });

  relevanceDropzone.addEventListener("dragleave", () => {
    relevanceDropzone.classList.remove("dragging");
  });

  relevanceDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    relevanceDropzone.classList.remove("dragging");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      handleRelevanceFile(file);
    }
  });
}

function handleRelevanceFile(file) {
  externalRelevanceFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    relevanceImg.src = e.target.result;
    relevanceEmptyView.classList.add("hidden");
    relevanceImagePreview.classList.remove("hidden");
    checkRelevanceBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

if (checkRelevanceBtn) {
  checkRelevanceBtn.addEventListener("click", async () => {
    if (!externalRelevanceFile) return;

    checkRelevanceBtn.disabled = true;
    checkRelevanceBtn.innerHTML = `
      <span class="spin-icon">⚙️</span>
      Analyzing Visual Content & Relevance...
    `;

    relevanceResults.classList.remove("hidden");
    relevanceResults.innerHTML = `
      <div class="result-section-card" style="text-align: center; padding: 32px;">
        <p style="color: var(--primary);">Examining external visual components, extracting semantic entities, and cross-referencing indexed documents...</p>
      </div>
    `;

    const formData = new FormData();
    formData.append("file", externalRelevanceFile);
    if (relevanceFocusQuery?.value?.trim()) {
      formData.append("query", relevanceFocusQuery.value.trim());
    }

    try {
      const res = await fetch("/api/vision/check-relevance", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${res.status}`);
      }

      const data = await res.json();
      renderRelevanceResults(data);
    } catch (err) {
      relevanceResults.innerHTML = `
        <div class="result-section-card" style="border-color: var(--danger);">
          <h4>Error during analysis</h4>
          <p style="color: var(--danger);">${escapeHtml(err.message)}</p>
        </div>
      `;
    } finally {
      checkRelevanceBtn.disabled = false;
      checkRelevanceBtn.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
          <polyline points="22 4 12 14.01 9 11.01"/>
        </svg>
        Check Relevance to Documents
      `;
    }
  });
}

function renderRelevanceResults(data) {
  let scoreClass = "moderate";
  if (data.relevance_score >= 70) scoreClass = "high";
  else if (data.relevance_score < 35) scoreClass = "not-relevant";

  let matchingHtml = "";
  if (data.matching_chunks && data.matching_chunks.length > 0) {
    matchingHtml = data.matching_chunks
      .map(
        (chunk) => `
      <div class="matching-item">
        <div class="matching-item-header">
          <span>${escapeHtml(chunk.filename)} (page ${chunk.page})</span>
          <span>Similarity: ${Math.round((1 - chunk.distance) * 100)}%</span>
        </div>
        <div class="matching-item-snippet">${escapeHtml(chunk.text)}</div>
      </div>
    `
      )
      .join("");
  } else {
    matchingHtml = `<p style="font-size: 13px; color: var(--text-muted);">No matching document excerpts found.</p>`;
  }

  relevanceResults.innerHTML = `
    <div class="verdict-banner ${scoreClass}">
      <div>
        <div class="verdict-title">${escapeHtml(data.relevance_label)}</div>
        <span style="font-size: 12px; color: var(--text-muted);">Relevance Verdict</span>
      </div>
      <div class="verdict-score">${data.relevance_score}%</div>
    </div>

    <div class="result-section-card">
      <h4>AI Visual Content Summary</h4>
      <p>${escapeHtml(data.image_description)}</p>
    </div>

    <div class="result-section-card">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <h4>Relevance Rationale</h4>
        <button type="button" class="tiny-btn" id="speak-relevance-btn">🔊 Read Rationale</button>
      </div>
      <p style="white-space: pre-wrap;">${escapeHtml(data.analysis)}</p>
    </div>

    <div class="result-section-card">
      <h4>Top Matching Document Sections</h4>
      <div class="matching-list">
        ${matchingHtml}
      </div>
    </div>

    <button type="button" id="relevance-to-chat-btn" class="action-btn-secondary" style="margin-top: 8px;">
      💬 Discuss This Image in Chat with Documents
    </button>
  `;

  document.querySelector("#speak-relevance-btn").addEventListener("click", function () {
    speakText(data.analysis, this);
  });

  document.querySelector("#relevance-to-chat-btn").addEventListener("click", () => {
    switchView("chat");
    if (externalRelevanceFile) {
      setAttachedImage(externalRelevanceFile, relevanceImg.src);
    }
    questionInput.value = `Based on our indexed documents, how does this external image relate to our findings?`;
    autoResizeTextarea(questionInput);
    questionInput.focus();
  });
}

// ============================================================================
// Chat & Multimodal Q&A
// ============================================================================
function autoResizeTextarea(el) {
  if (!el) return;
  el.style.height = "auto";
  el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
}

if (questionInput) {
  questionInput.addEventListener("input", () => autoResizeTextarea(questionInput));

  questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitChat();
    }
  });
}

// Prompt Chips Handler
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    if (questionInput) {
      questionInput.value = chip.dataset.prompt;
      autoResizeTextarea(questionInput);
      submitChat();
    }
  });
});

// Image Attachment in Chat
if (imageInput) {
  imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      setAttachedImage(file, e.target.result);
    };
    reader.readAsDataURL(file);
  });
}

function setAttachedImage(file, base64Url) {
  state.attachedImageFile = file;
  state.attachedImageBase64 = base64Url;
  attachmentImg.src = base64Url;
  attachmentName.textContent = file.name;
  attachmentPreview.classList.remove("hidden");
}

function clearAttachedImage() {
  state.attachedImageFile = null;
  state.attachedImageBase64 = null;
  if (imageInput) imageInput.value = "";
  if (attachmentPreview) attachmentPreview.classList.add("hidden");
}

if (removeAttachmentBtn) removeAttachmentBtn.addEventListener("click", clearAttachedImage);

// Append message bubbles to chat
function appendMessage(role, text, sources = [], figures = [], attachedImgSrc = null) {
  const welcomeCard = document.querySelector(".welcome-card");
  if (welcomeCard) welcomeCard.remove();

  const article = document.createElement("article");
  article.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";

  if (attachedImgSrc) {
    const imgEl = document.createElement("img");
    imgEl.src = attachedImgSrc;
    imgEl.className = "message-attached-image";
    imgEl.alt = "Attached question image";
    bubble.appendChild(imgEl);
  }

  const textDiv = document.createElement("div");
  textDiv.className = "message-text-content";
  textDiv.innerHTML = formatMarkdown(text);
  bubble.appendChild(textDiv);

  // If figures were linked in assistant answer
  if (figures && figures.length > 0) {
    attachFiguresToBubble(bubble, figures);
  }

  // Sources pills
  if (sources && sources.length > 0) {
    attachSourcesToBubble(bubble, sources);
  }

  article.appendChild(bubble);

  // Assistant message actions: Read aloud & Copy
  if (role === "assistant" && !text.includes("Searching documents")) {
    attachActionsToArticle(article, text);
  }

  messagesContainer.appendChild(article);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
  return article;
}

function attachFiguresToBubble(bubble, figures) {
  const existing = bubble.querySelector(".message-figures-container");
  if (existing) existing.remove();

  const figuresBox = document.createElement("div");
  figuresBox.className = "message-figures-container";
  figuresBox.innerHTML = `<span class="figures-header-label">Linked Document Visuals (${figures.length})</span>`;

  const grid = document.createElement("div");
  grid.className = "message-figures-grid";

  for (const fig of figures) {
    const card = document.createElement("div");
    card.className = "figure-chat-card";
    card.innerHTML = `
      <img class="figure-chat-thumb" src="${escapeHtml(fig.image_url)}" alt="${escapeHtml(fig.caption)}" loading="lazy" />
      <div class="figure-chat-meta">
        <div class="figure-chat-badges">
          <span class="badge-kind">${escapeHtml(formatKind(fig.kind))}</span>
          <span class="badge-page">Page ${fig.page}</span>
        </div>
        <span class="figure-caption-snippet">${escapeHtml(fig.caption || fig.vlm_description || "Visual figure")}</span>
      </div>
    `;
    card.addEventListener("click", () => openFigureModal(fig));
    grid.appendChild(card);
  }

  figuresBox.appendChild(grid);
  bubble.appendChild(figuresBox);
}

function attachSourcesToBubble(bubble, sources) {
  const existing = bubble.querySelector(".sources-box");
  if (existing) existing.remove();

  const uniqueSources = [...new Set(sources.filter(Boolean))];
  if (uniqueSources.length === 0) return;

  const sourcesBox = document.createElement("div");
  sourcesBox.className = "sources-box";
  sourcesBox.innerHTML = `<span class="sources-label">Sources:</span>`;
  for (const src of uniqueSources) {
    const pill = document.createElement("span");
    pill.className = "source-pill";
    pill.textContent = src;
    sourcesBox.appendChild(pill);
  }
  bubble.appendChild(sourcesBox);
}

function attachActionsToArticle(article, text) {
  const existing = article.querySelector(".message-actions");
  if (existing) existing.remove();

  const actionsRow = document.createElement("div");
  actionsRow.className = "message-actions";

  const readBtn = document.createElement("button");
  readBtn.type = "button";
  readBtn.className = "action-tiny-btn";
  readBtn.innerHTML = `🔊 Read`;
  readBtn.addEventListener("click", () => {
    if (readBtn.classList.contains("speaking")) {
      stopSpeaking();
    } else {
      speakText(text, readBtn);
    }
  });

  const copyBtn = document.createElement("button");
  copyBtn.type = "button";
  copyBtn.className = "action-tiny-btn";
  copyBtn.innerHTML = `📋 Copy`;
  copyBtn.addEventListener("click", () => {
    navigator.clipboard.writeText(text);
    copyBtn.innerHTML = `✓ Copied`;
    setTimeout(() => (copyBtn.innerHTML = `📋 Copy`), 2000);
  });

  actionsRow.appendChild(readBtn);
  actionsRow.appendChild(copyBtn);
  article.appendChild(actionsRow);

  // Auto-read if enabled by user
  if (state.speechSettings.autoSpeak) {
    speakText(text, readBtn);
  }
}

// Markdown formatting helper
function formatMarkdown(text) {
  if (!text) return "";
  let html = escapeHtml(text);

  // Bold
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  // Italic
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  // Inline code
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  // Line breaks to paragraphs
  const paragraphs = html.split(/\n\n+/);
  return paragraphs.map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`).join("");
}

function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Submit Chat with Real-Time Streaming & Live Progress Timer
function submitChat() {
  if (!questionInput) return;
  const question = questionInput.value.trim();
  if (!question) return;

  const attachedImg = state.attachedImageBase64;
  appendMessage("user", question, [], [], attachedImg);

  questionInput.value = "";
  autoResizeTextarea(questionInput);
  clearAttachedImage();

  sendButton.disabled = true;

  // Create live pending bubble with timer
  const welcomeCard = document.querySelector(".welcome-card");
  if (welcomeCard) welcomeCard.remove();

  const article = document.createElement("article");
  article.className = "message assistant";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";

  const textDiv = document.createElement("div");
  textDiv.className = "message-text-content";

  // Animated loader with elapsed seconds counter
  let elapsed = 0;
  textDiv.innerHTML = `
    <div class="thinking-loader">
      <span class="spin-icon">⚙️</span>
      <span class="thinking-text">Searching documents & thinking...</span>
      <span class="thinking-timer" id="timer-val">0s</span>
    </div>
  `;
  bubble.appendChild(textDiv);
  article.appendChild(bubble);
  messagesContainer.appendChild(article);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  const timerInterval = setInterval(() => {
    elapsed += 1;
    const timerEl = article.querySelector("#timer-val");
    if (timerEl) timerEl.textContent = `${elapsed}s`;
  }, 1000);

  // Stream query
  handleStreamingQuery(question, attachedImg, article, bubble, textDiv, timerInterval);
}

async function handleStreamingQuery(question, attachedImg, article, bubble, textDiv, timerInterval) {
  let fullAnswer = "";
  let receivedTokens = false;
  let sources = [];
  let figures = [];

  try {
    const payload = { question };
    if (attachedImg) payload.image = attachedImg;

    const response = await fetch("/api/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      // Fallback to standard /api/query if stream endpoint has an issue
      clearInterval(timerInterval);
      return handleStandardQueryFallback(question, attachedImg, article, bubble, textDiv);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop(); // Keep incomplete line

      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const event = JSON.parse(line);

          if (event.type === "init") {
            sources = event.sources || [];
            figures = event.figures || [];
            if (figures.length > 0) attachFiguresToBubble(bubble, figures);
            if (sources.length > 0) attachSourcesToBubble(bubble, sources);
          } else if (event.type === "token") {
            if (!receivedTokens) {
              clearInterval(timerInterval);
              receivedTokens = true;
              textDiv.innerHTML = "";
            }
            fullAnswer += event.token;
            textDiv.innerHTML = formatMarkdown(fullAnswer) + `<span class="typing-cursor"></span>`;
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
          } else if (event.type === "done") {
            if (event.sources && event.sources.length > 0) {
              sources = event.sources;
              attachSourcesToBubble(bubble, sources);
            }
          }
        } catch (_) {}
      }
    }

    clearInterval(timerInterval);

    // Finalize
    const cursor = textDiv.querySelector(".typing-cursor");
    if (cursor) cursor.remove();

    if (!fullAnswer.trim()) {
      fullAnswer = "No answer returned. Please check that your documents are indexed.";
      textDiv.innerHTML = formatMarkdown(fullAnswer);
    } else {
      textDiv.innerHTML = formatMarkdown(fullAnswer);
    }

    attachActionsToArticle(article, fullAnswer);

  } catch (err) {
    clearInterval(timerInterval);
    console.error("Stream error, falling back to standard query:", err);
    handleStandardQueryFallback(question, attachedImg, article, bubble, textDiv);
  } finally {
    sendButton.disabled = false;
    questionInput.focus();
  }
}

async function handleStandardQueryFallback(question, attachedImg, article, bubble, textDiv) {
  try {
    const payload = { question };
    if (attachedImg) payload.image = attachedImg;

    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Query failed with status ${response.status}`);
    }

    const result = await response.json();
    textDiv.innerHTML = formatMarkdown(result.answer);

    if (result.figures && result.figures.length > 0) {
      attachFiguresToBubble(bubble, result.figures);
    }
    if (result.sources && result.sources.length > 0) {
      attachSourcesToBubble(bubble, result.sources);
    }

    attachActionsToArticle(article, result.answer);
  } catch (err) {
    textDiv.innerHTML = `<p style="color: var(--danger);">Error: ${escapeHtml(err.message)}</p>`;
  } finally {
    sendButton.disabled = false;
    questionInput.focus();
  }
}

if (chatForm) {
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    submitChat();
  });
}

// ============================================================================
// Boot
// ============================================================================
initSettings();
loadDocuments();
loadAllFigures();
