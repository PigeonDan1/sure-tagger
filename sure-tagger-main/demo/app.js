const data = window.DEMO_DATA;

const state = {
  filter: "all",
  activeId: data.samples[0]?.sampleId,
  tagGroup: "overview",
};

const tagGroups = [
  ["overview", "Overview"],
  ["basic_acoustic", "Basic acoustic"],
  ["audio_quality", "Audio quality"],
  ["room_acoustic", "Room acoustic"],
  ["sound_field_scene", "Sound field"],
  ["language_content", "Language"],
  ["speaker", "Speaker"],
];

const datasetColors = new Map();
const palette = ["#16796f", "#2d62a3", "#c67828", "#6a5aa8", "#b14b4b", "#557a2f", "#8b5b2e", "#52616b"];
const waveformCache = new Map();
let audioContext;
let waveformState = null;

const els = {
  metricGrid: document.querySelector("#metricGrid"),
  filterRow: document.querySelector("#filterRow"),
  sampleList: document.querySelector("#sampleList"),
  visibleCount: document.querySelector("#visibleCount"),
  activeDataset: document.querySelector("#activeDataset"),
  activeTitle: document.querySelector("#activeTitle"),
  activeNote: document.querySelector("#activeNote"),
  activeSampleId: document.querySelector("#activeSampleId"),
  audioPlayer: document.querySelector("#audioPlayer"),
  waveform: document.querySelector("#waveform"),
  transcriptText: document.querySelector("#transcriptText"),
  tagToolbar: document.querySelector("#tagToolbar"),
  tagGrid: document.querySelector("#tagGrid"),
  inputJson: document.querySelector("#inputJson"),
  rawJson: document.querySelector("#rawJson"),
  datasetBars: document.querySelector("#datasetBars"),
  coverageBars: document.querySelector("#coverageBars"),
};

function init() {
  data.summary.datasets.forEach((dataset, index) => {
    datasetColors.set(dataset.name, palette[index % palette.length]);
  });
  renderMetrics();
  renderFilters();
  renderSampleList();
  renderTagToolbar();
  renderBars();
  renderActiveSample();
  setupWaveformEvents();
}

function getActiveSample() {
  return data.samples.find((sample) => sample.sampleId === state.activeId) || data.samples[0];
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "未生成";
  if (typeof value !== "number") return String(value);
  if (Math.abs(value) >= 100) return value.toFixed(0);
  return value.toFixed(digits).replace(/\.00$/, "");
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "未生成";
  return `${Math.round(value * 100)}%`;
}

function formatDuration(value) {
  if (value === null || value === undefined) return "未知时长";
  return `${formatNumber(value, 2)}s`;
}

function formatUnit(value, unit, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "未生成";
  return `${formatNumber(value, digits)} ${unit}`;
}

function compactDataset(name) {
  return name.replace("TUT Urban Acoustic Scenes 2018", "TUT 2018").replace("WHAM! noise", "WHAM");
}

function hasTranscript(sample) {
  return Boolean(sample.transcript && sample.transcript.trim());
}

function renderMetrics() {
  const metrics = [
    ["Phase2 samples", data.summary.sampleCount, `${data.summary.datasetCount} 个数据集，每个 5 条样本`],
    ["Playable audio", data.summary.selectedCount, "phase2 全量音频，可播放和查看波形"],
    ["Noise categories", `${data.summary.coverage.sound_field_scene}/${data.summary.sampleCount}`, "DASS 噪音类别（docs/DASS.md ②–⑦），具体标签见 composition"],
    ["Speaker labels", `${data.summary.speakerCount}/${data.summary.sampleCount}`, `${data.summary.speakerMultiCount} 条 multi-speaker`],
  ];

  els.metricGrid.innerHTML = metrics
    .map(
      ([label, value, caption]) => `
        <article class="metric-card">
          <p class="metric-label">${label}</p>
          <div class="metric-value">${value}</div>
          <p class="metric-caption">${caption}</p>
        </article>
      `,
    )
    .join("");
}

function renderFilters() {
  const filters = [["all", "全部"], ...data.summary.datasets.map((dataset) => [dataset.name, compactDataset(dataset.name)])];
  els.filterRow.innerHTML = filters
    .map(
      ([value, label]) => `
        <button type="button" data-filter="${escapeAttr(value)}" class="${state.filter === value ? "is-active" : ""}">
          ${label}
        </button>
      `,
    )
    .join("");

  els.filterRow.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.filter = button.dataset.filter;
      const filtered = getFilteredSamples();
      if (!filtered.some((sample) => sample.sampleId === state.activeId)) {
        state.activeId = filtered[0]?.sampleId || data.samples[0]?.sampleId;
      }
      renderFilters();
      renderSampleList();
      renderActiveSample();
    });
  });
}

function getFilteredSamples() {
  if (state.filter === "all") return data.samples;
  return data.samples.filter((sample) => sample.dataset === state.filter);
}

function renderSampleList() {
  const samples = getFilteredSamples();
  els.visibleCount.textContent = `${samples.length} 条`;
  els.sampleList.innerHTML = samples
    .map((sample) => {
      const active = sample.sampleId === state.activeId ? "is-active" : "";
      const transcriptState = hasTranscript(sample) ? "transcript" : "no transcript";
      const noise = sample.tags.sound_field_scene?.external_noise_type || [];
      const noiseState = noise.length ? `noise: ${formatNoiseTypes(noise)}` : "noise: none";
      const speaker = sample.tags.speaker;
      const speakerState =
        speaker && speaker.speaker_count !== null && speaker.speaker_count !== undefined
          ? `speaker: ${speaker.speaker_count}`
          : "speaker: none";
      const color = datasetColors.get(sample.dataset) || "#16796f";
      return `
        <button type="button" class="sample-button ${active}" data-sample="${escapeAttr(sample.sampleId)}">
          <div class="sample-title">
            <span><span class="dataset-dot" style="background:${color}"></span> ${sample.title}</span>
          </div>
          <div class="sample-meta">
            <span>${compactDataset(sample.dataset)}</span>
            <span>${formatDuration(sample.durationSec)}</span>
            <span>${transcriptState}</span>
            <span>${noiseState}</span>
            <span>${speakerState}</span>
          </div>
        </button>
      `;
    })
    .join("");

  els.sampleList.querySelectorAll(".sample-button").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeId = button.dataset.sample;
      renderSampleList();
      renderActiveSample();
    });
  });
}

function renderTagToolbar() {
  els.tagToolbar.innerHTML = tagGroups
    .map(
      ([value, label]) => `
        <button type="button" role="tab" data-group="${value}" class="${state.tagGroup === value ? "is-active" : ""}">
          ${label}
        </button>
      `,
    )
    .join("");
  els.tagToolbar.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.tagGroup = button.dataset.group;
      renderTagToolbar();
      renderTags(getActiveSample());
    });
  });
}

function renderActiveSample() {
  const sample = getActiveSample();
  if (!sample) return;

  els.activeDataset.textContent = sample.dataset;
  els.activeTitle.textContent = sample.title;
  els.activeNote.textContent = sample.note;
  els.activeSampleId.textContent = sample.sampleId;
  els.audioPlayer.src = sample.audio;
  els.audioPlayer.load();

  if (hasTranscript(sample)) {
    els.transcriptText.classList.remove("is-empty");
    els.transcriptText.textContent = sample.transcript;
  } else {
    els.transcriptText.classList.add("is-empty");
    els.transcriptText.textContent = "该样本没有 transcript，语言层只保留空结果，主要展示声学与声场标签。";
  }

  els.inputJson.textContent = JSON.stringify(buildInputRecord(sample), null, 2);
  els.rawJson.textContent = JSON.stringify(sample.tags, null, 2);
  renderTags(sample);
  loadWaveform(sample);
}

function buildInputRecord(sample) {
  return {
    corpus: {
      dataset_name: sample.dataset,
      source_urls: {
        article: [],
        github: [],
        huggingface: [],
        dataset_card: [],
      },
      native_metadata: {},
    },
    sample: {
      sample_id: sample.sampleId,
      audio: {
        path: manifestAudioPath(sample),
      },
      text: {
        transcript: sample.transcript || "",
      },
      native_metadata: sample.nativeMetadata || {},
    },
  };
}

function manifestAudioPath(sample) {
  const path = sample.audio || "";
  if (path.startsWith("assets/")) return path.slice("assets/".length);
  return path;
}

function renderTags(sample) {
  const cards = getCardsForGroup(sample, state.tagGroup);
  els.tagGrid.innerHTML = cards
    .map(
      (card) => `
        <article class="tag-card ${card.isNull ? "is-null" : ""}">
          <p class="tag-label">${card.label}</p>
          <div class="tag-value">${card.value}</div>
          <p class="tag-subtext">${card.subtext || ""}</p>
        </article>
      `,
    )
    .join("");
}

function getCardsForGroup(sample, group) {
  const tags = sample.tags;
  const basic = tags.basic_acoustic || {};
  const quality = tags.audio_quality || {};
  const room = tags.room_acoustic || {};
  const scene = tags.sound_field_scene || {};
  const lang = tags.language_content || {};
  const speaker = tags.speaker || {};

  if (group === "overview") {
    return [
      card("Duration", formatDuration(basic.duration_sec), `${formatNumber(basic.sample_rate_hz, 0)} Hz / ${basic.channels ?? "?"} channel`),
      card("SNR", formatUnit(quality.snr_db, "dB", 2), "更高通常表示背景噪声更少", quality.snr_db == null),
      card("Non-speech ratio", formatPercent(basic.nonspeech_ratio), `${(basic.silence_segments || []).length} non-speech segment(s)`),
      card("DNSMOS OVRL", formatNumber(quality.dnsmos_ovrl, 2), "模型估计的整体语音质量", quality.dnsmos_ovrl == null),
      card("Language", lang.language || "未生成", hasTranscript(sample) ? `${lang.word_count ?? 0} words/chars` : "缺少 transcript", !lang.language),
      card("External noise", formatNoiseTypes(scene.external_noise_type || []), "DASS 噪音类别（docs/DASS.md ②–⑦）", !(scene.external_noise_type || []).length),
      card("RT60", formatUnit(room.rt60_sec, "s", 3), "Rec-RIR reverberation estimate", room.rt60_sec == null),
      card("Speaker count", speaker.speaker_count ?? "未生成", speakerStatus(speaker), speaker.speaker_count == null),
    ];
  }

  if (group === "basic_acoustic") {
    return [
      card("duration_sec", formatNumber(basic.duration_sec, 3), "音频时长"),
      card("sample_rate_hz", formatNumber(basic.sample_rate_hz, 0), "采样率"),
      card("channels", basic.channels ?? "未生成", "声道数"),
      card("nonspeech_ratio", formatPercent(basic.nonspeech_ratio), "metadata 优先；缺失时退回 FireRed VAD"),
      card("silence_segments", listSegments(basic.silence_segments), "未检测到人类语音的片段，波形中以琥珀色显示", !basic.silence_segments?.length),
    ];
  }

  if (group === "audio_quality") {
    return [
      card("snr_db", formatUnit(quality.snr_db, "dB", 3), "Brouhaha 逐帧 SNR 均值", quality.snr_db == null),
      card("dnsmos_sig", formatNumber(quality.dnsmos_sig, 3), "speech quality", quality.dnsmos_sig == null),
      card("dnsmos_bak", formatNumber(quality.dnsmos_bak, 3), "background quality", quality.dnsmos_bak == null),
      card("dnsmos_ovrl", formatNumber(quality.dnsmos_ovrl, 3), "overall quality", quality.dnsmos_ovrl == null),
      card("dnsmos_p808", formatNumber(quality.dnsmos_p808, 3), "P.808 MOS estimate", quality.dnsmos_p808 == null),
    ];
  }

  if (group === "room_acoustic") {
    return [
      card("rt60_sec", formatUnit(room.rt60_sec, "s", 4), "Rec-RIR 混响衰减时间（T20 拟合外推 -60 dB）", room.rt60_sec == null),
      card("c50_db", formatNumber(room.c50_db, 4), "基于估计 RIR 的清晰度（直达声后 50ms 早/晚能量比）", room.c50_db == null),
      card("far_field", room.far_field ?? "未生成", "预留字段", room.far_field == null),
    ];
  }

  if (group === "sound_field_scene") {
    return [
      card("speech_music_events", listValue(scene.speech_music_events), "FireRed AED 检出类别（speech/singing/music 固定顺序）", !scene.speech_music_events?.length),
      card("music_present", scene.music_present ?? "未生成", "AED 是否检出音乐；同时门控 noise_composition 的音乐桶", scene.music_present == null),
      card("external_noise_type", formatNoiseTypes(scene.external_noise_type || []), "DASS 噪音类别键（docs/DASS.md ②–⑦；未被排除且 ≥0.25 的标签所归类别，人类/未归类不公开）", !scene.external_noise_type?.length),
      card("noise_composition", formatComposition(scene.noise_composition), "展开 external_noise_type 各类别的具体标签；每类 top-3 ≥ 0.25（与类别阈值对齐），音乐以 FireRed AED 门控", scene.noise_composition == null),
    ];
  }

  if (group === "language_content") {
    return [
      card("language", lang.language || "未生成", hasTranscript(sample) ? "FireRed LID 音频语言识别" : "缺少 transcript", !lang.language),
      card("word_count", lang.word_count ?? "未生成", "英文按 token，中文按字符/分词结果"),
      card("filler", lang.filler ?? "未生成", "um/uh/okay 等填充表达"),
      card("punctuation", punctuationValue(lang.punctuation), "标点统计", !lang.punctuation),
      card("repetition", repetitionValue(lang.repetition), "重复词统计", !lang.repetition),
    ];
  }

  const speakerReady = speakerHasValues(speaker);
  return [
    card("speaker_count", speaker.speaker_count ?? "未生成", "解析出的说话人数（quality-shadow：Sortformer 主判）", !speakerReady),
    card("speaker_present", speaker.speaker_present ?? "未生成", "由 speaker_count 确定性派生", speaker.speaker_present == null),
    card("multi_speaker", speaker.multi_speaker ?? "未生成", "是否包含两个或更多说话人", !speakerReady),
    card("speaker_change_count", speaker.speaker_change_count ?? "未生成", "说话人切换次数", !speakerReady),
    card("speaker_change", speaker.speaker_change ?? "未生成", "是否发生说话人切换", !speakerReady),
    card("overlap_ratio", formatPercent(speaker.overlap_ratio), "重叠发言时长 / 有效语音时长", speaker.overlap_ratio == null),
    card("speaker_overlap", speaker.speaker_overlap ?? "未生成", "是否多人同时发言（Pyannote 主判）", !speakerReady),
    card("profiles", speaker.profiles ? JSON.stringify(speaker.profiles) : "未生成", "确定性说话人语速、相对音高和相对音量", speaker.profiles == null),
    card("asr_transcript", speaker.asr_transcript ?? "未生成", "MOSS 全音频时间线文本，按时间顺序拼接", !speaker.asr_transcript),
  ];
}

function card(label, value, subtext, isNull = false) {
  return { label, value: String(value), subtext, isNull };
}

function listValue(value, limit) {
  if (!Array.isArray(value) || value.length === 0) return "[]";
  const shown = limit ? value.slice(0, limit) : value;
  const suffix = limit && value.length > limit ? ` +${value.length - limit}` : "";
  return shown.join(", ") + suffix;
}

function listSegments(segments) {
  if (!Array.isArray(segments) || segments.length === 0) return "[]";
  return segments.map((segment) => `${formatNumber(segment.start_sec, 2)}-${formatNumber(segment.end_sec, 2)}s`).join(", ");
}

const NOISE_CATEGORY_LABELS = {
  music: "音乐",
  animal: "动物",
  mechanical: "机械",
  nature: "自然",
  formless: "无明确声源",
  channel_environment: "声道/环境",
};

function formatNoiseTypes(keys) {
  if (!Array.isArray(keys) || !keys.length) return "空（无检出类别）";
  return keys.map((key) => NOISE_CATEGORY_LABELS[key] || key).join(", ");
}

function formatComposition(value) {
  if (!value || typeof value !== "object") return "未生成";
  const parts = Object.entries(value)
    .filter(([, labels]) => Array.isArray(labels) && labels.length > 0)
    .map(([key, labels]) => `${key}: ${labels.join(", ")}`);
  return parts.length ? parts.join(" · ") : "空（无检出类别）";
}

function punctuationValue(value) {
  if (!value) return "未生成";
  return `${value.punctuation_count ?? 0} punctuation / terminal ${value.has_terminal_punctuation ? "yes" : "no"}`;
}

function repetitionValue(value) {
  if (!value) return "未生成";
  return `${value.repetition_count ?? 0} repetition / ${value.has_repetition ? "detected" : "none"}`;
}

function speakerStatus(value) {
  if (!value) return "未生成";
  return speakerHasValues(value) ? "已生成" : "未生成";
}

function speakerHasValues(value) {
  if (!value) return false;
  return Object.values(value).some((item) => item !== null && item !== undefined);
}

function renderBars() {
  const maxDataset = Math.max(...data.summary.datasets.map((dataset) => dataset.count));
  els.datasetBars.innerHTML = data.summary.datasets
    .map((dataset) => renderBar(dataset.name, dataset.count, maxDataset, `${dataset.count} samples`))
    .join("");

  const groupLabels = {
    basic_acoustic: "Basic acoustic",
    audio_quality: "Audio quality",
    room_acoustic: "Room acoustic",
    sound_field_scene: "Sound field scene",
    language_content: "Language content",
    speaker: "Speaker diarization",
  };
  els.coverageBars.innerHTML = Object.entries(data.summary.coverage)
    .map(([group, count]) => renderBar(groupLabels[group] || group, count, data.summary.sampleCount, `${count}/${data.summary.sampleCount}`))
    .join("");
}

function renderBar(label, value, max, text) {
  const width = max > 0 ? Math.max(2, Math.round((value / max) * 100)) : 0;
  return `
    <div class="bar-row">
      <div class="bar-label">${label}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
      <div class="bar-value">${text}</div>
    </div>
  `;
}

function setupWaveformEvents() {
  window.addEventListener("resize", () => {
    if (waveformState) drawWaveform(waveformState);
  });

  els.audioPlayer.addEventListener("timeupdate", () => {
    if (waveformState) drawWaveform(waveformState);
  });

  els.waveform.addEventListener("click", (event) => {
    if (!waveformState || !els.audioPlayer.duration) return;
    const rect = els.waveform.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    els.audioPlayer.currentTime = ratio * els.audioPlayer.duration;
    drawWaveform(waveformState);
  });
}

async function loadWaveform(sample) {
  waveformState = null;
  drawLoadingWaveform("正在解码音频波形...");

  try {
    if (!waveformCache.has(sample.sampleId)) {
      if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
      }
      const response = await fetch(sample.audio);
      const arrayBuffer = await response.arrayBuffer();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      waveformCache.set(sample.sampleId, buildWaveformPeaks(audioBuffer));
    }
    waveformState = {
      peaks: waveformCache.get(sample.sampleId),
      duration: sample.tags.basic_acoustic?.duration_sec || els.audioPlayer.duration || sample.durationSec,
      silenceSegments: sample.tags.basic_acoustic?.silence_segments || [],
    };
    drawWaveform(waveformState);
  } catch (error) {
    drawLoadingWaveform("无法读取波形；请通过本地 HTTP 服务访问 demo。");
  }
}

function buildWaveformPeaks(audioBuffer) {
  const channel = audioBuffer.getChannelData(0);
  const bucketCount = 900;
  const bucketSize = Math.max(1, Math.floor(channel.length / bucketCount));
  const peaks = [];
  for (let i = 0; i < bucketCount; i += 1) {
    let min = 1;
    let max = -1;
    const start = i * bucketSize;
    const end = Math.min(channel.length, start + bucketSize);
    for (let j = start; j < end; j += 1) {
      const value = channel[j];
      if (value < min) min = value;
      if (value > max) max = value;
    }
    peaks.push(Math.max(Math.abs(min), Math.abs(max)));
  }
  return peaks;
}

function resizeCanvas() {
  const canvas = els.waveform;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(320, Math.floor(rect.width * dpr));
  canvas.height = Math.floor(160 * dpr);
  return { canvas, width: canvas.width, height: canvas.height, dpr };
}

function drawLoadingWaveform(message) {
  const { canvas, width, height, dpr } = resizeCanvas();
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#101716";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "#dfe8e4";
  ctx.font = `${14 * dpr}px system-ui, sans-serif`;
  ctx.fillText(message, 18 * dpr, 84 * dpr);
}

function drawWaveform(current) {
  const { canvas, width, height, dpr } = resizeCanvas();
  const ctx = canvas.getContext("2d");
  const centerY = height / 2;
  const barWidth = Math.max(1, Math.floor(width / current.peaks.length));

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#101716";
  ctx.fillRect(0, 0, width, height);

  if (current.duration) {
    ctx.fillStyle = "rgba(198, 120, 40, 0.34)";
    for (const segment of current.silenceSegments) {
      const start = (segment.start_sec / current.duration) * width;
      const end = (segment.end_sec / current.duration) * width;
      ctx.fillRect(start, 0, Math.max(1, end - start), height);
    }
  }

  ctx.strokeStyle = "rgba(255, 255, 255, 0.16)";
  ctx.beginPath();
  ctx.moveTo(0, centerY);
  ctx.lineTo(width, centerY);
  ctx.stroke();

  ctx.fillStyle = "#8fd0c6";
  current.peaks.forEach((peak, index) => {
    const x = index * barWidth;
    const barHeight = Math.max(2 * dpr, peak * height * 0.78);
    ctx.fillRect(x, centerY - barHeight / 2, Math.max(1, barWidth - 1), barHeight);
  });

  const duration = els.audioPlayer.duration || current.duration;
  if (duration && els.audioPlayer.currentTime) {
    const x = (els.audioPlayer.currentTime / duration) * width;
    ctx.fillStyle = "#f2d39a";
    ctx.fillRect(x, 0, 2 * dpr, height);
  }

  canvas.title = "点击波形跳转播放";
}

function escapeAttr(value) {
  return String(value).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/* ------------------------------------------------------------------ */
/* Captioner comparison (caption_pairs_3000, first 20 clips)          */
/* ------------------------------------------------------------------ */

const captionData = window.CAPTION_COMPARE;
const captionState = { activeId: null };

function initCaptionCompare() {
  if (!captionData || !captionData.samples?.length) return;
  captionState.activeId = captionData.samples[0].sampleId;
  renderCaptionMetrics();
  renderCaptionList();
  renderCaptionSample();
}

function renderCaptionMetrics() {
  const s = captionData.summary;
  const noise = (ours) =>
    `${s.bothNoise} 两侧均有 / ${s.onlyCaptioner} 仅 captioner / ${s.onlyOurs} 仅 sure-tagger`;
  const metrics = [
    ["对比样本", s.sampleCount, "caption_pairs_3000 前 20 条，纯音频仅跑 sound_field 标签"],
    ["音乐判定一致", `${s.musicAgree}/${s.sampleCount}`, `冲突 ${s.musicConflict} 条，captioner 未提及 ${s.musicUnmentioned} 条`],
    ["噪音检出对比", `${s.bothNoise + s.onlyOurs}/${s.sampleCount}`, noise()],
    ["captioner 噪音标签", `${s.onlyCaptioner + s.bothNoise}/${s.sampleCount}`, "qwen-omni 结构化提取的 external_noise_info.type"],
  ];
  const grid = document.querySelector("#captionMetrics");
  if (!grid) return;
  grid.innerHTML = metrics
    .map(
      ([label, value, caption]) => `
        <article class="metric-card">
          <p class="metric-label">${label}</p>
          <div class="metric-value">${value}</div>
          <p class="metric-caption">${caption}</p>
        </article>
      `,
    )
    .join("");
}

function renderCaptionList() {
  const list = document.querySelector("#captionList");
  const count = document.querySelector("#captionCount");
  if (!list) return;
  count.textContent = captionData.samples.length;
  list.innerHTML = captionData.samples
    .map(
      (sample) => `
        <button type="button" data-sample-id="${escapeAttr(sample.sampleId)}"
          class="${sample.sampleId === captionState.activeId ? "is-active" : ""}">
          <span class="sample-id">${escapeHtml(sample.sampleId)}</span>
        </button>
      `,
    )
    .join("");
  list.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      captionState.activeId = button.dataset.sampleId;
      renderCaptionList();
      renderCaptionSample();
    });
  });
}

function getCaptionSample() {
  return (
    captionData.samples.find((sample) => sample.sampleId === captionState.activeId) ||
    captionData.samples[0]
  );
}

function renderCaptionSample() {
  const sample = getCaptionSample();
  const elsCaption = {
    title: document.querySelector("#captionSampleTitle"),
    note: document.querySelector("#captionSampleNote"),
    id: document.querySelector("#captionSampleId"),
    audio: document.querySelector("#captionAudio"),
    grid: document.querySelector("#captionCompareGrid"),
    text: document.querySelector("#captionText"),
  };
  if (!elsCaption.grid) return;

  elsCaption.title.textContent = sample.sampleId;
  elsCaption.id.textContent = `#${captionData.samples.indexOf(sample) + 1} / ${captionData.samples.length}`;
  elsCaption.audio.src = sample.audio;

  const ours = sample.ours;
  const capt = sample.captioner;

  const categoryLabels = sample.ours.categoryLabels || {};
  const ourNoise = [];
  for (const category of ours.categories || []) {
    const labels = (ours.composition && ours.composition[category]) || [];
    ourNoise.push({
      key: category,
      name: categoryLabels[category] || category,
      labels,
    });
  }

  const musicNote = [];
  musicNote.push(
    ours.music == null ? "sure-tagger: AED 未运行" : `sure-tagger: AED 判 ${ours.music ? "有音乐" : "无音乐"}`,
  );
  musicNote.push(
    capt.musicState == null ? "captioner: 未提及" : `captioner: ${capt.musicState}`,
  );
  elsCaption.note.textContent = musicNote.join(" · ");

  const renderTag = (value) =>
    value == null || value === ""
      ? `<span class="caption-tag is-missing">未标注</span>`
      : `<span class="caption-tag">${escapeHtml(value)}</span>`;

  const listValue = (values) =>
    !values?.length ? `<span class="caption-tag is-missing">未检出</span>` : values.map((v) => `<span class="caption-tag">${escapeHtml(v)}</span>`).join(" ");

  elsCaption.grid.innerHTML = `
    <div class="caption-source">
      <h3>sure-tagger（FireRed AED + DASS）</h3>
      <dl>
        <dt>speech / singing / music 事件</dt>
        <dd>${listValue(ours.events)}</dd>
        <dt>music_present</dt>
        <dd>${ours.music == null ? `<span class="caption-tag is-missing">未生成</span>` : `<span class="caption-tag">${ours.music ? "true" : "false"}</span>`}</dd>
        <dt>external_noise_type</dt>
        <dd>${listValue(ours.categories)}</dd>
        <dt>noise_composition</dt>
        <dd>${
          ourNoise.length
            ? ourNoise
                .map(
                  (item) =>
                    `<span class="caption-tag is-category">${escapeHtml(item.name)}</span> ${listValue(item.labels)}`,
                )
                .join("<br>")
            : `<span class="caption-tag is-missing">全部类别为空</span>`
        }</dd>
      </dl>
    </div>
    <div class="caption-source">
      <h3>qwen-omni captioner</h3>
      <dl>
        <dt>sound_event</dt>
        <dd>${renderTag(capt.soundEvent)}</dd>
        <dt>music_state</dt>
        <dd>${renderTag(capt.musicState)}</dd>
        <dt>external_noise_info.type</dt>
        <dd>${listValue(capt.noiseLabels)}</dd>
      </dl>
    </div>
  `;

  elsCaption.text.textContent = capt.caption;
}

init();
initCaptionCompare();
