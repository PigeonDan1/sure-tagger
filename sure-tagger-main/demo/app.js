(function () {
  "use strict";

  const copy = {
    en: {
      heroEyebrow: "Interactive demo · curated audio",
      heroTitle: "From speech evidence to useful tags.",
      heroLead: "Listen to a real clip, inspect the structured fields and descriptions, then combine tags to find comparable recordings.",
      heroPillOne: "5 audio clips",
      heroPillTwo: "3 prediction sources",
      heroPillThree: "Tag-driven queries",
      listenEyebrow: "Listen",
      listenTitle: "Choose a recording",
      listenLead: "Listen to each clip and inspect its saved predictions.",
      openAudio: "Open audio ↗",
      openPredictions: "Full predictions ↗",
      transcriptLabel: "ASR transcript",
      tagsEyebrow: "Representation",
      tagsTitle: "What the tags say",
      tagsLead: "Typed fields make conditions queryable. A short description preserves context; unknown values stay unknown.",
      tagDescription: "Evidence-linked description",
      showJson: "View complete SURE-Tagger JSON",
      queryEyebrow: "Compose conditions",
      queryTitle: "Find a speech scene",
      queryLead: "Combine tag values to make a small, comparable subset. This preview does not compute ASR error rates.",
      speakerFilter: "Speakers",
      overlapFilter: "Overlap",
      languageFilter: "Language",
      noiseFilter: "External noise tag",
      clearFilters: "Clear filters",
      compareEyebrow: "Same audio",
      compareTitle: "Compare descriptions",
      compareLead: "Read each system's output for the selected clip. Differences are visible without hiding the full records.",
      footerCaveat: "These clips are examples, not a performance summary. Some labels are uncertain or disagree with the available references.",
      sourceLink: "Samples and references ↗",
      attributionLink: "Audio attribution ↗",
      usageLink: "Run this demo ↗",
      duration: "Duration",
      speakerCount: "Speakers",
      overlap: "Overlap",
      language: "Language",
      unknown: "Unknown",
      noTranscript: "No speech transcript is available for this clip.",
      noCaption: "No description is available in this record.",
      speakersGroup: "Speaker interaction",
      languageGroup: "Language and text",
      acousticGroup: "Acoustic evidence",
      soundGroup: "Sound field",
      count: "Speaker count",
      changes: "Speaker changes",
      words: "Word count",
      nonSpeech: "Non-speech",
      snr: "SNR",
      reverberation: "RT60",
      soundEvents: "Detected events",
      music: "Music present",
      noise: "External noise categories",
      all: "Any",
      noSpeakers: "No speakers",
      oneSpeaker: "One speaker",
      multipleSpeakers: "Multiple speakers",
      yes: "True",
      no: "False",
      tagPresent: "Tag present",
      tagEmpty: "Empty array",
      matches: "matching clips",
      noMatches: "No clips match these conditions. Try removing a filter.",
      selectClip: "Inspect clip",
      notAvailable: "Not available in this data file.",
      reportedSpeakers: "Reported speakers",
      localMissing: "Local data.js was not found; showing the bundled examples.",
      oneTagSource: "1 tag source",
      localEyebrow: "Interactive demo · local data",
      localCaveat: "Tags may be uncertain. Inspect the source audio and full JSON before drawing conclusions.",
    },
    zh: {
      heroEyebrow: "交互演示 · 精选音频",
      heroTitle: "从语音证据到可用的标签。",
      heroLead: "试听真实片段，查看结构字段与描述，再组合标签寻找可比较的录音。",
      heroPillOne: "5 条音频",
      heroPillTwo: "3 个预测来源",
      heroPillThree: "标签组合查询",
      listenEyebrow: "试听",
      listenTitle: "选择一段录音",
      listenLead: "试听每条音频，并查看保存的预测标签。",
      openAudio: "打开音频 ↗",
      openPredictions: "完整预测 ↗",
      transcriptLabel: "ASR 转写",
      tagsEyebrow: "半结构化表征",
      tagsTitle: "标签记录了什么",
      tagsLead: "有类型的字段可用于查询，简短描述保留语境；无法确认的值仍保留为未知。",
      tagDescription: "有证据支撑的描述",
      showJson: "查看完整 SURE-Tagger JSON",
      queryEyebrow: "组合条件",
      queryTitle: "寻找语音场景",
      queryLead: "组合标签值，选出可比较的小样本子集。此预览不计算 ASR 错误率。",
      speakerFilter: "说话人数",
      overlapFilter: "重叠语音",
      languageFilter: "语言",
      noiseFilter: "外部噪声标签",
      clearFilters: "清除筛选",
      compareEyebrow: "同一音频",
      compareTitle: "对比三种描述",
      compareLead: "阅读当前音频的三种系统输出；完整记录可在上方链接中查看。",
      footerCaveat: "这些片段仅用于演示，不代表整体性能。部分标签存在不确定性，或与可用参考信息不一致。",
      sourceLink: "样本与参考信息 ↗",
      attributionLink: "音频来源 ↗",
      usageLink: "运行演示页 ↗",
      duration: "时长",
      speakerCount: "说话人数",
      overlap: "重叠语音",
      language: "语言",
      unknown: "未知",
      noTranscript: "这段音频没有可用的语音转写。",
      noCaption: "这份记录没有可用的描述。",
      speakersGroup: "说话人交互",
      languageGroup: "语言与文本",
      acousticGroup: "声学证据",
      soundGroup: "声场与事件",
      count: "说话人数",
      changes: "说话人切换",
      words: "词数",
      nonSpeech: "非语音占比",
      snr: "信噪比",
      reverberation: "RT60 混响",
      soundEvents: "检出事件",
      music: "存在音乐",
      noise: "外部噪声类别",
      all: "不限",
      noSpeakers: "无人声",
      oneSpeaker: "单人",
      multipleSpeakers: "多人",
      yes: "true",
      no: "false",
      tagPresent: "有类别",
      tagEmpty: "空数组",
      matches: "条匹配音频",
      noMatches: "没有音频满足这些条件，请减少筛选条件。",
      selectClip: "查看音频",
      notAvailable: "这份数据没有提供此输出。",
      reportedSpeakers: "报告的说话人数",
      localMissing: "未找到本地 data.js，已显示仓库自带样本。",
      oneTagSource: "1 个标签来源",
      localEyebrow: "交互演示 · 本地数据",
      localCaveat: "标签可能存在不确定性。得出结论前，请核对原始音频与完整 JSON。",
    },
  };

  const params = new URLSearchParams(window.location.search);
  let language = params.get("lang") === "zh" ? "zh" : "en";
  let samples = [];
  let bundledSource = true;
  let activeId = params.get("sample");
  const filters = { speakers: "all", overlap: "all", language: "all", noise: "all" };
  const byId = (id) => document.getElementById(id);
  const tr = (key) => copy[language][key] || key;

  function element(tag, className, text) {
    const result = document.createElement(tag);
    if (className) result.className = className;
    if (text !== undefined && text !== null) result.textContent = String(text);
    return result;
  }

  function localized(value) {
    if (typeof value === "string") return value;
    if (value && typeof value === "object") return value[language] || value.en || value.zh || "";
    return "";
  }

  function currentSample() {
    return samples.find((sample) => sample.sampleId === activeId) || samples[0];
  }

  function tagsOf(sample) {
    return sample.tags || sample.predictions?.sure_tagger || {};
  }

  function displayValue(value) {
    if (value === null || value === undefined) return tr("unknown");
    if (Array.isArray(value)) return value.length ? value.join(", ") : "[]";
    if (typeof value === "boolean") return String(value);
    if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(2);
    return String(value);
  }

  function displayLanguage(code) {
    if (!code) return tr("unknown");
    const names = {
      en: { en: "English", zh: "英语" },
      ru: { en: "Russian", zh: "俄语" },
      pt: { en: "Portuguese", zh: "葡萄牙语" },
      zh: { en: "Chinese", zh: "中文" },
    };
    return names[code]?.[language] || code;
  }

  function writeUrl() {
    try {
      const url = new URL(window.location.href);
      url.searchParams.set("lang", language);
      if (activeId) url.searchParams.set("sample", activeId);
      window.history.replaceState(null, "", url);
    } catch (_error) {
      // A file preview may not permit history changes.
    }
  }

  function renderStaticText() {
    document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
    document.title = language === "zh" ? "SURE-Tagger · 交互演示" : "SURE-Tagger · Interactive demo";
    document.querySelectorAll("[data-i18n]").forEach((node) => {
      node.textContent = tr(node.dataset.i18n);
    });
    document.querySelectorAll("[data-language]").forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.language === language));
    });
    document.querySelector('[data-i18n="heroPillOne"]').textContent =
      samples.length + (language === "zh" ? " 条音频" : samples.length === 1 ? " audio clip" : " audio clips");
    if (!bundledSource) {
      document.querySelector('[data-i18n="heroEyebrow"]').textContent = tr("localEyebrow");
      document.querySelector('[data-i18n="heroPillTwo"]').textContent = tr("oneTagSource");
      document.querySelector('[data-i18n="footerCaveat"]').textContent = tr("localCaveat");
    }
    byId("readmeLink").href = language === "zh" ? "../../README.zh-CN.md" : "../../README.md";
  }

  function renderSampleList() {
    const list = byId("sampleList");
    list.replaceChildren();
    samples.forEach((sample, index) => {
      const button = element("button", "sample-button");
      button.type = "button";
      button.setAttribute("aria-pressed", String(sample.sampleId === activeId));
      button.append(
        element("span", "sample-index", String(index + 1).padStart(2, "0")),
        element("span", "sample-name", localized(sample.title) || sample.dataset),
        element("span", "sample-dataset", sample.dataset || ""),
      );
      button.addEventListener("click", () => selectSample(sample.sampleId));
      list.append(button);
    });
  }

  function addFact(container, label, value) {
    const card = element("div", "fact");
    card.append(element("span", "fact-label", label), element("strong", "fact-value", value));
    container.append(card);
  }

  function renderPlayer(sample) {
    const tags = tagsOf(sample);
    const player = byId("audioPlayer");
    byId("activeDataset").textContent = sample.dataset || "";
    byId("activeTitle").textContent = localized(sample.title) || sample.sampleId;
    byId("activeFocus").textContent = localized(sample.focus) || sample.note || "";
    byId("activeTrack").textContent = sample.track || "";
    byId("activeTrack").hidden = !sample.track;
    if (player.getAttribute("src") !== sample.audio) {
      player.pause();
      player.src = sample.audio || "";
      player.load();
    }
    byId("audioLink").href = sample.audio || "#";
    byId("predictionsLink").href = sample.predictionsUrl || "#";
    byId("predictionsLink").hidden = !sample.predictionsUrl;
    const facts = byId("quickFacts");
    facts.replaceChildren();
    const duration = tags.basic_acoustic?.duration_sec ?? sample.durationSec;
    addFact(facts, tr("duration"), duration == null ? tr("unknown") : displayValue(duration) + " s");
    addFact(facts, tr("speakerCount"), displayValue(tags.speaker?.speaker_count));
    addFact(facts, tr("overlap"), displayValue(tags.speaker?.speaker_overlap));
    addFact(facts, tr("language"), displayLanguage(tags.language_content?.language));
    byId("transcriptText").textContent = sample.transcript || tags.speaker?.asr_transcript || tr("noTranscript");
  }

  function addTagCard(container, titleKey, color, fields) {
    const card = element("article", "tag-card " + color);
    card.append(element("h3", "", tr(titleKey)));
    const list = element("dl", "field-list");
    fields.forEach(([labelKey, path, value]) => {
      const row = element("div", "field-row");
      const term = element("dt", "", tr(labelKey));
      term.append(element("code", "field-path", path));
      row.append(term, element("dd", "", value));
      list.append(row);
    });
    card.append(list);
    container.append(card);
  }

  function renderTags(sample) {
    const tags = tagsOf(sample);
    const grid = byId("tagGrid");
    grid.replaceChildren();
    const speaker = tags.speaker || {};
    const lang = tags.language_content || {};
    const basic = tags.basic_acoustic || {};
    const quality = tags.audio_quality || {};
    const room = tags.room_acoustic || {};
    const sound = tags.sound_field_scene || {};
    addTagCard(grid, "speakersGroup", "teal", [
      ["count", "speaker.speaker_count", displayValue(speaker.speaker_count)],
      ["overlap", "speaker.speaker_overlap", displayValue(speaker.speaker_overlap)],
      ["changes", "speaker.speaker_change_count", displayValue(speaker.speaker_change_count)],
    ]);
    addTagCard(grid, "languageGroup", "blue", [
      ["language", "language_content.language", displayLanguage(lang.language)],
      ["words", "language_content.word_count", displayValue(lang.word_count)],
    ]);
    addTagCard(grid, "acousticGroup", "orange", [
      ["duration", "basic_acoustic.duration_sec", basic.duration_sec == null ? tr("unknown") : displayValue(basic.duration_sec) + " s"],
      ["nonSpeech", "basic_acoustic.nonspeech_ratio", basic.nonspeech_ratio == null ? tr("unknown") : Math.round(basic.nonspeech_ratio * 100) + "%"],
      ["snr", "audio_quality.snr_db", quality.snr_db == null ? tr("unknown") : displayValue(quality.snr_db) + " dB"],
      ["reverberation", "room_acoustic.rt60_sec", room.rt60_sec == null ? tr("unknown") : displayValue(room.rt60_sec) + " s"],
    ]);
    addTagCard(grid, "soundGroup", "violet", [
      ["soundEvents", "sound_field_scene.speech_music_events", displayValue(sound.speech_music_events)],
      ["music", "sound_field_scene.music_present", displayValue(sound.music_present)],
      ["noise", "sound_field_scene.external_noise_type", displayValue(sound.external_noise_type)],
    ]);
    byId("tagCaption").textContent = tags.descriptive?.caption || tr("noCaption");
    byId("rawJson").textContent = JSON.stringify(tags, null, 2);
  }

  function renderComparison(sample) {
    const grid = byId("comparisonGrid");
    grid.replaceChildren();
    const tags = tagsOf(sample);
    const records = [
      ["SURE-Tagger", tags.descriptive?.caption, tags.speaker?.speaker_count],
      ["Captioner", sample.comparisons?.captioner?.caption, sample.comparisons?.captioner?.speakerCount],
      ["Qwen3-Omni", sample.comparisons?.qwen3_omni?.caption, sample.comparisons?.qwen3_omni?.speakerCount],
    ];
    records.forEach(([name, caption, count], index) => {
      const card = element("article", "comparison-card" + (index === 0 ? " is-primary" : ""));
      card.append(element("h3", "", name), element("p", "comparison-caption", caption || tr("notAvailable")));
      if (count !== null && count !== undefined) {
        card.append(element("p", "comparison-meta", tr("reportedSpeakers") + ": " + count));
      }
      grid.append(card);
    });
  }

  function selectSample(id) {
    activeId = id;
    renderSampleList();
    const sample = currentSample();
    renderPlayer(sample);
    renderTags(sample);
    renderComparison(sample);
    writeUrl();
  }

  function queryOptions() {
    const languageCodes = [...new Set(samples.map((sample) => tagsOf(sample).language_content?.language).filter(Boolean))].sort();
    return {
      speakers: [["all", tr("all")], ["none", tr("noSpeakers")], ["one", tr("oneSpeaker")], ["multi", tr("multipleSpeakers")]],
      overlap: [["all", tr("all")], ["true", tr("yes")], ["false", tr("no")]],
      language: [["all", tr("all")], ...languageCodes.map((code) => [code, displayLanguage(code)])],
      noise: [["all", tr("all")], ["present", tr("tagPresent")], ["empty", tr("tagEmpty")]],
    };
  }

  function renderFilterOptions() {
    const options = queryOptions();
    for (const [name, values] of Object.entries(options)) {
      const select = byId(name === "speakers" ? "speakerFilter" : name + "Filter");
      select.replaceChildren();
      values.forEach(([value, label]) => {
        const option = element("option", "", label);
        option.value = value;
        select.append(option);
      });
      select.value = filters[name];
    }
  }

  function matches(sample) {
    const tags = tagsOf(sample);
    const speakerCount = tags.speaker?.speaker_count;
    if (filters.speakers === "none" && speakerCount !== 0) return false;
    if (filters.speakers === "one" && speakerCount !== 1) return false;
    if (filters.speakers === "multi" && !(speakerCount >= 2)) return false;
    if (filters.overlap !== "all" && tags.speaker?.speaker_overlap !== (filters.overlap === "true")) return false;
    if (filters.language !== "all" && tags.language_content?.language !== filters.language) return false;
    const noise = tags.sound_field_scene?.external_noise_type;
    if (filters.noise === "present" && (!Array.isArray(noise) || noise.length === 0)) return false;
    if (filters.noise === "empty" && (!Array.isArray(noise) || noise.length !== 0)) return false;
    return true;
  }

  function renderQueryResults() {
    const results = samples.filter(matches);
    byId("matchCount").textContent = results.length + " / " + samples.length + " " + tr("matches");
    const container = byId("queryResults");
    container.replaceChildren();
    if (!results.length) {
      container.append(element("p", "empty-message", tr("noMatches")));
      return;
    }
    results.forEach((sample) => {
      const button = element("button", "query-result");
      button.type = "button";
      button.append(
        element("span", "result-name", localized(sample.title) || sample.sampleId),
        element("span", "result-meta", sample.dataset + " · " + tr("selectClip") + " ↗"),
      );
      button.addEventListener("click", () => {
        selectSample(sample.sampleId);
        byId("listen").scrollIntoView({ behavior: "smooth", block: "start" });
      });
      container.append(button);
    });
  }

  function setLanguage(next) {
    language = next;
    renderStaticText();
    renderSampleList();
    renderPlayer(currentSample());
    renderTags(currentSample());
    renderComparison(currentSample());
    renderFilterOptions();
    renderQueryResults();
    writeUrl();
  }

  function init(payload) {
    samples = Array.isArray(payload?.samples) ? payload.samples : [];
    bundledSource = payload?.source === "curated Tag-Bench README fixtures";
    if (!bundledSource) {
      document.querySelector('.footer-links a[href$="tagbench/README.md"]').hidden = true;
      document.querySelector('.footer-links a[href$="tagbench/ATTRIBUTION.md"]').hidden = true;
    }
    byId("compare").hidden = !samples.some((sample) =>
      sample.comparisons?.captioner?.caption || sample.comparisons?.qwen3_omni?.caption,
    );
    if (!samples.length) {
      byId("sampleList").append(element("p", "empty-message", "No demo samples were found."));
      return;
    }
    if (!samples.some((sample) => sample.sampleId === activeId)) activeId = samples[0].sampleId;
    document.querySelectorAll("[data-language]").forEach((button) => {
      button.addEventListener("click", () => setLanguage(button.dataset.language));
    });
    for (const [name, id] of Object.entries({
      speakers: "speakerFilter",
      overlap: "overlapFilter",
      language: "languageFilter",
      noise: "noiseFilter",
    })) {
      byId(id).addEventListener("change", (event) => {
        filters[name] = event.target.value;
        renderQueryResults();
      });
    }
    byId("clearFilters").addEventListener("click", () => {
      Object.keys(filters).forEach((key) => { filters[key] = "all"; });
      renderFilterOptions();
      renderQueryResults();
    });
    setLanguage(language);
  }

  const bundled = window.DEMO_DATA;
  if (params.get("data") === "local") {
    const script = document.createElement("script");
    script.src = "data.js";
    script.onload = () => init(window.DEMO_DATA);
    script.onerror = () => {
      window.DEMO_DATA = bundled;
      init(bundled);
      const message = element("p", "load-notice", tr("localMissing"));
      document.querySelector(".hero-copy").append(message);
    };
    document.body.append(script);
  } else {
    init(bundled);
  }
})();
