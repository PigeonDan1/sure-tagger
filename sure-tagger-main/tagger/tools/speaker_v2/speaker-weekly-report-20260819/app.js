(() => {
  "use strict";

  const report = window.REPORT_DATA;
  const publicFields = report.meta.publicFields;
  const modules = ["pipeline", "sure", "qwen", "demo"];
  const state = {
    module: modules.includes(location.hash.slice(1)) ? location.hash.slice(1) : "pipeline",
    pipelineClaim: "C",
    sureView: "cmo",
    qwenCase: 0,
    demoSample: 0,
    demoView: "candidate",
    timelineSource: "count_multi",
  };

  const palette = ["#124b38", "#ff6b35", "#6c8cff", "#a079ff", "#f5b942", "#29a9a0"];
  const sourceNames = {
    count_multi: "Sortformer · C / M decision",
    change: "MOSS · X decision",
    overlap: "Pyannote · O decision",
    native: "Native reference · post-inference",
  };

  const escapeHtml = (value) =>
    String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

  const displayValue = (value) => {
    if (value === null || value === undefined) return "null";
    if (typeof value === "boolean") return value ? "true" : "false";
    if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
    return String(value);
  };

  const percent = (value, digits = 1) => `${(Number(value) * 100).toFixed(digits)}%`;
  const number = (value, digits = 3) => Number(value).toFixed(digits);
  const json = (value) => JSON.stringify(value, null, 2);
  const boolShort = (value) => (value === null ? "—" : value ? "T" : "F");

  function sourceFooter(items) {
    return `<footer class="footer-evidence">${items
      .map(([label, value]) => `<span><b>${escapeHtml(label)}</b> ${escapeHtml(value)}</span>`)
      .join("")}</footer>`;
  }

  function metricStrip(items) {
    return `<div class="metric-strip">${items
      .map(
        (item) => `<div class="metric">
          <span class="metric-value ${item.accent ? "metric-accent" : ""}">${escapeHtml(item.value)}${item.suffix ? `<small>${escapeHtml(item.suffix)}</small>` : ""}</span>
          <span class="metric-label">${escapeHtml(item.label)}</span>
        </div>`,
      )
      .join("")}</div>`;
  }

  function hero({ index, kicker, title, summary, aside, asideNote, source, titleId }) {
    return `<header class="module-hero" data-index="${escapeHtml(index)}">
      <div>
        <p class="module-kicker">${escapeHtml(kicker)}</p>
        <h2 id="${escapeHtml(titleId)}">${title}</h2>
        <p class="hero-summary">${escapeHtml(summary)}</p>
      </div>
      <div class="hero-aside">
        <strong>${aside}</strong>
        <p>${escapeHtml(asideNote)}</p>
        <div class="source-line">${escapeHtml(source)}</div>
      </div>
    </header>`;
  }

  function sectionHeading(label, title, description = "") {
    return `<div class="section-heading"><div><p class="section-label">${escapeHtml(label)}</p><h3>${escapeHtml(title)}</h3></div>${description ? `<p>${escapeHtml(description)}</p>` : ""}</div>`;
  }

  function renderPipeline() {
    const data = report.pipeline;
    const fields = data.publicFields
      .map(
        (field) => `<article class="field-card">
          <div class="field-card-top"><span class="claim-chip">${escapeHtml(field.claim)}</span><span class="type-chip">${escapeHtml(field.type)}</span></div>
          <code>${escapeHtml(field.key)}</code><p>${escapeHtml(field.meaning)}</p>
        </article>`,
      )
      .join("");
    const stages = data.stages
      .map(
        (stage) => `<article class="stage-card"><span class="stage-index">${escapeHtml(stage.index)}</span><h4>${escapeHtml(stage.name)}</h4><p>${escapeHtml(stage.detail)}</p></article>`,
      )
      .join("");
    const contractColumns = [
      ["推理输入", data.inputContract.consumed, ""],
      ["明确排除", data.inputContract.excluded, "excluded"],
      ["公开 schema", data.inputContract.publicOutput, "output"],
      ["内部证据", data.inputContract.internalOnly, "internal"],
    ]
      .map(
        ([title, values, className]) => `<article class="contract-column ${className}"><h4>${escapeHtml(title)}</h4><ul>${values.map((value) => `<li><code>${escapeHtml(value)}</code></li>`).join("")}</ul></article>`,
      )
      .join("");
    const capabilities = data.internalCapabilities
      .map(
        (item) => `<article class="field-card"><div class="field-card-top"><span class="claim-chip">${escapeHtml(item.key)}</span><span class="micro-chip">${escapeHtml(item.state)}</span></div><code>${escapeHtml(item.name)}</code><p>${escapeHtml(item.model)}</p></article>`,
      )
      .join("");

    document.querySelector("#pipelineContent").innerHTML = `
      ${hero({
        index: "01",
        kicker: "LABEL CONTRACT / QUALITY-SHADOW",
        title: "六字段，四条 claim，<br><span>零生产发布</span>",
        summary: "Speaker v2 已完成 C/M/O/X 的 per-claim 专家路由：说话人数、多人、换人、重叠分别选择最合适的 timeline 模型；但当前仍是严格的 shadow 评测链路。",
        aside: "Candidate 可评估，<br>Public adapter 全部为 null。",
        asideNote: "六字段已有公开 schema 席位，但当前实现无法产出 production-certified 值。",
        source: data.source,
        titleId: "pipelineTitle",
      })}
      <section class="section-block">
        ${metricStrip([
          { value: "6", label: "public speaker schema fields", accent: true },
          { value: "4", label: "formal claims · C / M / O / X" },
          { value: "3", label: "timeline specialists" },
          { value: "0", label: "production-published values" },
        ])}
      </section>
      <section class="section-block">
        ${sectionHeading("PUBLIC CONTRACT", "六个公开字段", "公开对象只含结果值。模型、路由、evidence ID、置信与 artifact 路径全部留在内部。")}
        <div class="field-grid">${fields}</div>
      </section>
      <section class="section-block">
        ${sectionHeading("PIPELINE", "从音频到状态边界", "每个 claim 只接收一条被选中的 decision timeline；数值字段从对应 O / X timeline 确定性派生。")}
        <div class="pipeline-flow">${stages}</div>
        <div class="metric-strip" style="margin-top:12px">${data.timelineRules
          .map(
            (rule) => `<div class="metric"><span class="metric-value" style="font-size:22px">${escapeHtml(rule.value)}</span><span class="metric-label">${escapeHtml(rule.label)}</span></div>`,
          )
          .join("")}</div>
      </section>
      <section class="section-block">
        ${sectionHeading("ROUTING", "按标签选择专家", "点击 claim 查看 primary、fallback、guard 与审计来源。Guard 当前只记录 observation，不改变 candidate。")}
        <div class="route-explorer">
          <div class="route-list" id="routeList">${data.claimRoutes.map(routeButton).join("")}</div>
          <div class="route-detail" id="routeDetail"></div>
        </div>
      </section>
      <section class="section-block">
        ${sectionHeading("BOUNDARIES", "输入、输出与内部证据", "native metadata 和 supplied transcript 可以存在于原始 manifest，但不会进入 inference view 或 resolver。")}
        <div class="contract-grid">${contractColumns}</div>
      </section>
      <section class="section-block">
        ${sectionHeading("INTERNAL ONLY", "I / V / A / D 不写公开六字段", "Identity、coverage 和 lexical timeline 只用于诊断；full diarization 尚未接入 v2 正式 claim。")}
        <div class="field-grid">${capabilities}</div>
      </section>
      <section class="section-block">
        ${sectionHeading("READ BEFORE USE", "状态与口径限制")}
        <div class="notice-list">${data.warnings.map((warning) => `<div class="notice">${escapeHtml(warning)}</div>`).join("")}</div>
      </section>
      ${sourceFooter([
        ["inventory", data.source],
        ["sha256", data.sourceSha256],
        ["profile", data.profile],
      ])}`;
    renderRouteDetail();
  }

  function routeButton(route) {
    const active = route.claim === state.pipelineClaim;
    return `<button type="button" class="route-button ${active ? "is-active" : ""}" data-claim="${escapeHtml(route.claim)}" aria-pressed="${active}">
      <span class="claim-chip">${escapeHtml(route.claim)}</span><span><b>${escapeHtml(route.fields.join(" + "))}</b><small>${escapeHtml(route.headline)}</small></span>
    </button>`;
  }

  function renderRouteDetail() {
    const route = report.pipeline.claimRoutes.find((item) => item.claim === state.pipelineClaim);
    const nodes = [
      ["primary", "PRIMARY", route.primary],
      ["fallback", "FALLBACK", route.fallback],
      ["guard", "GUARD / WITNESS", route.guard],
      ["excluded", "EXCLUDED / AUDIT", route.excluded],
    ]
      .map(([className, label, value]) => `<div class="route-node ${className}"><span>${escapeHtml(label)}</span><b>${escapeHtml(value)}</b></div>`)
      .join("");
    document.querySelector("#routeList").innerHTML = report.pipeline.claimRoutes.map(routeButton).join("");
    document.querySelector("#routeDetail").innerHTML = `
      <div class="route-detail-head"><div><h4>${escapeHtml(route.claim)} · ${escapeHtml(route.fields.join(" / "))}</h4><p>first usable · decision timeline 单选</p></div><span class="route-score">${escapeHtml(route.headline)}</span></div>
      <div class="route-diagram">${nodes}</div>
      <p class="route-rule">Primary 可用时直接产生 candidate；只有 primary 不可用时才走有序 fallback。Guard 不投票，excluded 仅保留审计。派生数值只读取实际 decision evidence。</p>`;
  }

  const sureViews = [
    ["cmo", "结构 C/M/O"],
    ["change", "换人 X"],
    ["der", "Diarization"],
    ["vad", "VAD"],
    ["identity", "Identity"],
    ["asr", "ASR"],
  ];

  function renderSure() {
    const data = report.sure;
    document.querySelector("#sureContent").innerHTML = `
      ${hero({
        index: "02",
        kicker: "SURE / FROZEN BENCHMARK",
        title: "模型没有总冠军，<br><span>只有能力分工</span>",
        summary: "冻结 AMI 1k benchmark 覆盖 ASR、说话人结构、换人点、完整 diarization、VAD 与身份验证。路由选择依据能力指标，而不是把不同任务压成一个总分。",
        aside: "Sortformer 做结构，Pyannote 做重叠，MOSS 做换人。",
        asideNote: "Brouhaha 偏综合 VAD，ECAPA 在身份验证八项指标全部领先。",
        source: data.source,
        titleId: "sureTitle",
      })}
      <section class="section-block">
        ${metricStrip([
          { value: "1,000", label: "frozen utterance cuts", accent: true },
          { value: "7.7514", suffix: "hours", label: "AMI evaluation audio" },
          { value: "167", label: "source meetings" },
          { value: "23,815", label: "cross-meeting identity trials" },
        ])}
      </section>
      <section class="section-block">
        ${sectionHeading("CAPABILITY MATRIX", "选择任务，比较模型", "所有空预测都保留在分母；每个表只在自己的协议内比较，不跨任务合成排名。")}
        <div class="toolbar"><div class="segmented" id="sureTabs">${sureViews.map(([key, label]) => `<button type="button" class="segment-button ${state.sureView === key ? "is-active" : ""}" data-sure-view="${key}">${escapeHtml(label)}</button>`).join("")}</div><span class="toolbar-note">cached predictions · local SURE staging</span></div>
        <div class="chart-panel"><div id="sureChart"></div></div>
        <div class="table-wrap" id="sureTable" style="margin-top:14px"></div>
      </section>
      <section class="section-block">
        ${sectionHeading("DECISIONS", "这张矩阵如何进入 v2 选型")}
        <div class="insight-grid">
          <article class="insight-card accent"><h4>模型分工</h4><ul class="insight-list">${data.takeaways.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></article>
          <article class="insight-card"><h4>不可忽略的限制</h4><ul class="insight-list">${data.limitations.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></article>
        </div>
      </section>
      ${sourceFooter([
        ["matrix", data.source],
        ["manifest", data.dataset.manifestSha256],
        ["matrix sha256", data.sourceSha256],
      ])}`;
    renderSureView();
  }

  function sureChartRows() {
    const tables = report.sure.tables;
    if (state.sureView === "cmo") {
      return report.sure.modelChart.map((row) => ({
        name: row.shortName,
        values: [
          ["count accuracy", row.countAccuracy, "count"],
          ["multi accuracy", row.multiAccuracy, "multi"],
          ["overlap accuracy", row.overlapAccuracy, "overlap"],
        ],
      }));
    }
    if (state.sureView === "change") {
      return tables.change.map((row) => ({
        name: shortModel(row["模型"]),
        values: [
          ["change bool", Number(row["change_bool_accuracy↑"]), "change"],
          ["F1 @ 0.25s", Number(row["change_point_f1_025↑"]), "count"],
          ["F1 @ 0.50s", Number(row["change_point_f1_05↑"]), "multi"],
        ],
      }));
    }
    if (state.sureView === "der") {
      return tables.der.map((row) => ({ name: shortModel(row["模型"]), values: [["DER ↓", Number(row["DER↓"]), "der", true]] }));
    }
    if (state.sureView === "vad") {
      return tables.vad.map((row) => ({
        name: shortModel(row["模型"]),
        values: [
          ["F1", Number(row["f1↑"]), "multi"],
          ["1 − Pmiss", 1 - Number(row["p_miss↓"]), "count"],
          ["1 − DCF", 1 - Number(row["dcf_nist↓"]), "overlap"],
        ],
      }));
    }
    if (state.sureView === "identity") {
      return tables.identity.map((row) => ({
        name: shortModel(row["模型"]),
        values: [
          ["ROC-AUC", Number(row["ROC-AUC↑"]), "count"],
          ["test accuracy", Number(row["test accuracy↑"]), "multi"],
          ["TPR @ FAR 1%", Number(row["TPR@FAR=1%↑"]), "overlap"],
        ],
      }));
    }
    return tables.asr.map((row) => ({
      name: shortModel(row["模型"]),
      values: [
        ["1 − CER zh", 1 - Number(row["CER↓（zh，7,176 条）"]), "count"],
        ["1 − WER en", 1 - Number(row["WER↓（en，2,619 条）"]), "multi"],
        ["coverage en", Number(row["word_coverage↑（en）"]), "overlap"],
      ],
    }));
  }

  function shortModel(value) {
    const text = String(value);
    if (text.startsWith("MOSS")) return "MOSS";
    if (text.startsWith("NVIDIA")) return "Sortformer";
    if (text.toLowerCase().startsWith("pyannote")) return "Pyannote";
    if (text.startsWith("SpeechBrain")) return "ECAPA";
    if (text.startsWith("CAM++")) return "CAM++";
    if (text.startsWith("openai")) return "Whisper";
    if (text.startsWith("FireRed")) return "FireRed";
    if (text.startsWith("Brouhaha")) return "Brouhaha";
    return text;
  }

  function renderSureView() {
    document.querySelectorAll("[data-sure-view]").forEach((button) => button.classList.toggle("is-active", button.dataset.sureView === state.sureView));
    const rows = sureChartRows();
    const legendItems = new Map();
    rows.forEach((row) => row.values.forEach(([label, , color]) => legendItems.set(label, color)));
    const chart = `<div class="chart-legend">${[...legendItems].map(([label, color]) => `<span><i style="background:${paletteFor(color)}"></i>${escapeHtml(label)}</span>`).join("")}</div>
      <div class="grouped-chart">${rows
        .map(
          (row) => `<div class="chart-row"><div class="chart-row-label"><b>${escapeHtml(row.name)}</b><small>${row.values.some((value) => value[3]) ? "lower is better" : "normalized score"}</small></div><div class="bar-stack">${row.values
            .map(([label, value, color, lower]) => {
              const width = Math.min(100, Math.max(0, value * 100));
              return `<div class="bar-line"><span class="bar-label">${escapeHtml(label)}</span><div class="bar-track"><i class="bar-fill ${color}" style="width:${width}%"></i></div><span class="bar-number">${lower ? number(value, 4) : percent(value)}</span></div>`;
            })
            .join("")}</div></div>`,
        )
        .join("")}</div>`;
    document.querySelector("#sureChart").innerHTML = chart;
    const tableKey = state.sureView;
    const values = report.sure.tables[tableKey];
    document.querySelector("#sureTable").innerHTML = renderTable(values);
  }

  function paletteFor(name) {
    return { count: "#124b38", multi: "#24b76d", overlap: "#ff6b35", change: "#6c8cff", der: "#a079ff" }[name] || "#124b38";
  }

  function renderTable(rows) {
    if (!rows?.length) return "";
    const headers = Object.keys(rows[0]);
    return `<table class="data-table"><thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("")}</tr></thead><tbody>${rows
      .map((row) => `<tr>${headers.map((header) => `<td>${escapeHtml(row[header])}</td>`).join("")}</tr>`)
      .join("")}</tbody></table>`;
  }

  function renderQwen() {
    const data = report.qwen;
    const multi = data.metrics.boolean.multi_speaker;
    const overlap = data.metrics.boolean.speaker_overlap;
    document.querySelector("#qwenContent").innerHTML = `
      ${hero({
        index: "03",
        kicker: "PSEUDO-GT / QWEN-DERIVED REFERENCE",
        title: "多人检测可用，<br><span>重叠误报突出</span>",
        summary: "在 100 条 Qwen-captioner / Qwen-derived reference 上，quality-shadow 全部完成推理。多人检测达到 92.78% accuracy；但 overlap 只有 21.62% precision，是下一轮最明确的修正目标。",
        aside: "92.78% accuracy / 91.36% F1",
        asideNote: "97 条有效 multi_speaker reference；2 个误报、5 个漏报。",
        source: data.reportSource,
        titleId: "qwenTitle",
      })}
      <section class="section-block">
        ${metricStrip([
          { value: "100 / 100", label: "successful quality-shadow runs", accent: true },
          { value: percent(multi.accuracy_on_covered, 2), label: "multi_speaker accuracy" },
          { value: percent(multi.f1, 2), label: "multi_speaker F1" },
          { value: "29", label: "speaker_overlap false positives" },
        ])}
      </section>
      <section class="section-block">
        ${sectionHeading("REFERENCE CONTRACT", "这不是人工 gold", "reference 的每个字段都允许 null；只有 caption 明确陈述的事实才进入打分，因此六个字段的有效分母不同。")}
        <div class="reference-band">
          <div class="reference-label">QWEN-DERIVED<br>PSEUDO-GT</div>
          <div class="reference-step"><b>01 · Qwen-captioner 音频描述</b><p>原始 captioner 具体模型与版本未在现有 artifact 中固化。</p></div>
          <div class="reference-step"><b>02 · Qwen3-8B 严格抽取</b><p>caption → company speaker schema；100 个 JSON 与上游 tag_extracted 逐字节一致。</p></div>
        </div>
        <div class="notice" style="margin-top:12px">${escapeHtml(data.groundTruth.caveat)}</div>
      </section>
      <section class="section-block">
        ${sectionHeading("ALL SIX FIELDS", "字段级测评", "覆盖率按该字段的非 null pseudo-GT 分母计算；overlap_ratio 有 2 条 candidate 缺失。")}
        <div class="score-grid">${data.fieldCards.map(scoreCard).join("")}</div>
      </section>
      <section class="section-block">
        ${sectionHeading("PRIMARY SIGNAL", "Multi-speaker confusion matrix", "正例 42、负例 55，另有 3 条 null 不进入该字段统计。")}
        <div class="confusion-layout">
          <article class="confusion-card"><h4>Predicted × Reference</h4><div class="confusion-matrix"><span></span><span>PRED +</span><span>PRED −</span><span>REF +</span><span class="cell good">${multi.true_positive}<small>TP</small></span><span class="cell bad">${multi.false_negative}<small>FN</small></span><span>REF −</span><span class="cell bad">${multi.false_positive}<small>FP</small></span><span class="cell good">${multi.true_negative}<small>TN</small></span></div></article>
          <article class="risk-callout"><span class="risk-title">OVERLAP RISK / INVESTIGATE</span><strong>${overlap.false_positive} 个 overlap 误报，precision 仅 ${percent(overlap.precision, 2)}</strong><p>${escapeHtml(data.keyRisk)}</p></article>
        </div>
      </section>
      <section class="section-block">
        ${sectionHeading("ERROR EXPLORER", "听四个代表性误差", "所有音频和 GT / prediction 对照都来自这次 100 条完整批次，不使用手工构造示例。")}
        <div class="case-explorer"><div class="case-list" id="qwenCaseList"></div><div class="case-detail" id="qwenCaseDetail"></div></div>
      </section>
      ${sourceFooter([
        ["metrics", data.metricsSource],
        ["metrics sha256", data.metricsSha256],
        ["predictions sha256", data.predictionsSha256],
      ])}`;
    renderQwenCase();
  }

  function scoreCard(item) {
    const shown = item.lowerIsBetter ? number(item.value, 4) : percent(item.value, 2);
    const meter = item.lowerIsBetter ? Math.max(5, (1 - item.value) * 100) : item.value * 100;
    const risk = item.field === "speaker_overlap";
    return `<article class="score-card ${risk ? "risk" : ""}"><div class="score-card-head"><code>${escapeHtml(item.field)}</code><span>n=${item.eligible}</span></div><div class="score-number">${shown}<small> ${escapeHtml(item.primary)}</small></div><div class="score-meter"><i style="width:${meter}%"></i></div><div class="score-card-foot"><span>coverage ${percent(item.coverage, 2)}</span><span>${escapeHtml(item.secondary)}</span></div></article>`;
  }

  function renderQwenCase() {
    const cases = report.qwen.cases;
    const active = cases[state.qwenCase];
    document.querySelector("#qwenCaseList").innerHTML = cases
      .map(
        (item, index) => `<button type="button" class="case-button ${index === state.qwenCase ? "is-active" : ""}" data-qwen-case="${index}"><b>${escapeHtml(item.title)}</b><code>${escapeHtml(item.sampleId)}</code></button>`,
      )
      .join("");
    const gt = active.groundTruthRecord.annotation[0].speaker;
    const pred = Object.fromEntries(publicFields.map((field) => [field, active.comparison[`pred_${field}`]]));
    const compareBoxes = [
      ["gt", "Qwen-derived pseudo-GT", gt],
      ["pred", "speaker_v2 candidate", pred],
    ]
      .map(
        ([className, title, values]) => `<article class="compare-box ${className}"><h5>${escapeHtml(title)}</h5><dl class="key-value-list">${publicFields
          .map((field) => {
            const matches = displayValue(gt[field]) === displayValue(pred[field]);
            return `<dt>${escapeHtml(field)}</dt><dd class="${matches ? "value-match" : "value-mismatch"}">${escapeHtml(displayValue(values[field]))}</dd>`;
          })
          .join("")}</dl></article>`,
      )
      .join("");
    document.querySelector("#qwenCaseDetail").innerHTML = `
      <div class="case-detail-header"><div><p class="section-label">${escapeHtml(active.kind)}</p><h4>${escapeHtml(active.title)}</h4></div><audio controls preload="metadata" src="${escapeHtml(active.audio)}"></audio></div>
      <div class="compare-grid">${compareBoxes}</div>
      <div class="caption-evidence"><strong>Reference provenance</strong>${escapeHtml(report.qwen.groundTruth.provenance)}<div class="source-line" style="margin-top:9px">${escapeHtml(active.source)}</div></div>`;
  }

  function renderDemo() {
    const data = report.demo;
    document.querySelector("#demoContent").innerHTML = `
      ${hero({
        index: "04",
        kicker: "REAL RUN / AMI EN2001a",
        title: "布尔标签全对，<br><span>高人数仍低估</span>",
        summary: "5 条代表性 AMI utterance 覆盖 1–5 个 native 说话人。实际 claim 路由与 quality-shadow 完全一致；布尔标签 5/5，但 4 人和 5 人样本都被输出为 3 人。",
        aside: "5 success / 0 failure · RTF 1.65",
        asideNote: "这是定向功能 demo，不是总体精度估计；候选输出禁止生产发布。",
        source: data.source,
        titleId: "demoTitle",
      })}
      <section class="section-block">
        ${metricStrip([
          { value: "3 / 5", label: "speaker_count exact", accent: true },
          { value: "0.6", label: "speaker_count MAE" },
          { value: "5 / 5", label: "M / O / X boolean accuracy" },
          { value: "30 / 30", label: "enabled evidence usable" },
        ])}
      </section>
      <section class="section-block">
        ${sectionHeading("RUN EXPLORER", "逐条检查输入、路由和输出", "选择样本后可播放原始 WAV、切换三条 decision timeline、比较 native reference 与 candidate，并查看完整 fusion metadata。")}
        <div class="demo-shell"><aside class="demo-rail"><div class="demo-rail-head"><b>AMI EN2001a</b><span>5 SELECTED UTTERANCES</span></div><div id="demoSampleList"></div></aside><div class="demo-detail" id="demoDetail"></div></div>
      </section>
      <section class="section-block">
        ${sectionHeading("BOUNDARIES", "这次 demo 能说明什么")}
        <div class="notice-list">${data.warnings.map((warning) => `<div class="notice">${escapeHtml(warning)}</div>`).join("")}</div>
      </section>
      ${sourceFooter([
        ["summary", data.source],
        ["results sha256", data.resultsSha256],
        ["policy", data.run.policyVersion],
      ])}`;
    renderDemoSample();
  }

  function renderDemoSample() {
    const samples = report.demo.samples;
    const sample = samples[state.demoSample];
    document.querySelector("#demoSampleList").innerHTML = samples
      .map(
        (item, index) => `<button type="button" class="demo-sample-button ${index === state.demoSample ? "is-active" : ""}" data-demo-sample="${index}"><span class="speaker-badge">${item.nativeReference.speaker_count}p</span><span><b>${escapeHtml(item.sampleId.replace("EN2001a_utterance_", "UTT "))}</b><small>${escapeHtml(item.title)} · ${number(item.media.durationSeconds, 1)}s</small></span><span class="correct-mark ${item.countCorrect ? "" : "bad"}">${item.countCorrect ? "✓" : "−"}</span></button>`,
      )
      .join("");
    const scoreCells = publicFields
      .map((field) => {
        const gt = sample.nativeReference[field];
        const pred = sample.candidateOutput[field];
        const comparable = gt !== null;
        const match = !comparable || displayValue(gt) === displayValue(pred);
        return `<div class="demo-score-cell"><span>${escapeHtml(field)}</span><b class="${match ? "" : "mismatch"}">${escapeHtml(displayValue(pred))}</b><span>GT ${escapeHtml(displayValue(gt))}</span></div>`;
      })
      .join("");
    const timelineOptions = ["native", "count_multi", "change", "overlap"].filter((key) => key === "native" || sample.modelTimelines[key]);
    if (!timelineOptions.includes(state.timelineSource)) state.timelineSource = "count_multi";
    document.querySelector("#demoDetail").innerHTML = `
      <div class="demo-heading"><div><p class="section-label">${sample.countCorrect ? "COUNT MATCH" : "COUNT UNDER-ESTIMATE"}</p><h4>${escapeHtml(sample.sampleId)}</h4><p>${sample.media.sampleRateHz} Hz · ${sample.media.channels} channel · ${number(sample.media.durationSeconds, 3)} sec</p></div><audio id="demoAudio" controls preload="metadata" src="${escapeHtml(sample.audio)}"></audio></div>
      <div class="demo-scoreline">${scoreCells}</div>
      <div class="timeline-toolbar"><div class="segmented">${timelineOptions.map((key) => `<button type="button" class="segment-button ${state.timelineSource === key ? "is-active" : ""}" data-timeline-source="${key}">${escapeHtml(sourceNames[key])}</button>`).join("")}</div><span class="toolbar-note">点击片段查看时间与说话人</span></div>
      <div class="timeline-wrap" id="demoTimeline"></div>
      <div class="view-tabs">${[
        ["candidate", "Evaluation candidate"],
        ["public", "Public output"],
        ["routes", "Route evidence"],
        ["metadata", "Complete metadata"],
        ["input", "Input boundary"],
      ]
        .map(([key, label]) => `<button type="button" class="view-tab ${state.demoView === key ? "is-active" : ""}" data-demo-view="${key}">${escapeHtml(label)}</button>`)
        .join("")}</div>
      <div class="view-panel" id="demoViewPanel"></div>`;
    renderTimeline();
    renderDemoView();
  }

  function renderTimeline() {
    const sample = report.demo.samples[state.demoSample];
    const duration = sample.media.durationSeconds;
    const source = state.timelineSource;
    const segments = source === "native" ? sample.nativeTimeline : sample.modelTimelines[source].segments;
    const speakerIds = [...new Set(segments.map((item) => item.speaker_id))];
    const laneHeight = Math.max(16, Math.min(20, 82 / Math.max(1, speakerIds.length)));
    const trackHeight = Math.max(62, speakerIds.length * laneHeight + 12);
    const buttons = segments
      .map((segment, index) => {
        const lane = speakerIds.indexOf(segment.speaker_id);
        const left = Math.max(0, (segment.start_sec / duration) * 100);
        const right = Math.min(100, (segment.end_sec / duration) * 100);
        const width = Math.max(.35, right - left);
        const title = `${segment.speaker_id} · ${number(segment.start_sec, 2)}–${number(segment.end_sec, 2)}s${segment.text ? ` · ${segment.text}` : ""}`;
        return `<button type="button" class="timeline-segment speaker-${lane % palette.length}" data-timeline-index="${index}" style="left:${left}%;width:${width}%;top:${6 + lane * laneHeight}px;height:${Math.max(11, laneHeight - 4)}px;background:${palette[lane % palette.length]}" title="${escapeHtml(title)}" aria-label="${escapeHtml(title)}"></button>`;
      })
      .join("");
    document.querySelector("#demoTimeline").innerHTML = `<div class="timeline-axis"><span>0.0s</span><span>${number(duration / 2, 1)}s</span><span>${number(duration, 1)}s</span></div><div class="timeline-track" style="height:${trackHeight}px">${buttons}</div><div class="timeline-readout" id="timelineReadout">${escapeHtml(sourceNames[source])} · ${segments.length} segments · ${speakerIds.length} speaker IDs</div>`;
  }

  function renderDemoView() {
    const sample = report.demo.samples[state.demoSample];
    const panel = document.querySelector("#demoViewPanel");
    document.querySelectorAll("[data-demo-view]").forEach((button) => button.classList.toggle("is-active", button.dataset.demoView === state.demoView));
    if (state.demoView === "candidate") {
      panel.innerHTML = `<div class="json-panel"><p class="panel-note">仅用于离线评测的非认证 candidate；<code>production_eligible=false</code>。</p><pre>${escapeHtml(json(sample.candidateOutput))}</pre></div>`;
      return;
    }
    if (state.demoView === "public") {
      panel.innerHTML = `<div class="empty-public"><span class="empty-icon">0</span><div><strong>Public adapter 没有发布任何候选值</strong><p>六个 schema 字段完整保留，但均为 <code>null</code>。这与 evaluation candidate 是两个独立对象。</p></div></div><div class="json-panel" style="margin-top:12px"><pre>${escapeHtml(json(orderedPublicOutput(sample.publicOutput)))}</pre></div>`;
      return;
    }
    if (state.demoView === "routes") {
      panel.innerHTML = `<div class="route-records">${Object.entries(sample.routeEvidence)
        .map(
          ([claim, route]) => `<article class="route-record"><h5>${escapeHtml(claim)}</h5><dl><dt>selection</dt><dd>${escapeHtml(route.selection)}</dd><dt>decision</dt><dd>${escapeHtml((route.decision_sources || []).join(", "))}</dd><dt>fallback</dt><dd>${escapeHtml(route.fallback_reason || "none")}</dd><dt>guards affect</dt><dd>${escapeHtml(displayValue(route.guards_affect_candidate))}</dd></dl></article>`,
        )
        .join("")}</div>`;
      return;
    }
    if (state.demoView === "metadata") {
      const value = json(sample.metadata);
      const bytes = new Blob([value]).size;
      panel.innerHTML = `<div class="metadata-toolbar"><span>${value.split("\n").length} lines · ${bytes.toLocaleString()} bytes · exact fusion artifact</span><button type="button" class="copy-button" id="copyMetadata">复制完整 JSON</button></div><textarea class="metadata-textarea" id="fullMetadata" readonly spellcheck="false"></textarea><div class="source-line" style="margin-top:9px">${escapeHtml(sample.metadataSource)}</div>`;
      document.querySelector("#fullMetadata").value = value;
      return;
    }
    panel.innerHTML = `<div class="input-boundary"><article class="boundary-box"><h5>实际进入 inference view</h5><pre>${escapeHtml(json(sample.rawInput))}</pre></article><article class="boundary-box excluded"><h5>仅在推理完成后用于评测</h5><pre>${escapeHtml(json(sample.excludedInput))}</pre></article></div>`;
  }

  function orderedPublicOutput(value) {
    return Object.fromEntries(publicFields.map((field) => [field, value[field]]));
  }

  function showModule(module, updateHash = true) {
    if (!modules.includes(module)) return;
    state.module = module;
    document.querySelectorAll("[data-panel]").forEach((panel) => {
      const active = panel.dataset.panel === module;
      panel.hidden = !active;
      panel.classList.toggle("is-active", active);
    });
    document.querySelectorAll("[data-module]").forEach((button) => {
      const active = button.dataset.module === module;
      button.classList.toggle("is-active", active);
      if (active) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
    });
    if (updateHash) history.replaceState(null, "", `#${module}`);
    document.documentElement.scrollTop = 0;
    document.body.scrollTop = 0;
  }

  function toast(message) {
    const element = document.querySelector("#toast");
    element.textContent = message;
    element.classList.add("is-visible");
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => element.classList.remove("is-visible"), 1800);
  }

  async function copyText(value) {
    if (navigator.clipboard?.writeText) {
      try {
        await navigator.clipboard.writeText(value);
        toast("完整 metadata 已复制");
        return;
      } catch (_) {
      }
    }
    const textarea = document.querySelector("#fullMetadata");
    textarea.focus();
    textarea.select();
    const success = document.execCommand("copy");
    toast(success ? "完整 metadata 已复制" : "已选中，请按 Ctrl/Cmd+C 复制");
  }

  function handleClick(event) {
    const moduleButton = event.target.closest("[data-module]");
    if (moduleButton) return showModule(moduleButton.dataset.module);

    const routeButtonElement = event.target.closest("[data-claim]");
    if (routeButtonElement) {
      state.pipelineClaim = routeButtonElement.dataset.claim;
      renderRouteDetail();
      return;
    }

    const sureButton = event.target.closest("[data-sure-view]");
    if (sureButton) {
      state.sureView = sureButton.dataset.sureView;
      renderSureView();
      return;
    }

    const qwenButton = event.target.closest("[data-qwen-case]");
    if (qwenButton) {
      state.qwenCase = Number(qwenButton.dataset.qwenCase);
      renderQwenCase();
      return;
    }

    const sampleButton = event.target.closest("[data-demo-sample]");
    if (sampleButton) {
      state.demoSample = Number(sampleButton.dataset.demoSample);
      state.timelineSource = "count_multi";
      renderDemoSample();
      return;
    }

    const timelineButton = event.target.closest("[data-timeline-source]");
    if (timelineButton) {
      state.timelineSource = timelineButton.dataset.timelineSource;
      document.querySelectorAll("[data-timeline-source]").forEach((button) => button.classList.toggle("is-active", button.dataset.timelineSource === state.timelineSource));
      renderTimeline();
      return;
    }

    const timelineSegment = event.target.closest("[data-timeline-index]");
    if (timelineSegment) {
      const sample = report.demo.samples[state.demoSample];
      const segments = state.timelineSource === "native" ? sample.nativeTimeline : sample.modelTimelines[state.timelineSource].segments;
      const segment = segments[Number(timelineSegment.dataset.timelineIndex)];
      document.querySelector("#timelineReadout").textContent = `${segment.speaker_id} · ${number(segment.start_sec, 3)}–${number(segment.end_sec, 3)}s · ${number(segment.end_sec - segment.start_sec, 3)}s${segment.text ? ` · ${segment.text}` : ""}`;
      return;
    }

    const viewButton = event.target.closest("[data-demo-view]");
    if (viewButton) {
      state.demoView = viewButton.dataset.demoView;
      renderDemoView();
      return;
    }

    if (event.target.closest("#copyMetadata")) {
      copyText(document.querySelector("#fullMetadata").value);
    }
  }

  function handleKeydown(event) {
    if (state.module !== "demo" || ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    const direction = event.key === "ArrowRight" ? 1 : -1;
    state.demoSample = (state.demoSample + direction + report.demo.samples.length) % report.demo.samples.length;
    state.timelineSource = "count_multi";
    renderDemoSample();
  }

  if (!report) {
    document.body.innerHTML = "<p>Report data is unavailable.</p>";
    return;
  }

  renderPipeline();
  renderSure();
  renderQwen();
  renderDemo();
  showModule(state.module, false);
  document.addEventListener("click", handleClick);
  document.addEventListener("keydown", handleKeydown);
  window.addEventListener("hashchange", () => showModule(location.hash.slice(1), false));
})();
