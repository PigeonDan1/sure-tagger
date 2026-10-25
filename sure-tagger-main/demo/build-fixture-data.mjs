import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const demoDir = path.dirname(fileURLToPath(import.meta.url));
const sampleDir = path.resolve(demoDir, "../../assets/readme/tagbench/samples");
const syntheticDir = path.resolve(demoDir, "../../assets/readme/synthetic-samples");
const fixtures = [
  {
    id: "ES2012d_utterance_00018",
    dataset: "AMI",
    track: "T1",
    title: { en: "Four speakers and overlap", zh: "四人对话与重叠" },
    focus: { en: "The reference has four speakers; compare what each system describes.", zh: "参考标注为四位说话人；可对比不同系统的描述。" },
  },
  {
    id: "EN2001e_utterance_00071",
    dataset: "AMI",
    track: "T1",
    title: { en: "Two speakers and brief overlap", zh: "两人对话与短暂重叠" },
    focus: { en: "Follow speaker changes and short overlapping turns.", zh: "留意说话人切换与短暂的重叠语音。" },
  },
  {
    id: "fleurs_t2_0347",
    dataset: "FLEURS",
    track: "T2",
    title: { en: "Russian speech", zh: "俄语语音" },
    focus: { en: "Inspect language identification and speech content.", zh: "查看语种识别与语音内容。" },
  },
  {
    id: "t3_desed_0112",
    dataset: "DESED",
    track: "T3",
    title: { en: "Speech and an event reference", zh: "语音与事件参考标签" },
    focus: { en: "The reference includes a mechanical event; inspect the predicted sound tags.", zh: "参考信息包含机械声事件；对照预测的声场标签。" },
  },
  {
    id: "t3_fsd50k_0045",
    dataset: "FSD50K",
    track: "T3",
    title: { en: "Short music reference", zh: "短音乐事件参考标签" },
    focus: { en: "The reference marks music while the tagger reports music_present=false.", zh: "参考信息标有音乐，打标器却输出 music_present=false。" },
  },
];

const recordedSamples = fixtures.map((fixture) => {
  const folder = path.join(sampleDir, fixture.id);
  const audioFile = path.join(folder, "audio.wav");
  const predictionFile = path.join(folder, "full_predictions.json");
  if (!fs.existsSync(audioFile) || !fs.existsSync(predictionFile)) {
    throw new Error("Missing demo asset for " + fixture.id);
  }
  const predictions = JSON.parse(fs.readFileSync(predictionFile, "utf8"));
  const tags = predictions.sure_tagger;
  if (!tags || tags.sample_id !== fixture.id) {
    throw new Error("Prediction and audio ID mismatch for " + fixture.id);
  }
  const relative = "../../assets/readme/tagbench/samples/" + fixture.id + "/";
  const summary = (prediction) => ({
    caption: prediction?.descriptive?.caption ?? null,
    speakerCount: prediction?.speaker?.speaker_count ?? null,
  });
  return {
    sampleId: fixture.id,
    dataset: fixture.dataset,
    track: fixture.track,
    title: fixture.title,
    focus: fixture.focus,
    audio: relative + "audio.wav",
    predictionsUrl: relative + "full_predictions.json",
    transcript: tags.speaker?.asr_transcript ?? "",
    durationSec: tags.basic_acoustic?.duration_sec ?? null,
    tags,
    comparisons: {
      captioner: summary(predictions.captioner),
      qwen3_omni: summary(predictions.qwen3_omni),
    },
  };
});

const syntheticFixtures = [
  {
    id: "SCN-0006",
    title: { en: "Station announcement", zh: "车站换乘广播" },
    focus: { en: "One Mandarin speaker with transit ambience at 15 dB target SNR.", zh: "一位中文说话人，伴随公共交通环境声；目标信噪比为 15 dB。" },
  },
  {
    id: "SCN-0198",
    title: { en: "Warranty conversation", zh: "设备保修对话" },
    focus: { en: "Two Mandarin speakers overlap over background music.", zh: "两位中文说话人有重叠，背景带有音乐。" },
  },
  {
    id: "SCN-0360",
    title: { en: "Three voices with music", zh: "三人交谈与背景音乐" },
    focus: { en: "Three Mandarin speakers, overlap, music, and 5 dB target SNR.", zh: "三位中文说话人、重叠语音与音乐；目标信噪比为 5 dB。" },
  },
  {
    id: "SCN-0367",
    title: { en: "Code-switching lesson", zh: "中英切换的课堂片段" },
    focus: { en: "Three speakers switch between Mandarin and English with overlap.", zh: "三位说话人在中文与英文间切换，并有重叠语音。" },
  },
  {
    id: "SCN-0469",
    title: { en: "Arabic in open space", zh: "开放环境中的阿拉伯语" },
    focus: { en: "One Arabic speaker in open-space ambience at 5 dB target SNR.", zh: "一位阿拉伯语说话人，伴随开放环境声；目标信噪比为 5 dB。" },
  },
];

function reviewedValue(review, field, system) {
  for (const section of review.comparison || []) {
    const entry = section.fields?.find((item) => item.field === field);
    if (entry) return entry.systems?.[system]?.value ?? null;
  }
  return null;
}

function reviewedTags(review) {
  const tags = {};
  for (const section of review.comparison || []) {
    for (const entry of section.fields || []) {
      if (entry.gt_only || !entry.systems?.sure_tagger || !/^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$/.test(entry.field)) continue;
      const parts = entry.field.split(".");
      let target = tags;
      for (const part of parts.slice(0, -1)) target = target[part] ||= {};
      target[parts.at(-1)] = entry.systems.sure_tagger.value;
    }
  }
  tags.descriptive = review.descriptive?.sure_tagger?.fields || {};
  return tags;
}

const syntheticSamples = syntheticFixtures.map((fixture) => {
  const folder = path.join(syntheticDir, fixture.id);
  const audioFile = path.join(folder, "audio.wav");
  const groundTruthFile = path.join(folder, "ground_truth.json");
  const reviewFile = path.join(folder, "review.json");
  if (![audioFile, groundTruthFile, reviewFile].every(fs.existsSync)) {
    throw new Error("Missing synthetic demo asset for " + fixture.id);
  }
  const groundTruth = JSON.parse(fs.readFileSync(groundTruthFile, "utf8"));
  const review = JSON.parse(fs.readFileSync(reviewFile, "utf8"));
  if (groundTruth.sample_id !== fixture.id || review.sample_id !== fixture.id || review.audio !== "audio.wav") {
    throw new Error("Synthetic demo asset mismatch for " + fixture.id);
  }
  const tags = reviewedTags(review);
  const relative = "../../assets/readme/synthetic-samples/" + fixture.id + "/";
  return {
    sampleId: fixture.id,
    dataset: "LayeredSpeech-500",
    track: "SYN",
    sourceType: "synthetic",
    title: fixture.title,
    focus: fixture.focus,
    audio: relative + "audio.wav",
    predictionsUrl: relative + "review.json",
    groundTruthUrl: relative + "ground_truth.json",
    transcript: tags.speaker?.asr_transcript || "",
    durationSec: tags.basic_acoustic?.duration_sec ?? null,
    tags,
    comparisons: {
      captioner: {
        caption: review.descriptive?.captioner?.fields?.caption ?? null,
        speakerCount: reviewedValue(review, "speaker.speaker_count", "captioner"),
      },
      qwen3_omni: {
        caption: review.descriptive?.qwen3_omni?.fields?.caption ?? null,
        speakerCount: reviewedValue(review, "speaker.speaker_count", "qwen3_omni"),
      },
    },
  };
});

const samples = [...recordedSamples, ...syntheticSamples];
const payload = { source: "curated Tag-Bench demo fixtures", samples };
const output = path.join(demoDir, "fixture-data.js");
fs.writeFileSync(
  output,
  "// Generated by build-fixture-data.mjs from recorded and synthetic Tag-Bench examples.\n" +
    "window.DEMO_DATA = " + JSON.stringify(payload, null, 2) + ";\n",
  "utf8",
);
console.log("Wrote " + output + " with " + samples.length + " samples");
