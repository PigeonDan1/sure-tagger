# -*- coding: utf-8 -*-
"""Summarize human layer-preference IAA and system-human agreement."""
from __future__ import print_function

import csv
import io
import json
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

ROOT = Path("/hpc_stor03/sjtu_home/weihan.chen/share/compare")
sys.path.insert(0, str(ROOT / "eval"))
import score_benchmark_v1 as S  # noqa: E402

ANS = ROOT / "dataset/benchmark/demo/answer"
SAMPLES = ROOT / "dataset/benchmark/demo/samples"
BENCH = ROOT / "dataset/benchmark"
DOC_DOCS = ROOT / "docs"
DOC_DEMO = ROOT / "dataset/benchmark/demo/docs"
DOC_DEMO.mkdir(parents=True, exist_ok=True)

PRED_PATH = {
    "A": ROOT / "evaluation/benchmark-v1/sure_tagger_full_20260918_main_pipeline_w2_b8/result_with_id_descriptive.jsonl",
    "B": ROOT / "docs/report/result/qwen-captioner/results_t1_t3.jsonl",
    "C": ROOT / "docs/report/result/qwen3-omni-v2-multi-api12/final_results.jsonl",
}
CHOICE_NAME = {"A": "Sure-Tagger", "B": "Captioner", "C": "Omni", "D": "无法判断"}
CATS = ["A", "B", "C", "D"]
LAYERS = ["L1", "L2", "L3", "L5", "D"]
LAYER_NAME = {
    "L1": "L1 词汇内容",
    "L2": "L2 说话人",
    "L3": "L3 时间结构",
    "L5": "L5 声学场景与事件",
    "D": "D 描述性",
}
STRUCT_FIELDS = {
    "L1": S.LAYER_DEFINITIONS[0][2],
    "L2": S.LAYER_DEFINITIONS[1][2],
    "L3": S.LAYER_DEFINITIONS[2][2],
    "L5": (
        "basic_acoustic.nonspeech_ratio",
        "sound_field_scene.speech_music_events",
        "sound_field_scene.music_present",
        "sound_field_scene.external_noise_type",
    ),
}
PAIRS = [
    ("set-1", "第 1 套", "gaoshuo", "set-1-answers-gaoshuo.json", "whf", "set-1-answers-whf.json"),
    ("set-2", "第 2 套", "bicheng", "set-2-answers-bicheng.json", "whf", "set-2-answers-whf.json"),
    ("set-3", "第 3 套", "djh", "set-3-answers-djh.csv", "wenwen", "set-3-answers-wenwen.csv"),
    ("set-4", "第 4 套", "chaolei", "set-4-answers-chaolei.csv", "djh", "set-4-answers-djh.csv"),
]


def load_rows(path):
    raw = path.read_bytes()
    if path.suffix == ".json":
        rows = json.loads(raw.decode("utf-8"))
    else:
        text = None
        for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            raise RuntimeError("decode fail %s" % path)
        rows = list(csv.DictReader(io.StringIO(text)))
    out = []
    for r in rows:
        out.append({
            "set": (r.get("set") or "").strip(),
            "q": (r.get("题号") or "").strip(),
            "sample_id": (r.get("sample_id") or "").strip(),
            "track": (r.get("track") or "").strip(),
            "layer": (r.get("层") or "").strip(),
            "layer_name": (r.get("层名称") or "").strip(),
            "a": (r.get("答案") or "").strip().upper(),
        })
    return out


def qkey(q):
    return int(q[1:])


def pct(num, den, digits=1):
    if den == 0:
        return "—"
    return ("%." + str(digits) + "f%%") % (100.0 * num / float(den))


def frac(num, den, digits=1):
    if den == 0:
        return "—"
    return "%d / %d (%s)" % (num, den, pct(num, den, digits))


def kfmt(x):
    if x is None:
        return "—"
    return "%.3f" % x


def kappa_label(k):
    if k is None:
        return "—"
    if k < 0:
        return "差（低于偶然）"
    if k < 0.20:
        return "轻微"
    if k < 0.40:
        return "一般"
    if k < 0.60:
        return "中等"
    if k < 0.80:
        return "较充分"
    return "几乎完全一致"


def cohens_kappa(pairs):
    n = len(pairs)
    if n == 0:
        return None, None, None
    po = sum(1 for a, b in pairs if a == b) / float(n)
    p1 = Counter(a for a, b in pairs)
    p2 = Counter(b for a, b in pairs)
    pe = 0.0
    for k in CATS:
        pe += (p1[k] / float(n)) * (p2[k] / float(n))
    if abs(1.0 - pe) < 1e-12:
        kappa = 1.0 if abs(po - 1.0) < 1e-12 else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)
    return po, pe, kappa


def gwet_ac1(pairs):
    n = len(pairs)
    if n == 0:
        return None, None, None
    po = sum(1 for a, b in pairs if a == b) / float(n)
    counts = Counter()
    for a, b in pairs:
        counts[a] += 1
        counts[b] += 1
    K = len(CATS)
    pe = 0.0
    for k in CATS:
        pk = counts[k] / float(2 * n)
        pe += pk * (1.0 - pk)
    pe = pe / float(K - 1)
    if abs(1.0 - pe) < 1e-12:
        ac1 = 1.0 if abs(po - 1.0) < 1e-12 else 0.0
    else:
        ac1 = (po - pe) / (1.0 - pe)
    return po, pe, ac1


def winner_from(scores):
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    top, topv = ranked[0]
    if len(ranked) > 1 and abs(ranked[1][1] - topv) < 1e-9:
        return "TIE", ranked
    return top, ranked


def load_jsonl_wanted(path, wanted):
    out = {}
    with open(str(path)) as handle:
        for line in handle:
            obj = json.loads(line)
            if set(obj.keys()) == {"sample_id", "tags"}:
                obj = {"sample_id": obj["sample_id"]}
                obj.update(json.loads(line)["tags"] if False else {})
            if "tags" in obj and set(list(obj.keys())) <= {"sample_id", "tags"}:
                row = {"sample_id": obj["sample_id"]}
                row.update(obj["tags"])
                obj = row
            sid = obj.get("sample_id")
            if sid in wanted:
                out[sid] = obj
            if len(out) == len(wanted):
                break
    return out


def load_preds(path, wanted):
    out = {}
    with open(str(path)) as handle:
        for line in handle:
            obj = json.loads(line)
            if "tags" in obj and list(obj.keys()) == ["sample_id", "tags"]:
                row = {"sample_id": obj["sample_id"]}
                row.update(obj["tags"])
                obj = row
            sid = obj.get("sample_id")
            if sid in wanted:
                out[sid] = obj
            if len(out) == len(wanted):
                break
    return out


def field_score(gt, pred, path):
    if path not in gt["gt_mask"]:
        return None
    spec = S.FIELD_SPECS[path]
    state, predicted = S.prediction_state(pred, path, spec)
    reference = gt["values"].get(path)
    metric = spec["metric"]
    if metric == "gender_sequence":
        if state == "result_null":
            return 1.0
        if state != "result_known":
            return 0.0
        reference_t = tuple(reference)
        hypothesis = tuple(predicted)
        alignment = S.gender_sequence_alignment(reference_t, hypothesis)
        denom = len(reference_t)
        if denom:
            return 2.0 * max(0.0, 1.0 - alignment["distance"] / float(denom))
        return 2.0 if not hypothesis else 0.0
    if metric == "multilabel":
        mask = gt.get("label_mask_by_field", {}).get(path, {})
        scores = []
        for label in S.MULTILABEL_CLASSES[path]:
            if not mask.get(label, False):
                continue
            ref_b = label in set(reference or [])
            if state == "result_null":
                scores.append(1.0)
            elif state != "result_known":
                scores.append(0.0)
            else:
                pred_b = label in predicted
                scores.append(2.0 if pred_b == ref_b else 0.0)
        if not scores:
            return None
        return sum(scores) / float(len(scores))
    return S.unit_score(metric, state, predicted, reference, spec, gt.get("language"))


def layer_rate(gt, pred, fields):
    vals = []
    for path in fields:
        score = field_score(gt, pred, path)
        if score is not None:
            vals.append(score)
    if not vals:
        return None, 0
    return sum(vals) / (2.0 * len(vals)), len(vals)


def struct_winner(rates):
    valid = {k: v for k, v in rates.items() if v is not None}
    if not valid:
        return "NA", []
    mx = max(valid.values())
    tops = sorted(k for k, v in valid.items() if abs(v - mx) < 1e-12)
    if len(tops) == 1:
        return tops[0], tops
    return "TIE", tops


def stats_block(subset):
    pairs = [(it["a1"], it["a2"]) for it in subset]
    po, pe_k, kappa = cohens_kappa(pairs)
    _, _, ac1 = gwet_ac1(pairs)
    n = len(subset)
    agree = sum(1 for it in subset if it["agree"])
    votes = Counter(it["a1"] for it in subset) + Counter(it["a2"] for it in subset)
    conf = Counter((it["a1"], it["a2"]) for it in subset)
    both_abc = [it for it in subset if it["a1"] in "ABC" and it["a2"] in "ABC"]
    return {
        "n": n,
        "agree": agree,
        "po": po,
        "kappa": kappa,
        "ac1": ac1,
        "votes": votes,
        "conf": conf,
        "both_abc_n": len(both_abc),
        "both_abc_agree": sum(1 for it in both_abc if it["agree"]),
    }


def match_auto(subset, auto_key, human_key, unique_only=True):
    hit = n = hit_eff = n_eff = 0
    for it in subset:
        pred = it[auto_key]
        if unique_only and pred not in "ABC":
            continue
        if pred == "NA":
            continue
        hum = it[human_key]
        if pred == "TIE":
            tops = it.get("struct_tied") or []
            ok = hum in tops
        else:
            ok = hum == pred
        n += 1
        if ok:
            hit += 1
        if hum in "ABC":
            n_eff += 1
            if ok:
                hit_eff += 1
    return hit, n, hit_eff, n_eff


def main():
    ids = sorted(p.name for p in SAMPLES.iterdir() if p.is_dir())
    wanted = set(ids)
    gt_all = S.build_ground_truth(BENCH)
    preds = {k: load_preds(path, wanted) for k, path in PRED_PATH.items()}
    for k in "ABC":
        missing = [sid for sid in ids if sid not in preds[k]]
        if missing:
            raise RuntimeError("missing preds %s %s" % (k, missing))

    struct = {}
    for sid in ids:
        gt = gt_all[sid]
        rec = {"track": gt["track"], "layers": {}}
        for layer, fields in STRUCT_FIELDS.items():
            rates = {}
            n_fields = 0
            for letter in "ABC":
                rate, n_fields = layer_rate(gt, preds[letter][sid], fields)
                rates[letter] = rate
            winner, tops = struct_winner(rates)
            rec["layers"][layer] = {
                "n_fields": n_fields,
                "A": rates["A"],
                "B": rates["B"],
                "C": rates["C"],
                "winner": winner,
                "tied": tops,
            }
        struct[sid] = rec

    desc = {}
    for sid in ids:
        comp = json.loads((SAMPLES / sid / "comparison.json").read_text(encoding="utf-8"))
        npct = comp["normalized_percent"]
        rec = {"deepseek": {}, "grok": {}}
        rec["deepseek"]["A"] = npct["sure-tagger"]["deepseek"] / 100.0
        rec["deepseek"]["B"] = npct["qwen-captioner"]["deepseek"] / 100.0
        rec["deepseek"]["C"] = npct["qwen3-omni"]["deepseek"] / 100.0
        rec["grok"]["A"] = npct["sure-tagger"]["grok"] / 100.0
        rec["grok"]["B"] = npct["qwen-captioner"]["grok"] / 100.0
        rec["grok"]["C"] = npct["qwen3-omni"]["grok"] / 100.0
        rec["deepseek_winner"], _ = winner_from(rec["deepseek"])
        rec["grok_winner"], _ = winner_from(rec["grok"])
        desc[sid] = rec

    items = []
    for set_id, set_name, n1, f1, n2, f2 in PAIRS:
        d1 = {row["q"]: row for row in load_rows(ANS / f1)}
        d2 = {row["q"]: row for row in load_rows(ANS / f2)}
        for q in sorted(d1, key=qkey):
            a = d1[q]
            b = d2[q]
            sid = a["sample_id"]
            layer = a["layer"]
            st = struct[sid]["layers"].get(layer, {})
            dc = desc[sid]
            if layer in STRUCT_FIELDS:
                auto = st.get("winner", "NA")
                tied = st.get("tied") or []
                sa, sb, sc = st.get("A"), st.get("B"), st.get("C")
                n_fields = st.get("n_fields", 0)
            else:
                auto = dc["deepseek_winner"]
                tied = []
                sa, sb, sc = dc["deepseek"]["A"], dc["deepseek"]["B"], dc["deepseek"]["C"]
                n_fields = 10
            items.append({
                "set": set_id,
                "set_name": set_name,
                "q": q,
                "sample_id": sid,
                "track": a["track"],
                "layer": layer,
                "layer_name": a["layer_name"] or LAYER_NAME.get(layer, layer),
                "rater1": n1,
                "rater2": n2,
                "a1": a["a"],
                "a2": b["a"],
                "agree": a["a"] == b["a"],
                "struct_A": st.get("A") if layer in STRUCT_FIELDS else None,
                "struct_B": st.get("B") if layer in STRUCT_FIELDS else None,
                "struct_C": st.get("C") if layer in STRUCT_FIELDS else None,
                "struct_winner": st.get("winner") if layer in STRUCT_FIELDS else None,
                "struct_tied": st.get("tied") if layer in STRUCT_FIELDS else [],
                "struct_n": st.get("n_fields") if layer in STRUCT_FIELDS else None,
                "deepseek_winner": dc["deepseek_winner"],
                "grok_winner": dc["grok_winner"],
                "ds_A": dc["deepseek"]["A"],
                "ds_B": dc["deepseek"]["B"],
                "ds_C": dc["deepseek"]["C"],
                "gk_A": dc["grok"]["A"],
                "gk_B": dc["grok"]["B"],
                "gk_C": dc["grok"]["C"],
                "auto_struct": auto if layer in STRUCT_FIELDS else None,
            })

    assert len(items) == 100

    csv_path = DOC_DEMO / "human-preference-item-table.csv"
    with open(str(csv_path), "w") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "set", "题号", "sample_id", "track", "层", "评者1", "评者1答案", "评者2", "评者2答案", "两人一致",
            "结构第一", "结构字段数", "结构A", "结构B", "结构C",
            "DeepSeek第一", "Grok第一", "DeepSeek_A", "DeepSeek_B", "DeepSeek_C", "Grok_A", "Grok_B", "Grok_C",
        ])
        for it in items:
            def rf(x):
                return "" if x is None else ("%.4f" % x)
            writer.writerow([
                it["set"], it["q"], it["sample_id"], it["track"], it["layer"],
                it["rater1"], it["a1"], it["rater2"], it["a2"], "yes" if it["agree"] else "no",
                it["struct_winner"] or "", it["struct_n"] if it["struct_n"] is not None else "",
                rf(it["struct_A"]), rf(it["struct_B"]), rf(it["struct_C"]),
                it["deepseek_winner"], it["grok_winner"],
                "%.4f" % it["ds_A"], "%.4f" % it["ds_B"], "%.4f" % it["ds_C"],
                "%.4f" % it["gk_A"], "%.4f" % it["gk_B"], "%.4f" % it["gk_C"],
            ])

    score_path = DOC_DEMO / "structural-layer-scores.json"
    dump = {}
    for sid in ids:
        rec = {"track": struct[sid]["track"], "layers": {}}
        for layer, payload in struct[sid]["layers"].items():
            rec["layers"][layer] = {
                "n_fields": payload["n_fields"],
                "score_rate": {
                    "A": payload["A"],
                    "B": payload["B"],
                    "C": payload["C"],
                },
                "winner": payload["winner"],
                "tied": payload["tied"],
            }
        dump[sid] = rec
    score_path.write_text(json.dumps(dump, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    all_s = stats_block(items)
    by_set = OrderedDict()
    for set_id, set_name, n1, f1, n2, f2 in PAIRS:
        sub = [it for it in items if it["set"] == set_id]
        by_set[set_id] = (set_name, n1, n2, stats_block(sub))
    by_layer = OrderedDict()
    for layer in LAYERS:
        sub = [it for it in items if it["layer"] == layer]
        by_layer[layer] = (LAYER_NAME[layer], stats_block(sub), sub)
    by_track = OrderedDict()
    for tr in sorted(set(it["track"] for it in items)):
        by_track[tr] = stats_block([it for it in items if it["track"] == tr])

    votes = all_s["votes"]
    agree_items = [it for it in items if it["agree"]]
    agree_votes = Counter(it["a1"] for it in agree_items)
    dis = [it for it in items if not it["agree"]]
    d_items = [it for it in items if it["layer"] == "D"]
    struct_items = [it for it in items if it["layer"] in STRUCT_FIELDS]
    struct_unique = [it for it in struct_items if it["struct_winner"] in "ABC"]
    struct_na = [it for it in struct_items if it["struct_winner"] == "NA"]
    struct_tie = [it for it in struct_items if it["struct_winner"] == "TIE"]

    ds_win = Counter(desc[sid]["deepseek_winner"] for sid in ids)
    gk_win = Counter(desc[sid]["grok_winner"] for sid in ids)
    ds_gk_agree = sum(1 for sid in ids if desc[sid]["deepseek_winner"] == desc[sid]["grok_winner"])
    jg_dis = [sid for sid in ids if desc[sid]["deepseek_winner"] != desc[sid]["grok_winner"]]
    d_votes = Counter(it["a1"] for it in d_items) + Counter(it["a2"] for it in d_items)

    def two_human_match(subset, getter):
        """getter(it) -> auto label in ABC/TIE/NA."""
        hit = n = hit_eff = n_eff = cons_hit = cons_n = any_m = both_m = 0
        n_items = 0
        for it in subset:
            pred = getter(it)
            if pred not in "ABC":
                continue
            n_items += 1
            for hum in (it["a1"], it["a2"]):
                n += 1
                if hum == pred:
                    hit += 1
                if hum in "ABC":
                    n_eff += 1
                    if hum == pred:
                        hit_eff += 1
            if it["agree"] and it["a1"] in "ABC":
                cons_n += 1
                if it["a1"] == pred:
                    cons_hit += 1
            if it["a1"] == pred or it["a2"] == pred:
                any_m += 1
            if it["a1"] == pred and it["a2"] == pred:
                both_m += 1
        return {
            "hit": hit, "n": n, "hite": hit_eff, "ne": n_eff,
            "cons_hit": cons_hit, "cons_n": cons_n,
            "any": any_m, "both": both_m, "n_items": n_items,
        }

    struct_match = two_human_match(struct_items, lambda it: it["struct_winner"])
    d_ds = two_human_match(d_items, lambda it: it["deepseek_winner"])
    d_gk = two_human_match(d_items, lambda it: it["grok_winner"])

    mixed_ds = two_human_match(
        items,
        lambda it: it["struct_winner"] if it["layer"] in STRUCT_FIELDS else it["deepseek_winner"],
    )
    mixed_gk = two_human_match(
        items,
        lambda it: it["struct_winner"] if it["layer"] in STRUCT_FIELDS else it["grok_winner"],
    )

    layer_struct_match = OrderedDict()
    for layer in ["L1", "L2", "L3", "L5"]:
        sub = [it for it in items if it["layer"] == layer]
        layer_struct_match[layer] = (
            stats_block(sub),
            two_human_match(sub, lambda it: it["struct_winner"]),
            sum(1 for it in sub if it["struct_winner"] in "ABC"),
            sum(1 for it in sub if it["struct_winner"] == "NA"),
            sum(1 for it in sub if it["struct_winner"] == "TIE"),
        )

    # confusion human vs unique structural winner (pool 2 raters)
    conf_hs = Counter()
    for it in struct_unique:
        pred = it["struct_winner"]
        for hum in (it["a1"], it["a2"]):
            conf_hs[(hum, pred)] += 1

    md = []

    def w(line=""):
        md.append(line)

    w("# 人工分层偏好：评者一致性与系统自动分对齐")
    w()
    w("数据：`dataset/benchmark/demo/answer`。四套题共 100 道，每题两人独立作答。选项为 A Sure-Tagger、B Captioner、C Omni、D 无法判断。")
    w()
    w("本文同时报告两件事：")
    w()
    w("1. **人与人**：100 题双编码的观察一致率、Cohen $\\kappa$、Gwet AC1。")
    w("2. **人与系统**：把三套系统的自动得分变成 A/B/C 第一名，再和人类选择比。")
    w()
    w("- **L1 / L2 / L3 / L5**：用 Benchmark v1 结构 19 轴，在该样本、该层上对三套系统打分，取 argmax。计分与 [`benchmark-v1-29-metrics.md`](benchmark-v1-29-metrics.md) 相同：已知且对得 2 分，显式 `null` 得 1 分，缺失/错误得 0 分；层内对有 GT 的轴做微平均得到得分率。")
    w("- **D**：仍用描述 10 轴的 DeepSeek v4 Pro / Grok 4.7 归一化得分率 argmax。这两个评委**没有**给 L1–L5 打分。")
    w("- T2 的 L2/L3/L5、T3 的 L1/L2/L3 官方没有该层 GT，自动裁判记为 NA，**不计入**人机一致率的分母。")
    w()
    w("Sure-Tagger 为论文 **v1**（`result_with_id_descriptive.jsonl`，语种已经核验）。Captioner / Omni 结构预测分别来自 `docs/report/result/qwen-captioner/results_t1_t3.jsonl` 与 `docs/report/result/qwen3-omni-v2-multi-api12/final_results.jsonl`。")
    w()
    w("逐题表：[`human-preference-item-table.csv`](../dataset/benchmark/demo/docs/human-preference-item-table.csv)；结构分层分：[`structural-layer-scores.json`](../dataset/benchmark/demo/docs/structural-layer-scores.json)。")
    w()
    w("## 评者配对")
    w()
    w("| 套 | 评者 1 | 评者 2 | 题数 |")
    w("| --- | --- | --- | ---: |")
    for set_id, set_name, n1, f1, n2, f2 in PAIRS:
        w("| %s | `%s` | `%s` | 25 |" % (set_name, n1, n2))
    w()
    w("`whf` 做了第 1、2 套，`djh` 做了第 3、4 套。总体 kappa 是四对评者的双编码题池，不是同一对人标完全部 100 题。")
    w()
    w("## 人类选择分布")
    w()
    w("| 选项 | 次数 | 占比 |")
    w("| --- | ---: | ---: |")
    for k in CATS:
        w("| %s %s | %d | %s |" % (k, CHOICE_NAME[k], votes[k], pct(votes[k], 200)))
    w("| 合计 | 200 | 100.0% |")
    w()
    w("两人一致的 %d 题里，共识为 A %d、B %d、C %d、D %d。" % (
        all_s["agree"], agree_votes["A"], agree_votes["B"], agree_votes["C"], agree_votes["D"]))
    w()
    w("## 评者一致性")
    w()
    w("A 占 79.5%，Cohen $\\kappa$ 会被 kappa 悖论压低；Gwet AC1 更稳。Landis & Koch：κ 0.21–0.40 一般，0.41–0.60 中等，0.61–0.80 较充分。")
    w()
    w("### 总体")
    w()
    w("| 指标 | 数值 |")
    w("| --- | --- |")
    w("| 题数 | 100 |")
    w("| 两人相同 | %s |" % frac(all_s["agree"], all_s["n"]))
    w("| Cohen $\\kappa$ | %s（%s） |" % (kfmt(all_s["kappa"]), kappa_label(all_s["kappa"])))
    w("| Gwet AC1 | %s |" % kfmt(all_s["ac1"]))
    w("| 双方都选 A/B/C 的题 | %s，其中相同 %s |" % (
        frac(all_s["both_abc_n"], all_s["n"]),
        frac(all_s["both_abc_agree"], all_s["both_abc_n"])))
    w()
    w("### 分套")
    w()
    w("| 套 | 评者 | 相同 | $P_o$ | Cohen $\\kappa$ | Gwet AC1 |")
    w("| --- | --- | --- | ---: | ---: | ---: |")
    for set_id, set_name, n1, f1, n2, f2 in PAIRS:
        name, r1, r2, st = by_set[set_id]
        w("| %s | `%s` vs `%s` | %s | %s | %s | %s |" % (
            name, r1, r2, frac(st["agree"], st["n"]), pct(st["agree"], st["n"]),
            kfmt(st["kappa"]), kfmt(st["ac1"])))
    w()
    w("### 分层")
    w()
    w("| 层 | 题数 | 相同 | $P_o$ | Cohen $\\kappa$ | Gwet AC1 |")
    w("| --- | ---: | --- | ---: | ---: | ---: |")
    for layer in LAYERS:
        name, st, _sub = by_layer[layer]
        w("| %s | %d | %s | %s | %s | %s |" % (
            name, st["n"], frac(st["agree"], st["n"]), pct(st["agree"], st["n"]),
            kfmt(st["kappa"]), kfmt(st["ac1"])))
    w()
    w("### 分轨道")
    w()
    w("| 轨道 | 题数 | 相同 | $P_o$ | Cohen $\\kappa$ | Gwet AC1 |")
    w("| --- | ---: | --- | ---: | ---: | ---: |")
    for tr, st in by_track.items():
        w("| `%s` | %d | %s | %s | %s | %s |" % (
            tr, st["n"], frac(st["agree"], st["n"]), pct(st["agree"], st["n"]),
            kfmt(st["kappa"]), kfmt(st["ac1"])))
    w()
    w("### 混淆（评者1 × 评者2）")
    w()
    w("| 评者1 \\ 评者2 | A | B | C | D | 合计 |")
    w("| --- | ---: | ---: | ---: | ---: | ---: |")
    conf = all_s["conf"]
    for a in CATS:
        cells = [str(conf[(a, b)]) for b in CATS]
        w("| **%s** | %s | %d |" % (a, " | ".join(cells), sum(conf[(a, b)] for b in CATS)))
    w("| **合计** | %s | 100 |" % " | ".join(str(sum(conf[(a, b)] for a in CATS)) for b in CATS))
    w()
    w("不一致主要是 A↔C 和 A↔D。B 几乎无人选。")
    w()
    w("## 系统自动分 vs 人类")
    w()
    w("### 结构层可评范围")
    w()
    w("80 道结构题（20 条 × L1/L2/L3/L5）里：")
    w()
    w("| 自动裁判 | 题数 | 说明 |")
    w("| --- | ---: | --- |")
    w("| 唯一第一名 | %d | 计入人机一致率 |" % len(struct_unique))
    w("| 并列第一 | %d | 不计入严格一致率 |" % len(struct_tie))
    w("| 无该层 GT（NA） | %d | T2 的 L2/L3/L5、T3 的 L1/L2/L3 |" % len(struct_na))
    w()
    w("Captioner 结构字段大量显式 `null`，官方规则记 50% 半分。Sure-Tagger 只要高于 50% 就会在该层胜出。这是规则使然，不是 Captioner 得了「真的 50 分」。")
    w()
    w("### 主表：人类 vs 自动第一名")
    w()
    w("「评分」= 该层可裁判题 × 2 人。D 用描述评委；L1–L5 用结构 argmax。")
    w()
    w("| 范围 | 自动裁判 | 可评题 | 40/2 人评分对齐 | 去掉人类 D | 两人共识且非 D |")
    w("| --- | --- | ---: | --- | --- | --- |")

    def row(name, judge, st):
        w("| %s | %s | %d | %s | %s | %s |" % (
            name, judge, st["n_items"],
            frac(st["hit"], st["n"]),
            frac(st["hite"], st["ne"]),
            frac(st["cons_hit"], st["cons_n"])))

    row("L1–L5 合计", "结构 19 轴", struct_match)
    for layer in ["L1", "L2", "L3", "L5"]:
        _st, m, n_uni, n_na, n_tie = layer_struct_match[layer]
        row(LAYER_NAME[layer], "结构（NA %d / 并列 %d）" % (n_na, n_tie), m)
    row("D 描述性", "DeepSeek 描述分", d_ds)
    row("D 描述性", "Grok 描述分", d_gk)
    row("100 题混合", "结构 + DeepSeek(D)", mixed_ds)
    row("100 题混合", "结构 + Grok(D)", mixed_gk)
    w()
    w("结构层人机一致率明显高于 D 层：人类本来就大量选 A，结构自动第一名也大量是 A。描述评委则经常把 Omni 排第一，所以 D 层人对不齐。")
    w()
    w("### 人类选择 vs 结构第一名（可评题，两人评分）")
    w()
    w("| 人类 \\ 结构第一 | A | B | C | 合计 |")
    w("| --- | ---: | ---: | ---: | ---: |")
    for hum in CATS:
        cells = [str(conf_hs[(hum, pred)]) for pred in "ABC"]
        w("| **%s** | %s | %d |" % (hum, " | ".join(cells), sum(conf_hs[(hum, pred)] for pred in "ABC")))
    w("| **合计** | %s | %d |" % (
        " | ".join(str(sum(conf_hs[(hum, pred)] for hum in CATS)) for pred in "ABC"),
        sum(conf_hs.values())))
    w()
    w("### 20 条样本的结构层得分率（%）")
    w()
    w("| 样本 | 层 | 字段 | A | B | C | 第一 |")
    w("| --- | --- | ---: | ---: | ---: | ---: | --- |")

    def pf(x):
        return "—" if x is None else ("%.1f" % (100.0 * x))

    for sid in ids:
        for layer in ["L1", "L2", "L3", "L5"]:
            payload = struct[sid]["layers"][layer]
            w("| `%s` | %s | %d | %s | %s | %s | %s |" % (
                sid, layer, payload["n_fields"],
                pf(payload["A"]), pf(payload["B"]), pf(payload["C"]),
                payload["winner"]))
    w()
    w("### D 层 vs DeepSeek / Grok")
    w()
    w("20 条音频上 DeepSeek 与 Grok 没有并列第一；第一名相同 %s。" % frac(ds_gk_agree, 20))
    w()
    w("| 评委第一名 | DeepSeek | Grok |")
    w("| --- | ---: | ---: |")
    for k in ["A", "B", "C"]:
        w("| %s %s | %d | %d |" % (k, CHOICE_NAME[k], ds_win[k], gk_win[k]))
    w()
    if jg_dis:
        w("第一名不同的样本：")
        w()
        w("| sample_id | DeepSeek（A/B/C %） | Grok（A/B/C %） |")
        w("| --- | --- | --- |")
        for sid in jg_dis:
            dc = desc[sid]
            w("| `%s` | %s %.1f / %.1f / %.1f | %s %.1f / %.1f / %.1f |" % (
                sid, dc["deepseek_winner"],
                100 * dc["deepseek"]["A"], 100 * dc["deepseek"]["B"], 100 * dc["deepseek"]["C"],
                dc["grok_winner"],
                100 * dc["grok"]["A"], 100 * dc["grok"]["B"], 100 * dc["grok"]["C"]))
        w()
    w("人类 D 层 40 个评分：A %d、B %d、C %d、D %d。" % (d_votes["A"], d_votes["B"], d_votes["C"], d_votes["D"]))
    w()
    w("| 套 | 题 | 样本 | 人1 | 人2 | 结构第一 | DeepSeek | Grok |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for it in d_items:
        w("| %s | %s | `%s` | %s | %s | %s | %s | %s |" % (
            it["set_name"], it["q"], it["sample_id"], it["a1"], it["a2"],
            it["struct_winner"] or "—", it["deepseek_winner"], it["grok_winner"]))
    w()
    w("## 两人不一致的题")
    w()
    w("共 %d / 100 题两人答案不同。" % len(dis))
    w()
    w("| 套 | 题 | 层 | 样本 | 评者1 | 评者2 | 结构第一 | DeepSeek | Grok |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for it in dis:
        w("| %s | %s | %s | `%s` | %s `%s` | %s `%s` | %s | %s | %s |" % (
            it["set_name"], it["q"], it["layer"], it["sample_id"],
            it["rater1"], it["a1"], it["rater2"], it["a2"],
            it["struct_winner"] or "—", it["deepseek_winner"], it["grok_winner"]))
    w()
    w("## 结论")
    w()
    w("- 人类高度偏向 Sure-Tagger：200 个评分里 A 占 %s，B 只有 %d 次。" % (pct(votes["A"], 200), votes["B"]))
    w("- 两人相同 %s。Cohen κ=%s（%s）；Gwet AC1=%s。" % (
        frac(all_s["agree"], 100), kfmt(all_s["kappa"]), kappa_label(all_s["kappa"]), kfmt(all_s["ac1"])))
    w("- **结构层人机一致**：L1–L5 在有唯一自动第一名的题上，%s 的人类评分选中该系统；去掉 D 后 %s。共识非 D 时 %s。" % (
        frac(struct_match["hit"], struct_match["n"]),
        frac(struct_match["hite"], struct_match["ne"]),
        frac(struct_match["cons_hit"], struct_match["cons_n"])))
    w("- **描述层人机一致**明显更低：D 对 DeepSeek %s，对 Grok %s。人类仍偏 A，评委第一名里 Omni 更多。")
    w("- 混合口径（结构裁判 L1–L5 + DeepSeek 裁判 D）整体 %s。" % frac(mixed_ds["hit"], mixed_ds["n"]))
    w("- 80 道结构题里有 %d 道官方无该层 GT，不能做人机对比。Captioner 的 50%% 来自 `null` 半分，不代表识别对了一半。" % len(struct_na))
    w()
    # fill D rates in conclusion bullet - I used incomplete string. Fix below after join? I'll write complete bullets.
    # Actually the D bullet has unfinished format. Let me rewrite conclusions at the end more carefully.

    text = "\n".join(md) + "\n"
    # fix incomplete D conclusion if present
    old = "- **描述层人机一致**明显更低：D 对 DeepSeek %s，对 Grok %s。人类仍偏 A，评委第一名里 Omni 更多。\n"
    if "%s" in text.split("## 结论")[-1]:
        text = text.replace(
            "- **描述层人机一致**明显更低：D 对 DeepSeek %s，对 Grok %s。人类仍偏 A，评委第一名里 Omni 更多。\n",
            "- **描述层人机一致**明显更低：D 对 DeepSeek %s，对 Grok %s。人类仍偏 A，评委第一名里 Omni 更多。\n" % (
                frac(d_ds["hit"], d_ds["n"]), frac(d_gk["hit"], d_gk["n"])),
        )

    w2 = []
    # rebuild conclusion cleanly by replacing the conclusion section
    head, _sep, _tail = text.partition("## 结论\n")
    conclusion = []
    conclusion.append("## 结论")
    conclusion.append("")
    conclusion.append("- 人类高度偏向 Sure-Tagger：200 个评分里 A 占 %s，B 只有 %d 次。" % (pct(votes["A"], 200), votes["B"]))
    conclusion.append("- 两人相同 %s。Cohen κ=%s（%s）；Gwet AC1=%s。" % (
        frac(all_s["agree"], 100), kfmt(all_s["kappa"]), kappa_label(all_s["kappa"]), kfmt(all_s["ac1"])))
    conclusion.append("- **结构层 ↔ 人类**：有唯一结构第一名的题上，人类评分对齐 %s；去掉无法判断后 %s；两人共识且非 D 时 %s。自动第一名也以 A 为主，和人类同向。" % (
        frac(struct_match["hit"], struct_match["n"]),
        frac(struct_match["hite"], struct_match["ne"]),
        frac(struct_match["cons_hit"], struct_match["cons_n"])))
    conclusion.append("- **描述层 ↔ 人类**：D 对 DeepSeek %s，对 Grok %s。人类偏 A，评委常把 Omni 排第一，两边不是同一排序。" % (
        frac(d_ds["hit"], d_ds["n"]), frac(d_gk["hit"], d_gk["n"])))
    conclusion.append("- **混合人机**（L1–L5 用结构分，D 用 DeepSeek）%s；若 D 改用 Grok 则 %s。" % (
        frac(mixed_ds["hit"], mixed_ds["n"]), frac(mixed_gk["hit"], mixed_gk["n"])))
    conclusion.append("- 80 道结构题中 %d 道无该层 GT（NA），不能做人机对比。Captioner 结构分接近 50%% 是 `null` 半分，不是识别准确率。" % len(struct_na))
    conclusion.append("")
    conclusion.append("## 数据与口径")
    conclusion.append("")
    conclusion.append("| 项 | 路径 / 说明 |")
    conclusion.append("| --- | --- |")
    conclusion.append("| 人类答案 | `dataset/benchmark/demo/answer/` |")
    conclusion.append("| 结构计分 | `eval/score_benchmark_v1.py`，单样本层微平均 |")
    conclusion.append("| Sure-Tagger v1 | `evaluation/benchmark-v1/sure_tagger_full_20260918_main_pipeline_w2_b8/result_with_id_descriptive.jsonl` |")
    conclusion.append("| Captioner | `docs/report/result/qwen-captioner/results_t1_t3.jsonl` |")
    conclusion.append("| Omni | `docs/report/result/qwen3-omni-v2-multi-api12/final_results.jsonl` |")
    conclusion.append("| 描述分 | 各样本 `comparison.json`（DeepSeek v4 Pro / Grok 4.7） |")
    conclusion.append("| 逐题表 | `dataset/benchmark/demo/docs/human-preference-item-table.csv` |")
    conclusion.append("| 结构分层分 | `dataset/benchmark/demo/docs/structural-layer-scores.json` |")
    conclusion.append("")
    text = head + "\n".join(conclusion) + "\n"

    p1 = DOC_DOCS / "human-layer-preference-agreement.md"
    p2 = DOC_DEMO / "human-layer-preference-agreement.md"
    # demo copy uses local csv links
    t1 = text
    t2 = text.replace(
        "[`human-preference-item-table.csv`](../dataset/benchmark/demo/docs/human-preference-item-table.csv)",
        "[`human-preference-item-table.csv`](human-preference-item-table.csv)",
    ).replace(
        "[`structural-layer-scores.json`](../dataset/benchmark/demo/docs/structural-layer-scores.json)",
        "[`structural-layer-scores.json`](structural-layer-scores.json)",
    ).replace(
        "[`benchmark-v1-29-metrics.md`](benchmark-v1-29-metrics.md)",
        "[`benchmark-v1-29-metrics.md`](../../../docs/benchmark-v1-29-metrics.md)",
    )
    p1.write_text(t1, encoding="utf-8")
    p2.write_text(t2, encoding="utf-8")
    print("wrote", p1)
    print("wrote", p2)
    print("csv", csv_path)
    print("scores", score_path)
    print("struct_match", struct_match)
    print("d_ds", d_ds)
    print("d_gk", d_gk)
    print("mixed_ds", mixed_ds)
    print("mixed_gk", mixed_gk)
    print("unique", len(struct_unique), "na", len(struct_na), "tie", len(struct_tie))
    for layer in ["L1", "L2", "L3", "L5"]:
        print(layer, layer_struct_match[layer][1], "uni", layer_struct_match[layer][2], "na", layer_struct_match[layer][3])


if __name__ == "__main__":
    main()
