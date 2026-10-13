"""Exercise shell dispatch with real shard preparation/validation and fake inference.

No model, GPU, benchmark data or API credentials are needed. Each test builds
650 tiny WAV files and launches the production shell entry in an isolated tree.
Subset checks reduce the input manifests before launching the same entry.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave


SOURCE = Path(__file__).resolve().parent

FAKE_PYTHON = r'''
import json
import os
from pathlib import Path
import runpy
import sys

args = sys.argv[1:]
if args[0] in ("-c", "-"):
    os.execv(os.environ["TEST_REAL_PYTHON"], [os.environ["TEST_REAL_PYTHON"]] + args)

script = Path(args[0]).name
argv = args[1:]

def option(name):
    return argv[argv.index(name) + 1]

def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]

def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

def event(kind, **values):
    values.update(kind=kind, pid=os.getpid())
    descriptor = os.open(os.environ["TEST_EVENTS"], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, (json.dumps(values) + "\n").encode())
    finally:
        os.close(descriptor)

if script == "run_tagger.py":
    from finalize_sure_tagger import GROUP_FIELDS
    stage = "retry" if "--sample-id" in argv else ("aux" if "--input-tags" in argv else "core")
    manifest = Path(option("--manifest"))
    rows = read_rows(manifest)
    event(stage, gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
          workers=int(option("--workers")), manifest=manifest.name,
          device=option("--nemotron-device") if "--nemotron-device" in argv else None,
          gender_device=os.environ.get("TAGGER_GENDER_DEVICE"))
    if stage == "core" and os.environ.get("TEST_FAIL_FIRST_CORE") == "1":
        marker = Path(os.environ["TEST_EVENTS"]).with_suffix(".failed")
        if not marker.exists():
            marker.touch()
            raise SystemExit(1)
    if stage == "core":
        tags = {group: {field: None for field in fields} for group, fields in GROUP_FIELDS.items()}
        output = [tags for row in rows]
    else:
        output = read_rows(option("--input-tags"))
        assert len(output) == len(rows), "aux/retry ran without complete core output"
    write_rows(option("--output"), output)
    print(json.dumps({"processed_sample_count": len(rows)}))
elif script == "find_sure_tagger_retry_samples.py":
    if os.environ.get("TEST_SPEAKER_RETRY") == "1" and "--expected-manifest-dir" in argv:
        print("shard_00\tsample_0000")
elif script == "score_benchmark_v1.py":
    count = len(read_rows(option("--predictions")))
    Path(option("--output")).write_text(json.dumps({
        "prediction_row_count": count, "missing_result_rows": 0, "input_contract_failed": False}))
    Path(option("--report")).write_text("Test score\n")
elif script == "run_sure_tagger_deepseek_batch.py":
    event("descriptions", workers=int(option("--workers")), prompt=option("--prompt-version"))
    rows = read_rows(option("--input-manifest"))
    output = Path(option("--output-dir"))
    write_rows(output / "results.jsonl", [{"sample_id": row["sample_id"]} for row in rows])
    (output / "summary.json").write_text(json.dumps({"success_count": len(rows), "failed_count": 0}))
elif script == "merge_sure_tagger_descriptive.py":
    write_rows(Path(option("--output-dir")) / "result_with_id_descriptive.jsonl",
               read_rows(option("--structural-input")))
elif script == "run_descriptive_semantic_scoring_batch.py":
    from run_descriptive_semantic_scoring_batch import SCORED_FIELDS
    event("semantic_scores", workers=int(option("--workers")))
    rows = read_rows(option("--predictions"))
    output = Path(option("--output-dir"))
    write_rows(output / "results.jsonl", [{"sample_id": row["sample_id"], "status": "success",
               "scores": {field: 1 for field in SCORED_FIELDS}} for row in rows])
    (output / "summary.json").write_text(json.dumps({
        "status": "complete", "scoring_success_count": len(rows), "scoring_failed_count": 0,
        "score_denominator_sample_count": len(rows), "excluded_missing_ground_truth_count": 0,
        "excluded_missing_ground_truth_sample_ids": []}))
else:
    sys.argv = args
    runpy.run_path(args[0], run_name="__main__")
'''

FAKE_SCORING_MODULE = '''
import json
from pathlib import Path
CAPABILITY_FIELDS = ["test_capability"]
CONTRACT_FIELDS = ["test_contract"]
def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]
def keyed(rows, key, label):
    result = {key(row): row for row in rows}
    assert len(result) == len(rows), "duplicate " + label
    return result
def build_ground_truth(root):
    return keyed(load_jsonl(root / "benchmark.manifest.jsonl"),
                 lambda row: row["sample"]["sample_id"], "GT")
'''

FAKE_SEMANTIC_MODULE = '''
SCORED_FIELDS = ["test_description"]
def load_ready_ground_truth(items, expected):
    return {sample_id: {} for sample_id in expected}, {}
'''


class PipelineLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sure-tagger-launcher-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.eval = self.root / "compare" / "eval"
        self.eval.mkdir(parents=True)
        for name in (
            "run_sure_tagger_nemotron_benchmark.sh", "run_sure_tagger_pipeline.sh",
            "run_sure_tagger_pipeline_1.sh", "run_sure_tagger_pipeline_8.sh",
            "prepare_sure_tagger_stages.py", "finalize_sure_tagger.py",
        ):
            shutil.copy2(SOURCE / name, self.eval / name)
        (self.eval / "score_benchmark_v1.py").write_text(FAKE_SCORING_MODULE)
        (self.eval / "run_descriptive_semantic_scoring_batch.py").write_text(FAKE_SEMANTIC_MODULE)
        for name in ("deepseek_descriptive_semantic_scoring_prompt.txt",
                     "sure_tagger_deepseek_language_descriptive_prompt.txt"):
            (self.eval / name).write_text("Test prompt\n")

        runtime = self.root / "runtime"
        audio = runtime / "audio"
        audio.mkdir(parents=True)
        rows = []
        for number in range(650):
            sample_id = "sample_%04d" % number
            path = audio / (sample_id + ".wav")
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(b"\x00\x00" * (number % 8 + 1))
            rows.append({
                "corpus": {"dataset_name": "", "native_metadata": {},
                           "source_urls": {key: [] for key in ("article", "dataset_card", "github", "huggingface")}},
                "sample": {"sample_id": sample_id, "audio": {"path": str(path)},
                           "native_metadata": {}, "text": {"transcript": ""}},
            })
        contents = "".join(json.dumps(row) + "\n" for row in rows)
        (runtime / "runtime.manifest.jsonl").write_text(contents)
        benchmark = self.root / "benchmark"
        benchmark.mkdir()
        (benchmark / "benchmark.manifest.jsonl").write_text(contents)

        binaries = self.root / "bin"
        binaries.mkdir()
        python = binaries / "fake_python"
        self.executable(python, "#!" + sys.executable + "\n" + FAKE_PYTHON)
        self.executable(binaries / "nvidia-smi", "#!/bin/bash\nprintf '%s\\n' ${TEST_GPUS:-0 1 2 3 4 5 6 7}\n")
        tagger = self.root / "sure-tagger-main"
        wrappers = tagger / "scripts" / "gpu_wrappers"
        wrappers.mkdir(parents=True)
        for name in ("firered_gpu_python", "firered_lid_gpu_python", "dnsmos_gpu_python", "dass_gpu_python"):
            self.executable(wrappers / name, "#!/bin/bash\nexit 99\n")
        model = self.root / "model"
        model.mkdir()
        (model / "model.safetensors").touch()
        api = self.root / "external-api.txt"
        api.write_text("dummy-test-only\n")
        self.output = self.root / "output"
        self.stages = self.root / "staged"
        self.events_path = self.root / "events.jsonl"
        self.env = {key: value for key, value in os.environ.items() if key not in {
            "GPU_COUNT", "GPU_ID", "SAMPLE_WORKERS", "DEEPSEEK_WORKERS", "SCORING_WORKERS",
            "RESUME_EXISTING", "SKIP_STRUCTURAL", "ALLOW_FROZEN_GT_EXCLUSIONS", "PROMPT_VERSION",
            "CUDA_VISIBLE_DEVICES", "SCORING_PROMPT", "EXPECTED_SAMPLE_COUNT",
        }}
        self.env.update({
            "PATH": str(binaries) + os.pathsep + os.environ.get("PATH", ""),
            "PYTHONPATH": str(self.eval), "PYTHON_BIN": str(python),
            "TEST_REAL_PYTHON": sys.executable, "TEST_EVENTS": str(self.events_path),
            "RUNTIME_ROOT": str(runtime), "BENCHMARK_ROOT": str(benchmark),
            "GT_ROOT": str(self.root / "gt"), "SURE_TAGGER_ROOT": str(tagger),
            "GPU_WRAPPER_ROOT": str(wrappers), "NEMOTRON_PYTHON": sys.executable,
            "NEMOTRON_MODEL_DIR": str(model), "OUTPUT_ROOT": str(self.output),
            "STAGE_ROOT": str(self.stages), "DEEPSEEK_ROOT": str(self.root / "deepseek"),
            "DESCRIPTIVE_SCORE_ROOT": str(self.root / "semantic"),
            "DEEPSEEK_API_FILE": str(api), "SCORING_API_FILE": str(api),
        })

    @staticmethod
    def executable(path, content):
        path.write_text(content)
        path.chmod(0o755)

    def launch(self, script="run_sure_tagger_nemotron_benchmark.sh", expect=0, **variables):
        env = dict(self.env, **{name: str(value) for name, value in variables.items()})
        result = subprocess.run(["bash", str(self.eval / script)], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=45)
        logs = "\n".join(path.read_text() for path in self.output.glob("logs/*.log"))
        self.assertEqual(result.returncode, expect, result.stdout + "\n" + logs)
        return result

    def events(self):
        if not self.events_path.exists():
            return []
        return [json.loads(line) for line in self.events_path.read_text().splitlines()]

    def summary(self, name="finalize.summary.json"):
        return json.loads((self.output / name).read_text())

    def keep_first_samples(self, count):
        for directory, filename in (("runtime", "runtime.manifest.jsonl"),
                                    ("benchmark", "benchmark.manifest.jsonl")):
            manifest = self.root / directory / filename
            manifest.write_text("\n".join(manifest.read_text().splitlines()[:count]) + "\n")

    def test_single_gpu_uses_selected_physical_card_for_core_aux_and_speaker_retry(self):
        self.launch(GPU_COUNT=1, GPU_ID=3, TEST_SPEAKER_RETRY=1)
        events = self.events()
        self.assertEqual([row["kind"] for row in events], ["core", "aux", "retry"])
        self.assertEqual({row["gpu"] for row in events}, {"3"})
        self.assertEqual({row["manifest"] for row in events}, {"manifest_00.jsonl"})
        self.assertEqual({row["workers"] for row in events}, {1})
        self.assertEqual({row["gender_device"] for row in events}, {"cuda:0"})
        self.assertEqual([row["device"] for row in events if row["kind"] != "aux"], ["cuda:0", "cuda:0"])
        self.assertEqual(len({row["pid"] for row in events}), 3)
        self.assertEqual(self.summary()["shard_counts"], [650])
        self.assertEqual(self.summary()["gpu_count"], 1)

    def test_eight_gpu_defaults_dispatch_core_then_aux_per_card(self):
        self.launch()
        events = self.events()
        self.assertEqual(len(events), 16)
        self.assertEqual({row["gpu"] for row in events}, {str(number) for number in range(8)})
        for gpu in range(8):
            selected = [row for row in events if row["gpu"] == str(gpu)]
            self.assertEqual([row["kind"] for row in selected], ["core", "aux"])
            self.assertEqual({row["workers"] for row in selected}, {2})
            self.assertEqual({row["manifest"] for row in selected}, {"manifest_%02d.jsonl" % gpu})
        self.assertEqual(self.summary()["gpu_count"], 8)
        self.assertEqual(sum(self.summary()["shard_counts"]), 650)

    def test_resume_skips_validated_core_and_aux(self):
        self.launch(GPU_COUNT=1, GPU_ID=3)
        before = self.events()
        self.launch(GPU_COUNT=1, GPU_ID=3, RESUME_EXISTING=1)
        self.assertEqual(self.events(), before)

    def test_resume_reruns_invalid_aux_without_rerunning_core(self):
        self.launch(GPU_COUNT=1)
        (self.stages / "aux" / "tags_00.jsonl").write_text("{}\n")
        self.launch(GPU_COUNT=1, RESUME_EXISTING=1)
        self.assertEqual([row["kind"] for row in self.events()], ["core", "aux", "aux"])
        self.assertEqual(self.summary()["sample_count"], 650)

    def test_failed_core_retries_with_one_worker_before_aux(self):
        self.launch(GPU_COUNT=1, GPU_ID=3, SAMPLE_WORKERS=2, TEST_FAIL_FIRST_CORE=1)
        events = self.events()
        self.assertEqual([row["kind"] for row in events], ["core", "core", "aux"])
        self.assertEqual([row["workers"] for row in events], [2, 1, 2])
        self.assertEqual({row["gpu"] for row in events}, {"3"})

    def test_missing_selected_gpu_fails_before_inference(self):
        self.launch(GPU_COUNT=1, GPU_ID=3, TEST_GPUS="0 1", expect=2)
        self.assertEqual(self.events(), [])
        self.assertFalse(self.output.exists())

    def test_full_single_gpu_wrapper_runs_downstream_and_reports_one_gpu(self):
        self.launch("run_sure_tagger_pipeline_1.sh", GPU_ID=3, GPU_COUNT=8)
        summary = self.summary("pipeline.full.summary.json")
        self.assertEqual(summary["gpu_count"], 1)
        self.assertEqual(summary["gpu_ids"], [3])
        self.assertEqual(summary["sample_workers_per_gpu"], 1)
        self.assertEqual(summary["sample_count"], 650)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual([(row["kind"], row["workers"]) for row in self.events()],
                         [("core", 1), ("aux", 1), ("descriptions", 300), ("semantic_scores", 300)])
        self.assertEqual(self.events()[2]["prompt"], "v1")

    def test_full_single_gpu_wrapper_runs_three_sample_subset(self):
        self.keep_first_samples(3)
        self.launch("run_sure_tagger_pipeline_1.sh", GPU_ID=3, EXPECTED_SAMPLE_COUNT=3)
        summary = self.summary("pipeline.full.summary.json")
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["evaluation_scope"], "benchmark_subset")
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["semantic_score_denominator_sample_count"], 3)
        self.assertEqual(summary["gpu_count"], 1)
        self.assertEqual(summary["gpu_ids"], [3])
        self.assertEqual(self.summary()["shard_counts"], [3])
        self.assertEqual([row["kind"] for row in self.events()],
                         ["core", "aux", "descriptions", "semantic_scores"])
        results = [json.loads(line) for line in
                   (self.output / "result_with_id_descriptive.jsonl").read_text().splitlines()]
        self.assertEqual([row["sample_id"] for row in results],
                         ["sample_0000", "sample_0001", "sample_0002"])

    def test_three_sample_subset_requires_explicit_expected_count(self):
        self.keep_first_samples(3)
        for script, expected_exit, diagnostic in (
            ("run_sure_tagger_pipeline_1.sh", 1, "expected 650 benchmark samples, got 3"),
            ("run_sure_tagger_nemotron_benchmark.sh", 2, "expected 650 audio-only benchmark rows"),
        ):
            with self.subTest(script=script):
                result = self.launch(script, GPU_COUNT=1, expect=expected_exit)
                self.assertIn(diagnostic, result.stdout)
                self.assertEqual(self.events(), [])
                self.assertFalse(self.output.exists())

    def test_subset_rejects_full_gt_config_before_inference(self):
        self.keep_first_samples(3)
        gt_root = Path(self.env["GT_ROOT"])
        gt_root.mkdir()
        (gt_root / "run_config.json").write_text(json.dumps({"sample_count": 650}))
        result = self.launch("run_sure_tagger_pipeline_1.sh", EXPECTED_SAMPLE_COUNT=3, expect=1)
        self.assertIn("descriptive GT run_config sample_count differs from benchmark", result.stdout)
        self.assertEqual(self.events(), [])
        self.assertFalse(self.output.exists())
        self.assertFalse(self.stages.exists())

    def test_invalid_expected_count_fails_before_inference(self):
        for script in ("run_sure_tagger_pipeline_1.sh", "run_sure_tagger_nemotron_benchmark.sh"):
            for count in ("0", "-1", "abc", "3.0"):
                with self.subTest(script=script, count=count):
                    result = self.launch(script, GPU_COUNT=1, EXPECTED_SAMPLE_COUNT=count, expect=2)
                    self.assertIn("EXPECTED_SAMPLE_COUNT must be positive integers", result.stdout)
                    self.assertEqual(self.events(), [])
                    self.assertFalse(self.output.exists())

    def test_full_eight_gpu_wrapper_preserves_default_topology_and_api_workers(self):
        self.launch("run_sure_tagger_pipeline_8.sh", GPU_COUNT=1, GPU_ID=3)
        summary = self.summary("pipeline.full.summary.json")
        self.assertEqual(summary["gpu_count"], 8)
        self.assertEqual(summary["gpu_ids"], list(range(8)))
        self.assertEqual(summary["sample_workers_per_gpu"], 2)
        self.assertEqual(summary["deepseek_workers"], 300)
        self.assertEqual(summary["scoring_workers"], 300)
        self.assertEqual(summary["prompt_version"], "v1")
        self.assertEqual(len([row for row in self.events() if row["kind"] == "core"]), 8)

    def test_single_gpu_entry_rejects_resume_of_eight_gpu_structural_results(self):
        self.launch(GPU_COUNT=8)
        before = self.events()
        self.launch("run_sure_tagger_pipeline_1.sh", SKIP_STRUCTURAL=1,
                    RESUME_EXISTING=1, expect=1)
        self.assertEqual(self.events(), before)
        self.assertFalse((self.output / "pipeline.full.summary.json").exists())


if __name__ == "__main__":
    unittest.main()
