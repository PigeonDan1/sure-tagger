import json
import os
import tempfile
import unittest

from sure_tagger.cli import build_parser, cmd_tag, pipeline_tag_names
from sure_tagger.datasets.ami import parse_words_file, resolve_href
from sure_tagger.datasets.ami_utterance import (
    build_utterance_manifest,
    record_from_utterance,
)
from sure_tagger.io.jsonl import JsonlWriter, read_jsonl
from sure_tagger.meeting_manifest import aggregate_meeting_records
from sure_tagger.schemas import validate_manifest_record
from sure_tagger.tags.topic import (
    is_non_content_utterance,
    load_taxonomy,
    merge_chunk_payloads,
    resolve_topic_schema_path,
    split_text_for_llm,
    tag as topic_tag,
    validate_topic_payload,
)
from sure_tagger.llm.prompts import build_topic_prompt
from sure_tagger.text.context import ContextBuilder
from sure_tagger.text.normalize import join_tokens
from sure_tagger.text.tokenizer import punctuation_counts, word_tokens


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PipelineSmokeTest(unittest.TestCase):
    def test_join_tokens(self):
        text = join_tokens(["Hi", ",", "I", "'m", "David", "."])
        self.assertIn("Hi,", text)
        self.assertTrue(text.endswith("."))

    def test_word_and_punctuation_counts(self):
        text = "Hi, I'm David."
        self.assertEqual(len(word_tokens(text)), 3)
        counts = punctuation_counts(text)
        self.assertEqual(counts["comma"], 1)
        self.assertTrue(counts["has_terminal_punctuation"])

    def test_ami_href_resolution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            words_path = os.path.join(tmpdir, "words.xml")
            with open(words_path, "w", encoding="utf-8") as f:
                f.write(
                    "<root>"
                    "<w id=\"w0\" starttime=\"0.0\" endtime=\"0.1\">Hi</w>"
                    "<w id=\"w1\" starttime=\"0.1\" endtime=\"0.2\">there</w>"
                    "<w id=\"w2\" punc=\"true\" starttime=\"0.2\" endtime=\"0.2\">.</w>"
                    "</root>"
                )
            words = parse_words_file(words_path)
        items = resolve_href("words.xml#id(w0)..id(w2)", words)
        self.assertEqual(items[0]["text"], "Hi")
        self.assertEqual(items[-1]["text"], ".")

    def test_manifest_record_shape(self):
        record = {
            "corpus": {"dataset_name": "X"},
            "sample": {
                "sample_id": "X:1",
                "audio": {"path": "a.wav", "start_sec": 0.0, "end_sec": 1.0},
                "text": {"transcript": "hello"},
                "native_metadata": {},
                "provenance": {"source_path": "x", "source_split": ""},
            },
        }
        self.assertTrue(validate_manifest_record(record))

    def test_topic_taxonomy_validation(self):
        taxonomy = load_taxonomy(os.path.join(ROOT, "configs/topic_taxonomy_general.yaml"))
        payload = {
            "major_topic": "academic_research",
            "minor_topic": "computer_science",
            "confidence": 0.9,
            "topic_keywords": ["ASR"],
            "proper_nouns": ["AMI"],
            "reason_short": "test",
        }
        clean = validate_topic_payload(payload, taxonomy)
        self.assertEqual(clean["major_topic"], "academic_research")

    def test_meeting_manifest_aggregation(self):
        records = [
            {
                "corpus": {"dataset_name": "AMI", "native_metadata": {}},
                "sample": {
                    "sample_id": "AMI:M1:A:s2",
                    "audio": {"path": "m1.wav", "start_sec": 10.0, "end_sec": 12.0},
                    "text": {"transcript": "second turn"},
                    "native_metadata": {"meeting_id": "M1", "speaker_id": "A", "meeting_type": "scenario"},
                    "provenance": {"source_path": "segments.xml", "source_split": ""},
                },
            },
            {
                "corpus": {"dataset_name": "AMI", "native_metadata": {}},
                "sample": {
                    "sample_id": "AMI:M1:B:s1",
                    "audio": {"path": "m1.wav", "start_sec": 1.0, "end_sec": 2.0},
                    "text": {"transcript": "first turn"},
                    "native_metadata": {"meeting_id": "M1", "speaker_id": "B", "meeting_type": "scenario"},
                    "provenance": {"source_path": "segments.xml", "source_split": ""},
                },
            },
        ]
        meetings = aggregate_meeting_records(records, source_manifest_path="manifest.jsonl")
        self.assertEqual(len(meetings), 1)
        meeting = meetings[0]
        self.assertTrue(validate_manifest_record(meeting))
        self.assertEqual(meeting["sample"]["sample_id"], "AMI:M1")
        self.assertEqual(meeting["sample"]["native_metadata"]["granularity"], "meeting")
        self.assertEqual(meeting["sample"]["native_metadata"]["source_sample_count"], 2)
        self.assertTrue(meeting["sample"]["text"]["speaker_labeled_transcript"].splitlines()[0].endswith("B: first turn"))

    def test_topic_chunk_split_and_merge(self):
        chunks = split_text_for_llm("A: alpha\nB: beta\nC: gamma", 12)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 12 for chunk in chunks))
        taxonomy = load_taxonomy(os.path.join(ROOT, "configs/topic_taxonomy_general.yaml"))
        merged = merge_chunk_payloads([
            {
                "major_topic": "technology_engineering",
                "minor_topic": "product_design",
                "confidence": 0.8,
                "topic_keywords": ["remote", "design"],
                "proper_nouns": ["AMI"],
                "reason_short": "test",
            },
            {
                "major_topic": "business_management",
                "minor_topic": "project_management",
                "confidence": 0.5,
                "topic_keywords": ["schedule"],
                "proper_nouns": [],
                "reason_short": "test",
            },
        ], taxonomy, [100, 20])
        self.assertEqual(merged["major_topic"], "technology_engineering")
        self.assertEqual(merged["minor_topic"], "product_design")

    def test_topic_schema_can_be_disabled_for_compatible_gateways(self):
        self.assertIsNone(resolve_topic_schema_path({"use_json_schema": False}))
        self.assertIsNone(resolve_topic_schema_path({"model": {"use_json_schema": "false"}}))
        self.assertEqual(
            resolve_topic_schema_path({"schema_path": "custom_schema.json"}),
            "custom_schema.json",
        )

    def test_run_meeting_pipeline_tag_defaults(self):
        parser = build_parser()
        args = parser.parse_args([
            "run-meeting-pipeline",
            "--dataset", "ami",
            "--root", os.path.join(ROOT, "AMI"),
        ])
        self.assertEqual(pipeline_tag_names(args), ["language", "word_count", "punctuation", "filler", "repetition"])
        args = parser.parse_args([
            "run-meeting-pipeline",
            "--dataset", "ami",
            "--root", os.path.join(ROOT, "AMI"),
            "--include-topic",
        ])
        self.assertIn("topic", pipeline_tag_names(args))

    def test_ami_utterance_record_mapping(self):
        raw = {
            "utt_id": "ES2002a_utt_00002",
            "audio_id": "ES2002a",
            "speaker": "B",
            "start": 55.415,
            "end": 77.456,
            "text": "Um well this is the kick-off meeting.",
            "words": [{"w": "Um", "start": 55.98, "end": 56.53}],
        }

        record = record_from_utterance(raw, "/tmp/ES2002a.jsonl", 3)

        self.assertTrue(validate_manifest_record(record))
        self.assertEqual(record["sample"]["sample_id"], "ES2002a_utt_00002")
        self.assertEqual(record["sample"]["audio"], {"path": ""})
        self.assertEqual(record["sample"]["native_metadata"]["speaker"], "B")
        self.assertEqual(record["sample"]["native_metadata"]["start"], 55.415)

    def test_build_utterance_manifest_from_jsonl_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = os.path.join(tmpdir, "all")
            os.makedirs(root)
            source_path = os.path.join(root, "ES2002a.jsonl")
            rows = [
                {
                    "utt_id": "ES2002a_utt_00000",
                    "audio_id": "ES2002a",
                    "speaker": "B",
                    "start": 1.0,
                    "end": 2.0,
                    "text": "Okay.",
                    "words": [{"w": "Okay", "start": 1.1, "end": 1.8}],
                },
                {
                    "utt_id": "ES2002a_utt_00001",
                    "audio_id": "ES2002a",
                    "speaker": "A",
                    "start": 3.0,
                    "end": 4.0,
                    "text": "Let's begin.",
                    "words": [{"w": "Let's", "start": 3.1, "end": 3.4}],
                },
            ]
            with open(source_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row))
                    f.write("\n")
            output_path = os.path.join(tmpdir, "utterance_manifest.jsonl")
            bad_path = os.path.join(tmpdir, "bad.jsonl")

            with JsonlWriter(output_path) as out, JsonlWriter(bad_path) as bad:
                stats = build_utterance_manifest(
                    root,
                    out,
                    bad_writer=bad,
                    meetings=["ES2002a"],
                )

            records = list(read_jsonl(output_path))
            self.assertEqual(stats["records"], 2)
            self.assertEqual(stats["bad_records"], 0)
            self.assertEqual(stats["meetings_seen"], 1)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["sample"]["native_metadata"]["utt_id"], "ES2002a_utt_00000")

    def test_context_builder_utterance_window(self):
        records = [
            record_from_utterance(
                {
                    "utt_id": "M1_utt_00000",
                    "audio_id": "M1",
                    "speaker": "A",
                    "start": 0.0,
                    "end": 1.0,
                    "text": "We need a remote control.",
                    "words": [],
                },
                "M1.jsonl",
                1,
            ),
            record_from_utterance(
                {
                    "utt_id": "M1_utt_00001",
                    "audio_id": "M1",
                    "speaker": "B",
                    "start": 2.0,
                    "end": 3.0,
                    "text": "The buttons should be simple.",
                    "words": [],
                },
                "M1.jsonl",
                2,
            ),
        ]
        context = ContextBuilder(records).build(records[1], meeting_window_sec=5)

        self.assertEqual(context["target_granularity"], "utterance")
        self.assertEqual(context["meeting_id"], "M1")
        self.assertEqual(context["speaker_id"], "B")
        self.assertEqual(context["evidence_scope"], "utterance_context")
        self.assertIn("remote control", context["neighbor_window_text"])

    def test_run_utterance_pipeline_tag_defaults(self):
        parser = build_parser()
        args = parser.parse_args([
            "run-utterance-pipeline",
            "--dataset", "ami_utterance",
            "--root", "/tmp/all",
        ])
        self.assertEqual(pipeline_tag_names(args), ["language", "word_count", "punctuation", "filler", "repetition"])
        args = parser.parse_args([
            "run-utterance-pipeline",
            "--dataset", "ami_utterance",
            "--root", "/tmp/all",
            "--include-topic",
        ])
        self.assertIn("topic", pipeline_tag_names(args))
        args = parser.parse_args([
            "run-utterance-pipeline",
            "--dataset", "ami_utterance",
            "--root", "/tmp/all",
            "--workers", "5",
        ])
        self.assertEqual(args.workers, 5)

    def test_cmd_tag_parallel_workers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "manifest.jsonl")
            output_path = os.path.join(tmpdir, "tags.jsonl")
            records = [
                record_from_utterance(
                    {
                        "utt_id": "M1_utt_00000",
                        "audio_id": "M1",
                        "speaker": "A",
                        "start": 0.0,
                        "end": 1.0,
                        "text": "Hello there.",
                        "words": [],
                    },
                    "M1.jsonl",
                    1,
                ),
                record_from_utterance(
                    {
                        "utt_id": "M1_utt_00001",
                        "audio_id": "M1",
                        "speaker": "B",
                        "start": 1.0,
                        "end": 2.0,
                        "text": "Okay.",
                        "words": [],
                    },
                    "M1.jsonl",
                    2,
                ),
            ]
            with JsonlWriter(manifest_path) as out:
                for rec in records:
                    out.write(rec)

            cmd_tag(type("Args", (object,), {
                "manifest": manifest_path,
                "config": None,
                "tags": "language,word_count",
                "output": output_path,
                "bad_samples": None,
                "report": None,
                "limit": None,
                "run_id": "parallel_test",
                "topic_provider": None,
                "dry_run": False,
                "workers": 2,
            })())

            tagged = list(read_jsonl(output_path))
            self.assertEqual(len(tagged), 2)
            self.assertEqual(tagged[0]["sample_id"], records[0]["sample"]["sample_id"])
            self.assertIn("language", tagged[0]["tags"])

    def test_topic_non_content_utterance_guard(self):
        record = record_from_utterance(
            {
                "utt_id": "M1_utt_00000",
                "audio_id": "M1",
                "speaker": "A",
                "start": 0.0,
                "end": 1.0,
                "text": "Yeah. Yeah.",
                "words": [],
            },
            "M1.jsonl",
            1,
        )
        config = {
            "taxonomy_path": os.path.join(ROOT, "configs/topic_taxonomy_general.yaml"),
        }
        context = {
            "target_granularity": "utterance",
            "neighbor_window_text": "B: We are designing a remote control prototype.",
        }
        result = topic_tag(record, config, context=context)

        self.assertTrue(is_non_content_utterance("Mm-hmm.", config))
        self.assertFalse(is_non_content_utterance("Slim.", config))
        self.assertEqual(result["method"], "deterministic_non_content_utterance_guard")
        self.assertEqual(result["value"]["major_topic"], "other")
        self.assertEqual(result["value"]["minor_topic"], "insufficient_context")

    def test_topic_prompt_utterance_rules(self):
        taxonomy = load_taxonomy(os.path.join(ROOT, "configs/topic_taxonomy_general.yaml"))
        prompt = build_topic_prompt(
            taxonomy,
            "Okay.",
            {
                "target_granularity": "utterance",
                "neighbor_window_text": "A: We are discussing the remote control prototype.",
            },
        )
        payload = json.loads(prompt)

        self.assertEqual(payload["target_granularity"], "utterance")
        self.assertTrue(any("target utterance" in rule for rule in payload["rules"]))
        self.assertFalse(any("dominant topic of the whole meeting" in rule for rule in payload["rules"]))


if __name__ == "__main__":
    unittest.main()
