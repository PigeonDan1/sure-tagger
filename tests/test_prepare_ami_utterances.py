import json
from pathlib import Path
import struct
import tempfile
import unittest
import wave

from scripts.prepare_ami_utterances import prepare_dataset
from tagger.input_schema import validate_input_record


class PrepareAmiUtterancesTests(unittest.TestCase):
    def test_groups_complete_utt_ids_and_rebases_all_times(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_audio = root / "meeting.wav"
            annotations = root / "meeting.jsonl"
            output_dir = root / "output"
            with wave.open(str(source_audio), "wb") as sink:
                sink.setnchannels(1)
                sink.setsampwidth(2)
                sink.setframerate(10)
                sink.writeframes(struct.pack("<600h", *range(600)))

            rows = [
                _utterance("utt_0", "A", 1.0, 36.0, "Long turn.", 1.5, 2.5),
                _utterance("utt_1", "B", 30.0, 34.0, "Overlap.", 31.0, 32.0),
                _utterance("utt_2", "B", 38.0, 42.0, "Next turn.", 39.0, 40.0),
                _utterance("utt_3", "A", 43.0, 48.0, "Final turn.", 44.0, 45.0),
            ]
            annotations.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            summary = prepare_dataset(annotations, source_audio, output_dir)

            self.assertEqual(summary["source_utt_id_count"], 4)
            self.assertEqual(summary["utterance_count"], 2)
            self.assertEqual(summary["over_max_duration_count"], 1)
            manifest = [
                json.loads(line)
                for line in (output_dir / "manifest.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            for record in manifest:
                validate_input_record(record)

            first = manifest[0]["sample"]
            self.assertEqual(first["sample_id"], "meeting_utterance_00000")
            self.assertEqual(
                [row["utt_id"] for row in first["native_metadata"]["utterances"]],
                ["utt_0", "utt_1"],
            )
            self.assertEqual(first["native_metadata"]["start"], 0.0)
            self.assertEqual(first["native_metadata"]["end"], 36.5)
            shifted_long = first["native_metadata"]["utterances"][0]
            self.assertEqual((shifted_long["start"], shifted_long["end"]), (0.5, 35.5))
            self.assertEqual(
                (
                    shifted_long["words"][0]["start"],
                    shifted_long["words"][0]["end"],
                ),
                (1.0, 2.0),
            )

            second = manifest[1]["sample"]
            self.assertEqual(second["native_metadata"]["end"], 11.5)
            self.assertEqual(
                [
                    (row["utt_id"], row["start"], row["end"])
                    for row in second["native_metadata"]["utterances"]
                ],
                [("utt_2", 1.0, 5.0), ("utt_3", 6.0, 11.0)],
            )

            with wave.open(
                str(output_dir / "audio" / "meeting_utterance_00000.wav"), "rb"
            ) as cut:
                self.assertEqual(cut.getnframes(), 365)
                self.assertEqual(struct.unpack("<h", cut.readframes(1))[0], 5)

    def test_refuses_to_replace_existing_output_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_audio = root / "meeting.wav"
            annotations = root / "meeting.jsonl"
            output_dir = root / "output"
            with wave.open(str(source_audio), "wb") as sink:
                sink.setnchannels(1)
                sink.setsampwidth(2)
                sink.setframerate(10)
                sink.writeframes(struct.pack("<200h", *range(200)))
            annotations.write_text(
                json.dumps(_utterance("utt_0", "A", 1.0, 15.0, "Test.", 2.0, 3.0))
                + "\n",
                encoding="utf-8",
            )

            prepare_dataset(annotations, source_audio, output_dir)
            with self.assertRaises(FileExistsError):
                prepare_dataset(annotations, source_audio, output_dir)

    def test_clamps_endpoint_padding_to_source_duration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_audio = root / "meeting.wav"
            annotations = root / "meeting.jsonl"
            output_dir = root / "output"
            with wave.open(str(source_audio), "wb") as sink:
                sink.setnchannels(1)
                sink.setsampwidth(2)
                sink.setframerate(10)
                sink.writeframes(struct.pack("<120h", *range(120)))
            annotations.write_text(
                json.dumps(_utterance("utt_0", "A", 1.0, 11.8, "Test.", 2.0, 3.0))
                + "\n",
                encoding="utf-8",
            )

            summary = prepare_dataset(annotations, source_audio, output_dir)

            self.assertEqual(summary["utterance_count"], 1)
            self.assertEqual(summary["max_duration_sec"], 11.5)
            with wave.open(str(output_dir / "audio" / "meeting_utterance_00000.wav"), "rb") as cut:
                self.assertEqual(cut.getnframes(), 115)


def _utterance(utt_id, speaker, start, end, text, word_start, word_end):
    return {
        "utt_id": utt_id,
        "audio_id": "meeting",
        "speaker": speaker,
        "start": start,
        "end": end,
        "text": text,
        "words": [{"w": text.split()[0], "start": word_start, "end": word_end}],
    }


if __name__ == "__main__":
    unittest.main()
