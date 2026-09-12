def sample_sort_key(record):
    sample = record.get("sample", {})
    meta = sample.get("native_metadata", {})
    start = sample_start(sample)
    if start is None:
        start = -1
    return (
        meeting_id_from_metadata(meta),
        float(start),
        speaker_id_from_metadata(meta),
        sample.get("sample_id", ""),
    )


def meeting_id_from_metadata(meta):
    return meta.get("meeting_id") or meta.get("audio_id") or ""


def speaker_id_from_metadata(meta):
    return meta.get("speaker_id") or meta.get("speaker") or ""


def sample_start(sample):
    meta = sample.get("native_metadata", {})
    audio = sample.get("audio", {})
    return audio.get("start_sec", meta.get("start"))


def sample_end(sample):
    meta = sample.get("native_metadata", {})
    audio = sample.get("audio", {})
    return audio.get("end_sec", meta.get("end"))


def sample_granularity(meta):
    if meta.get("granularity"):
        return meta.get("granularity")
    if meta.get("utt_id"):
        return "utterance"
    if meta.get("utterances"):
        return "utterance"
    return "sample"


class ContextBuilder(object):
    def __init__(self, records):
        self.records = sorted(records, key=sample_sort_key)
        self.by_meeting = {}
        for rec in self.records:
            meta = rec["sample"]["native_metadata"]
            meeting_id = meeting_id_from_metadata(meta)
            self.by_meeting.setdefault(meeting_id, []).append(rec)

    def build(self, record, meeting_window_sec=120, speaker_neighbor_segments=3, max_context_chars=6000):
        sample = record["sample"]
        meta = sample["native_metadata"]
        meeting_id = meeting_id_from_metadata(meta)
        speaker_id = speaker_id_from_metadata(meta)
        start = sample_start(sample)
        end = sample_end(sample)
        current_text = sample.get("text", {}).get("transcript", "")
        meeting_records = self.by_meeting.get(meeting_id, [])
        granularity = sample_granularity(meta)

        meeting_window = []
        speaker_window = []
        if start is not None and end is not None:
            lo = float(start) - float(meeting_window_sec)
            hi = float(end) + float(meeting_window_sec)
            for rec in meeting_records:
                rec_sample = rec["sample"]
                rec_start = sample_start(rec_sample)
                rec_end = sample_end(rec_sample)
                if rec_start is None or rec_end is None:
                    continue
                if float(rec_end) >= lo and float(rec_start) <= hi:
                    meeting_window.append(rec)

        same_speaker = [
            r
            for r in meeting_records
            if speaker_id_from_metadata(r["sample"]["native_metadata"]) == speaker_id
        ]
        ids = [r["sample"]["sample_id"] for r in same_speaker]
        try:
            idx = ids.index(sample["sample_id"])
            lo_i = max(0, idx - int(speaker_neighbor_segments))
            hi_i = min(len(same_speaker), idx + int(speaker_neighbor_segments) + 1)
            speaker_window = same_speaker[lo_i:hi_i]
        except ValueError:
            speaker_window = [record]

        meeting_text = self._records_to_text(meeting_window, max_context_chars)
        speaker_text = self._records_to_text(speaker_window, max_context_chars // 2)
        evidence_scope = "meeting_window" if meeting_window else "sample"
        if granularity == "utterance":
            evidence_scope = "utterance_context" if meeting_window else "utterance"

        return {
            "sample_text": current_text,
            "utterance_text": current_text,
            "utterance_start_sec": start,
            "utterance_end_sec": end,
            "speaker_id": speaker_id,
            "meeting_id": meeting_id,
            "target_granularity": granularity,
            "meeting_window_text": meeting_text,
            "speaker_window_text": speaker_text,
            "neighbor_window_text": meeting_text,
            "same_speaker_window_text": speaker_text,
            "evidence_scope": evidence_scope,
            "evidence_sample_count": len(meeting_window) if meeting_window else 1,
        }

    def _records_to_text(self, records, max_chars):
        parts = []
        total = 0
        for rec in records:
            sample = rec["sample"]
            meta = sample["native_metadata"]
            text = sample.get("text", {}).get("transcript", "")
            if not text:
                continue
            prefix = "%s: " % (speaker_id_from_metadata(meta) or "?")
            piece = prefix + text
            if total + len(piece) > max_chars:
                break
            parts.append(piece)
            total += len(piece)
        return "\n".join(parts)
