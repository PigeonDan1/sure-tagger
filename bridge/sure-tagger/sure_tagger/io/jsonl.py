import json


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError as exc:
                raise ValueError("Invalid JSON at %s:%d: %s" % (path, line_no, exc))


class JsonlWriter(object):
    def __init__(self, path):
        self.path = path
        self._f = None

    def __enter__(self):
        self._f = open(self.path, "w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._f:
            self._f.close()

    def write(self, obj):
        self._f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True))
        self._f.write("\n")
