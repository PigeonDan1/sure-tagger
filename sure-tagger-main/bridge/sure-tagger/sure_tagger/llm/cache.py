import hashlib
import json
import os
import threading


_CACHE_LOCK = threading.Lock()


def stable_hash(obj):
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class JsonlCache(object):
    def __init__(self, path):
        self.path = path
        self.items = {}
        if path and os.path.exists(path):
            with _CACHE_LOCK:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except ValueError:
                            continue
                        key = rec.get("key")
                        if key:
                            self.items[key] = rec.get("value")

    def get(self, key):
        return self.items.get(key)

    def set(self, key, value):
        if not self.path:
            self.items[key] = value
            return
        parent = os.path.dirname(self.path)
        with _CACHE_LOCK:
            if parent and not os.path.exists(parent):
                os.makedirs(parent)
            self.items[key] = value
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False, sort_keys=True))
                f.write("\n")
