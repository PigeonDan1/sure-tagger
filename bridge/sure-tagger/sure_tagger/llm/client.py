import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request


class LLMError(Exception):
    pass


def _parse_toml_value(raw):
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1]
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    return raw


def load_codex_config(path=None):
    path = path or os.environ.get("CODEX_CONFIG_PATH")
    if not path:
        path = os.path.expanduser("~/.codex/config.toml")
    result = {"top": {}, "sections": {}}
    if not os.path.exists(path):
        return result
    section = "top"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip()
                result["sections"].setdefault(section, {})
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = _parse_toml_value(value)
            if section == "top":
                result["top"][key] = value
            else:
                result["sections"].setdefault(section, {})[key] = value
    return result


def extract_json_object(text):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("No JSON object found")


class LLMClient(object):
    def __init__(self, provider_config=None):
        self.config = provider_config or {}
        self.provider = self.config.get("provider", "heuristic")

    def complete_json(self, prompt, schema_path=None):
        if self.provider == "codex_cli":
            return self._complete_codex_cli(prompt, schema_path)
        if self.provider in ("openai_http", "openai_responses"):
            return self._complete_openai_responses(prompt, schema_path)
        if self.provider == "dry_run":
            raise LLMError("dry_run provider does not call a model")
        raise LLMError("Unsupported LLM provider: %s" % self.provider)

    def _read_first_line(self, path):
        if not path:
            return None
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                value = line.strip()
                if value:
                    return value
        return None

    def _complete_codex_cli(self, prompt, schema_path=None):
        codex = shutil.which("codex")
        if not codex:
            raise LLMError("codex CLI not found")
        out = tempfile.NamedTemporaryFile(prefix="sure_tagger_codex_", suffix=".json", delete=False)
        out_path = out.name
        out.close()
        cmd = [
            codex,
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--output-last-message",
            out_path,
        ]
        if schema_path:
            cmd.extend(["--output-schema", schema_path])
        cmd.extend(["-C", os.getcwd(), "-"])
        timeout = int(self.config.get("timeout_sec", 180))
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            stdout, stderr = proc.communicate(prompt.encode("utf-8"), timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            raise LLMError("codex_cli timed out after %s sec stderr=%s stdout=%s" % (
                timeout,
                stderr.decode("utf-8", "replace")[-2000:],
                stdout.decode("utf-8", "replace")[-2000:],
            ))
        if proc.returncode != 0:
            raise LLMError("codex_cli failed rc=%s stderr=%s stdout=%s" % (
                proc.returncode,
                stderr.decode("utf-8", "replace")[-2000:],
                stdout.decode("utf-8", "replace")[-2000:],
            ))
        with open(out_path, "r", encoding="utf-8") as f:
            message = f.read()
        try:
            os.unlink(out_path)
        except OSError:
            pass
        return extract_json_object(message)

    def _complete_openai_responses(self, prompt, schema_path=None):
        settings = self._resolve_openai_settings()
        api_key = settings.get("api_key")
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set and no key found in Codex config")
        model = settings.get("model")
        if not model or model == "default":
            raise LLMError("OpenAI API provider requires model name in config or OPENAI_MODEL")
        if model == "gpt5.5":
            model = "gpt-5.5"

        schema = None
        if schema_path:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)

        text_format = {"type": "json_object"}
        if schema:
            text_format = {
                "type": "json_schema",
                "name": "topic_response",
                "schema": schema,
                "strict": True,
            }

        body = {
            "model": model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": "Return strict JSON only."}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ],
            "text": {"format": text_format},
            "store": False,
        }
        if "temperature" in self.config:
            body["temperature"] = float(self.config.get("temperature", 0))
        base_url = settings.get("base_url", "https://api.openai.com/v1").rstrip("/")
        req = urllib.request.Request(
            base_url + "/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": "Bearer %s" % api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=int(self.config.get("timeout_sec", 120))) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise LLMError("OpenAI Responses API HTTP %s: %s" % (exc.code, body[:2000]))
        content = data.get("output_text")
        if not content:
            parts = []
            for item in data.get("output", []):
                for piece in item.get("content", []):
                    if piece.get("type") in ("output_text", "text"):
                        parts.append(piece.get("text", ""))
            content = "\n".join(parts)
        if not content:
            raise LLMError("OpenAI Responses API returned no output_text")
        return extract_json_object(content)

    def _resolve_openai_settings(self):
        codex = load_codex_config(self.config.get("codex_config_path"))
        top = codex.get("top", {})
        sections = codex.get("sections", {})
        provider_name = self.config.get("model_provider") or os.environ.get("OPENAI_MODEL_PROVIDER") or top.get("model_provider")
        provider_section = sections.get("model_providers.%s" % provider_name, {}) if provider_name else {}
        provider_env = sections.get("model_providers.%s.env" % provider_name, {}) if provider_name else {}
        return {
            "api_key": (
                os.environ.get("OPENAI_API_KEY")
                or self.config.get("api_key")
                or self._read_first_line(self.config.get("api_key_path"))
                or provider_env.get("OPENAI_API_KEY")
            ),
            "model": (
                os.environ.get("OPENAI_MODEL")
                or self.config.get("name")
                or top.get("model")
            ),
            "base_url": (
                os.environ.get("OPENAI_BASE_URL")
                or self.config.get("base_url")
                or provider_section.get("base_url")
                or "https://api.openai.com/v1"
            ),
            "model_provider": provider_name,
        }
