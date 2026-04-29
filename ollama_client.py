from __future__ import annotations
import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CallRecord:
    label: str          # caller tag: "intent", "design", "implement", "raw"
    prompt_tokens: int
    completion_tokens: int
    elapsed: float


class OllamaClient:
    def __init__(self, model: str, endpoint: str = "http://localhost:11434",
                 timeout: int = 180) -> None:
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self._log: list[CallRecord] = []

    # ── stats ─────────────────────────────────────────────────
    def reset_stats(self) -> None:
        self._log.clear()

    def log(self) -> list[CallRecord]:
        return list(self._log)

    def stats_summary(self) -> dict:
        return {
            "turns":             len(self._log),
            "prompt_tokens":     sum(r.prompt_tokens     for r in self._log),
            "completion_tokens": sum(r.completion_tokens for r in self._log),
            "total_tokens":      sum(r.prompt_tokens + r.completion_tokens for r in self._log),
            "elapsed_s":         sum(r.elapsed           for r in self._log),
        }

    # ── internals ─────────────────────────────────────────────
    def _messages(self, prompt: str, system: str) -> list[dict]:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _post(self, body: dict) -> dict:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.endpoint}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.URLError as e:
            raise RuntimeError(f"ollama request failed: {e}")

    def _record(self, resp: dict, label: str, t0: float) -> None:
        self._log.append(CallRecord(
            label=label,
            prompt_tokens=resp.get("prompt_eval_count", 0),
            completion_tokens=resp.get("eval_count", 0),
            elapsed=time.time() - t0,
        ))

    # ── public API ────────────────────────────────────────────
    def complete(self, prompt: str, *, system: str = "",
                 max_tokens: int = 2000, temperature: float = 0.0,
                 label: str = "raw") -> str:
        body = {
            "model": self.model,
            "messages": self._messages(prompt, system),
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        t0 = time.time()
        resp = self._post(body)
        self._record(resp, label, t0)
        return resp["message"]["content"]

    def chat(self, messages: list[dict], *, system: str = "",
             max_tokens: int = 2000, temperature: float = 0.0,
             label: str = "raw") -> str:
        """Multi-turn: accepts full messages list (role/content dicts).
        Prepends system message if provided."""
        full = []
        if system:
            full.append({"role": "system", "content": system})
        full.extend(messages)
        body = {
            "model": self.model,
            "messages": full,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        t0 = time.time()
        resp = self._post(body)
        self._record(resp, label, t0)
        return resp["message"]["content"]

    def complete_json(self, prompt: str, *, system: str = "",
                      max_tokens: int = 2000, temperature: float = 0.0,
                      label: str = "raw") -> dict:
        body = {
            "model": self.model,
            "messages": self._messages(prompt, system),
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        t0 = time.time()
        resp = self._post(body)
        self._record(resp, label, t0)
        return json.loads(resp["message"]["content"])
