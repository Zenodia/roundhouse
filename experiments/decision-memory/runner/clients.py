# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model clients. Three, all returning the same `Reply`:

    responses  Roundhouse's /v1/responses (OpenAI Responses dialect), streamed;
               usage carries input_tokens_details.cached_tokens, which is
               Roundhouse's own cache view (Dynamo KV events for a local
               worker, the provider's report for a hosted one).
    chat       any OpenAI-compatible /v1/chat/completions: Dynamo's frontend
               directly, NVIDIA's inference API, OpenAI.
    stub       no model: returns the reference solution (`stub:reference`) or
               nothing (`stub:none`), so the whole loop runs on a CPU.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Reply:
    text: str
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    model: str | None = None
    error: str | None = None
    raw_usage: dict = field(default_factory=dict)


class ResponsesClient:
    def __init__(self, base_url: str, key: str | None = None, key_header: str = "x-roundhouse-key", model: str | None = None, timeout: int = 900):
        self.base_url = base_url.rstrip("/")
        self.key = key
        self.key_header = key_header
        self.model = model
        self.timeout = timeout

    def complete(self, instructions: str, user: str, cache_key: str) -> Reply:
        body = {"instructions": instructions, "input": [{"type": "message", "role": "user", "content": user}], "stream": True, "prompt_cache_key": cache_key}
        if self.model:
            body["model"] = self.model
        headers = {"content-type": "application/json"}
        if self.key:
            headers[self.key_header] = self.key
        req = urllib.request.Request(f"{self.base_url}/v1/responses", data=json.dumps(body).encode(), method="POST", headers=headers)
        text, usage, model, error = [], {}, None, None
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        ev = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    et = ev.get("type", "")
                    if et == "response.output_text.delta":
                        text.append(ev.get("delta", ""))
                    elif et == "response.completed":
                        r = ev.get("response", {})
                        usage = r.get("usage", {}) or {}
                        model = r.get("model")
                    elif et == "response.failed":
                        error = json.dumps(ev.get("response", {}).get("error", ev))
        except urllib.error.HTTPError as exc:
            error = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:500]}"
        except urllib.error.URLError as exc:
            error = f"connection error: {exc}"
        return Reply(
            text="".join(text),
            input_tokens=int(usage.get("input_tokens", 0)),
            cached_tokens=int((usage.get("input_tokens_details") or {}).get("cached_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            latency_s=time.perf_counter() - t0,
            model=model,
            error=error,
            raw_usage=usage,
        )


class ChatClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: int = 900, max_tokens: int = 6000, temperature: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(self, instructions: str, user: str, cache_key: str) -> Reply:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": instructions}, {"role": "user", "content": user}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/v1/chat/completions", data=json.dumps(body).encode(), method="POST", headers=headers)
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return Reply("", latency_s=time.perf_counter() - t0, error=f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:500]}")
        except urllib.error.URLError as exc:
            return Reply("", latency_s=time.perf_counter() - t0, error=f"connection error: {exc}")
        usage = data.get("usage", {}) or {}
        msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
        text = msg.get("content") or ""
        details = usage.get("prompt_tokens_details") or {}
        return Reply(
            text=text,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            cached_tokens=int(details.get("cached_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            latency_s=time.perf_counter() - t0,
            model=data.get("model"),
            raw_usage=usage,
        )


class StubClient:
    """`stub:reference` answers with the reference files for the task's
    `files`; `stub:none` answers with no files (the agent did nothing)."""

    def __init__(self, mode: str, reference_dir: Path, task_files: list[str] | None = None):
        self.mode = mode
        self.reference_dir = reference_dir
        self.task_files = task_files or []

    def complete(self, instructions: str, user: str, cache_key: str) -> Reply:
        if self.mode == "none":
            return Reply("No changes.", input_tokens=len(instructions) // 4 + len(user) // 4)
        blocks = []
        for rel in self.task_files:
            p = self.reference_dir / Path(rel).relative_to("ledgerline")
            if p.exists():
                blocks.append(f"<<<FILE {rel}>>>\n{p.read_text(encoding='utf-8')}\n<<<END FILE>>>")
        return Reply("\n\n".join(blocks), input_tokens=len(instructions) // 4 + len(user) // 4, output_tokens=sum(len(b) for b in blocks) // 4, model="stub:reference")


def make_client(spec: str, model: str | None, reference_dir: Path, task_files: list[str]):
    """`spec` is `responses:<base_url>`, `chat:<base_url>` or `stub:<mode>`."""
    kind, _, rest = spec.partition(":")
    if kind == "stub":
        return StubClient(rest or "reference", reference_dir, task_files)
    if kind == "responses":
        return ResponsesClient(rest, key=os.environ.get("ROUNDHOUSE_API_KEY"), model=model)
    if kind == "chat":
        if not model:
            raise SystemExit("--model is required for a chat client")
        return ChatClient(rest, model, api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("INFERENCE_API_KEY"))
    raise SystemExit(f"unknown client spec {spec!r}; use responses:<url>, chat:<url> or stub:<reference|none>")
