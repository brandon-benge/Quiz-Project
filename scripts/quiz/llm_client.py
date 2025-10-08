#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

DEFAULT_HTTP_TIMEOUT = 240
OLLAMA_URL = "http://localhost:11434/api/generate"


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now().isoformat(timespec='seconds')


def _log(level: str, msg: str) -> None:
    print(f"[{_now_iso()}] [{level}] {msg}")


@dataclass
class DumpOptions:
    dump_prompt_path: Optional[str] = None
    dump_payload_path: Optional[str] = None
    dump_response_path: Optional[str] = None


# OpenAI support removed


@dataclass
class OllamaGeneratePayload:
    model: str
    options: Dict[str, Any]
    prompt: Optional[str] = None
    prompt_template: Optional[str] = None
    variables: Optional[Dict[str, Any]] = None
    debug_payload: bool = False
    iteration: Optional[int] = None
    theme: Optional[str] = None
    # Transport-level retry settings (not formatting retries)
    retries: int = 2
    retry_delay: float = 1.0


class LLMClient:
    """Ollama LLM client.

    Contract:
    - questions_ollama(model, prompt, options, *, debug_payload, iteration, theme, do_validate) -> (json_text, raw_response, duration)

    Raw response is provider-specific JSON-able object for optional logging.
    """

    def __init__(self, dumps: DumpOptions | None = None):
        self._requests = None
        # OpenAI removed
        self.dumps = dumps or DumpOptions()
        # Lazy dependencies
        try:
            import requests  # type: ignore
            self._requests = requests
        except Exception:
            self._requests = None
        # No OpenAI dependency

    # --------------- Common utils ---------------
    def _dump_payload(self, prompt: str, payload: dict, response: Optional[Any] = None) -> None:
        if self.dumps.dump_prompt_path:
            try:
                Path(self.dumps.dump_prompt_path).write_text(prompt, encoding='utf-8')
                _log("debug", f"Wrote full LLM prompt -> {self.dumps.dump_prompt_path}")
            except Exception as e:
                _log("warn", f"Could not write prompt: {e}")
        if self.dumps.dump_payload_path:
            try:
                with open(self.dumps.dump_payload_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
            except Exception as e:
                _log("warn", f"Could not append payload: {e}")
        if self.dumps.dump_response_path and response is not None:
            try:
                try:
                    text = json.dumps(response, indent=2, ensure_ascii=False)
                except Exception:
                    text = str(response)
                with open(self.dumps.dump_response_path, 'a', encoding='utf-8') as f:
                    f.write(text + '\n')
            except Exception as e:
                _log("warn", f"Could not append LLM response: {e}")

    def _render_text(self, inline: Optional[str], template_path: Optional[str], variables: Optional[Dict[str, Any]]) -> str:
        if template_path:
            try:
                tpl = Path(template_path).read_text(encoding='utf-8')
                from string import Template
                return Template(tpl).safe_substitute(variables or {})
            except Exception as e:
                _log("warn", f"Template render failed ({template_path}): {e}")
        return inline or ""

    # OpenAI methods removed

    # --------------- Ollama ---------------
    def run_ollama(self, payload: OllamaGeneratePayload) -> Tuple[str, Any, float]:
        if not self._requests:
            raise RuntimeError('requests package not available')
        prompt_text = self._render_text(payload.prompt, payload.prompt_template, payload.variables)
        req = {'model': payload.model, 'prompt': prompt_text, 'stream': False, 'options': payload.options}
        self._dump_payload(prompt_text, req, None)
        if payload.debug_payload:
            trunc = prompt_text[:600] + (f"... [truncated, total {len(prompt_text)} chars]" if len(prompt_text) > 600 else "")
            try:
                _log("debug", "Ollama request payload (truncated prompt):")
                print(json.dumps({**req, "prompt": trunc}, indent=2)[:4000])
            except Exception:
                pass
        start = time.time()
        last_err: Optional[Exception] = None
        for attempt in range((payload.retries or 0) + 1):
            try:
                resp = self._requests.post(OLLAMA_URL, json=req, timeout=DEFAULT_HTTP_TIMEOUT)
                if resp.status_code != 200:
                    raise RuntimeError(f'Ollama HTTP {resp.status_code}: {resp.text[:200]}')
                data = resp.json()
                break
            except Exception as e:
                last_err = e
                if attempt < (payload.retries or 0):
                    delay = (payload.retry_delay or 1.0) * (2 ** attempt)
                    time.sleep(delay)
                    continue
                raise
        content = data.get('response', '') or ''
        m = re.search(r"```json\n(.*)```", content, re.DOTALL)
        json_text = m.group(1) if m else content
        duration = time.time() - start
        iter_str = f"[{(payload.iteration or 0)+1}/?] " if payload.iteration is not None else ""
        theme_str = f", theme: {payload.theme}" if payload.theme else ""
        _log("info", f"{iter_str}LLM response time (ollama {payload.model}{theme_str}): {duration:.2f}s")
        self._dump_payload(prompt_text, req, data)
        return json_text, data, duration

    def questions_ollama(self, *, model: str, prompt: str, options: Dict[str, Any],
                          debug_payload: bool, iteration: Optional[int], theme: Optional[str],
                          do_validate: bool) -> Tuple[str, Any, float]:
        payload = OllamaGeneratePayload(
            model=model,
            prompt=prompt,
            options=options,
            debug_payload=debug_payload,
            iteration=iteration,
            theme=theme,
        )
        return self.run_ollama(payload)
