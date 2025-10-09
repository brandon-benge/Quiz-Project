from __future__ import annotations

import time
import uuid
from pathlib import Path
import os
import random
from typing import Any, Dict, List, Optional

# Support running as a module or as a script
try:
    from .config import Config  # type: ignore
    from .llm_client import LLMClient, DumpOptions, OllamaGeneratePayload  # type: ignore
    from .questions import Question  # type: ignore
    from .utils import log, _parse_model_questions  # type: ignore
    from .rag import RAG  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    import sys
    from pathlib import Path as _P
    sys.path.append(str(_P(__file__).parent))
    from config import Config  # type: ignore
    from llm_client import LLMClient, DumpOptions, OllamaGeneratePayload  # type: ignore
    from questions import Question  # type: ignore
    from utils import log, _parse_model_questions  # type: ignore
    from rag import RAG  # type: ignore


class Providers:
    """Provider wrapper (Ollama-only) responsible for prompt rendering and parsing with retries."""

    def __init__(self, cfg: Config, rag: Optional["RAG"] = None):
        self.cfg = cfg
        # Optional RAG instance for retry-time context rebuilding
        self.rag = rag
        self._client = LLMClient(
            DumpOptions(
                dump_prompt_path=cfg.dump_ollama_prompt,
                dump_payload_path=cfg.dump_llm_payload,
                dump_response_path=cfg.dump_llm_response,
            ),
            base_url=getattr(cfg, 'ollama_url', None),
            http_timeout=getattr(cfg, 'http_timeout', None),
        )

    def ollama_questions(self, files: Dict[str, str], count: int, model: str,
                         token: str, recent_norm: List[str], temperature: float,
                         *, snippet_chars: int, corpus_chars: int, num_predict: Optional[int],
                         top_k: Optional[int], top_p: Optional[float], compact_json: bool,
                         debug_payload: bool, iteration: Optional[int] = None, theme: Optional[str] = None,
                         themes: Optional[List[str]] = None, theme_index: Optional[int] = None) -> List[Question]:
        parts, total = [], 0
        max_chars = None if corpus_chars == -1 else 28000
        for pth, txt in files.items():
            trimmed = txt if snippet_chars == -1 or pth == 'RAG_CONTEXT.md' else txt[:snippet_chars]
            part = f"\n# FILE: {pth}\n{trimmed}\n"
            if corpus_chars != -1 and total + len(part) > corpus_chars:
                break
            if max_chars is not None and total + len(part) > max_chars:
                break
            parts.append(part)
            total += len(part)
        corpus = ''.join(parts)
        window = self.cfg.avoid_recent_window
        recent_clause = ("Avoid reusing these prior question phrasings: " + '; '.join(recent_norm[-window:])) if recent_norm else ''
        if compact_json:
            style_clause = 'Return ONLY a strict compact JSON object; no markdown, no extra text.'
        else:
            style_clause = 'Return ONLY a strict JSON object; no markdown, no extra text.'
        question_index = (iteration + 1) if iteration is not None else 1
        # Ensure theme is a single scalar string for both logging and template variables
        safe_theme: Optional[str] = None
        if theme is not None:
            if isinstance(theme, (list, tuple)):
                safe_theme = str(theme[0]) if len(theme) > 0 else None
            elif isinstance(theme, str):
                t = theme.strip()
                if t.startswith('[') and t.endswith(']'):
                    try:
                        import json as _json
                        arr = _json.loads(t)
                        if isinstance(arr, list) and arr:
                            safe_theme = str(arr[0])
                        else:
                            safe_theme = t
                    except Exception:
                        safe_theme = t
                else:
                    safe_theme = t
            else:
                safe_theme = str(theme)

        variables_base = {
            'token': token,
            'question_index': question_index,
            'recent_clause': recent_clause,
            'corpus': (corpus if corpus_chars == -1 else corpus[:corpus_chars]),
            'style_clause': style_clause,
            'iteration': iteration,
            'count': count,
            'model': model,
            'theme': safe_theme,
        }
        options: Dict[str, Any] = {'temperature': temperature}
        if num_predict is not None: options['num_predict'] = num_predict
        if top_k is not None: options['top_k'] = top_k
        if top_p is not None: options['top_p'] = top_p
        tpl_dir = Path(__file__).parent / 'templates'
        prompt_tpl = str(tpl_dir / 'ollama_prompt.tmpl')
        payload = OllamaGeneratePayload(
            model=model,
            options=options,
            prompt_template=prompt_tpl,
            variables=variables_base,
            debug_payload=debug_payload,
            iteration=iteration,
            theme=safe_theme,
            retries=self.cfg.llm_retries,
            keep_alive=self.cfg.ollama_keep_alive,
        )
        json_text, data, _ = self._client.run_ollama(payload)
        max_retries = getattr(self.cfg, 'max_retries', 2)
        attempt = 0
        while attempt <= max_retries:
            try:
                questions = _parse_model_questions(json_text, provider='ollama')
                for q in questions:
                    q.raw_response = data
                return questions
            except Exception as e:
                log("warn", f"LLM output parsing failed (attempt {attempt+1}/{max_retries+1}): {e}")
                attempt += 1
                if attempt <= max_retries:
                    try:
                        nonce = uuid.uuid4().hex[:8]
                    except Exception:
                        nonce = str(int(time.time() * 1000))
                    variables_retry = dict(variables_base)
                    variables_retry['retry_nonce'] = nonce
                    new_theme = None
                    new_files = None
                    # Use existing themes list to move to the next theme; do not refetch
                    try:
                        if themes and theme_index is not None and self.rag and self.rag._document_store:
                            next_idx = (theme_index + attempt) % len(themes)
                            new_theme = themes[next_idx]
                            new_files = self.rag.get_blocks_for_tag(new_theme, getattr(self.cfg, 'rag_k', 5)) or None
                    except Exception as _e:
                        log("debug", f"Retry-time next-theme/context rebuild failed: {_e}")
                    # Do not fall back to modifying the theme text; only switch if a next theme is available
                    if new_theme:
                        variables_retry['theme'] = new_theme
                    # Rebuild corpus from new_files if provided
                    if new_files:
                        parts, total = [], 0
                        max_chars = None if getattr(self.cfg, 'ollama_corpus_chars', -1) == -1 else 28000
                        for pth, txt in new_files.items():
                            trimmed = txt if getattr(self.cfg, 'ollama_snippet_chars', -1) == -1 or pth == 'RAG_CONTEXT.md' else txt[:getattr(self.cfg, 'ollama_snippet_chars', -1)]
                            part = f"\n# FILE: {pth}\n{trimmed}\n"
                            if getattr(self.cfg, 'ollama_corpus_chars', -1) != -1 and total + len(part) > getattr(self.cfg, 'ollama_corpus_chars', -1):
                                break
                            if max_chars is not None and total + len(part) > max_chars:
                                break
                            parts.append(part)
                            total += len(part)
                        corpus_retry = ''.join(parts)
                        variables_retry['corpus'] = (corpus_retry if getattr(self.cfg, 'ollama_corpus_chars', -1) == -1 else corpus_retry[:getattr(self.cfg, 'ollama_corpus_chars', -1)])
                    payload_retry = OllamaGeneratePayload(
                        model=model,
                        options=options,
                        prompt_template=prompt_tpl,
                        variables=variables_retry,
                        debug_payload=debug_payload,
                        iteration=iteration,
                        theme=new_theme or safe_theme,
                        retries=self.cfg.llm_retries,
                        keep_alive=self.cfg.ollama_keep_alive,
                    )
                    json_text, data, _ = self._client.run_ollama(payload_retry)
                else:
                    raise
