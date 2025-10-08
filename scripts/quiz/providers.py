from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Support running as a module or as a script
try:
    from .config import Config  # type: ignore
    from .llm_client import LLMClient, DumpOptions, OllamaGeneratePayload  # type: ignore
    from .questions import Question  # type: ignore
    from .utils import log, _parse_model_questions  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    import sys
    from pathlib import Path as _P
    sys.path.append(str(_P(__file__).parent))
    from config import Config  # type: ignore
    from llm_client import LLMClient, DumpOptions, OllamaGeneratePayload  # type: ignore
    from questions import Question  # type: ignore
    from utils import log, _parse_model_questions  # type: ignore


class Providers:
    """Provider wrapper (Ollama-only) responsible for prompt rendering and parsing with retries."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = LLMClient(
            DumpOptions(
                dump_prompt_path=cfg.dump_ollama_prompt,
                dump_payload_path=cfg.dump_llm_payload,
                dump_response_path=cfg.dump_llm_response,
            )
        )

    def ollama_questions(self, files: Dict[str, str], count: int, model: str,
                         token: str, recent_norm: List[str], temperature: float,
                         *, snippet_chars: int, corpus_chars: int, num_predict: Optional[int],
                         top_k: Optional[int], top_p: Optional[float], compact_json: bool,
                         debug_payload: bool, iteration: Optional[int] = None, theme: Optional[str] = None) -> List[Question]:
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
        variables_base = {
            'token': token,
            'question_index': question_index,
            'recent_clause': recent_clause,
            'corpus': (corpus if corpus_chars == -1 else corpus[:corpus_chars]),
            'style_clause': style_clause,
            'iteration': iteration,
            'count': count,
            'model': model,
            'theme': theme,
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
            theme=theme,
            retries=self.cfg.llm_retries,
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
                    if variables_base.get('theme'):
                        new_theme = f"{variables_base['theme']} (alt {nonce[:4]})"
                        variables_retry['theme'] = new_theme
                    payload_retry = OllamaGeneratePayload(
                        model=model,
                        options=options,
                        prompt_template=prompt_tpl,
                        variables=variables_retry,
                        debug_payload=debug_payload,
                        iteration=iteration,
                        theme=new_theme or theme,
                        retries=self.cfg.llm_retries,
                    )
                    json_text, data, _ = self._client.run_ollama(payload_retry)
                else:
                    raise
