from __future__ import annotations

import json
import re
from datetime import datetime as _dt
from pathlib import Path
from typing import Any, Dict, List

_orig_print = print

def log(level: str, msg: str) -> None:
    """Lightweight logger that prefixes messages with an ISO timestamp and a level."""
    ts = _dt.now().isoformat(timespec='seconds')
    _orig_print(f"[{ts}] [{level}] {msg}")


def print(*args, **kwargs):  # type: ignore
    """Wrap builtin print to prefix string messages with an ISO timestamp for consistency."""
    if args and isinstance(args[0], str):
        ts = _dt.now().isoformat(timespec='seconds')
        args = (f"[{ts}] {args[0]}",) + args[1:]
    return _orig_print(*args, **kwargs)


def write_outputs(questions, quiz_path: Path, answers_path: Path) -> None:
    """Write quiz.json and answer_key.json files from the generated Question objects."""
    quiz = [q.public_dict() for q in questions]
    quiz_path.write_text(json.dumps(quiz, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    key: Dict[str, Dict[str, Any]] = {}
    for q in questions:
        entry = q.answer_dict()
        raw = q.raw_response
        if isinstance(raw, str) and raw.strip().startswith('{'):
            try:
                entry['raw_response'] = json.loads(raw)
            except Exception:
                entry['raw_response'] = raw
        else:
            entry['raw_response'] = raw
        key[q.id] = entry
    key = _pretty_and_parse_raw_response(key)
    answers_path.write_text(json.dumps(key, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def _pretty_and_parse_raw_response(key: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Normalize and pretty-print embedded raw LLM responses within the answer key for readability."""
    for _, entry in key.items():
        raw = entry.get('raw_response', '')
        obj: Any = raw
        if isinstance(raw, str) and raw.strip().startswith('{'):
            try:
                obj = json.loads(raw)
            except Exception:
                obj = raw
        if isinstance(obj, dict):
            obj.pop('context', None)
            resp = obj.get('response')
            if isinstance(resp, str):
                text = resp.strip()
                if text.startswith('[') or text.startswith('{'):
                    try:
                        obj['response'] = json.loads(text)
                    except Exception:
                        obj['response'] = text.replace('\\n', '\n')
                else:
                    obj['response'] = text.replace('\\n', '\n')
        entry['raw_response'] = obj
    return key


def _parse_model_questions(raw_json: str, provider: str):
    """Parse provider JSON text into a list of Question objects and validate basic schema.

    Note: Question class imported lazily to avoid circular imports.
    """
    try:
        from .questions import Question  # type: ignore  # local import to avoid cycles
    except Exception:
        # Fallback for script mode
        from questions import Question  # type: ignore
    try:
        # Preprocess: extract JSON from fenced blocks or surrounding text if needed
        text = (raw_json or '').strip()
        if not text:
            raise RuntimeError(f'{provider}: empty response')

        # Prefer fenced code blocks first (labelled or unlabelled)
        fence_patterns = [
            r"```\s*json\s*\n(.*?)```",
            r"```\s*JSON\s*\n(.*?)```",
            r"```\s*\n(.*?)```",
        ]
        extracted = None
        for pat in fence_patterns:
            m = re.search(pat, text, re.DOTALL)
            if m:
                extracted = m.group(1).strip()
                break

        if extracted is None:
            # Not fenced: find first balanced JSON object or array.
            # Prefer an object slice to avoid accidentally grabbing the first
            # nested array (e.g., the value of the `options` field).
            def _balanced_slice(s0: str, open_char: str, close_char: str):
                start_idx = s0.find(open_char)
                if start_idx == -1:
                    return None
                depth = 0
                in_str = False
                esc = False
                for i in range(start_idx, len(s0)):
                    ch = s0[i]
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == '\\':
                            esc = True
                        elif ch == '"':
                            in_str = False
                    else:
                        if ch == '"':
                            in_str = True
                        elif ch == open_char:
                            depth += 1
                        elif ch == close_char:
                            depth -= 1
                            if depth == 0:
                                return s0[start_idx:i+1].strip()
                return None

            # Prefer object; fall back to array, then to whole text
            candidate_obj = _balanced_slice(text, '{', '}')
            candidate_arr = _balanced_slice(text, '[', ']')
            extracted = (candidate_obj or candidate_arr or text)

        # Attempt to parse JSON, with a light cleanup retry for trailing commas
        def _loads_with_cleanup(src: str):
            try:
                return json.loads(src)
            except Exception:
                # Remove trailing commas before closing object/array brackets
                cleaned = re.sub(r",\s*([}\]])", r"\1", src)
                return json.loads(cleaned)

        data = _loads_with_cleanup(extracted)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and provider == 'ollama':
            items = [data]
        else:
            raise RuntimeError(f'{provider}: expected a list of questions, got {type(data)}')
        out: List[Question] = []
        def format_options(options):
            # Accept either a list of strings, or a dict like {"A": "...", ...}
            if isinstance(options, list) and all(isinstance(opt, str) for opt in options):
                return [opt.strip() for opt in options]
            if isinstance(options, dict):
                ordered = [options.get(k) for k in ['A', 'B', 'C', 'D']]
                if all(isinstance(v, str) and v.strip() for v in ordered):
                    return [v.strip() for v in ordered]
            raise RuntimeError('Options must be a list of strings or a dict with A-D keys')
        for idx, obj in enumerate(items, start=1):
            if not isinstance(obj, dict):
                continue
            qid = str(obj.get('id') or f'Q{idx}')
            question = (obj.get('question') or '').strip() or f'Placeholder question {idx}'
            raw_opts = obj.get('options')
            options = format_options(raw_opts)
            explicit_letter = str(obj.get('answer_letter', '')).strip().upper()
            answer_field = str(obj.get('answer', '')).strip().upper()
            if explicit_letter in ['A', 'B', 'C', 'D']:
                answer_letter = explicit_letter
            elif answer_field in ['A', 'B', 'C', 'D']:
                answer_letter = answer_field
            else:
                raise RuntimeError("answer or answer_letter must be one of A, B, C, D")
            topic = (obj.get('topic') or 'General').strip() or 'General'
            difficulty = (obj.get('difficulty') or 'medium').strip() or 'medium'
            explanation = (obj.get('explanation') or '').strip()
            out.append(Question(
                id=qid,
                question=question,
                options=options,
                topic=topic,
                difficulty=difficulty,
                answer=answer_letter,
                explanation=explanation
            ))
        if not out:
            raise RuntimeError(f'{provider}: no questions parsed from model output')
        return out
    except Exception as e:
        log("error", f"LLM output parsing failed: {e}")
        # Best-effort dump handled by caller; raise to trigger retry if any
        raise
