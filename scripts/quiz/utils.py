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
        # Normalizer: remove any leading option-label like "A)", "B.", "C:", "D -" (case-insensitive) and trim
        def _clean_option_text(s: str) -> str:
            try:
                # Strip enclosing whitespace/newlines first
                s0 = str(s).strip()
                # Remove leading label patterns like "A)", "b.", "C :", "d -"
                s1 = re.sub(r"^\s*[A-Da-d]\s*[\)\.:\-]\s*", "", s0)
                # Collapse internal whitespace sequences to single spaces around edges
                return s1.strip()
            except Exception:
                return str(s).strip()
        def format_options(options):
            # Accept either a list of strings, or a dict like {"A": "...", ...}
            if isinstance(options, list) and all(isinstance(opt, str) for opt in options):
                cleaned_list = []
                for opt in options:
                    cleaned = _clean_option_text(opt)
                    # Fallback to original trim if cleaning results in empty
                    cleaned_list.append(cleaned or str(opt).strip())
                return cleaned_list
            if isinstance(options, dict):
                ordered = [options.get(k) for k in ['A', 'B', 'C', 'D']]
                if all(isinstance(v, str) and v.strip() for v in ordered):
                    cleaned_ordered = []
                    for v in ordered:
                        cleaned = _clean_option_text(v)
                        cleaned_ordered.append(cleaned or str(v).strip())
                    return cleaned_ordered
            raise RuntimeError('Options must be a list of strings or a dict with A-D keys')
        for idx, obj in enumerate(items, start=1):
            if not isinstance(obj, dict):
                continue
            qid = str(obj.get('id') or f'Q{idx}')
            question = (obj.get('question') or '').strip() or f'Placeholder question {idx}'
            raw_opts = obj.get('options')
            options = format_options(raw_opts)
            # Expect 'answer' to be the full correct option text; avoid letters.
            # However, for robustness, if the model includes a leading label (e.g., "B. ...")
            # or a single letter (e.g., "B"), normalize it to the option text and warn.
            answer_raw = str(obj.get('answer', '')).strip()
            if not answer_raw:
                raise RuntimeError("answer must be the full correct option text")
            answer_text = answer_raw
            # Primary: exact text match (case-insensitive)
            norm = answer_text.strip().lower()
            match_idx = -1
            for i, opt in enumerate(options):
                if norm == str(opt).strip().lower():
                    match_idx = i
                    break
            # Fallback 1: strip any leading label from the answer text and retry
            if match_idx == -1:
                cleaned_ans = _clean_option_text(answer_text)
                if cleaned_ans and cleaned_ans.lower() != norm:
                    c_norm = cleaned_ans.strip().lower()
                    for i, opt in enumerate(options):
                        if c_norm == str(opt).strip().lower():
                            match_idx = i
                            answer_text = str(opt)
                            log("warn", f"{provider}: normalized answer text by stripping label for {qid}")
                            break
            # Fallback 2: map single-letter answers (A-D) to the indexed option
            if match_idx == -1:
                letter = answer_raw.strip()
                if re.fullmatch(r"[A-Da-d]", letter) or re.fullmatch(r"[A-Da-d]\s*[\)\.:\-]?", letter):
                    idx_letter = ord(letter.strip()[0].upper()) - ord('A')
                    if 0 <= idx_letter < len(options):
                        match_idx = idx_letter
                        answer_text = str(options[idx_letter])
                        log("warn", f"{provider}: mapped single-letter answer to option text for {qid}")
            if match_idx == -1:
                raise RuntimeError("answer text must exactly match one of the provided options")
            topic = (obj.get('topic') or 'General').strip() or 'General'
            difficulty = (obj.get('difficulty') or 'medium').strip() or 'medium'
            explanation = (obj.get('explanation') or '').strip()

            # Optional: reconcile explanation with answer text is not required now
            out.append(Question(
                id=qid,
                question=question,
                options=options,
                topic=topic,
                difficulty=difficulty,
                answer=answer_text,
                explanation=explanation
            ))
        if not out:
            raise RuntimeError(f'{provider}: no questions parsed from model output')
        return out
    except Exception as e:
        log("error", f"LLM output parsing failed: {e}")
        # Best-effort dump handled by caller; raise to trigger retry if any
        raise
