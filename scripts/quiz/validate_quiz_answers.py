#!/usr/bin/env python3
"""Auto-validate quiz answers using the LLM (no interactive input).

For each question, we ask the LLM to determine whether the provided answer is correct,
or if any other option is closer to correct, or if none of the options make sense.
The LLM must respond strictly with True or False using the contract described in the prompt.

If the LLM responds True (meaning the given answer is correct and reasonable, with no better option),
the question is appended to an output file configured via params.yaml (global key: validated_quiz).
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import re
import sqlite3
import uuid

def parse_args(argv: List[str]):
    p = argparse.ArgumentParser(description="Validate quiz answers")
    p.add_argument("--quiz", default="quiz.json")
    p.add_argument("--answers", default="answer_key.json")
    p.add_argument("--validated-out", help="Output file for validated questions (overrides params.yaml validated_quiz)")
    p.add_argument("--stdin", action="store_true", help="Read combined JSON from stdin: {quiz: [...], answer_key: {...}}")
    return p.parse_args(argv)

def load_json(path: Path): return json.loads(path.read_text(encoding="utf-8"))

# Support running as a module or as a script for LLM client import
try:  # pragma: no cover - import flexibility
    from .llm_client import LLMClient, DumpOptions, OllamaGeneratePayload  # type: ignore
    from .rag import RAG  # type: ignore
except Exception:
    import sys
    from pathlib import Path as _P
    sys.path.append(str(_P(__file__).parent))
    from llm_client import LLMClient, DumpOptions, OllamaGeneratePayload  # type: ignore
    from rag import RAG  # type: ignore

def format_raw_response(raw: Any, mode: str, truncate: int) -> str:
    if mode == "none":
        return ""
    if not isinstance(raw, dict):
        s = str(raw)
        return (s[:truncate] + ("... [truncated]" if len(s) > truncate else "")) if s else ""
    obj = dict(raw)
    resp = obj.get("response")
    if isinstance(resp, str):
        text = resp.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                import json as _json
                obj["response"] = _json.loads(text)
            except Exception:
                obj["response"] = text.replace("\\n", "\n")
    def _summarize(o: Any) -> Any:
        if mode == "summary" and isinstance(o, list) and len(o) > 20:
            return f"<list of {len(o)} items>"
        return o
    if mode == "summary":
        for k in ["context", "tokens", "prompt", "prompt_token_ids"]:
            if k in obj and isinstance(obj[k], list):
                obj[k] = f"<list of {len(obj[k])} items>"
        if isinstance(obj.get("response"), list):
            short = []
            for item in obj["response"][:3]:
                if isinstance(item, dict):
                    short.append({
                        k: _summarize(v) if k in ("options",) else v
                        for k, v in item.items() if k in ("id", "question", "answer", "topic", "difficulty", "options")
                    })
                else:
                    short.append(item)
            total = len(obj["response"])  # type: ignore
            obj["response"] = {
                "items_preview": short,
                "total_items": total
            }
    import json as _json
    s = _json.dumps(obj, indent=2, ensure_ascii=False)
    return (s[:truncate] + ("... [truncated]" if len(s) > truncate else ""))

from string import Template

def _load_prompt_template() -> str:
    """Load the validation prompt template from scripts/quiz/templates/validate_prompt.tmpl.

    No environment overrides or built-in fallbacks. If the file is missing, raise a clear error.
    """
    try:
        base = Path(__file__).resolve().parent
    except Exception:
        base = Path.cwd()
    p = base / 'templates' / 'validate_prompt.tmpl'
    if not p.exists():
        raise FileNotFoundError(f"Validation prompt template not found at {p}. This file is required.")
    return p.read_text(encoding='utf-8')

def _build_validation_prompt(tpl_text: str, q: Dict[str, Any], key_entry: Dict[str, Any], rag_context: str = "") -> str:
    question = q.get('question','').strip()
    options = q.get('options', [])
    provided_text = str(key_entry.get('answer','')).strip()
    expl = key_entry.get('explanation','').strip()
    # Render options without letter prefixes (text-only)
    opts_str = '\n'.join([f"- {opt}" for opt in options])
    # Provided answer is already text
    provided_answer_text = provided_text
    context_section = f"Context (from knowledge base):\n{rag_context}\n\n" if rag_context else ""
    tmpl = Template(tpl_text)
    return tmpl.safe_substitute(
        context_section=context_section,
        question=question,
        options=opts_str,
        provided_answer_text=provided_answer_text,
        explanation=expl,
    )

def _ensure_db(conn: sqlite3.Connection) -> None:
    # Enforce foreign keys and apply schema from schema.sql for readability
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
    except Exception:
        pass
    try:
        # Try to locate schema.sql at repo root (2 dirs up from this file)
        root = Path(__file__).resolve().parents[2]
        schema_path = root / 'schema.sql'
        if schema_path.exists():
            sql = schema_path.read_text(encoding='utf-8')
            conn.executescript(sql)
            conn.commit()
    except Exception as e:
        print(f"[warn] Could not load schema.sql: {e}")

def auto_validate(quiz: List[Dict[str, Any]], key: Dict[str, Any], *, out_path: Path) -> int:
    # Build a minimal client with shared params from env (master.py passes them via CLI -> env)
    # Pull config from environment or defaults
    ollama_url = os.environ.get('OLLAMA_URL')
    http_timeout = os.environ.get('OLLAMA_HTTP_TIMEOUT')
    model = os.environ.get('OLLAMA_MODEL', 'mistral')
    if not ollama_url or not http_timeout:
        print("[error] OLLAMA_URL or OLLAMA_HTTP_TIMEOUT env not set; run via master.py")
        return 1
    dumps = DumpOptions(
        dump_prompt_path=os.environ.get('DUMP_OLLAMA_PROMPT'),
        dump_payload_path=os.environ.get('DUMP_LLM_PAYLOAD'),
        dump_response_path=os.environ.get('DUMP_LLM_RESPONSE'),
    )
    client = LLMClient(dumps, base_url=ollama_url, http_timeout=int(http_timeout))
    # RAG setup (question-only retrieval) is REQUIRED
    rag = None
    try:
        cfg_like = type('Cfg', (), {})()
        setattr(cfg_like, 'rag_persist', os.environ.get('RAG_PERSIST', '../.chroma/sentence-transformers-all-mpnet-base-v2'))
        setattr(cfg_like, 'rag_embed_model', os.environ.get('RAG_EMBED_MODEL', 'sentence-transformers/all-mpnet-base-v2'))
        setattr(cfg_like, 'rag_k', int(os.environ.get('RAG_K', '5')))
        setattr(cfg_like, 'no_rag', False)
        rag = RAG(cfg_like)  # type: ignore
        # Initialize the RAG components once (document store, retriever, embedder)
        rag._init()
        if not getattr(rag, '_document_store', None):
            print("[error] RAG document store is not available for validation; ensure rag_persist is correct and dependencies are installed.")
            return 2
    except Exception as e:
        print(f"[error] RAG init for validation failed: {e}")
        return 2
    options = {}
    valid: List[Dict[str, Any]] = []
    # Optional DB target from env
    db_path = os.environ.get('SQLITE_DB')
    conn: sqlite3.Connection | None = None
    if db_path:
        try:
            conn = sqlite3.connect(db_path)
            _ensure_db(conn)
        except Exception as e:
            print(f"[warn] Could not open SQLite DB at {db_path}: {e}")
            conn = None
    # Load prompt template once
    tpl_text = _load_prompt_template()
    for q in quiz:
        qid = q.get('id')
        key_entry = key.get(qid, {})
        if not key_entry: continue
        provided_text = str(key_entry.get('answer','')).strip()
        # Basic sanity warnings for answer text and options
        try:
            opts = q.get('options', [])
            if not isinstance(opts, list) or len(opts) == 0:
                print(f"[warn] Question {qid} has no options; validation may be unreliable.")
            elif provided_text:
                norm = provided_text.strip().lower()
                if all(norm != str(o).strip().lower() for o in (opts or [])):
                    print(f"[warn] Question {qid} provided answer text does not match any option.")
        except Exception:
            pass
        # Build context by querying ONLY with the question text
        context_text = ""
        if rag and isinstance(q.get('question'), str):
            try:
                maybe = rag.get_blocks_for_query(q['question'], int(os.environ.get('RAG_K', '5')))
                if isinstance(maybe, dict):
                    # Use only the core context content; do not include external header templates
                    raw_ctx = maybe.get('RAG_CONTEXT.md', '') or ''
                    # Strip any leading header markers or template banners to keep the validation context aligned
                    # with the validator's simpler decision task.
                    lines = [ln for ln in str(raw_ctx).splitlines() if ln.strip()]
                    # Heuristic: drop any first line that looks like a title/banner (e.g., starts with '#', '===', '---')
                    if lines and (lines[0].startswith('#') or lines[0].startswith('===') or lines[0].startswith('---')):
                        lines = lines[1:]
                    context_text = "\n".join(lines).strip()
            except Exception as e:
                print(f"[debug] RAG retrieval failed for {qid}: {e}")
        prompt = _build_validation_prompt(tpl_text, q, key_entry, rag_context=context_text)
        retries = int(os.environ.get('LLM_RETRIES', '1'))
        keep_alive = os.environ.get('OLLAMA_KEEP_ALIVE')
        payload = OllamaGeneratePayload(model=model, options=options, prompt=prompt, retries=retries, keep_alive=keep_alive)
        try:
            json_text, raw, _ = client.run_ollama(payload)
            answer = json_text.strip()
            # Normalize possible formats: true/false, True/False, quoted strings, or JSON literals
            ans_norm = answer.strip()
            if ans_norm.startswith('"') and ans_norm.endswith('"'):
                ans_norm = ans_norm[1:-1].strip()
            if ans_norm.startswith("'") and ans_norm.endswith("'"):
                ans_norm = ans_norm[1:-1].strip()
            low = ans_norm.lower()
            if low in {"true", "false"}:
                verdict = (low == "true")
            else:
                # Last resort: try to parse JSON literal
                try:
                    import json as _json
                    parsed = _json.loads(answer)
                    verdict = bool(parsed) is True
                except Exception:
                    verdict = False
            if verdict:
                # Store the question with the answer TEXT and explanation
                try:
                    enriched = dict(q)
                except Exception:
                    enriched = q
                if isinstance(enriched, dict):
                    enriched['answer'] = provided_text
                    if 'explanation' not in enriched or not enriched['explanation']:
                        enriched['explanation'] = key_entry.get('explanation','')
                valid.append(enriched)
                # Insert into SQLite if available
                if conn is not None:
                    try:
                        question_text = str(q.get('question',''))
                        # Generate a random UUID for DB, independent of displayed question id
                        question_uuid = str(uuid.uuid4())
                        explanation_text = str(key_entry.get('explanation',''))
                        options_list = q.get('options', []) or []
                        # Determine correct option index by matching text
                        correct_idx = -1
                        try:
                            if provided_text:
                                n = provided_text.strip().lower()
                                for i, opt in enumerate(options_list):
                                    if n == str(opt).strip().lower():
                                        correct_idx = i
                                        break
                        except Exception:
                            correct_idx = -1
                        cur = conn.cursor()
                        # Insert parent row first (no delete/update)
                        cur.execute(
                            "INSERT INTO test_questions (question, question_uuid, explanation) VALUES (?, ?, ?)",
                            (question_text, question_uuid, explanation_text),
                        )
                        # Insert child rows, relying on FK constraint
                        for i, opt in enumerate(options_list):
                            is_true = 1 if i == correct_idx else 0
                            cur.execute(
                                "INSERT INTO test_answers (question_uuid, option, true_or_false) VALUES (?, ?, ?)",
                                (question_uuid, str(opt), is_true),
                            )
                        conn.commit()
                    except Exception as e:
                        print(f"[warn] SQLite insert failed for {qid}: {e}")
        except Exception as e:
            print(f"[warn] Validation call failed for {qid}: {e}")
            continue
    # Write validated questions list
    try:
        out_path.write_text(json.dumps(valid, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"[ok] Wrote {len(valid)} validated questions -> {out_path}")
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        return 0
    except Exception as e:
        print(f"[error] Could not write validated questions: {e}")
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass
        return 1

def main(argv):
    args = parse_args(argv)
    # Support stdin mode for in-memory pipeline
    if getattr(args, 'stdin', False):
        import sys as _sys
        try:
            data = json.loads(_sys.stdin.read())
            quiz = data.get('quiz', [])
            key = data.get('answer_key', {})
        except Exception as e:
            print(f"[error] Failed to parse stdin JSON: {e}")
            return 1
    else:
        quiz_path = Path(args.quiz); ans_path = Path(args.answers)
        if not quiz_path.exists() or not ans_path.exists(): print("[error] quiz or answer key path does not exist"); return 1
        quiz = load_json(quiz_path); key = load_json(ans_path)
    if not isinstance(quiz, list) or not isinstance(key, dict): print("[error] Invalid JSON structure"); return 1
    # Resolve output path: CLI overrides env; env should be set by master.py from params.yaml
    out_name = args.validated_out or os.environ.get('VALIDATED_QUIZ', 'validated_quiz.json')
    out_path = Path(out_name)
    return auto_validate(quiz, key, out_path=out_path)

if __name__ == "__main__":  # pragma: no cover
    import sys
    try: raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt: print("\nInterrupted."); raise SystemExit(130)

