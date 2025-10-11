#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import typing as _t
import sqlite3

try:
    import yaml  # type: ignore
except Exception as _e:  # pragma: no cover
    yaml = None  # allow import even if dependency missing; try venv re-exec later

ROOT = Path(__file__).parent
QUIZ_DIR = ROOT / 'scripts' / 'quiz'
BIN = ROOT / 'scripts' / 'bin' / 'run_venv.sh'
PARAMS = ROOT / 'params.yaml'
_DISABLED = object()


def load_params(p: Path) -> dict:
    """Load configuration from params.yaml.

    Falls back to legacy quiz.params INI-like format if YAML file is missing,
    to ease transition.
    """
    if p.exists():
        if yaml is None:
            # Try to re-exec under the repo venv if available
            venv_py = (ROOT.parent / '.venv' / 'bin' / 'python')
            if venv_py.exists():
                import os
                os.execv(str(venv_py), [str(venv_py), str(Path(__file__).resolve()), *sys.argv[1:]])
            raise RuntimeError("PyYAML is required to read params.yaml. Please install 'pyyaml' or run via scripts/bin/run_venv.sh")
        data = yaml.safe_load(p.read_text(encoding='utf-8'))
        return data or {}

    # Fallback: support legacy quiz.params if present
    legacy = ROOT / 'quiz.params'
    if legacy.exists():
        cur: dict[str, dict[str, str]] = {}
        sect: _t.Optional[str] = None
        for line in legacy.read_text(encoding='utf-8').splitlines():
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            if s.startswith('[') and s.endswith(']'):
                sect = s.strip('[]')
                cur[sect] = {}
            elif '=' in s and sect:
                k, v = s.split('=', 1)
                cur[sect][k.strip()] = v.strip()
        return cur

    raise FileNotFoundError(f"No configuration found. Expected {p} (YAML) or legacy quiz.params.")


def _is_true(v: object, default: bool = False) -> bool:
    """Return a normalized boolean from YAML or legacy string values.

    Accepts booleans, numeric 0/1, and common truthy/falsey strings.
    """
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    return bool(v)


_DUMP_KEYS = {"dump_llm_payload", "dump_llm_response", "dump_ollama_prompt", "dump_rag_context"}


def _norm_none(key: str, v: object) -> object:
    """Normalize string sentinels like 'none'/'null' to None.

    Keeps other values as-is. Empty strings also treated as None.
    For dump/path toggles, the explicit string 'none' means DISABLED (no fallback).
    """
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"null", ""}:
            return None
        if s == "none" and key in _DUMP_KEYS:
            # Special sentinel to indicate explicit disable
            return _DISABLED
    return v


def _fallback(section: dict, root: dict, key: str, default: object | None = None) -> object | None:
    """Return section[key] if set (and not 'none'), else root[key], else default.

    For file-path toggles like dump_llm_payload, a value of 'none' disables it.
    """
    try:
        sv = _norm_none(key, section.get(key)) if isinstance(section, dict) else None
    except Exception:
        sv = None
    # If explicitly disabled, return None without falling back
    if sv is _DISABLED:
        return None
    if sv is not None:
        return sv
    try:
        rv = _norm_none(key, root.get(key)) if isinstance(root, dict) else None
    except Exception:
        rv = None
    if rv is not None:
        return rv
    return default



def run_prepare(cfg: dict) -> int:
    args = cfg.get('prepare', {})
    # Shared defaults fallback from root-level keys
    rag_k = _fallback(args, cfg, 'rag_k', 5)
    llm_retries = _fallback(args, cfg, 'llm_retries', 2)
    rag_embed_model = _fallback(args, cfg, 'rag_embed_model')
    dump_payload = _fallback(args, cfg, 'dump_llm_payload')
    dump_response = _fallback(args, cfg, 'dump_llm_response')
    model = _fallback(args, cfg, 'model', 'mistral')
    rag_persist = _fallback(args, cfg, 'rag_persist', '../.chroma')
    ollama_keep_alive = _fallback(args, cfg, 'ollama_keep_alive', '5m')
    ollama_url = _fallback(args, cfg, 'ollama_url')
    http_timeout = _fallback(args, cfg, 'http_timeout')
    out = [
        str(BIN),
        str(QUIZ_DIR / 'generate_quiz.py'),
        '--count', str(args.get('count', 5)),
        '--quiz', str(_fallback(args, cfg, 'quiz', 'quiz.json')),
        '--answers', str(_fallback(args, cfg, 'answers', 'answer_key.json')),
        '--avoid-recent-window', str(args.get('avoid_recent_window', 5)),
        '--rag-persist', str(rag_persist),
        '--rag-k', str(rag_k),
        '--max-retries', str(args.get('max_retries', 2)),
        '--llm-retries', str(llm_retries),
        '--ollama-keep-alive', str(ollama_keep_alive),
    ]
    if ollama_url:
        out += ['--ollama-url', str(ollama_url)]
    if http_timeout:
        out += ['--http-timeout', str(http_timeout)]
    if rag_embed_model:
        out += ['--rag-embed-model', str(rag_embed_model)]
    # Provider fixed to Ollama; no standalone --ollama flag
    out += ['--ollama-model', str(model)]

    if _is_true(args.get('verify', False)):
        out += ['--verify']

    # RAG is local-only
    out += ['--rag-local']

    # Optional dump settings
    if dump_payload:
        out += ['--dump-llm-payload', str(dump_payload)]
    if dump_response:
        out += ['--dump-llm-response', str(dump_response)]
    dump_prompt = args.get('dump_ollama_prompt')
    if dump_prompt:
        out += ['--dump-ollama-prompt', str(dump_prompt)]
    dump_rag_ctx = args.get('dump_rag_context')
    if dump_rag_ctx:
        out += ['--dump-rag-context', str(dump_rag_ctx)]


    return exec_cmd(out)


# export and parse functionality removed


def run_validate(cfg: dict) -> int:
    raw_args = cfg.get('validate')
    args = raw_args if isinstance(raw_args, dict) else {}
    sqlite_db = _fallback(args, cfg, 'sqlite_db', 'quiz.db')
    ollama_url = _fallback(args, cfg, 'ollama_url')
    http_timeout = _fallback(args, cfg, 'http_timeout')
    model = _fallback(args, cfg, 'model', 'mistral')
    llm_retries = _fallback(args, cfg, 'llm_retries', 2)
    ollama_keep_alive = _fallback(args, cfg, 'ollama_keep_alive', '5m')
    validated_out = _fallback(args, cfg, 'validated_quiz', 'validated_quiz.json')
    rag_persist = _fallback(args, cfg, 'rag_persist', '../.chroma/baai-bge-base-en-v1-5')
    rag_k = _fallback(args, cfg, 'rag_k', 5)
    rag_embed_model = _fallback(args, cfg, 'rag_embed_model', 'BAAI/bge-base-en-v1.5')
    no_rag = _fallback(args, cfg, 'no_rag', False)
    dump_payload = _fallback(args, cfg, 'dump_llm_payload')
    dump_response = _fallback(args, cfg, 'dump_llm_response')
    dump_prompt = _fallback(args, cfg, 'dump_ollama_prompt')
    out = [
        str(BIN), str(QUIZ_DIR / 'validate_quiz_answers.py'),
        '--quiz', str(_fallback(args, cfg, 'quiz', 'quiz.json')),
        '--answers', str(_fallback(args, cfg, 'answers', 'answer_key.json')),
        '--validated-out', str(validated_out),
    ]
    env = None
    try:
        import os as _os
        env = dict(_os.environ)
        # Ensure SQLite DB exists and has the required schema
        if sqlite_db:
            try:
                _ensure_sqlite_db(Path(str(sqlite_db)))
                env['SQLITE_DB'] = str(sqlite_db)
            except Exception as _e:
                print(f"[warn] Could not initialize SQLite DB at {sqlite_db}: {_e}")
        if ollama_url:
            env['OLLAMA_URL'] = str(ollama_url)
        if http_timeout:
            env['OLLAMA_HTTP_TIMEOUT'] = str(http_timeout)
        if model:
            env['OLLAMA_MODEL'] = str(model)
        if llm_retries is not None:
            env['LLM_RETRIES'] = str(llm_retries)
        if ollama_keep_alive:
            env['OLLAMA_KEEP_ALIVE'] = str(ollama_keep_alive)
        if validated_out:
            env['VALIDATED_QUIZ'] = str(validated_out)
        if rag_persist:
            env['RAG_PERSIST'] = str(rag_persist)
        if rag_k is not None:
            env['RAG_K'] = str(rag_k)
        if rag_embed_model:
            env['RAG_EMBED_MODEL'] = str(rag_embed_model)
        if _is_true(no_rag, False):
            env['NO_RAG'] = '1'
        if dump_payload:
            env['DUMP_LLM_PAYLOAD'] = str(dump_payload)
        if dump_response:
            env['DUMP_LLM_RESPONSE'] = str(dump_response)
        if dump_prompt:
            env['DUMP_OLLAMA_PROMPT'] = str(dump_prompt)
    except Exception:
        env = None
    return exec_cmd(out, env=env)


def run_chat(cfg: dict) -> int:
    args = cfg.get('chat', {})
    # Shared defaults fallback from root-level keys
    rag_k = _fallback(args, cfg, 'rag_k', 5)
    rag_embed_model = _fallback(args, cfg, 'rag_embed_model', 'BAAI/bge-base-en-v1.5')
    llm_retries = _fallback(args, cfg, 'llm_retries', 2)
    dump_payload = _fallback(args, cfg, 'dump_llm_payload')
    dump_response = _fallback(args, cfg, 'dump_llm_response')
    model = _fallback(args, cfg, 'model', 'mistral')
    rag_persist = _fallback(args, cfg, 'rag_persist', '../.chroma/baai-bge-base-en-v1-5')
    ollama_keep_alive = _fallback(args, cfg, 'ollama_keep_alive', '5m')
    ollama_url = _fallback(args, cfg, 'ollama_url')
    http_timeout = _fallback(args, cfg, 'http_timeout')
    out = [
        str(BIN), str(QUIZ_DIR / 'chat.py'),
        '--window', str(args.get('window', 6)),
        '--model', str(model),
        '--temperature', str(args.get('temperature', 0.2)),
        '--rag-persist', str(rag_persist),
        '--rag-k', str(rag_k),
        '--rag-embed-model', str(rag_embed_model),
        '--llm-retries', str(llm_retries),
        '--ollama-keep-alive', str(ollama_keep_alive),
    ]
    if ollama_url:
        out += ['--ollama-url', str(ollama_url)]
    if http_timeout:
        out += ['--http-timeout', str(http_timeout)]
    if _is_true(args.get('no_rag', False)):
        out += ['--no-rag']
    if dump_payload:
        out += ['--dump-llm-payload', str(dump_payload)]
    if dump_response:
        out += ['--dump-llm-response', str(dump_response)]
    dump_prompt = args.get('dump_ollama_prompt')
    if dump_prompt:
        out += ['--dump-ollama-prompt', str(dump_prompt)]
    return exec_cmd(out)


def exec_cmd(cmd: list[str], *, env: dict[str,str] | None = None) -> int:
    import subprocess
    print('[run]', ' '.join(cmd))
    try:
        r = subprocess.run(cmd, check=False, env=env)
        return r.returncode
    except KeyboardInterrupt:
        print('\nInterrupted.')
        return 130


def _ensure_sqlite_db(db_path: Path) -> None:
    """Create the SQLite database and schema if they do not already exist.

    Schema:
      - test_questions(question TEXT, question_uuid TEXT PRIMARY KEY, explanation TEXT)
      - test_answers(question_uuid TEXT, option TEXT, true_or_false INTEGER,
                     FOREIGN KEY(question_uuid) REFERENCES test_questions(question_uuid))
    """
    try:
        # Create parent directories if needed
        if db_path.parent and not db_path.parent.exists():
            db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(db_path)) as conn:
            # Enforce foreign keys
            conn.execute("PRAGMA foreign_keys = ON;")
            schema_path = ROOT / 'schema.sql'
            if not schema_path.exists():
                raise FileNotFoundError(f"schema.sql not found at {schema_path}")
            sql = schema_path.read_text(encoding='utf-8')
            conn.executescript(sql)
            conn.commit()
    except Exception as e:
        raise


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('target', choices=['prepare','validate','chat','prepare-validate'])
    a = ap.parse_args(argv)
    cfg = load_params(PARAMS)

    if a.target == 'prepare':
        return run_prepare(cfg)
    if a.target == 'validate':
        return run_validate(cfg)
    if a.target == 'chat':
        return run_chat(cfg)
    if a.target == 'prepare-validate':
        # Run prepare in emit-stdout mode and pipe to validate with --stdin
        import subprocess, os as _os
        print('[info] Running prepare→validate in-memory. This can take a while depending on question count and model speed...')
        # Build prepare command using existing function but override to include --emit-stdout
        args = cfg.get('prepare', {})
        rag_k = _fallback(args, cfg, 'rag_k', 5)
        llm_retries = _fallback(args, cfg, 'llm_retries', 2)
        rag_embed_model = _fallback(args, cfg, 'rag_embed_model')
        model = _fallback(args, cfg, 'model', 'mistral')
        rag_persist = _fallback(args, cfg, 'rag_persist', '../.chroma')
        ollama_keep_alive = _fallback(args, cfg, 'ollama_keep_alive', '5m')
        ollama_url = _fallback(args, cfg, 'ollama_url')
        http_timeout = _fallback(args, cfg, 'http_timeout')
        prep_cmd = [
            str(BIN), str(QUIZ_DIR / 'generate_quiz.py'),
            '--count', str(args.get('count', 5)),
            '--quiz', str(_fallback(args, cfg, 'quiz', 'quiz.json')),
            '--answers', str(_fallback(args, cfg, 'answers', 'answer_key.json')),
            '--avoid-recent-window', str(args.get('avoid_recent_window', 5)),
            '--rag-persist', str(rag_persist),
            '--rag-k', str(rag_k),
            '--max-retries', str(args.get('max_retries', 2)),
            '--llm-retries', str(llm_retries),
            '--ollama-keep-alive', str(ollama_keep_alive),
            '--ollama-model', str(model),
            '--rag-local',
            '--emit-stdout',
        ]
        if ollama_url:
            prep_cmd += ['--ollama-url', str(ollama_url)]
        if http_timeout:
            prep_cmd += ['--http-timeout', str(http_timeout)]
        if rag_embed_model:
            prep_cmd += ['--rag-embed-model', str(rag_embed_model)]

        # Build validate command to read from stdin and still use env wiring
        raw_args = cfg.get('validate')
        v_args = raw_args if isinstance(raw_args, dict) else {}
        validated_out = _fallback(v_args, cfg, 'validated_quiz', 'validated_quiz.json')
        v_env = dict(_os.environ)
        # Reuse env wiring from run_validate
        ollama_url = _fallback(v_args, cfg, 'ollama_url')
        http_timeout = _fallback(v_args, cfg, 'http_timeout')
        model = _fallback(v_args, cfg, 'model', 'mistral')
        llm_retries = _fallback(v_args, cfg, 'llm_retries', 2)
        ollama_keep_alive = _fallback(v_args, cfg, 'ollama_keep_alive', '5m')
        rag_persist = _fallback(v_args, cfg, 'rag_persist', '../.chroma/baai-bge-base-en-v1-5')
        rag_k = _fallback(v_args, cfg, 'rag_k', 5)
        rag_embed_model = _fallback(v_args, cfg, 'rag_embed_model', 'BAAI/bge-base-en-v1.5')
        sqlite_db = _fallback(v_args, cfg, 'sqlite_db', 'quiz.db')
        if ollama_url: v_env['OLLAMA_URL'] = str(ollama_url)
        if http_timeout: v_env['OLLAMA_HTTP_TIMEOUT'] = str(http_timeout)
        if model: v_env['OLLAMA_MODEL'] = str(model)
        if llm_retries is not None: v_env['LLM_RETRIES'] = str(llm_retries)
        if ollama_keep_alive: v_env['OLLAMA_KEEP_ALIVE'] = str(ollama_keep_alive)
        if validated_out: v_env['VALIDATED_QUIZ'] = str(validated_out)
        if rag_persist: v_env['RAG_PERSIST'] = str(rag_persist)
        if rag_k is not None: v_env['RAG_K'] = str(rag_k)
        if rag_embed_model: v_env['RAG_EMBED_MODEL'] = str(rag_embed_model)
        if sqlite_db:
            try:
                _ensure_sqlite_db(Path(str(sqlite_db)))
                v_env['SQLITE_DB'] = str(sqlite_db)
            except Exception as _e:
                print(f"[warn] Could not initialize SQLite DB at {sqlite_db}: {_e}")
        dump_payload = _fallback(v_args, cfg, 'dump_llm_payload')
        dump_response = _fallback(v_args, cfg, 'dump_llm_response')
        dump_prompt = _fallback(v_args, cfg, 'dump_ollama_prompt')
        if dump_payload: v_env['DUMP_LLM_PAYLOAD'] = str(dump_payload)
        if dump_response: v_env['DUMP_LLM_RESPONSE'] = str(dump_response)
        if dump_prompt: v_env['DUMP_OLLAMA_PROMPT'] = str(dump_prompt)

        val_cmd = [str(BIN), str(QUIZ_DIR / 'validate_quiz_answers.py'), '--stdin', '--validated-out', str(validated_out)]

        print('[run]', ' | '.join([' '.join(prep_cmd), ' '.join(val_cmd)]))
        # Run prepare, capture stdout (logs + JSON), and extract the JSON payload only
        prep = subprocess.run(prep_cmd, stdout=subprocess.PIPE, check=False)
        prep_rc = prep.returncode
        if prep_rc != 0:
            return prep_rc
        raw_out = prep.stdout.decode('utf-8', errors='ignore') if isinstance(prep.stdout, (bytes, bytearray)) else str(prep.stdout)
        # Extract first balanced JSON object from the output
        def _extract_json(s: str) -> str | None:
            start = s.find('{')
            if start == -1:
                return None
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(s)):
                ch = s[i]
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
                    elif ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            return s[start:i+1]
            return None
        json_text = _extract_json(raw_out)
        if not json_text:
            print('[error] Could not locate JSON payload in prepare output')
            return 1
        # Best-effort: echo any non-JSON logs to the console so users still see timing/info
        try:
            before = raw_out.split(json_text, 1)[0]
            after = raw_out.split(json_text, 1)[1]
            leftover = (before + after).strip()
            if leftover:
                print(leftover)
        except Exception:
            pass
        # Start validate, send only the JSON via stdin
        val = subprocess.Popen(val_cmd, stdin=subprocess.PIPE, env=v_env)
        _, _ = val.communicate(input=json_text.encode('utf-8'))
        return val.returncode
    return 0

if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
