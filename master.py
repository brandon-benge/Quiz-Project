#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import typing as _t

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
        '--quiz', str(args.get('quiz', 'quiz.json')),
        '--answers', str(args.get('answers', 'answer_key.json')),
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
    args = cfg.get('validate', {})
    out = [
        str(BIN), str(QUIZ_DIR / 'validate_quiz_answers.py'),
        '--quiz', str(args.get('quiz', 'quiz.json')),
        '--answers', str(args.get('answers', 'answer_key.json')),
        '--raw', 'summary',
        '--show-correct-first'
    ]
    return exec_cmd(out)


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


def exec_cmd(cmd: list[str]) -> int:
    import subprocess
    print('[run]', ' '.join(cmd))
    try:
        r = subprocess.run(cmd, check=False)
        return r.returncode
    except KeyboardInterrupt:
        print('\nInterrupted.')
        return 130


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('target', choices=['prepare','validate','chat'])
    a = ap.parse_args(argv)
    cfg = load_params(PARAMS)

    if a.target == 'prepare':
        return run_prepare(cfg)
    if a.target == 'validate':
        return run_validate(cfg)
    if a.target == 'chat':
        return run_chat(cfg)
    return 0

if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
