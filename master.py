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


def run_prepare(cfg: dict) -> int:
    args = cfg.get('prepare', {})
    out = [
        str(BIN),
        str(QUIZ_DIR / 'generate_quiz.py'),
        '--count', str(args.get('count', 5)),
        '--quiz', str(args.get('quiz', 'quiz.json')),
        '--answers', str(args.get('answers', 'answer_key.json')),
        '--avoid-recent-window', str(args.get('avoid_recent_window', 5)),
        '--rag-persist', str(args.get('rag_persist', '../.chroma')),
        '--rag-k', str(args.get('rag_k', 5)),
        '--max-retries', str(args.get('max_retries', 2)),
        '--llm-retries', str(args.get('llm_retries', 2)),
    ]
    if args.get('rag_embed_model'):
        out += ['--rag-embed-model', str(args.get('rag_embed_model'))]
    # Provider fixed to Ollama; no standalone --ollama flag
    out += ['--ollama-model', str(args.get('model', 'mistral'))]

    if _is_true(args.get('verify', False)):
        out += ['--verify']

    # RAG is local-only
    out += ['--rag-local']

    # Optional dump settings
    dump_payload = args.get('dump_llm_payload')
    if dump_payload:
        out += ['--dump-llm-payload', str(dump_payload)]
    dump_response = args.get('dump_llm_response')
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
    ap.add_argument('target', choices=['prepare','validate'])
    a = ap.parse_args(argv)
    cfg = load_params(PARAMS)

    if a.target == 'prepare':
        return run_prepare(cfg)
    if a.target == 'validate':
        return run_validate(cfg)
    return 0

if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
