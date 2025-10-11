#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Support running as a module or as a script
try:
    from .config import Config, parse_args  # type: ignore
    from .questions import Question, HISTORY_FILE, save_history  # type: ignore
    from .quiz_core import Quiz  # type: ignore
    from .utils import log, write_outputs  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    sys.path.append(str(Path(__file__).parent))
    from config import Config, parse_args  # type: ignore
    from questions import Question, HISTORY_FILE, save_history  # type: ignore
    from quiz_core import Quiz  # type: ignore
    from utils import log, write_outputs  # type: ignore

def main(argv: List[str]) -> int:
    """CLI entrypoint: parse args, run quiz generation, validate, and write outputs."""
    try:
        cfg = parse_args(argv)
        # No OpenAI embeddings path
        quiz = Quiz(cfg)
        questions, _ = quiz.run()
        err = quiz.validate(questions, cfg.count)
        if err:
            log("error", f"Validation failed: {err}")
            return 1
        if cfg.emit_stdout:
            # Emit a combined JSON object to stdout for piping into validate
            try:
                payload = {
                    "quiz": [q.public_dict() for q in questions],
                    "answer_key": {q.id: q.answer_dict() for q in questions},
                }
                print(json.dumps(payload, ensure_ascii=False))
                return 0
            except Exception as e:
                log("error", f"Could not emit stdout payload: {e}")
                return 1
        if cfg.dry_run:
            log("info", "Dry run complete.")
            return 0
        write_outputs(questions, cfg.quiz, cfg.answers)
        save_history(questions)
        log("ok", f"Wrote {len(questions)} questions -> {cfg.quiz} and answer key -> {cfg.answers}")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as e:
        log("error", f"Generation failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

