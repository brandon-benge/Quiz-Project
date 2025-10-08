from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# History file for deduplication of recent questions
HISTORY_FILE = Path('.quiz_history.json')


@dataclass
class Question:
    id: str
    question: str
    options: List[str]
    topic: str
    difficulty: str
    answer: str
    explanation: str
    raw_response: Optional[Any] = None

    def public_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "options": self.options,
            "topic": self.topic,
            "difficulty": self.difficulty,
        }

    def answer_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "explanation": self.explanation,
        }


def recent_history(avoid_recent_window: int) -> List[str]:
    """Return normalized recent question texts limited by avoid_recent_window."""
    if not HISTORY_FILE.exists():
        return []
    try:
        loaded = json.loads(HISTORY_FILE.read_text(encoding='utf-8'))
        if isinstance(loaded, list):
            return [re.sub(r'\s+', ' ', q.lower()).strip() for q in loaded][-avoid_recent_window:]
    except Exception:
        return []
    return []


def save_history(questions: List[Question], keep_last: int = 100) -> None:
    """Append generated questions to history for deduplication, keeping only the last keep_last entries."""
    try:
        items = [re.sub(r'\s+', ' ', q.question.lower()).strip() for q in questions]
        if HISTORY_FILE.exists():
            try:
                existing = json.loads(HISTORY_FILE.read_text(encoding='utf-8'))
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []
        else:
            existing = []
        HISTORY_FILE.write_text(json.dumps((existing + items)[-keep_last:], indent=2), encoding='utf-8')
    except Exception:
        # Non-fatal if history can't be written
        pass
