from __future__ import annotations

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
