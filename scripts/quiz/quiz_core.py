from __future__ import annotations

import uuid
import os
import random
from typing import Dict, List, Optional, Tuple

# Support running as a module or as a script
try:
    from .config import Config  # type: ignore
    from .questions import Question, HISTORY_FILE, recent_history, save_history  # type: ignore
    from .providers import Providers  # type: ignore
    from .rag import RAG  # type: ignore
    from .utils import log  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from config import Config  # type: ignore
    from questions import Question, HISTORY_FILE, recent_history, save_history  # type: ignore
    from providers import Providers  # type: ignore
    from rag import RAG  # type: ignore
    from utils import log  # type: ignore


class Quiz:
    def __init__(self, cfg: Config):
        """Orchestrate providers and RAG to generate and validate a quiz based on configuration."""
        self.cfg = cfg
        self.rag = RAG(cfg)
        self.providers = Providers(cfg, rag=self.rag)

    # history helpers are provided by questions.py

    def validate(self, questions: List[Question], expected: int) -> Optional[str]:
        """Validate count, option cardinality, and normalize answer letters when possible; return error string or None."""
        if len(questions) != expected:
            return f'Expected {expected} questions, got {len(questions)}'
        for q in questions:
            if q.answer.upper() not in ['A','B','C','D']:
                lower = q.answer.strip().lower()
                for idx,opt in enumerate(q.options):
                    if lower == opt.lower() or opt.lower().startswith(lower[:5]):
                        q.answer = chr(ord('A')+idx)
                        break
            if len(q.options) != 4:
                return f'Question {q.id} does not have 4 options'
        return None

    def _gen_one(self, files_for_q: Dict[str,str], token: str,
                 recent_norm: List[str], temperature: float, iteration_index: int, theme: Optional[str] = None,
                 themes: Optional[List[str]] = None, theme_index: Optional[int] = None) -> List[Question]:
        """Generate a single question via the Ollama provider using provided context and knobs."""
        return self.providers.ollama_questions(
                files_for_q, 1, self.cfg.ollama_model, token, recent_norm, temperature,
                snippet_chars=self.cfg.ollama_snippet_chars,
                corpus_chars=self.cfg.ollama_corpus_chars,
                num_predict=self.cfg.ollama_num_predict,
                top_k=self.cfg.ollama_top_k,
                top_p=self.cfg.ollama_top_p,
                compact_json=self.cfg.ollama_compact_json,
                debug_payload=self.cfg.debug_ollama_payload,
                iteration=iteration_index,
                theme=theme,
                themes=themes,
                theme_index=theme_index,
            )

    def run(self) -> Tuple[List[Question], Dict[str,str]]:
        """Main prepare flow: build RAG context, generate N questions, and return questions plus source context."""
        if self.cfg.count < 1:
            raise RuntimeError('--count must be at least 1')
        files_ctx, _ = self.rag.build_context({}, k=self.cfg.rag_k, count=self.cfg.count)
        recent_norm = recent_history(self.cfg.avoid_recent_window)
        base_temp = 0.4
        temperature = self.cfg.ollama_temperature if (self.cfg.ollama_temperature is not None) else base_temp
        questions: List[Question] = []
        for idx in range(self.cfg.count):
            token = str(uuid.uuid4())
            theme = None
            files_single = files_ctx
            # Recreate unique tags for each question
            db_path = os.path.join(self.rag.cfg.rag_persist, 'chroma.sqlite3')
            tags = self.rag._fetch_unique_tags(db_path)
            if tags:
                if self.cfg.include_tags:
                    req_lower = {t.strip().lower() for t in self.cfg.include_tags}
                    tag_pool = [t for t in tags if t.strip().lower() in req_lower]
                else:
                    tag_pool = list(tags)
                if tag_pool:
                    # Strategy: use the first theme in the list for the initial attempt,
                    # then retries will advance to the next entries.
                    theme_index = 0
                    q_theme = tag_pool[theme_index]
                    theme = q_theme
                    maybe = self.rag.get_blocks_for_tag(q_theme, self.cfg.rag_k)
                    if maybe:
                        files_single = maybe
            qlist = self._gen_one(files_single, token, recent_norm, temperature, idx, theme=theme, themes=tag_pool if tags else None, theme_index=0 if tags else None)
            if qlist:
                q = qlist[0]
                q.id = f"Q{idx+1}"
                questions.append(q)
        return questions, files_ctx
