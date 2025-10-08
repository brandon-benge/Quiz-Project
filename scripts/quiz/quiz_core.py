from __future__ import annotations

import uuid
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
        self.providers = Providers(cfg)
        self.rag = RAG(cfg)

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
                 recent_norm: List[str], temperature: float, iteration_index: int, theme: Optional[str] = None) -> List[Question]:
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
                theme=theme
            )

    def run(self) -> Tuple[List[Question], Dict[str,str]]:
        """Main prepare flow: build RAG context, generate N questions, and return questions plus source context."""
        if self.cfg.count < 1:
            raise RuntimeError('--count must be at least 1')
        files_ctx, queries = self.rag.build_context({}, k=self.cfg.rag_k, count=self.cfg.count)
        recent_norm = recent_history(self.cfg.avoid_recent_window)
        base_temp = 0.4
        temperature = self.cfg.ollama_temperature if (self.cfg.ollama_temperature is not None) else base_temp
        questions: List[Question] = []
        for idx in range(self.cfg.count):
            token = str(uuid.uuid4())
            theme = None
            files_single = files_ctx
            if queries and self.rag._document_store:
                if self.cfg.include_tags:
                    q = queries[idx % len(queries)]
                else:
                    import random as _r
                    q = _r.choice(queries)
                theme = q
                maybe = self.rag.get_blocks_for_tag(q, self.cfg.rag_k)
                if maybe:
                    files_single = maybe
            elif queries:
                if self.cfg.include_tags:
                    q = queries[idx % len(queries)]
                else:
                    import random as _r
                    q = _r.choice(queries)
                theme = q
            qlist = self._gen_one(files_single, token, recent_norm, temperature, idx, theme=theme)
            if qlist:
                q = qlist[0]
                q.id = f"Q{idx+1}"
                questions.append(q)
        return questions, files_ctx
