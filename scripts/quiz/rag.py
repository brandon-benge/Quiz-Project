from __future__ import annotations

import os
import sqlite3
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple

# Support running as a module or as a script
try:
    from .config import Config  # type: ignore
    from .utils import log  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    import sys
    from pathlib import Path as _P
    sys.path.append(str(_P(__file__).parent))
    from config import Config  # type: ignore
    from utils import log  # type: ignore


class RAG:
    def __init__(self, cfg: Config):
        """Set up RAG configuration and lazy-initialized components."""
        self.cfg = cfg
        self._document_store = None
        self._retriever = None
        self._embedding = None
        self._init_done = False

    def _init(self) -> None:
        """Lazy-initialize Chroma document store, retriever, and local embedder; continue if unavailable."""
        if self._init_done:
            return
        self._init_done = True
        try:
            from haystack_integrations.document_stores.chroma import ChromaDocumentStore  # type: ignore
            from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever  # type: ignore
            from haystack.components.embedders import SentenceTransformersTextEmbedder  # type: ignore
            self._embedding = SentenceTransformersTextEmbedder(model=self.cfg.rag_embed_model)
            self._embedding.warm_up()
            if not os.path.isdir(self.cfg.rag_persist):
                log("warn", f"RAG store '{self.cfg.rag_persist}' not found; proceeding without RAG.")
                return
            self._document_store = ChromaDocumentStore(persist_path=self.cfg.rag_persist)
            self._retriever = ChromaEmbeddingRetriever(document_store=self._document_store)
        except ModuleNotFoundError as e:
            log("warn", f"RAG modules missing ({e}); continuing without RAG.")
        except Exception as e:
            log("warn", f"RAG init error: {e}; continuing without RAG.")

    def _fetch_unique_tags(self, db_path: str) -> List[str]:
        """Read unique theme values from Chroma's SQLite metadata (STRICT: tags_% only).

        Notes:
        - If any string_value appears to be a JSON array or CSV, explode into individual tokens.
        - Preserve original case to allow strict equality filtering in retrieval.
        """
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            # Use DISTINCT values for all denormalized tag columns (tags_0, tags_1, ...), excluding tags_json
            # Order deterministically so retries can index into this list by attempt number.
            cur.execute("SELECT DISTINCT string_value FROM embedding_metadata WHERE key LIKE 'tags_%' AND key != 'tags_json' ORDER BY string_value ASC;")
            rows = cur.fetchall()
            # Randomize the order to ensure variety across runs
            random.shuffle(rows)
            conn.close()
            ordered: List[str] = []
            seen: set[str] = set()
            for (val,) in rows:
                s = (str(val) if val is not None else '').strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                ordered.append(s)
            # Truncate to initial attempt + max_retries so index == attempt works (0..max_retries)
            try:
                max_retries = int(getattr(self.cfg, 'max_retries', 0) or 0)
            except Exception:
                max_retries = 0
            desired = max(1, max_retries + 1)
            return ordered[:desired]
        except Exception as e:
            log("warn", f"Could not fetch tags: {e}")
            return []

    def get_blocks_for_tag(self, tag: str, k: int) -> Optional[Dict[str, str]]:
        """Retrieve top-k document snippets for a tag and build a contextual file for prompting."""
        if not self._document_store:
            return None
        q = tag.strip()
        if not q:
            return None
        try:
            query_embedding = self._embedding.run(text=q)["embedding"]
            # STRICT: only return docs tagged exactly with this theme under tags_% metadata keys
            or_filters = {
                "operator": "OR",
                "conditions": [
                    {"field": f"meta.tags_{i}", "operator": "==", "value": q} for i in range(64)
                ]
            }
            docs = self._retriever.run(
                query_embedding=query_embedding,
                top_k=k,
                filters=or_filters
            )["documents"]
        except Exception as e:
            log("warn", f"Per-question retrieval by tag failed: {e}")
            return None
        if not docs:
            return None
        seen, blocks = set(), []
        for d in docs:
            snippet = d.content[:1000].strip()
            if snippet in seen:
                continue
            seen.add(snippet)
            heading = d.meta.get('section_heading') or (snippet.split('\n',1)[0][:80])
            source  = (d.meta.get('source') or d.meta.get('rel_path') or d.meta.get('path') or 'unknown')
            blocks.append(f"[C1] (source: {source}, heading: {heading})\n{snippet}")
        if not blocks:
            return None
        # Render header from template with safe fallback
        header: str
        try:
            tpl_path = Path(__file__).parent / 'templates' / 'rag_context_header.tmpl'
            tpl_text = tpl_path.read_text(encoding='utf-8')
            from string import Template
            header = Template(tpl_text).safe_substitute({'tag': q})
        except Exception as e:
            log("warn", f"Could not render RAG header template; omitting header. Error: {e}")
            header = ""
        return {'RAG_CONTEXT.md': header + "\n\n---\n\n" + "\n\n".join(blocks)}

    def get_blocks_for_query(self, query: str, k: int, *, debug: bool = False):
        """Retrieve top-k snippets for a free-text query; optionally return debug info (embedding, docs).

        Returns:
          - If debug is False: Optional[Dict[str, str]] with a single 'RAG_CONTEXT.md' entry, or None
          - If debug is True: Optional[Tuple[Dict[str, str], Dict[str, object]]], or None
        """
        if not self._document_store:
            return None
        q = (query or '').strip()
        if not q:
            return None
        emb = None
        try:
            emb = self._embedding.run(text=q)["embedding"]
            try:
                docs = self._retriever.run(
                    query_embedding=emb,
                    top_k=k
                )["documents"]
            except Exception as e:
                log("debug", f"Query retrieval failed: {type(e).__name__}: {e}")
                docs = []
        except Exception as e:
            log("warn", f"Free-text retrieval failed: {e}")
            return None
        if not docs:
            return None
        seen, blocks = set(), []
        dbg_docs = []
        for d in docs:
            snippet = d.content[:1000].strip()
            if snippet in seen:
                continue
            seen.add(snippet)
            heading = d.meta.get('section_heading') or (snippet.split('\n',1)[0][:80])
            source  = (d.meta.get('source') or d.meta.get('rel_path') or d.meta.get('path') or 'unknown')
            blocks.append(f"[C1] (source: {source}, heading: {heading})\n{snippet}")
            if debug:
                score = getattr(d, 'score', None)
                dbg_docs.append({'source': source, 'heading': heading, 'score': score, 'snippet': snippet})
        if not blocks:
            return None
        try:
            tpl_path = Path(__file__).parent / 'templates' / 'rag_context_header.tmpl'
            tpl_text = tpl_path.read_text(encoding='utf-8')
            from string import Template
            header = Template(tpl_text).safe_substitute({'tag': q})
        except Exception as e:
            log("warn", f"Could not render RAG header template; omitting header. Error: {e}")
            header = ""
        files = {'RAG_CONTEXT.md': header + "\n\n---\n\n" + "\n\n".join(blocks)}
        if debug:
            debug_info = {'query': q, 'embedding': emb, 'documents': dbg_docs}
            return files, debug_info
        return files


    def build_context(self, files: Dict[str, str], *, k: int, count: int) -> Tuple[Dict[str, str], List[str]]:
        """Build per-run retrieval context and a list of candidate query tags for questions."""
        if self.cfg.no_rag:
            return files, []
        self._init()
        if not self._document_store:
            return files, []
        db_path = os.path.join(self.cfg.rag_persist, 'chroma.sqlite3')
        all_tags = self._fetch_unique_tags(db_path)
        if self.cfg.include_tags:
            req_lower = {t.strip().lower() for t in self.cfg.include_tags}
            tag_pool = [t for t in all_tags if t.strip().lower() in req_lower]
        else:
            tag_pool = list(all_tags)
        if not tag_pool:
            log("warn", "No tags found in store; RAG will proceed with empty context.")
            return {}, []
        queries = tag_pool
        return files, queries
