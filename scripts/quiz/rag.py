from __future__ import annotations

import os
import sqlite3
from pathlib import Path
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
        """Read unique tag values from Chroma's SQLite metadata table (best-effort)."""
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT string_value FROM embedding_metadata WHERE key='tags';")
            rows = cur.fetchall()
            conn.close()
            uniq: set[str] = set()
            for (val,) in rows:
                if not val:
                    continue
                for tok in str(val).split(','):
                    t = tok.strip().lower()
                    if t:
                        uniq.add(t)
            return sorted(uniq)
        except Exception as e:
            log("warn", f"Could not fetch tags: {e}")
            return []

    def get_blocks_for_tag(self, tag: str, k: int) -> Optional[Dict[str, str]]:
        """Retrieve top-k document snippets for a tag and build a contextual file for prompting."""
        if not self._document_store:
            return None
        q = tag.strip().lower()
        if not q:
            return None
        try:
            query_embedding = self._embedding.run(text=q)["embedding"]
            docs = []
            try:
                docs = self._retriever.run(
                    query_embedding=query_embedding,
                    top_k=k,
                    filters={"field": "tags", "operator": "==", "value": q}
                )["documents"]
            except Exception:
                docs = []
            if not docs:
                docs = self._retriever.run(
                    query_embedding=query_embedding,
                    top_k=k
                )["documents"]
        except Exception as e:
            log("warn", f"Per-question retrieval by tag failed: {e}")
            docs = []
        kept = []
        for d in docs:
            md = getattr(d, 'meta', {}) or {}
            tline = (md.get('tags') or '').lower()
            tag_list = [t.strip() for t in tline.split(',') if t.strip()]
            if q in tag_list or any((q == t or q in t) for t in tag_list):
                kept.append(d)
        docs = kept or docs
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
            req = [t.strip().lower() for t in self.cfg.include_tags]
            tag_pool = [t for t in all_tags if t in req]
        else:
            tag_pool = list(all_tags)
        if not tag_pool:
            log("warn", "No tags found in store; RAG will proceed with empty context.")
            return {}, []
        queries = tag_pool
        return files, queries
