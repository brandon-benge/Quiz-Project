#!/usr/bin/env python3
from __future__ import annotations

"""
Interactive Q&A chat with sliding window memory and per-question RAG.

Design:
- Keeps an in-memory deque of the last N exchanges (question+answer pairs) as context.
- For each new user question: builds a prompt with prior window, enriches with RAG snippets
  looked up using the combined context + current question; queries Ollama; prints the answer.
- No disk writes for Q/A; only optional dumps for prompt/payload/response as configured.
"""

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

try:
    from .config import Config  # type: ignore
    from .llm_client import LLMClient, DumpOptions, OllamaGeneratePayload  # type: ignore
    from .rag import RAG  # type: ignore
    from .utils import log  # type: ignore
except Exception:  # pragma: no cover - fallback
    import sys
    from pathlib import Path as _P
    sys.path.append(str(_P(__file__).parent))
    from config import Config  # type: ignore
    from llm_client import LLMClient, DumpOptions, OllamaGeneratePayload  # type: ignore
    from rag import RAG  # type: ignore
    from utils import log  # type: ignore


@dataclass
class ChatConfig:
    window_size: int
    model: str
    temperature: float
    num_predict: Optional[int]
    top_k: Optional[int]
    top_p: Optional[float]
    rag_persist: str
    rag_k: int
    rag_embed_model: str
    no_rag: bool
    dump_ollama_prompt: Optional[str]
    dump_llm_payload: Optional[str]
    dump_llm_response: Optional[str]
    llm_retries: int
    ollama_keep_alive: Optional[str]
    ollama_url: Optional[str]
    http_timeout: Optional[int]


def _render_chat_prompt(history: List[Tuple[str, str]], question: str, rag_files: Dict[str, str]) -> str:
    """Render the chat prompt: history + optional RAG files + current question.

    History is a list of (user, assistant) pairs.
    """
    context_parts = []
    if rag_files:
        for p, txt in rag_files.items():
            context_parts.append(f"\n# FILE: {p}\n{txt}\n")
    if history:
        hist_lines = []
        for u, a in history:
            hist_lines.append(f"User: {u}\nAssistant: {a}")
        context_parts.append("\n# CONVERSATION_HISTORY\n" + "\n\n".join(hist_lines) + "\n")
    context_parts.append("\n# CURRENT_QUESTION\n" + question.strip() + "\n")
    # System guidance for concise answer
    context_parts.append("\n# INSTRUCTIONS\nAnswer the CURRENT_QUESTION using the provided context and conversation history.\nBe concise and factual. If unsure, say you don't know.\n")
    return "".join(context_parts)


def run_chat(cfg: ChatConfig) -> int:
    # Initialize LLM client and RAG
    client = LLMClient(DumpOptions(
        dump_prompt_path=cfg.dump_ollama_prompt,
        dump_payload_path=cfg.dump_llm_payload,
        dump_response_path=cfg.dump_llm_response,
    ), base_url=cfg.ollama_url, http_timeout=cfg.http_timeout)
    rag = RAG(type("_Tmp", (), {
        "rag_embed_model": cfg.rag_embed_model,
        "rag_persist": cfg.rag_persist,
        "no_rag": cfg.no_rag,
        "include_tags": None,
    }))
    # Initialize RAG components so embeddings/retriever are available
    try:
        rag._init()  # type: ignore[attr-defined]
    except Exception:
        pass

    history: Deque[Tuple[str, str]] = deque(maxlen=cfg.window_size)

    log("info", f"Chat started (model={cfg.model}, window={cfg.window_size}, rag_k={cfg.rag_k})")
    try:
        while True:
            try:
                q = input("You('exit' to quit): ").strip()
            except EOFError:
                print()
                break
            if not q:
                continue
            if q.lower() in {"exit", "quit", "/exit", "/quit", ":q", ":wq"}:
                break

            # Build RAG context for the combined text of history + question
            rag_query = ("\n".join([h[0] + "\n" + h[1] for h in list(history)]) + "\n" + q).strip()
            rag_debug: Optional[Dict[str, object]] = None
            if not cfg.no_rag:
                if cfg.dump_llm_payload:
                    out = rag.get_blocks_for_query(rag_query, cfg.rag_k, debug=True)
                    if out:
                        rag_files, rag_debug = out
                    else:
                        rag_files = None
                else:
                    rag_files = rag.get_blocks_for_query(rag_query, cfg.rag_k)  # type: ignore[assignment]
            else:
                rag_files = None
            prompt_text = _render_chat_prompt(list(history), q, rag_files or {})

            options: Dict[str, object] = {"temperature": cfg.temperature}
            if cfg.num_predict is not None:
                options["num_predict"] = cfg.num_predict
            if cfg.top_k is not None:
                options["top_k"] = cfg.top_k
            if cfg.top_p is not None:
                options["top_p"] = cfg.top_p

            payload = OllamaGeneratePayload(
                model=cfg.model,
                options=options,  # type: ignore[arg-type]
                prompt=prompt_text,
                retries=cfg.llm_retries,
                keep_alive=cfg.ollama_keep_alive,
            )
            # If we're dumping payloads, append a separate debug entry with embeddings and chat context
            if cfg.dump_llm_payload:
                try:
                    import json as _json
                    dbg = {
                        'type': 'chat_debug',
                        'chat_window': [{'user': u, 'assistant': a} for (u, a) in list(history)],
                        'rag_debug': rag_debug or {},
                    }
                    with open(cfg.dump_llm_payload, 'a', encoding='utf-8') as f:
                        f.write(_json.dumps(dbg, ensure_ascii=False) + '\n')
                except Exception:
                    pass
            text, raw, _ = client.run_ollama(payload)
            # Basic post-process: strip fences if any (client already tries), keep as answer text
            answer = text.strip()
            print(f"Assistant: {answer}")
            history.append((q, answer))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    return 0


def parse_args(argv: List[str]) -> ChatConfig:
    p = argparse.ArgumentParser(description="Interactive chat with sliding window and RAG")
    p.add_argument('--window', type=int, default=6, help='Sliding window size for prior Q/A pairs')
    p.add_argument('--model', default='mistral', help='Ollama model name')
    p.add_argument('--temperature', type=float, default=0.2)
    p.add_argument('--num-predict', type=int)
    p.add_argument('--top-k', type=int)
    p.add_argument('--top-p', type=float)
    p.add_argument('--rag-persist', default='../.chroma/sentence-transformers-all-mpnet-base-v2')
    p.add_argument('--rag-k', type=int, default=5)
    p.add_argument('--rag-embed-model', default='sentence-transformers/all-mpnet-base-v2')
    p.add_argument('--no-rag', action='store_true')
    p.add_argument('--dump-ollama-prompt')
    p.add_argument('--dump-llm-payload')
    p.add_argument('--dump-llm-response')
    p.add_argument('--llm-retries', type=int, default=2)
    p.add_argument('--ollama-keep-alive', help='Pass-through keep_alive duration for Ollama (e.g., 5m)')
    p.add_argument('--ollama-url', help='Override Ollama API URL (e.g., http://localhost:11434/api/generate)')
    p.add_argument('--http-timeout', type=int, help='HTTP timeout in seconds for LLM requests')
    a = p.parse_args(argv)
    return ChatConfig(
        window_size=a.window,
        model=a.model,
        temperature=a.temperature,
        num_predict=a.num_predict,
        top_k=a.top_k,
        top_p=a.top_p,
        rag_persist=a.rag_persist,
        rag_k=a.rag_k,
        rag_embed_model=a.rag_embed_model,
        no_rag=a.no_rag,
        dump_ollama_prompt=a.dump_ollama_prompt,
        dump_llm_payload=a.dump_llm_payload,
        dump_llm_response=a.dump_llm_response,
        llm_retries=a.llm_retries,
        ollama_keep_alive=a.ollama_keep_alive,
        ollama_url=a.ollama_url,
        http_timeout=a.http_timeout,
    )


def main(argv: List[str]) -> int:
    cfg = parse_args(argv)
    return run_chat(cfg)


if __name__ == '__main__':
    import sys
    raise SystemExit(main(sys.argv[1:]))
