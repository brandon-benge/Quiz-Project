from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Config:
    count: int
    quiz: Path
    answers: Path
    no_random_component: bool
    model: str
    ollama_model: str
    ollama_temperature: Optional[float]
    ollama_num_predict: Optional[int]
    ollama_top_k: Optional[int]
    ollama_top_p: Optional[float]
    ollama_snippet_chars: int
    ollama_corpus_chars: int
    ollama_compact_json: bool
    debug_ollama_payload: bool
    dump_ollama_prompt: Optional[str]
    dump_llm_payload: Optional[str]
    dump_llm_response: Optional[str]
    template: bool
    seed: int
    avoid_recent_window: int
    verify: bool
    dry_run: bool
    rag_persist: str
    rag_k: int
    rag_queries: Optional[List[str]]
    rag_max_queries: Optional[int]
    rag_local: bool
    rag_embed_model: str
    no_rag: bool
    restrict_sources: Optional[List[str]]
    include_tags: Optional[List[str]]
    include_h1: Optional[List[str]]
    dump_rag_context: Optional[str]
    max_retries: int
    llm_retries: int
    ollama_keep_alive: str
    ollama_url: Optional[str]
    http_timeout: Optional[int]


def parse_args(argv: List[str]) -> Config:
    """Parse CLI arguments into a Config dataclass for quiz generation."""
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=5)
    p.add_argument('--quiz', default='quiz.json')
    p.add_argument('--answers', default='answer_key.json')
    p.add_argument('--no-random-component', action='store_true')
    p.add_argument('--model', default='mistral')
    p.add_argument('--ollama-model', default='mistral')
    p.add_argument('--ollama-temperature', type=float)
    p.add_argument('--ollama-num-predict', type=int)
    p.add_argument('--ollama-top-k', type=int)
    p.add_argument('--ollama-top-p', type=float)
    p.add_argument('--ollama-snippet-chars', type=int, default=-1)
    p.add_argument('--ollama-corpus-chars', type=int, default=-1)
    p.add_argument('--ollama-compact-json', action='store_true')
    p.add_argument('--debug-ollama-payload', action='store_true')
    p.add_argument('--dump-ollama-prompt')
    p.add_argument('--dump-llm-payload')
    p.add_argument('--dump-llm-response')
    p.add_argument('--template', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--avoid-recent-window', type=int, required=True, help='Avoid reusing the last N questions (required int argument)')
    p.add_argument('--max-retries', type=int, default=0, help='Maximum number of retries for LLM output parsing failures')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--rag-persist', default='../.chroma')
    p.add_argument('--rag-k', type=int, default=4)
    p.add_argument('--rag-queries', nargs='+')
    p.add_argument('--rag-max-queries', type=int)
    p.add_argument('--rag-local', dest='rag_local', action='store_true', default=True)
    p.add_argument('--rag-embed-model', default='sentence-transformers/all-mpnet-base-v2')
    p.add_argument('--no-rag', dest='no_rag', action='store_true')
    p.add_argument('--restrict-sources', nargs='+')
    p.add_argument('--include-tags', nargs='+')
    p.add_argument('--include-h1', nargs='+')
    p.add_argument('--dump-rag-context')
    p.add_argument('--llm-retries', type=int, default=2, help='Transport-level retries for provider calls (OpenAI/Ollama)')
    p.add_argument('--ollama-keep-alive', default='5m', help='Duration to keep Ollama model loaded (e.g., 5m, 30s, 1h)')
    p.add_argument('--ollama-url', help='Override Ollama API URL (e.g., http://localhost:11434/api/generate)')
    p.add_argument('--http-timeout', type=int, help='HTTP timeout in seconds for LLM requests')

    a = p.parse_args(argv)
    return Config(
        count=a.count,
        quiz=Path(a.quiz),
        answers=Path(a.answers),
        no_random_component=a.no_random_component,
        model=a.model,
        ollama_model=a.ollama_model,
        ollama_temperature=a.ollama_temperature,
        ollama_num_predict=a.ollama_num_predict,
        ollama_top_k=a.ollama_top_k,
        ollama_top_p=a.ollama_top_p,
        ollama_snippet_chars=a.ollama_snippet_chars,
        ollama_corpus_chars=a.ollama_corpus_chars,
        ollama_compact_json=a.ollama_compact_json,
        debug_ollama_payload=a.debug_ollama_payload,
        dump_ollama_prompt=a.dump_ollama_prompt,
        dump_llm_payload=a.dump_llm_payload,
        dump_llm_response=a.dump_llm_response,
        template=a.template,
        seed=a.seed,
        avoid_recent_window=a.avoid_recent_window,
        verify=a.verify,
        dry_run=a.dry_run,
        rag_persist=a.rag_persist,
        rag_k=a.rag_k,
        rag_queries=a.rag_queries,
        rag_max_queries=a.rag_max_queries,
        rag_local=True,
        rag_embed_model=a.rag_embed_model,
        no_rag=a.no_rag,
        restrict_sources=a.restrict_sources,
        include_tags=a.include_tags,
        include_h1=a.include_h1,
        dump_rag_context=a.dump_rag_context,
        max_retries=a.max_retries,
        llm_retries=a.llm_retries,
        ollama_keep_alive=a.ollama_keep_alive,
        ollama_url=a.ollama_url,
        http_timeout=a.http_timeout,
    )
