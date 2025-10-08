# Quiz Generation (Split Project)

This project contains the quiz-only workflow. It assumes you have already built a Chroma vector store in the parent directory (../.chroma) using the RAG-Workflow project.

Commands:

./scripts/bin/run_venv.sh               # create venv and install deps
./master.py prepare                     # generate quiz.json + answer_key.json (uses ../.chroma)
./master.py validate                    # interactive validation of your answers

Edit params.yaml to tweak models, counts, and retrieval.

Notes:
- LLM calls (Ollama-only) are centralized in scripts/quiz/llm_client.py for clarity.
- You can enable optional dumps for debugging:
	- dump_ollama_prompt: write prompt text to a file
	- dump_llm_payload: append the JSON request payloads
	- dump_llm_response: append the raw responses from the providers
 - Prompt templates are used by default and live in scripts/quiz/templates/.
	- Ollama: ollama_prompt.tmpl
	Templates use Python string.Template and support variables like ${token}, ${iteration}, ${count}, ${model}, ${theme}, ${recent_clause}, ${corpus}, ${style_clause}, and ${retry_nonce} on retries.
 - Retries:
	- Transport-level retries (prepare.llm_retries) apply to Ollama POSTs.
	- Parsing-level retries (prepare.max_retries) re-run the prompt and vary the theme to diversify outputs.
 - Embedding model: RAG embeddings are local-only (SentenceTransformers). If your Chroma store was built with a 768-dim encoder (e.g., all-mpnet-base-v2), ensure --rag-embed-model matches to avoid dimension mismatch errors. Default is sentence-transformers/all-mpnet-base-v2.
