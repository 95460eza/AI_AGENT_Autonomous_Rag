# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

A LlamaIndex-based autonomous AI agent that runs **locally** (privacy-focused) and reasons over a knowledge base of PDF documents using Mistral-Small-3.1-24B-Instruct. Capabilities include document summarization, targeted information retrieval, and multi-step comparative analysis.

Tested on Azure `Standard_NC24ads_A100_v4` (NVIDIA A100 40GB GPU, 24 vCPUs).

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `HUGFACE_AUTH_TOKEN` | Yes | HuggingFace auth token for downloading Mistral weights |
| `MISTRAL_API_KEY` | No | For Mistral API (alternative to local inference) |

## Running Locally

```bash
export HUGFACE_AUTH_TOKEN=<your_huggingface_token>
python app/app.py
```

## Docker Build & Run

```bash
# Build
docker build --build-arg PYTHON_DOCKER_IMAGE=python:3.12.9 \
  -t image-ai_agent_autonomous_reasoning_rag:latest .

# Run (GPU required)
docker run --gpus all -p 8000:5000 \
  -e HUGFACE_AUTH_TOKEN=<token> \
  image-ai_agent_autonomous_reasoning_rag:latest
```

The container runs Gunicorn with 4 workers on port 5000 internally.

## Architecture

### Data Flow

```
Input Query
  → FunctionCallingAgentWorker (LlamaIndex agent)
  → ObjectIndex (tool retrieval via vector embeddings)
  → Tool Execution: VectorStoreIndex or SummaryIndex per document
  → Multi-step LLM reasoning with tool calls
  → Final Response
```

### Key Components

**`app/app.py`** — Main entry point:
- `MistralLlamaIndexWrapper`: Custom `FunctionCallingLLM` subclass that wraps local Mistral inference, implements tool-calling by parsing structured output, and provides both sync and async interfaces.
- `create_tools_by_processing_documents()`: Iterates over PDFs in `llamaindex_datasets/`, calls `get_doc_tools()` for each, returns a flat list of all tools.
- `agent_with_tools_in_vector_store()`: Embeds tools into an `ObjectIndex` for semantic tool selection, then creates a `FunctionCallingAgentWorker` wrapped in an `AgentRunner`.

**`app/utilities/utils.py`** — `get_doc_tools()`:
- Loads a PDF → chunks with `SentenceSplitter` (chunk_size=1024)
- Builds a `VectorStoreIndex` (semantic search) and a `SummaryIndex` (full-doc summarization)
- Returns two LlamaIndex `FunctionTool` objects per document

### Document Storage

- `llamaindex_datasets/`: Place PDF files here for RAG ingestion (currently contains `metagpt.pdf` and `swebench.pdf`)
- `model_weights/`: Local Mistral model weights are downloaded here at runtime

### Key Dependencies

- **llama-index 0.12.29**: RAG orchestration (indices, agents, tools)
- **mistral-inference 1.6.0**: Local Mistral model inference
- **transformers** (latest from GitHub): HuggingFace model loading
- **torch 2.6.0** with CUDA 12.4: GPU inference
- **sentence-transformers 4.0.2**: Embedding model for vector indices

## Changing the Query

The agent query is hardcoded in `app.py`. To test with a different question, modify the `query` variable passed to `agent.query()` at the bottom of `app.py`.

## Adding Documents

Drop additional PDF files into `llamaindex_datasets/`. The `create_tools_by_processing_documents()` function automatically discovers and processes all PDFs in that directory.
