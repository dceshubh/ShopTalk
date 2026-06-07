# ShopTalk

**ShopTalk** is a multimodal, conversational shopping assistant built around Retrieval-Augmented
Generation (RAG). It lets a user describe what they're looking for in natural language —
including attributes that only show up in product *images*, not in catalog text — and returns
relevant products through a multi-turn, memory-aware conversation with optional voice
interaction.

The system is built on the [Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
dataset, a large open catalog of real product listings paired with images, attributes, and
metadata.

---

## What it does

A typical interaction looks like:

> **User:** "I'm looking for a mid-century style coffee table, something in a walnut finish."
> **ShopTalk:** *(returns matching products with images, names, and key attributes; remembers
> the conversation)*
> **User:** "Do you have anything similar but smaller?"
> **ShopTalk:** *(resolves "anything similar" from the prior turn, narrows by size)*

Underneath, ShopTalk combines:

- **Semantic + structured retrieval** — a vector index over product descriptions, augmented
  with metadata filters (color, material, category, dimensions).
- **Vision-language integration** — product images are captioned by a vision-language model
  and the captions are folded into the retrieval signal, so visual attributes that are absent
  from catalog text ("a tufted velvet headboard," "a brushed-steel finish") become searchable.
- **A fine-tuned retrieval model** — a sentence embedding model adapted (via parameter-efficient
  fine-tuning) on catalog-derived query/product pairs to sharpen attribute-level discrimination
  beyond what an off-the-shelf encoder achieves.
- **An agentic conversation layer** — a graph-based agent that rewrites follow-up queries using
  conversation history, extracts structured search filters, calls retrieval as a tool, and
  phrases results conversationally — with both short-term (in-session) and persistent
  (cross-session) memory.
- **Optional voice interaction** — speech-to-text input and text-to-speech responses.

---

## Architecture

ShopTalk is split into an **offline indexing pipeline** (run once, on a GPU, to build the
searchable index) and an **online inference service** (a REST API + UI that loads all models
once at startup and serves requests with measured latency).

```text
┌──────────────────── OFFLINE — build once (GPU) ────────────────────┐
│  Product catalog ─▶ preprocessing & EDA ─▶ canonical product docs   │
│  Product images  ─▶ vision-language captioning ──────────┐          │
│                                                           ▼          │
│                         enriched doc = catalog text + visual caption│
│                                                           │          │
│           hard-negative mining ─▶ fine-tune embedding model (LoRA)  │
│                                                           │          │
│                                                           ▼          │
│                              encode all docs ─▶ vector index (Chroma)│
└──────────────────────────────────────────────────────────────────────┘
                          │ artifacts: encoder weights, vector index, configs
                          ▼
┌──────────────── ONLINE — models loaded once at startup ────────────┐
│  Chat / voice UI                                                     │
│      │ text or audio                                                 │
│      ▼                                                               │
│  speech-to-text ─▶ REST API ─▶ ┌─ conversational agent ───────────┐ │
│                                │ • history-aware query rewriting   │ │
│  text-to-speech ◀──────────────┤ • structured filter extraction    │ │
│      ▼                         │ • retrieval tool → vector index   │ │
│  audio playback                │ • optional re-ranking             │ │
│                                │ • response generation (local LLM) │ │
│                                │ • short-term + persistent memory  │ │
│                                └────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

A detailed design document — covering dataset analysis, model selection and comparisons,
the fine-tuning methodology, evaluation strategy, and a staged development and testing plan —
lives in [`docs/ShopTalk_Plan.md`](docs/ShopTalk_Plan.md).

---

## Tech stack

| Layer | Technology |
|---|---|
| Dataset | Amazon Berkeley Objects (ABO) — listings + product imagery |
| Image captioning | Vision-language model (BLIP-2), compared against alternatives |
| Text embeddings | Sentence-embedding encoders, compared and then fine-tuned with LoRA |
| Multimodal embeddings | CLIP joint embedding space (comparison approach) |
| Vector store | ChromaDB (ANN search + metadata filtering) |
| Conversational agent | LangGraph — stateful, tool-using, memory-aware |
| Response generation | Locally hosted LLM via Ollama |
| Re-ranking | Cross-encoder re-ranker (optional retrieval-quality experiment) |
| Speech-to-text | faster-whisper |
| Text-to-speech | Local TTS engine |
| Persistent memory | Redis |
| API | FastAPI (models loaded once at startup; Dockerized) |
| UI | Streamlit (chat history, product cards, filters, feedback) |
| Testing | pytest — correctness suites + P95/P99 latency benchmarking |

---

## Getting started

**Requirements:** Python 3.12+, ~10 GB free disk for models and indices (datasets and raw
images are kept out of the repository).

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv-shoptalk
source .venv-shoptalk/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the test suite
pytest
```

All paths, model identifiers, and tunable parameters are centralized in
[`configs/config.yaml`](configs/config.yaml) — every module reads from this single source of
truth, so configuration never needs to be duplicated or hunted down across the codebase.

> Comprehensive setup instructions — including dataset acquisition, building the index from
> scratch, running the API and UI, and reproducing experiments — will be published in
> `docs/ARCHITECTURE.md` as the corresponding components are completed.

---

## Project structure

```text
src/
  preprocess/   # Data cleaning, normalization, and canonical product-document construction
  embeddings/   # Embedding encoders and LoRA fine-tuning
  captioning/   # Vision-language image captioning
  index/        # Vector index construction and retrieval
  agent/        # Conversational agent graph, tools, and memory
  api/          # REST inference service
  ui/           # Chat application UI
  voice/        # Speech-to-text / text-to-speech integration
  common/       # Shared utilities: configuration loading, logging, latency timing
notebooks/      # Exploratory analysis and experiment notebooks
tests/          # Test suites — unit, component, and end-to-end correctness/latency
configs/        # config.yaml — single source of truth for paths, models, and parameters
docs/           # Design documents, architecture notes, and experiment write-ups
data/           # Datasets, indices, and model artifacts (not committed to version control)
```

The same data-preparation and embedding code paths are shared between the offline indexing
pipeline and the online inference service, guaranteeing that documents are represented
identically at index time and at query time.

---

## Development approach

The project is being built incrementally, with each stage gated behind its own automated
test suite covering correctness and — where relevant — latency. This keeps the system in a
known-good, demonstrable state at every step rather than arriving at a working build only at
the end. The full staged plan, including the rationale behind every architectural and modeling
decision, is documented in [`docs/ShopTalk_Plan.md`](docs/ShopTalk_Plan.md).

## License

See [`LICENSE`](LICENSE).
