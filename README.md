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
| Response generation | Groq-hosted LLM (`llama-3.1-8b-instant`) via a Pydantic-structured client |
| Re-ranking | Cross-encoder re-ranker (optional retrieval-quality experiment) |
| Speech-to-text | faster-whisper (CPU, int8) |
| Text-to-speech | Piper (local, ONNX voices) |
| Persistent memory | Redis |
| API | FastAPI (models loaded once at startup; Dockerized) |
| UI | Streamlit (chat history, product cards, filters, feedback, voice mode) |
| Testing | pytest — correctness suites + P95/P99 latency benchmarking |

---

## Running ShopTalk

ShopTalk has three places it can run, and they map to the three stages of the project's
lifecycle:

1. **Locally** (Apple Silicon / any machine with ~10 GB free) — the day-to-day dev loop:
   the API, the UI, the agent, voice mode, and `pytest` against a small dev-scale sample.
   This is where almost everything in the codebase is built and tested.
2. **Google Colab / Kaggle (free GPU)** — the two GPU-heavy *offline* steps that don't fit
   comfortably on a 16 GB-unified-memory laptop: full-catalog image captioning and LoRA
   fine-tuning of the retrieval encoder. Both produce small artifacts (a captions parquet,
   a LoRA adapter) that are pulled back to the laptop — the GPU is rented for an hour, not
   lived in.
3. **AWS (forward-looking)** — the only step that genuinely needs paid cloud compute: an
   official, reproducible P95/P99 latency benchmark of the fully-assembled service on a
   `g4dn.xlarge` GPU instance, plus a containerized deployment. Not yet built — see
   [AWS deployment](#3-aws-cloud-forward-looking) below for the planned shape of it.

`docs/ShopTalk_Plan.md` §8 ("Compute & cost strategy") captures the reasoning behind this
split in full — in short: **local-first, two short free-GPU bursts, one short paid AWS
window at the end**, for a total expected cloud cost of a few dollars.

### 1. Local setup

**Requirements:** Python 3.12+ (use Homebrew Python on macOS — the stock `python3` is 3.9
and breaks on modern type-hint syntax), ~10 GB free disk for models/weights/index, Redis.

```bash
# 1. Create and activate a project-specific virtual environment
python3.13 -m venv .venv-shoptalk
source .venv-shoptalk/bin/activate

# 2. Install dependencies (pinned — see requirements.txt for why each pin matters)
pip install -r requirements.txt

# 3. Configure secrets — copy the template and fill in a Groq API key
cp .env.example .env
# edit .env:
#   GROQ_API_KEY=<your key from https://console.groq.com/keys>   (free tier is plenty)
#   REDIS_URL=redis://localhost:6379/0                            (default, matches below)

# 4. Start Redis (no Docker required on macOS)
brew install redis && brew services start redis

# 5. Run the test suite — the fastest way to confirm the environment is sound
pytest -q
```

All paths, model identifiers, and tunables are centralized in
[`configs/config.yaml`](configs/config.yaml) — every module reads from this single source of
truth (via `src.common.config`), so nothing is hardcoded or duplicated across the codebase.

#### Acquiring the dataset

ShopTalk is built on the [Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
dataset. You need two of its archives, extracted under `data/raw/` (paths configured in
`configs/config.yaml: paths`):

| Archive | Size | Used for |
|---|---|---|
| `abo-listings.tar` (JSONL) | ~83 MB | Product metadata — name, brand, color, material, `product_type`, bullet points, keywords, dimensions, `main_image_id` |
| `abo-images-small.tar` (256px) | ~3 GB | Captioning input images (256px is plenty for BLIP-2 — avoids the ~100 GB full-res set) |

> **Disk tip for small machines:** the 3 GB image archive is only needed for the
> *captioning* stage, and that stage is designed to run on a free cloud GPU (see below) —
> not on the laptop. Don't unpack it locally; upload it directly to Kaggle/Colab as a
> dataset instead, and only pull back the small captions parquet it produces.

#### Running the offline pipeline (local, dev-scale)

Each stage is a runnable module that reads/writes the paths in `configs/config.yaml` and is
covered by its own `pytest` suite. Run them in order to go from raw archives to a queryable
index:

```bash
# Build the canonical products.parquet (cleaning, language filter, doc_text synthesis)
python -m src.preprocess.build

# Caption a local dev sample (~200 images) on Apple Silicon (MPS) — fast iteration;
# the FULL ~40K-product run is what you'd take to Kaggle (see below)
python -m src.captioning.enrich

# Compare encoders / retrieval approaches over the dev sample and write the sweep results
python -m src.eval.harness

# Build the Chroma vector index (src.index.build) — see its module docstring for the
# build_collection(...) call that wires encoder -> corpus_type -> persisted collection
python -c "from src.index.build import build_collection; build_collection(...)"
```

(Each of these has a corresponding `tests/test_*.py` suite that runs against tiny synthetic
or sampled data — `pytest -q` exercises the wiring of every stage without needing the full
40K-product run.)

#### Running the live service

The API and UI are two separate processes that talk over HTTP — exactly the shape the final
deployment takes, so "it works in dev" and "it works deployed" are the same claim.

```bash
# Terminal 1 — the inference API (loads all models ONCE at startup; ~15-20s cold start)
source .venv-shoptalk/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — the chat UI (talks to the API at configs/config.yaml: ui.api_base_url)
source .venv-shoptalk/bin/activate
streamlit run src/ui/app.py
```

Open the Streamlit URL it prints (typically `http://localhost:8501`). From there you can:
type a shopping query, narrow with sidebar filters (`product_type`/`color`/`material`),
👍/👎 individual product cards (this feeds personalization — see §6.10 of
`docs/PROJECT_REPORT.md`), and toggle **🎙️ Voice mode** to speak your query and hear the
response read back.

#### Voice mode — one-time setup

Voice mode (`faster-whisper` STT + Piper TTS) needs a Piper voice model downloaded once —
`requirements.txt` already pulls in `faster-whisper` and `piper-tts`, but voices are
~60 MB each and fetched separately:

```bash
python -m piper.download_voices --download-dir weights/piper en_US-lessac-low
```

This drops `en_US-lessac-low.onnx` + `.onnx.json` into `weights/piper/` (the path
`configs/config.yaml: paths.piper_voice_dir` expects). If you skip this step, voice mode's
checkbox still appears, but `load_speaker` raises a `FileNotFoundError` that names this
exact command — text chat is unaffected either way; speech is additive, never a hard
dependency.

#### Running the test suite

```bash
pytest -q                    # full suite — ~160 tests, ~4 minutes (some hit live APIs/models)
pytest tests/test_api.py -q  # one module
pytest -k "voice" -q         # by keyword
```

Tests follow a consistent "fake the expensive edge, test the wiring" split: cheap, fast unit
tests fake the one genuinely expensive call (an LLM API, a model load, an HTTP round trip)
at its module boundary, while everything else — real SQLite stores, real Redis (on `db=15`,
isolated from dev's `db=0`), real Chroma collections, real Pydantic validation — runs for
real. A handful of tests are explicitly *live* integration checks against the real Groq API
and the real dev-scale index; they `pytest.skip` gracefully if `GROQ_API_KEY` is unset or the
index hasn't been built yet, so the suite stays green in a minimal environment.

### 2. Google Colab / Kaggle (free GPU)

Two stages need more GPU memory than a 16 GB-unified-memory laptop comfortably offers, and
both are designed to run as **short, one-off bursts** on a free cloud GPU — rent the GPU for
an hour, pull back a small artifact, go back to working locally.

| Stage | Why it needs cloud GPU | What comes back |
|---|---|---|
| Full-catalog captioning (`Salesforce/blip2-opt-2.7b` over ~40K images) | The model is ~15 GB on disk / ~7-8 GB in fp16 — too large to dev-iterate against locally; the full run is also slow on CPU/MPS | `products_enriched.parquet` (captions folded into `doc_text`) — a few MB |
| LoRA fine-tuning of the retrieval encoder (`BAAI/bge-base-en-v1.5` + LoRA, per `configs/config.yaml: finetune`) | Needs a real GPU to be fast enough to iterate on (~20-60 min on a free T4 vs. much longer on CPU/MPS) | LoRA adapter weights — a few hundred MB |

**Kaggle is preferred over Colab** for this project: 30 free GPU-hours/week, P100 (16 GB) or
2×T4, ~12-hour sessions, and a far more stable scratch-disk story than Colab's aggressive
free-tier disconnects. [`notebooks/02_caption_comparison_kaggle.ipynb`](notebooks/02_caption_comparison_kaggle.ipynb)
is written specifically for Kaggle and documents the workflow end-to-end:

1. **Clone the repo** (`git clone --depth 1 https://github.com/dceshubh/ShopTalk.git`) —
   the notebook does this with an idempotent `rm -rf` first, to avoid a stale partial clone
   from an earlier attempt masking missing subpackages with a confusing `ModuleNotFoundError`.
2. **Don't `pip install -r requirements.txt`** — Kaggle's base image already ships
   torch/transformers/etc. at compatible (if not identical) versions, and reinstalling the
   full pinned set tends to trigger slow, fragile dependency resolution and C-ABI mismatches.
   Install only what's missing for the notebook's specific task.
3. **Force-load `numpy.random` first, before anything else** — numpy 2.x lazily defers
   loading that submodule via `__getattr__`, which can hide a C-ABI mismatch
   (`ValueError: numpy.dtype size changed...`) until it surfaces deep inside an unrelated
   import (e.g. `from src.captioning.enrich import build_enriched_dataset`). The notebook's
   first code cell exists purely to fail fast, with a clear message, before that happens.
4. **Check the GPU's compute capability before running fp16 ops** — Kaggle's free
   accelerators include both P100 (Pascal, `sm_60`) and T4 (Turing, `sm_75`); modern PyTorch
   wheels increasingly ship without `sm_60` kernel images to save space, producing
   `torch.AcceleratorError: CUDA error: no kernel image is available for execution on the
   device`. The notebook checks this up front — one GPU-second spent here saves a much more
   confusing failure later.
5. **Bring `products.parquet` in as an attached Kaggle Dataset**, not a re-clone of the raw
   ABO archives — upload `data/processed/products.parquet` once (small), attach it via
   *Add Input*, and the notebook copies it to the path `src.common.config` expects.
6. **Run both caption models over the identical sample** (`build_enriched_dataset` samples
   with the fixed `random_seed` from `configs/config.yaml: dataset`, so both runs see the
   exact same ~200 — or, for the full run, ~40,000 — images: a true apples-to-apples
   comparison, not two different catalog slices).
7. **Pull the resulting parquet(s) back down** to `data/kaggle_process/` (or
   `data/processed/` for the final enriched corpus) and resume working locally — the rest of
   the pipeline (encoding, indexing, agent, API, UI) runs entirely on the laptop from there.

The same shape applies to fine-tuning: attach the mined hard-negative triplets
(`src.eval.hard_negatives.mine_hard_negatives`, §6.10 of `docs/PROJECT_REPORT.md`) and the
base encoder as inputs, run the LoRA training loop from `configs/config.yaml: finetune`
against a free T4, and pull the resulting adapter weights back into `weights/`.

### 3. AWS cloud (forward-looking)

**Not yet built.** This is the only stage of the project that needs paid compute, and it's
deliberately scoped to a short, well-defined window at the end rather than something the
project lives in day-to-day. The plan (`docs/ShopTalk_Plan.md` §7-§8):

- **Target instance:** `g4dn.xlarge` (1× T4, 16 GB VRAM / 16 GB host RAM) — named directly
  in the assignment brief. Estimated inference-time footprint: a 4-bit ~7B local LLM
  (~5-6 GB VRAM) + the embedding encoder (~0.5 GB) + an optional cross-encoder reranker
  (~0.5 GB) + Whisper-small (~1 GB) ≈ 7-8 GB VRAM — fits with headroom. (Note: the *deployed*
  generator may differ from the dev-time Groq-hosted one — see `docs/PROJECT_REPORT.md` §5.1
  for why Groq was chosen for local dev, and why that choice is orthogonal to what runs on
  the GPU box.)
- **Cost:** ~$0.526/hr on-demand, ~$0.16/hr spot. A focused 4-8 hour window — enough to
  build the container, deploy, run the official P95/P99 latency benchmark, and capture
  evidence — comes to roughly **$2-4 on-demand, or under $1.50 on spot**. AWS's free tier
  has no GPU instances, so this is the one genuinely-paid step in the whole project.
- **Packaging:** a multi-stage Dockerfile plus `docker-compose` wiring the API, Redis, and
  (if used) a locally-hosted generator together — so "runs on my Mac" and "runs on the GPU
  box" are the same artifact, not two configurations that can drift apart.
- **What gets measured there:** the official P95/P99 end-to-end latency numbers
  (`docs/ShopTalk_Plan.md` §7) — captured on the *actual target hardware* the rubric names,
  not extrapolated from a dev laptop. Everything else (correctness suites, the dev-scale
  latency harness, the live demo) is designed to already be true before this window opens —
  the AWS step exists to *prove* it on the named hardware, not to discover whether it works.

This section will be filled in with real commands, a Dockerfile, and live-measured numbers
once that stage is built — see `docs/ShopTalk_Plan.md` for the up-to-date staged build order.

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
