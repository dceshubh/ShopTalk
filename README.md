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
   `g4dn.xlarge` GPU instance. The container it deploys is already built and runnable — see
   [Containerized deployment](#3-containerized-deployment-docker--aws) below for the real
   `docker compose` commands and the runbook for the live deploy.

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

**The one-command way — `scripts/shoptalk.sh`** (recommended; handles every starting state):

```bash
./scripts/shoptalk.sh          # (re)start: stops any stale ShopTalk processes, makes sure
                               #   Redis is running, boots the API + UI fresh in the
                               #   background. Run this whether nothing is running yet,
                               #   it's already running, or a previous run crashed and left
                               #   stale processes — all converge to the same clean state.
./scripts/shoptalk.sh status   # what's running right now, and the URLs to open
./scripts/shoptalk.sh down     # stop the API + UI — frees the ~2-3 GB of loaded models
                               #   (Groq client, bge-base-en-v1.5 encoder, Chroma index,
                               #   Whisper, Piper) when you're not actively using the app.
                               #   Redis is left running (it's lightweight, ~1 MB resident).
```

It logs to `/tmp/shoptalk_{api,ui}.log` (`tail -f` them to watch the ~15-20s model-loading
boot) and prints the URLs once both processes are launched:

- **API:** `http://localhost:8000/health`
- **UI:** `http://localhost:8501`

> **Port collisions:** the script identifies ShopTalk's own processes by full command line
> (scoped to the `.venv-shoptalk` path) — never by port — so it can never kill an unrelated
> project's server by mistake. It *can't*, however, fix a case where another project is
> already bound to `:8000` or `:8501`: if `http://localhost:8000/health` doesn't respond
> the way `docs/PROJECT_REPORT.md` §6.1 describes, run `lsof -nP -iTCP:8000 -sTCP:LISTEN`
> to see what's actually listening there, and either stop that other process or change
> `configs/config.yaml: api.port` / run Streamlit with `--server.port`.

**The manual way** (two terminals — useful for watching each process's logs directly):

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

### 3. Containerized deployment (Docker → AWS)

**The container is built and runnable; the live AWS deployment is the one remaining step**
(it needs an AWS account and a provisioned GPU instance — see "What's left for you to do"
below). The container is what makes "runs on my Mac," "runs via `docker compose`," and
"runs on an AWS g4dn box" the *same artifact* — not three configurations that can drift
apart — via the `REDIS_URL`/`API_BASE_URL` runtime overrides described in
`docs/PROJECT_REPORT.md` §8.

#### Run the whole stack with Docker, locally

This is the fastest way to prove "reproducible container, clean machine" before paying for
any cloud compute — and the *exact same* `docker compose up` is what the AWS runbook below
runs on the g4dn box.

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/)
(includes Compose v2), the dataset/index already built locally (`data/`, `weights/` —
see "Local setup" above; these are bind-mounted into the containers, not baked into the
image), and a filled-in `.env`.

```bash
# Build the image once (subsequent `up`s reuse the cached layers — only the final COPY
# layers rebuild when src/ changes)
docker compose build

# Bring up Redis -> API -> UI, in that dependency order (docker-compose.yml's
# `depends_on: condition: service_healthy` gates each service on the previous one's
# healthcheck — Redis must answer PING before the API starts; the API must answer
# /health before the UI starts)
docker compose up

# ...or detached, with logs on demand:
docker compose up -d
docker compose logs -f api ui
```

Open `http://localhost:8501` — identical experience to the local two-process run, because
it's the identical code behind it. Tear down with `docker compose down` (add `-v` to also
drop the named `redis-data` volume — preserved Redis persistence — if you want a truly
clean slate).

> **Not yet run end-to-end on this machine** — it has no Docker daemon installed. The
> `Dockerfile`/`docker-compose.yml`/`.dockerignore` are written, the image's dependency
> resolution matches the pinned `requirements.txt` exactly, and the runtime wiring
> (env-var address overrides, healthcheck-gated startup order, volume mounts) is unit-
> tested at the code level (`tests/test_memory.py::test_load_persistent_memory_*`,
> `tests/test_ui_app.py::test_config_*`) — but the literal `docker build && docker compose
> up` hasn't been run by me. That's first on the "what's left for you" list.

#### Deploying to AWS g4dn.xlarge (the runbook this will follow)

`g4dn.xlarge` (1× T4, 16 GB VRAM / 16 GB host RAM) is named directly in the assignment
brief. Estimated inference-time footprint there, summed across a 4-bit ~7B local LLM
(~5-6 GB VRAM, if you swap off the dev-time Groq-hosted generator — see
`docs/PROJECT_REPORT.md` §5.1 for why Groq was chosen for local dev and why that choice is
orthogonal to what runs on the GPU box), the embedding encoder (~0.5 GB), an optional
cross-encoder reranker (~0.5 GB), and Whisper-small (~1 GB): roughly 7-8 GB VRAM — fits
with headroom.

```bash
# On the g4dn instance (Deep Learning AMI — ships with NVIDIA drivers + Docker + the
# NVIDIA Container Toolkit preinstalled, so `docker compose` can already see the GPU):

git clone https://github.com/dceshubh/ShopTalk.git && cd ShopTalk
cp .env.example .env   # fill in GROQ_API_KEY (or point the generator at a local model)

# Pull the prebuilt index/model artifacts onto the box — e.g. `aws s3 cp --recursive
# s3://<your-bucket>/shoptalk-artifacts/{data,weights} ./{data,weights}/` — rather than
# rebuilding the whole offline pipeline on a billed-by-the-hour GPU instance.

docker compose up -d
curl http://localhost:8000/health   # smoke test: load_count == 1, all models reported
```

Then: open a security-group rule for `8501` (UI) — and `8000` if you want direct API
access — point a browser at `http://<instance-public-ip>:8501`, run the conversational +
voice smoke test end-to-end, and capture the official P95/P99 numbers (§7) by pointing the
existing latency harness at the live `/chat` endpoint instead of an in-process call.

**Cost:** ~$0.526/hr on-demand, ~$0.16/hr spot. A focused 4-8 hour window — provision,
deploy, smoke-test, capture P95/P99 evidence, tear down — comes to roughly **$2-4
on-demand, or under $1.50 on spot**. AWS's free tier has no GPU instances, so this is the
one genuinely-paid step in the whole project (`docs/ShopTalk_Plan.md` §8 has the full
local-first cost-strategy reasoning). **Stop or terminate the instance the moment you're
done** — a forgotten g4dn left running is the single most common way this kind of project
quietly burns $15-20+ overnight.

**HF Spaces fallback** (§2.7 of the plan): if AWS access is a blocker, a *trimmed* config —
hosted-LLM API (Groq, as in dev) instead of a local model, in-memory session state instead
of Redis — can run on HF Spaces' free CPU tier as a lightweight demo. Treat it as a backup
demo surface, not a parity deployment; the rubric names g4dn specifically, and that's what
the official P95/P99 numbers need to come from.

#### What's left for you to do

Everything above this line is built, tested, and (where the hardware allows) live-verified.
What remains needs things only you can provide — an AWS account, billing, and a Docker
daemon on this machine:

1. **Install Docker Desktop** on this Mac (or any machine with one) and run
   `docker compose build && docker compose up` — the one exit gate ("brings up the whole
   stack on a clean machine") that needs an actual Docker daemon to prove, not just inspect.
2. **An AWS account with billing enabled**, and either an access key pair or
   `aws configure sso` set up for the CLI/console — I can't create or fund this for you.
3. **Provision the `g4dn.xlarge` instance** (on-demand or spot — spot is ~3x cheaper and
   fine for a few-hour smoke-test window; just expect possible interruption), using a Deep
   Learning AMI so the NVIDIA driver/Container Toolkit story is solved for you.
4. **Open the security-group ports** (`8501`, optionally `8000`) to your IP — not `0.0.0.0/0`.
5. **Run the deploy runbook above**, smoke-test the live endpoint (including a voice-mode
   round trip), and capture the official P95/P99 latency numbers for §7 / the rubric.
6. **Stop/terminate the instance** the moment you're done capturing evidence — this is the
   step most likely to silently cost real money if forgotten.
7. *(Optional)* if you'd rather not spend anything, the HF Spaces fallback above needs a
   free Hugging Face account and a `git push` to a Space repo — zero AWS required, at the
   cost of not being "on the named target hardware" for the latency numbers.

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
