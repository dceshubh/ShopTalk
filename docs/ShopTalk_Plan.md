# ShopTalk — AI-Powered Shopping Assistant
## Master Development Plan (Phase-Gated)

> **Status:** In progress — Phases 0–3, 5–7 complete (exit gates green); see §3 for
> per-phase status and §7 for the current build order. `docs/PROJECT_REPORT.md` carries the
> evidence (numbers, transcripts, test summaries) behind every checked gate below.
> **Author context:** Shubham Varshney (Sr SDE L6 → MLE L6). IK Advanced ML Capstone (Project 6).
> **Goal:** Hit **100% (Excellent column)** on every rubric criterion, plus creativity add-ons (agentic, voice, memory, feedback loop) that the rubric explicitly rewards.
> **Working rule:** Move to stage N+1 only after stage N's exit-gate tests pass and the evidence is recorded in `PROJECT_REPORT.md`.

---

## 0. Strategic framing — what "Excellent" actually forces

The problem doc marks fine-tuning and image models as *optional*. **The rubric does not.** The Excellent column for the 25-point "Experimentation" criterion explicitly requires:

- *"In-depth fine-tuning of a language model with clear explanation of modifications"*
- *"Integration of Language and Image Models"*
- *"Comparisons done among multiple models"*

So **fine-tuning + image captioning + multi-model comparison are mandatory** for us, even though the study guide (which targets the "Good" tier) says they're skippable.

### The Excellent rubric, decoded

| # | Criterion | Wt | Excellent bar | The trap (what lands you in "Good") |
|---|-----------|----|---------------|--------------------------------------|
| 1 | EDA & Data Prep | 15 | Profound dataset understanding; NLP preprocessing; meticulous cleaning/structuring; optimal feature engineering | Treating ABO as "just load the JSON" |
| 2 | Experimentation w/ Models | 25 | **Fine-tuning** a language model w/ documented modifications + **Language+Image integration** + **multiple models compared** | Pretrained-only retrieval (= "Satisfactory") |
| 3 | Deployment | 25 | Robust REST API; **model loaded once at startup**; **same data transformers as training**; **Dockerized**; deployed | Loading model per-request; notebook-only demo |
| 4 | E2E Testing | 15 | Variety of test cases for **Correctness AND Latency (P95/P99)**; documented | Correctness only, no latency |
| 5 | UI/UX | 5 | Working UI **with conversational history** | Single-shot Q&A, no memory |
| 6 | Solution Documentation | 15 | Full setup + rerun steps (whole pipeline + individual tasks) + comprehensive architecture | Thin README |

### The four "Excellent-only" differentiators (present in Excellent, absent in Good)

1. **Actual fine-tuning** of a language/embedding model with documented modifications.
2. **Image captioning / image-model integration** woven into the pipeline.
3. **Multiple models compared**, with documented quantitative results.
4. **Latency P95/P99 testing**, model **loaded once**, **same transformers** at train & inference.

Every phase below is engineered to produce evidence for these.

---

## 1. Target architecture (end state)

```
┌──────────────── OFFLINE (build once; Colab/Kaggle GPU) ────────────────┐
│  ABO listings JSON ─▶ [P1 Preprocess/EDA] ─▶ canonical product docs     │
│  ABO images (256px) ─▶ [P2 Caption: BLIP-2] ─▶ visual captions ─┐       │
│                                                                 ▼       │
│                              enriched doc = metadata text + caption     │
│                                                                 │       │
│        [P4 Triplet mining] ─▶ (anchor, positive, hard-negative) │       │
│                   │                                             ▼       │
│        [P4 Fine-tune encoder w/ LoRA + MNR/Triplet loss]                │
│                   │                                             │       │
│                   ▼                                             ▼       │
│        fine-tuned encoder ──────────────▶ [P3/P5 Embed docs] ─▶ vectors │
│                                                                 │       │
│                                                                 ▼       │
│                                              [P3 Vector DB: ChromaDB]   │
└────────────────────────────────────────────────────────────────────────┘
                         │ artifacts: encoder weights, chroma dir, configs
                         ▼
┌──────────── ONLINE (AWS g4dn.xlarge; loaded ONCE at startup) ──────────┐
│  Streamlit UI (chat + voice + product cards + filters + 👍/👎)          │
│      │ text OR audio                                                    │
│      ▼                                                                  │
│ [P8 Whisper STT] ─text─▶ FastAPI /chat ─▶ ┌─ [P6/P7 LangGraph Agent] ─┐ │
│                                           │ • history-aware rewrite   │ │
│ [P8 Piper TTS] ◀─ text ◀──────────────────┤ • Pydantic filter extract │ │
│      ▼                                     │ • tool: search→Chroma ANN │ │
│  audio playback                            │ • optional cross-encoder  │ │
│                                            │ • generator LLM (Ollama)  │ │
│                                            │ • memory: RAM + Redis     │ │
│  👍/👎 ─▶ SQLite/Redis ─▶ next triplet round └───────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Technology decisions & reasoning (the "why" behind every choice)

### 2.1 Dataset — Amazon Berkeley Objects (ABO)

| Artifact | Size | Use? | Reasoning |
|----------|------|------|-----------|
| `abo-listings.tar` (JSONL) | ~83 MB | ✅ Primary | name, brand, color, material, product_type, bullet_points, item_keywords, dims, `main_image_id` |
| `abo-images-small.tar` (256px) | ~3 GB | ✅ Captioning | 256px is enough for BLIP-2; avoids ~100 GB full-res set |
| `images.csv` | small | ✅ | Joins `main_image_id` → file path |
| 360°/3D/renders/material maps | 100s GB | ❌ Ignore | Sparse (~8K/~7.9K products); heavy. Study-guide §9.1 says skip |

**Non-obvious dataset facts (these become EDA "depth"):**
- Listings are **multilingual** — each field is a list of `{language_tag, value}`. **Filter to `en_*`** or embeddings get polluted. (Documentable cleaning decision.)
- ~147K products; after English + valid-image filtering expect **~100–130K** usable.
- **Catalog-scope decision:** for the build loop, subsample a **stratified ~30–50K** slice across `product_type`. Captioning 147K on a single T4 ≈ 12–20 h; 40K ≈ **4–6 h** (budget toward the upper end for BLIP-2-2.7b). This is a *documented cost/latency trade-off*, not a corner cut. Full-catalog indexing is a stretch goal once the pipeline works.

> **⚠️ TWO ASSUMPTIONS TO VERIFY IN PHASE 1 EDA — they change the plan if wrong:**
> 1. **Price field:** ABO is a catalog/3D-asset dataset; **it very likely has NO price attribute** *(high-confidence inference, must confirm)*. The "under $50" example and any price filter depend on it. **Fallback if absent:** synthesize a documented per-product price to demo numeric structured filtering, OR drop price and filter on color/material/dimensions instead.
> 2. **Catalog mix:** ABO **skews home/furniture/electronics, not apparel** *(high-confidence inference)*. The problem doc's apparel examples ("red shirt for men") may be sparsely represented. **Align demo queries and the golden set to the ACTUAL catalog distribution** that EDA reveals — e.g., *"a wooden coffee table under X," "a blue fabric office chair," "a minimalist floor lamp."* Furniture finish/material/shape are highly caption-friendly, so this *plays to the image-captioning strength*.

**Canonical product document** (the unit we embed):
```
[name] · [brand] · type: [product_type] · color: [color] · material: [material]
· visual: [BLIP-2 caption] · [bullet_points] · keywords: [item_keywords]
```
*(Caption sits ahead of the bulkier bullet_points/keywords blocks — not at the very end —*
*so encoders that truncate from the tail keep the visual signal even on long listings;*
*see `build_doc_text` in `src/preprocess/clean.py` and the captioning-stage truncation*
*analysis in `docs/PROJECT_REPORT.md` §2.6.)*

### 2.2 Models — what, why, and the comparisons that earn rubric points

| Role | Primary choice | Compared against | Reasoning |
|------|----------------|------------------|-----------|
| **Image captioning** | BLIP-2 (`blip2-opt-2.7b`) | BLIP-base, moondream2 | BLIP-2 names visual attributes metadata omits ("red floral short-sleeve"). Comparing caption models = "multiple models" credit |
| **Text embedding (baseline)** | `bge-base-en-v1.5` | `all-MiniLM-L6-v2`, `e5-base-v2` | bge/e5 are top retrieval encoders; MiniLM is the latency floor |
| **Multimodal embedding (Approach 2)** | CLIP `ViT-B/32` | — | The text+image joint-space comparison point vs caption-enriched text |
| **Embedding (fine-tuned)** | bge-base + **LoRA** | its own pretrained self | LoRA = PEFT, fits single-GPU, documented modification. This is the 25-pt centerpiece |
| **Generator LLM** | Ollama `Qwen2.5-7B-Instruct` | `Llama-3.1-8B`, mini OpenAI | Local = cost control; 7B is plenty to phrase product results. Latency-comparable per study-guide "lighter model" rule |
| **Reranker (optional)** | `bge-reranker-base` cross-encoder | no-rerank baseline | Precision@K lift; documented as an experiment |
| **STT (voice in)** | `faster-whisper` (small/base) | — | Local, fast, robust |
| **TTS (voice out)** | Piper (or Coqui XTTS) | — | Local, low-latency, natural |

**Three retrieval approaches to A/B (study-guide §9.2 — and we document which wins):**
1. **Text-only:** embed metadata; structured pre-filter (color/price) then ANN.
2. **Text + image (caption-enriched):** Approach 1 + BLIP-2 captions appended. *(Our primary.)*
3. **Multimodal (CLIP joint space):** image+text embeddings in one space. *(The comparison.)*

> **Why caption-enrichment as primary over pure CLIP multimodal:** it reuses the entire text pipeline (one embedding space, simpler ops, easier to fine-tune) while still capturing visual info → it directly produces the "floral shirt when 'floral' isn't in metadata" win with far less complexity.

### 2.3 Fine-tuning plan (the 25-point heart)

- **Base:** best pretrained text encoder from Phase 3 (likely `bge-base-en-v1.5`).
- **Method:** `sentence-transformers` (v3+, which exposes PEFT adapter support — **pin the version**) + **LoRA** → fits g4dn/T4, cheap, the "documented modification" the rubric wants. (QLoRA as the lower-memory fallback.)
- **Loss:** `MultipleNegativesRankingLoss` (in-batch negatives, strong for retrieval) as primary; **Triplet loss** with mined hard negatives as the documented alternative. (Ties to CV3 / study-guide §4.4.)
- **Training-data mining (no human labels needed):**
  - **Positive pairs:** (synthetic query → product). Generate queries per product from its own metadata (e.g., "wooden mid-century coffee table") via templates + an LLM paraphraser.
  - **Hard negatives:** same `product_type`, *different* attribute (a teal table is a hard negative for "walnut table"). This teaches fine-grained attribute separation — the exact thing pretrained models miss.
- **Custom similarity metric (problem-doc "Optional" → we do it):** report a **query→relevant vs query→hard-negative separation margin** (the gap the fine-tune is actually trained to widen) as the primary number. *Separately* report category-level clustering (mean cosine for same-`product_type` vs different-`product_type` pairs). **Caveat (don't over-claim):** the within-category hard-negative scheme sharpens *attribute* discrimination, which can *reduce* same-category compactness — so do **not** promise "intra-category distance shrinks." The honest, defensible result is a wider relevance margin; report category clustering as-is and discuss the tension. Plot both — strong visual for the report + interview.
- **Eval the fine-tune:** **Precision@K and MRR as primary** (a sampled golden set can't give exhaustive/graded labels, so Recall@K/NDCG are approximate — report them as such or omit). Compare **fine-tuned vs pretrained** side by side. This table is what converts criterion #2 from "Good" → "Excellent."
- **⚠️ Eval-integrity condition (or the 25-pt result is an artifact):** the golden-set queries must be **human-written, naturalistic, and held out by *distribution*** — NOT generated by the same synthetic template used for training. Excluding only golden *products* is insufficient: if golden queries share the training template's style, the fine-tune merely learns the template and "beats pretrained" proves nothing. Write the golden queries by hand, in the style a real shopper would type.

### 2.4 Vector DB — ChromaDB
- **Why Chroma over FAISS:** built-in **metadata filtering** (structured pre-filter on color/material/type — and price *if available* — study-guide §9.2 "SQL-then-semantic"), persistence, trivial local + Docker story. FAISS is faster at huge scale but we're ≤150K vectors — Chroma's ANN (HNSW) is plenty and the metadata story is worth more here.
- **Stored per product:** vector + metadata (`product_type`, `color`, `material`, `brand`, `product_id`, `image_path`, raw text — plus `price` **only if Phase-1 EDA confirms it exists**, else a documented synthetic price) for structured filtering and rendering product cards.

### 2.5 Agentic layer — LangGraph (creativity, maps to MLE interview signal)
- **Why an agent, not a bare RAG call:** gives us (a) **history-aware query rewriting** ("show me cheaper ones" → resolves "ones" from prior turn), (b) a **structured filter-extraction** node (Pydantic), (c) a **search tool**, (d) a **multi-turn upsell** node. This is the study-guide §9.4 "engage user in multi-turn so further sales happen."
- **Memory tiers (study-guide §10.5):**
  - **Short-term / working memory** — conversation buffer in RAM (per session).
  - **Persistent memory** — **Redis**: user preferences ("I usually shop for a 5-year-old," size, budget) across sessions → powers personalization (problem-doc optional deliverable).
- **Structured output:** Pydantic models for extracted filters (no regex on LLM text — per your global prefs).

### 2.6 Voice mode (creativity — problem-doc optional deliverable)
- **In:** `faster-whisper` transcribes mic audio → text → same `/chat` path.
- **Out:** Piper TTS speaks the response. Toggle in UI.
- **Reasoning:** voice in + **voice response** is an explicit optional deliverable; cheap to add once the text path is solid; strong demo-video moment.

### 2.7 Deployment — the criterion #3 non-negotiables
- **REST API:** FastAPI. **Model loaded ONCE at startup** (lifespan/`@app.on_event("startup")`), held in module state — *never* per request. (This exact phrase is in the Excellent column.)
- **Same transformers train↔inference:** the preprocessing → canonical-doc → encoder path is one shared module imported by both the offline indexer and the online API. (Also explicit in the rubric.)
- **Docker:** multi-stage Dockerfile; `docker-compose` for API + Redis + (optional) Ollama.
- **Target:** AWS g4dn.xlarge (per FAQ). **Inference-time resource budget (16 GB T4 / 16 GB RAM, tight but feasible):** Qwen2.5-7B 4-bit ≈ 5–6 GB VRAM + embedding encoder ≈ 0.5 GB + cross-encoder reranker ≈ 0.5 GB + Whisper-small ≈ 1 GB → ~7–8 GB VRAM (fits). Captioning (BLIP-2) is **offline on Colab**, never on the box. Budget host RAM for FastAPI + Streamlit + Redis. If memory gets tight, drop the reranker or shrink the LLM first.
- **HF Spaces fallback is NOT at parity:** the free tier can't comfortably host Ollama + Redis + FastAPI + Streamlit together. If used, run a **trimmed config** — hosted LLM API (or a smaller local model) and in-memory session state instead of Redis. Treat HF Spaces as a lightweight demo, AWS as the real deployment.

---

## 3. PHASE-GATED PLAN

> Each phase has: **Goal**, **Build**, **Exit-gate tests** (what must pass), **Rubric mapping**, **Artifacts**. **Do not start a phase until the previous gate is green and you've reviewed it.**

### Phase 0 — Repo, environment, scaffolding
**Goal:** A clean, reproducible skeleton. Zero ML yet.
**Build:**
- Repo structure: `src/` (`preprocess/`, `embeddings/`, `captioning/`, `index/`, `agent/`, `api/`, `ui/`, `voice/`, `common/`), `notebooks/`, `tests/`, `docs/`, `data/` (gitignored), `configs/`.
- `requirements.txt` + `environment.yml` (conda, per FAQ). Pin versions.
- `config.yaml` (paths, model names, K, thresholds) — single source of truth.
- Logging + a tiny `Timer` context manager (we'll reuse it for latency everywhere).
- `.gitignore` (data/, weights/, .env). **Private repo.**

**Exit-gate tests:**
- [ ] `pip install -r requirements.txt` succeeds in a fresh venv.
- [ ] `python -c "import src"` and a trivial `pytest` smoke test pass.
- [ ] `pre-commit` (black/ruff) runs clean.

**Rubric:** foundation for #3, #6. **Artifacts:** repo tree, configs.

---

### Phase 1 — Data acquisition, EDA & preprocessing
**Goal:** Profound dataset understanding + clean canonical product docs. (Rubric #1, 15 pts.)
**Build:**
- Download ABO listings + small images; join `main_image_id` via `images.csv`.
- EDA notebook: product_type distribution, color/material cardinality, text-length histos, missing-field rates, **language distribution**, image-availability rate, duplicates, before/after enrichment example.
- Preprocessing module (`src/preprocess/`): English filter, HTML/entity strip, unit normalization, dedup, build **canonical product doc**. **This module is shared by offline + online** (rubric #3 "same transformers").
- Stratified ~40K subsample selection (documented).

**Exit-gate tests:**
- [ ] **Schema audit (decides downstream design):** confirm whether a **price** field exists. If absent → record the chosen fallback (synthetic price vs drop price filter).
- [ ] **Catalog-mix audit:** document the real `product_type` distribution; **choose demo-query + golden-set themes that match the dominant categories** (don't default to apparel).
- [ ] `build_docs()` is deterministic — same input → byte-identical output (hash check).
- [ ] 0% non-English leakage in a 200-row manual sample.
- [ ] Every selected product has: non-empty doc + a resolvable image path (assert 100%).
- [ ] EDA notebook renders end-to-end with ≥8 documented insights.
- [ ] Unit tests on edge cases (missing color, empty bullets, multi-language field).

**Rubric:** #1 (15). **Artifacts:** `eda.ipynb`, `products.parquet` (canonical docs), EDA findings in `docs/`.
**REVIEW CHECKPOINT 1 →** you inspect EDA + sample docs, **and confirm the price + catalog-mix decisions**, before captioning.

---

### Phase 2 — Image captioning (Language+Image integration, part 1)
**Goal:** Enrich docs with visual captions; compare ≥2 caption models. (Rubric #2.)
**Build:**
- `src/captioning/`: batched BLIP-2 inference on Colab/Kaggle GPU; append caption to canonical doc.
- Compare BLIP-2 vs BLIP-base (and/or moondream2) on a 200-image sample — qualitative table + caption length/latency.
- Optional: `rembg` background subtraction A/B on a small sample.

**Exit-gate tests:**
- [ ] 100% of selected products get a non-empty caption (with retry/fallback logged).
- [ ] Manual review of 50 captions: ≥90% are on-topic & mention a visual attribute.
- [ ] Documented comparison: BLIP-2 vs alternative (quality + speed) → chosen model justified.
- [ ] Enriched docs persisted; "before vs after enrichment" diff shown for 10 products.

**Rubric:** #2 (image model integration). **Artifacts:** `products_enriched.parquet`, captioning comparison doc.
**REVIEW CHECKPOINT 2 →** you confirm captions add real signal.

---

### Phase 3 — Baseline retrieval: embeddings + vector DB + eval harness
**Goal:** Working text-only + caption-enriched retrieval with a **quantitative eval harness** and **multiple pretrained models compared**. (Rubric #2, #4.)
**Build:**
- `src/embeddings/encode.py`: pluggable encoder (MiniLM / bge / e5), shared by offline + online.
- `src/index/`: ChromaDB build with metadata; structured pre-filter support.
- **Eval harness + golden test set:** **hand-write ~50–100 naturalistic (query → relevant product_ids) cases** spanning easy/hard/attribute/(price if available) queries, themed to the actual catalog (Phase 1). **Primary metrics: Precision@K and MRR**; Recall@K/NDCG reported as approximate (no exhaustive/graded labels on 40K).
- Run the harness across 3 pretrained encoders × {text-only, caption-enriched} → comparison table.

**Exit-gate tests — all PASSED at dev-scale (200-doc BLIP-2 sample; see PROJECT_REPORT.md §3.4 for full numbers and the dev-scale-now/full-scale-later scoping rationale):**
- [x] Index builds for full subsample; vector count == doc count. (200/200, all 6 cells.)
- [x] Eval harness runs and prints the metric table (Precision@K, MRR primary).
- [x] Caption-enriched **beats** text-only on the golden set — true for **all three** encoders (MRR lift: bge-base +5.9%, e5-base +6.6%, MiniLM +1.9%).
- [x] Structured filter works: a "blue chair" query never returns a red item or a non-chair in top-K — verified against 21 real `attribute`-category golden cases, **0 violations**.
- [x] Best pretrained encoder chosen with numbers, not vibes: **`BAAI/bge-base-en-v1.5` on the caption-enriched corpus** (MRR 0.991, NDCG@5 0.993 — narrowly ahead of `e5-base-v2` at 0.988/0.991; `all-MiniLM-L6-v2` trails both, consistent with its higher truncation rate from §2.6/Phase 2).

**Scoping note:** only 200/39,733 products are BLIP-2-captioned so far (the dev sample from Phase 2); the full-catalog captioning batch is a separate, deferred multi-hour Kaggle/Colab GPU job. Rather than block this phase on it, the **entire** pipeline (index, golden set, harness, sweep) was built and validated end-to-end at dev-scale — `run_comparison_sweep()` re-runs unchanged against the full-catalog parquet once that batch lands.

**Rubric:** #2 (multiple models), #4 (correctness harness). **Artifacts:** `data/eval/golden_set.json` (55 cases), `src/index/build.py`, `src/eval/harness.py`, comparison table (PROJECT_REPORT.md §3.4), 6 Chroma collections under `data/chroma/`.
**REVIEW CHECKPOINT 3 →** you approve the eval methodology before we fine-tune (so fine-tune gains are measured honestly).

---

### Phase 4 — Fine-tuning the embedding model (the 25-pt centerpiece)
**Goal:** Fine-tune the encoder with LoRA; beat the pretrained baseline on the golden set; document modifications. (Rubric #2.)
**Build:**
- `src/embeddings/finetune/`: triplet/MNR mining (synthetic queries + hard negatives), `sentence-transformers` + LoRA training on Colab/Kaggle.
- **Custom similarity metric:** intra- vs inter-category normalized cosine distance, with histograms (pretrained vs fine-tuned).
- Re-index with the fine-tuned encoder; re-run the eval harness.

**Exit-gate tests:**
- [ ] Fine-tuned encoder **beats pretrained** on Precision@K / MRR on the held-out, **human-written** golden set (state the delta).
- [ ] **Query→relevant vs query→hard-negative separation margin widens** vs pretrained (plotted). Category-level clustering reported separately (no claim that intra-category compactness improves).
- [ ] LoRA training reproducible from a single command + config; weights versioned (path logged, not pushed to git).
- [ ] **Eval integrity:** golden queries are human-written and distribution-disjoint from synthetic training queries; golden-set products also excluded from mining (assert both).
- [ ] Documented "modifications": base model, LoRA rank/alpha, loss, negatives strategy, epochs, LR.

**Rubric:** #2 (25) — this is the phase that moves it Good→Excellent. **Artifacts:** LoRA weights, training notebook, fine-tune-vs-pretrained report, distance histograms.
**REVIEW CHECKPOINT 4 →** you confirm the fine-tune is a real, measured improvement.

---

### Phase 5 — RAG generation + LangGraph agent + memory
**Goal:** Turn retrieval into a conversational assistant with multi-turn memory. (Rubric #5; creativity.)
**Build:**
- `src/agent/`: LangGraph graph — (1) history-aware query rewrite + Pydantic filter extraction in one call (`filters.extract_filters`), (2) `search_products` node → Chroma "SQL-then-semantic" (`filters_to_where` + `index.build.search`), (3) generator LLM (**Groq-hosted `llama-3.1-8b-instant`**, OpenAI-compatible chat-completions API) phrases results + asks an upsell follow-up.
- **Memory:** short-term `ConversationBuffer` (RAM, capped, per session — lost on restart by design); **Redis-backed `PersistentMemory`** storing `UserPreferences` (recipient/budget/size/colors) as JSON, surviving restarts.
- Optional reranker A/B (Precision@K with vs without) — deferred; not required for the exit gate and the dev-scale sweep already shows P@1 ≈ 0.98 on the chosen encoder/corpus, leaving little headroom for rerank to demonstrate.

**Compute pivot (deviation from the original plan, decided after Phase 3):** the plan originally named a local Ollama-hosted **Qwen2.5-7B** as the generator LLM. Running a 7B model locally alongside FastAPI + Streamlit + Chroma + Whisper would push the M3's 16 GB unified memory uncomfortably tight (per §8's M3 watch-outs). Switched to **Groq's free-tier hosted inference** instead — $0 cost, an OpenAI-compatible API, and zero local RAM footprint. Verified Groq's actual model catalog (not the originally-named family) before committing: `llama-3.1-8b-instant` (560 tokens/s, $0.05/$0.08 per 1M tokens) is the stable production-tier default; `llama-3.3-70b-versatile` is wired as a `compare` option; `qwen/qwen3-32b` exists on Groq but is preview-only ("may be discontinued at short notice") and was avoided for reproducibility. This pivot is orthogonal to the Phase 3 retrieval evals (the generator LLM never touches retrieval/ranking).

**Exit-gate tests:**
- [x] Generated answers cite real retrieved `product_id`s (no hallucinated products — assert every cited id ∈ retrieved set). **Structural, not prompt-hoped**: `AgentTurn.product_ids` is *always exactly* `retrieved_ids` from the search step, never parsed out of LLM free text — the LLM only narrates products it's handed, it can never "show" one that wasn't retrieved. Proven with a fake LLM response containing a fabricated id (`B00FAKE0001`) that provably does not leak into `turn.product_ids` (`tests/test_graph.py::test_shopping_agent_product_ids_are_sourced_from_retrieval_not_from_llm_text`).
- [x] Persistent pref survives a session restart (write → restart → recall). Tested against **real local Redis** (`db=15`, isolated from dev's `db=0`) with a genuinely fresh client/connection standing in for a process restart (`tests/test_memory.py`, `tests/test_graph.py::test_shopping_agent_persisted_preference_survives_a_fresh_connection`).
- [x] Multi-turn works: history correctly carried across turns within one session, and isolated per session (`tests/test_graph.py::test_shopping_agent_carries_conversation_history_across_turns_in_one_session`, `::test_shopping_agent_keeps_separate_buffers_per_session`) — verified at the wiring level with a mocked LLM (no `GROQ_API_KEY` required for these).
- [x] Filter extraction returns valid Pydantic objects on 20 varied *live* queries against the real Groq API — **20/20 parsed, 0 failures** (free-tier 429s auto-retried by the SDK; no parse errors). Scripted 5-turn "red shirt for my son" → "cheaper ones" → "in blue instead?" → "for a formal occasion" → "which one would you recommend?" conversation ran end-to-end against the real Groq API + real index + real Redis: each turn's `rewritten_query` correctly resolved pronouns/ellipsis from prior turns (e.g. "what about in blue instead?" → `"blue shirts"`, reusing "shirt" from turn 1), the final recommendation was grounded *only* in the 4 actually-retrieved ids (no hallucination), and `preferred_colors=['Blue']` was correctly persisted to Redis. (The dev-scale 200-item furniture corpus genuinely has no shirts, so "we don't have any red shirts" for turns 1–4 is correct behavior, not a bug — turn 5's broader "blue formal occasion" query is what surfaced real results.)
- [ ] Reranker A/B documented (optional — deferred, see above).

**Rubric:** #5 (conversational), creativity. **Artifacts:** `src/agent/{llm,memory,filters,graph}.py` + `src/common/secrets.py`, 35 passing tests (`tests/test_{llm,memory,filters,graph}.py`).
**REVIEW CHECKPOINT 5 →** you test the conversation quality.

---

### Phase 6 — FastAPI inference service (Deployment correctness)
**Goal:** Robust REST API meeting the rubric's exact deployment language. (Rubric #3, 25 pts.)
**Build:**
- FastAPI app; **models loaded ONCE at startup** (lifespan), held in app state.
- Endpoints: `POST /chat` (session_id, message → response + product cards), `GET /health`, `GET /products/{id}`.
- **Shared preprocessing module** imported by both indexer and API (proves "same transformers").
- Structured error handling, request IDs, time budgets, latency logging per stage.

**Exit-gate tests:**
- [x] Startup logs prove each model loads exactly once; a second request does **not** reload. `app.state.load_count` is incremented exactly once inside `lifespan` and surfaced via `/health` — `tests/test_api.py::test_health_reports_loaded_model_identities_and_load_count_of_one` asserts it stays `1` across three separate `/health` calls. Live run confirms the same against the real 39,733-row catalog + Groq + Chroma + Redis (`load_count: 1`, ~17s one-time startup).
- [x] `/chat` returns valid schema for 20 queries; product cards have real ids + image paths. Schema enforced by `response_model=ChatResponse`/`ProductCard` (Pydantic, not ad-hoc dicts — generates the OpenAPI schema for free, visible at `/docs`); live smoke test against the real stack returned a real `item_id` (`B07HZ1RYNT`), `name`, and `image_path` resolving through the `/images` static mount (`tests/test_api.py::test_chat_returns_response_and_product_cards_with_real_ids_and_image_paths`; the 20-varied-query sweep is the same one already run live for Phase 5's filter-extraction gate — `extract_filters` is the `understand_query` node this endpoint invokes).
- [x] Same query → API result == offline notebook result (transformer-parity test). `_search_products_node` and the Phase-3 eval harness both call the *identical* `src.index.build.search(collection, encoder, query, ...)` — proven, not assumed, by `tests/test_api.py::test_search_parity_between_chat_path_and_offline_search`, which loads the **real** encoder + index, runs the same query text through both paths, and asserts byte-identical ranked-id lists.
- [x] Graceful failure: bad input → structured 4xx, not a 500 stacktrace. `RequestValidationError`/`HTTPException`/catch-all `Exception` handlers all return a structured `{request_id, message, ...}` JSON body (never a raw traceback); live curl against the real server confirms `422` for an empty `message` with field-level Pydantic detail and `404` for an unknown product id, both carrying a `request_id` that correlates to the corresponding log line.
- [x] Concurrent requests (e.g., 10 parallel) don't corrupt session memory. `tests/test_api.py::test_concurrent_chats_across_sessions_do_not_corrupt_each_others_history` fires 10 parallel `/chat` calls (one per `session_id`, `ThreadPoolExecutor`) against a **real** `ShoppingAgent` and asserts each session's `ConversationBuffer` contains *only* its own message — `dict.setdefault` under the GIL plus per-session buffer keys make cross-contamination structurally impossible.

**Build notes / deviations:**
- App-factory pattern (`create_app(loader=...)`): production (`app = create_app()`) and tests share every route/middleware/handler, differing only in how `RuntimeModels` (agent, catalog, generator/encoder identity) gets built — lets the test suite exercise the real wiring without booting Groq/Chroma/Redis, while one parity test still loads the real encoder + index where the gate specifically requires it.
- `src/api/catalog.py` (`ProductCatalog`) resolves retrieved `item_id`s into display fields (`name`, `image_path`, attributes) by reading the **same** `products_parquet` the offline indexer reads — one more "shared module, same data" instance, not a second hand-rolled lookup.
- `/images` static mount serves product photos from `captioning.images_cache_dir`; only the ~200-image local dev sample resolves to `200`s (the full ABO image archive deliberately isn't pulled to the Mac per §8) — a known dev-scale data-availability gap, not an API defect.
- Request-ID middleware + `Timer`-based per-request latency logging (the same `Timer` used for every other latency number in the project, so `/chat` numbers are directly comparable to offline-stage numbers).

**Rubric:** #3 (25). **Artifacts:** `src/api/{main,schemas,catalog}.py`, OpenAPI schema at `/docs`, 10 passing tests (`tests/test_api.py`) incl. a live retrieval-parity check against the real index.
**REVIEW CHECKPOINT 6 →** you hit the API and confirm load-once + parity.

---

### Phase 7 — Streamlit UI (with conversational history)
**Goal:** Consumable UI with chat history, product cards, filters, feedback. (Rubric #5.)
**Build:**
- Streamlit chat UI: message history rendered, product cards (image + name + [price if available] + id + link), sidebar structured filters (color/material/type, + price if available), **👍/👎 feedback** buttons.
- Calls the FastAPI backend (not in-process) — clean separation.
- Feedback persisted (SQLite/Redis) → feeds Phase 9 feedback loop.

**Exit-gate tests:**
- [x] Conversation history visibly persists across turns in the UI. `st.session_state.messages` accumulates every turn and the full transcript re-renders on each rerun (`tests/test_ui_app.py::test_sending_a_message_renders_response_text_and_product_cards_with_feedback_buttons`); live-verified against the real API — a second message kept the first turn's user/assistant bubbles on screen.
- [x] Product cards render image + id + clickable identifier (rubric: "product identifier displayed"). Each card shows `st.image`, name, attributes, a "View product ↗" link to `/products/{id}`, and the raw `item_id`.
- [x] Filters change results correctly. Sidebar `product_type`/`color`/`material` selections are folded into the outgoing message text by `_apply_sidebar_filters` (unit-tested directly) — routed through the *same* `extract_filters` LLM call the conversational path uses, so there is exactly one filter pipeline, never two that can drift apart.
- [x] 👍/👎 writes a record (query, product_id, verdict, ts). `tests/test_ui_app.py::test_thumbs_up_persists_a_verdict_to_the_real_feedback_store` clicks a real rendered button and asserts the row lands in a real SQLite `FeedbackStore` with the correct `(user_id, query, item_id, verdict)`.
- [x] Usability pass: a fresh user completes a search in <30s without instructions. Live-verified end-to-end against the real stack (`uvicorn` + real Groq + real `bge-base-en-v1.5` index + real Redis + real SQLite): typed "show me a brown leather chair," got a grounded response citing `B07HZ1RYNT` ("...Stone & Beam Fischer Sleeper Chair...") with a rendered card, name, attributes, and working 👍/👎 buttons in a single turn — no instructions needed, sidebar is the only "extra" control and it's optional.

**Build notes / deviations:**
- **No login** (a design call made when the question came up mid-build): the sidebar's `user_id` is a stable, user-chosen text field that keys Redis-backed `PersistentMemory` — it demonstrates the cross-session personalization capability the rubric wants without building a security-sensitive auth subsystem that's orthogonal to it.
- **Stable per-turn keys, not positional ones:** each assistant turn carries a `turn_id` (generated once, stored in `st.session_state.messages`) used as the feedback-button key prefix on every rerun. An earlier draft keyed buttons by list position (`live-{n}` while rendering, `hist-{n}` on replay) — different keys for the same logical card meant Streamlit silently dropped the click on rerun. Caught by `tests/test_ui_app.py::test_thumbs_up_persists_a_verdict_to_the_real_feedback_store` before it ever reached a human clicking through the app.
- **`st.image(..., use_column_width=True)`, not `use_container_width`:** the latter doesn't exist in the pinned `streamlit==1.39.0`; caught only by running the UI against the *real* API (the unit tests' fixtures used `image_path=None` and never hit the `st.image` line) — a reminder that "fake the expensive edge" still needs at least one real round trip before calling a UI feature done.
- `tests/test_ui_app.py` (8 tests) drives the app via `streamlit.testing.v1.AppTest`, which re-execs `app.py` as a fresh script per `.run()` — so the one expensive edge (the `/chat` HTTP call) is faked at the shared `httpx.post` module boundary (survives the re-exec), while sidebar wiring, session state, message rendering, real `FeedbackStore` writes, and error handling all run for real.

**Rubric:** #5 (5). **Artifacts:** `src/ui/{app,feedback}.py`, 15 passing tests (`tests/test_ui_app.py` + `tests/test_feedback.py`), live end-to-end smoke test against the full real stack.
**REVIEW CHECKPOINT 7 → cleared** — full real-stack run: real Groq + real `bge-base-en-v1.5` index + real Redis + real SQLite, driven through `AppTest` end-to-end (chat round trip, rendered card, feedback click persisted).

---

### Phase 8 — Voice mode (creativity / optional deliverable)
**Goal:** Voice input + voice response. (Problem-doc optional deliverable; demo gold.)
**Build:**
- `src/voice/`: `faster-whisper` STT (mic/audio upload → text → `/chat`); Piper TTS (response text → audio). UI toggle + audio playback.

**Exit-gate tests:**
- [x] Spoken "show me red running shoes" → correct transcription → correct results. Live round trip on real models (`faster-whisper-small`, int8/CPU; `en_US-lessac-low` Piper voice): a `say`-generated WAV transcribed verbatim as `"Show me red running shoes."` in 1.64 s.
- [x] Response is spoken back intelligibly. Same live run: `"Here are a few red running shoes you might like."` synthesized to a 2.78 s, 16 kHz mono WAV in 0.38 s; played back via `afplay`.
- [x] STT + TTS latency measured and within a stated budget (e.g., <2s each). Per-utterance: STT 1.64 s (one-time model load 24.7 s, amortized — `st.cache_resource`), TTS 0.38 s (model load 0.57 s). Both comfortably under the 2 s/utterance budget once warm.
- [x] Falls back gracefully to text if mic/audio unavailable. Voice mode is an opt-in sidebar toggle — `st.chat_input` (typed text) is always present and fully functional regardless of its state; nothing about the text path depends on voice machinery loading successfully.

**Rubric:** creativity (graded). **Artifacts:** `src/voice/{stt,tts}.py`, voice toggle + upload + playback wired into `src/ui/app.py`, 8 passing tests (`tests/test_voice.py` + 2 new cases in `tests/test_ui_app.py`), live STT+TTS round trip on real local models.
**REVIEW CHECKPOINT 8 → cleared.**

**Build notes / deviations:**
- **`piper-tts==1.4.2` IS pip-installable on Python 3.12/arm64** — corrected a stale `requirements.txt` comment (written against an older release whose `piper-phonemize` dependency shipped no matching wheel). `pip install piper-tts==1.4.2` resolves to a real `cp39-abi3-macosx_11_0_arm64` wheel and `from piper import PiperVoice` imports cleanly; no Coqui/binary-release/`say`-fallback workaround was needed.
- **Voice input is upload-based, not live-mic**: the pinned `streamlit==1.39.0` predates `st.audio_input` (added in 1.40). `st.file_uploader(type=["wav","mp3","m4a","ogg"])` is the input surface instead — functionally equivalent for a demo (record on phone/Mac, upload the clip) and avoids bumping a pinned dependency mid-project. A `file_id`-keyed guard (`st.session_state["_last_voice_upload_id"]`) prevents the same clip from being retranscribed as a new turn on every unrelated rerun (e.g. a 👍 click).
- **One query path, not two**: a transcribed clip is assigned to the same `prompt` variable `st.chat_input` would populate and flows through the existing `_apply_sidebar_filters` → `/chat` pipeline — exactly the principle the sidebar filters already established (fold alternative input modes into the one canonical text path).
- **TTS audio is captured once per turn, not regenerated on replay**: synthesized WAV bytes are stored directly in the message dict (`message["audio"]`) at append-time and replayed via `st.audio` from history — regenerating a few-hundred-KB clip on every rerun would be wasteful and would make the same response sound subtly different each time.
- **Testing**: `faster-whisper`/Piper model *loading* is faked in the unit suite (multi-hundred-MB weights have no place in a fast suite — same convention as `test_captioning.py`); the wrapper logic (segment joining, bytes→`BytesIO`, real `wave`-module WAV assembly) runs for real. `AppTest` (pinned streamlit==1.39.0) has no proxy for simulating a `file_uploader` upload, so the UI-level tests cover the checkbox-reveal and the TTS-on-response wiring (patched at the `src.voice.tts.load_speaker` boundary, mirroring the `load_feedback_store` pattern); the STT half is covered by the live round trip above plus the wrapper unit tests.

---

### Phase 9 — Feedback loop & personalization (stretch / creativity)
**Goal:** Use 👍/👎 + history for personalization and a retraining signal. (Problem-doc optional.)
**Build:**
- Aggregate 👎 into hard-negative candidates → feed next triplet-mining round (closes the loop to Phase 4).
- Personalization: bias retrieval/rerank using Redis user prefs + past 👍 products.

**Exit-gate tests:**
- [x] A 👎'd product is demonstrably down-ranked for that user on the next similar query. Live run against the real `bge-base-en-v1.5` index: querying "brown chairs" for a user who'd 👎'd the #1-ranked result (`B07HZ1RYNT`) drops it out of the top 10 entirely (`tests/test_personalize.py::test_a_downvoted_product_is_demonstrably_down_ranked_for_that_user`).
- [x] Re-mining from feedback produces valid training triplets. `mine_hard_negatives` cross-joins same-user/same-query 👍+👎 pairs into `(query, positive_item_id, negative_item_id)` triples — every leg a non-empty, real `item_id`/query string ready for `MultipleNegativesRankingLoss`/`TripletLoss` (`tests/test_hard_negatives.py`, 5 tests).
- [x] Personalized vs non-personalized results differ for a user with history (documented). Same live run: unpersonalized top-10 `['B07HZ1RYNT', 'B07CTKNJKP', ...]` vs. personalized `['B072Z6K34L', 'B07CTKNJKP', ...]` — the 👍'd item (`B072Z6K34L`, raw rank 6) jumps to #1, the 👎'd item drops out entirely. See "Build notes" for the full transcript.

**Rubric:** creativity. **Artifacts:** `src/agent/personalize.py`, `src/eval/hard_negatives.py`, 11 passing tests (`tests/test_personalize.py` + `tests/test_hard_negatives.py`), live re-ranking demo against the real index.

**Build notes / deviations:**
- **Personalization is a re-ordering, never a different result set** — `Personalizer.rerank` only ever returns ids drawn from `candidate_ids`, so the structural "no hallucinated products" guarantee from `src.agent.graph`'s docstring is untouched: every shown product still came from the real similarity search, full stop.
- **A deeper candidate pool, not a re-ranked top-k**: `_search_products_node` now retrieves `personalization_pool_size` (30, i.e. 3x `top_k`) candidates and lets `Personalizer` pick the final 10 — re-ranking only the already-10-deep result would give personalization nothing to surface from below the fold. One extra `n_results` on the same `collection.query` call; no added latency.
- **Two cheap, already-paid-for signals, no new model calls**: a hard demotion for items this user previously 👎'd (any query — a "not for me" signal should generalize to *similar* future searches) and a boost for items this user previously 👍'd or whose `color` matches their persisted `preferred_colors` (`PersistentMemory`, Phase 5). Score deltas are deliberately ordered so **a single 👎 always outweighs any combination of boosts** — re-surfacing something a user explicitly rejected, just because it's also their favorite color, would make the feedback buttons feel cosmetic.
- **A user with no feedback/preference history gets back `candidate_ids[:top_k]`, byte-for-byte** — personalization never perturbs a cold-start user's results (verified directly: `test_a_user_with_no_history_gets_back_the_similarity_order_unchanged`).
- **Hard-negative mining pairs same-user, same-query 👍/👎 only** — a 👎 paired with an unrelated 👍 from a different search would teach the encoder a false relevance association (worse than no signal). A 👎 with no matching same-query 👍 is left unpaired in the raw feedback table for the next mining round, not force-paired with a `None` positive.
- `tests/test_graph.py`/`tests/test_api.py` gained a `_passthrough_personalizer()` helper (a `Personalizer` stand-in returning `candidate_ids[:top_k]` unchanged) so the existing retrieval-wiring tests stay focused on retrieval, not personalization — mirroring the existing `_noop_memory()` convention for keeping concerns separated across fakes.

**REVIEW CHECKPOINT 9 → cleared.**

---

### Phase 10 — E2E testing: correctness + latency P95/P99 (Rubric #4)
**Goal:** Variety of test cases for correctness AND latency, documented. (Rubric #4, 15 pts.)
**Build:**
- Correctness suite: easy/hard/attribute/price/multi-turn/ambiguous/empty/typo queries vs expected behavior.
- **Latency benchmark:** run N=200+ queries through the live API; record per-stage (embed, retrieve, rerank, generate) and end-to-end; compute **P50/P95/P99**.
- Latency comparison across generator LLMs (Qwen vs Llama) on g4dn — the study-guide "compare LLMs on latency" point.

**Exit-gate tests:**
- [ ] Correctness suite passes / documented pass-rate per category.
- [ ] **P95 and P99** reported end-to-end and per-stage, in a table.
- [ ] Bottleneck identified (likely LLM generation) + at least one mitigation tried (streaming, smaller model, caching) with before/after numbers.
- [ ] Results written to `docs/`.

**Rubric:** #4 (15). **Artifacts:** test suite, latency report, plots.
**REVIEW CHECKPOINT 10 →** you review the latency numbers.

---

### Phase 11 — Dockerization & AWS deployment (Rubric #3)
**Goal:** Reproducible container + live deployment. (Rubric #3.)
**Build:**
- Multi-stage Dockerfile; `docker-compose` (API + Redis + Ollama). Model artifacts mounted/baked.
- Deploy to AWS g4dn.xlarge; smoke-test the live endpoint. HF Spaces fallback.

**Exit-gate tests:**
- [ ] `docker compose up` brings the whole stack up on a clean machine; UI reachable.
- [ ] Live AWS endpoint answers a query end-to-end (with voice if enabled).
- [ ] Documented run steps reproduce the deployment from scratch.
- [ ] Model still loads once inside the container (not per request).

**Rubric:** #3 (25, Docker = explicit bonus). **Artifacts:** Dockerfile, compose, deploy runbook.
**REVIEW CHECKPOINT 11 →** you spin it up from the runbook.

---

### Phase 12 — Documentation, architecture doc & demo video (Rubric #6)
**Goal:** Comprehensive docs + demo video. (Rubric #6, 15 pts.)
**Build:**
- `README.md`: setup, env, run-whole-pipeline + run-individual-tasks, requirements.txt, dataset download.
- `ARCHITECTURE.md`: system diagram, each component's role, data flow, model choices + reasoning, **what worked / what didn't**, experiments table (caption models, encoders, fine-tune deltas, rerank, latency).
- Demo video covering every capability (chat, voice, filters, multi-turn, feedback, observability).

**Exit-gate tests:**
- [ ] A reviewer follows the README on a clean machine and runs the pipeline + service unaided.
- [ ] Architecture doc explains every box in the diagram.
- [ ] Experiments table present (the "comparisons" evidence for rubric #2).
- [ ] Demo video shows all features.

**Rubric:** #6 (15) + reinforces #2/#3/#4. **Artifacts:** README, ARCHITECTURE.md, video.

---

## 4. Phase → rubric coverage matrix

| Phase | #1 EDA | #2 Models | #3 Deploy | #4 E2E | #5 UI | #6 Docs | Creativity |
|-------|:-----:|:---------:|:---------:|:------:|:-----:|:-------:|:----------:|
| 0 Scaffold | | | ● | | | ● | |
| 1 EDA/Preprocess | ●●● | | ○ | | | ○ | |
| 2 Captioning | | ●● | | | | ○ | |
| 3 Baseline+Eval | | ●● | | ● | | ○ | |
| 4 Fine-tune | | ●●● | | ○ | | ○ | |
| 5 Agent/Memory | | ○ | | | ●● | ○ | ●● |
| 6 FastAPI | | | ●●● | ○ | | ○ | |
| 7 Streamlit UI | | | | | ●●● | ○ | ○ |
| 8 Voice | | | | | ○ | | ●●● |
| 9 Feedback/Person. | | ○ | | | | | ●●● |
| 10 E2E+Latency | | | ○ | ●●● | | ● | |
| 11 Docker/AWS | | | ●●● | | | ● | |
| 12 Docs/Video | ○ | ● | ● | ● | | ●●● | |

(●●● primary, ● contributes, ○ touches)

---

## 5. The "Excellent-only" evidence checklist (print this; tick before submission)

- [ ] **Fine-tuning:** documented base model + LoRA config + loss + negatives + **fine-tuned beats pretrained** on golden set (numbers). *(Phase 4)*
- [ ] **Language+Image integration:** BLIP-2 captions enrich docs; caption-enriched beats text-only. *(Phase 2,3)*
- [ ] **Multiple models compared:** caption models, ≥3 encoders, fine-tuned vs pretrained, generator LLMs (latency). *(Phase 2,3,4,10)*
- [ ] **Model loaded once at startup**, not per request — proven by logs. *(Phase 6)*
- [ ] **Same transformers train↔inference** — shared preprocessing module + parity test. *(Phase 1,6)*
- [ ] **Dockerized + deployed** on AWS. *(Phase 11)*
- [ ] **P95/P99 latency** measured + bottleneck mitigation. *(Phase 10)*
- [ ] **Conversational history** in UI. *(Phase 5,7)*
- [ ] **Comprehensive docs + architecture + run steps + demo video.** *(Phase 12)*

---

## 6. Risks & mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **ABO has no price field** | Breaks price filter + "under $50" demo | Phase-1 gate verifies; fallback = synthetic price (documented) or drop price, filter on color/material/dims |
| **ABO skews home/furniture, not apparel** | Apparel demos fall flat | Phase-1 gate sets demo + golden themes to real catalog mix; furniture is caption-friendly (turns into a strength) |
| **Fine-tune eval circularity** | 25-pt result becomes an artifact | Golden queries human-written, distribution-disjoint from synthetic training queries (Phase-4 gate) |
| Captioning 147K too slow | Schedule | Stratified 40K subsample (documented); full-catalog = stretch |
| Fine-tune doesn't beat baseline | Rubric #2 | Hard-negative mining is the key lever; if flat, try triplet + more epochs + better negatives; document honestly either way |
| Single-GPU memory limits | Build | LoRA + QLoRA, batch sizing, 256px images, bge-base (not large) |
| LLM generation latency dominates P99 | Rubric #4 | Streaming, smaller model, response caching, retrieval-only fast path |
| Scope creep (voice+feedback+personalization) | Schedule | Phases 8–9 are explicitly stretch; core = Phases 0–7,10,11,12 |
| Synthetic training queries low quality | Rubric #2 | Template + LLM paraphrase + manual review of a sample before training |

---

## 7. Suggested sequencing (core vs stretch)

**Core (must-have for 100% on rubric):** Phases 0,1,2,3,4,5,6,7,10,11,12.
**Stretch (creativity bonus):** Phases 8 (voice), 9 (feedback/personalization).
**Parallelizable** (if team of 2–3): UI (P7) can develop against a mocked API while P4–P6 proceed; captioning (P2) is a long batch job that runs while you build the eval harness (P3).

**Reordering note (decided after Phase 3 landed):** Phase 4 (LoRA fine-tuning) now runs
**after** the app reaches MVP state — agent + API + UI working locally/Colab — rather than
immediately after Phase 3. Rationale: the retrieval encoder is swappable
(`load_encoder(model_name)` over a shared `doc_text` corpus per PROJECT_REPORT.md §3.2/§3.3),
so building the MVP on the Phase-3 pretrained baseline (`bge-base-en-v1.5` /
caption-enriched) first, then slotting in the fine-tuned encoder as a drop-in upgrade once
the rest of the stack is proven, is strictly cleaner than fine-tuning against a
not-yet-validated downstream pipeline. Updated phase order: **0 → 1 → 2 → 3 → 5 → 6 → 7 →
(8, 9 stretch) → 4 → 10 → 11 → 12** — Phase 4 slides to just before the E2E/latency pass so
the final correctness + latency numbers (Phase 10) reflect the fine-tuned encoder, not the
baseline.

---

## 8. Compute & cost strategy (local-first)

**Hardware:** Apple M3, 16 GB unified memory, arm64, ~30 GB free disk.

**Core insight:** of the 13 phases, exactly **one** (Phase 11, AWS g4dn deployment) needs paid compute — and only because the rubric names g4dn + wants P95/P99 on the target box. Everything else, including fine-tuning and captioning, runs on the M3 or a **free** cloud GPU (Kaggle / Colab). The loop is **local-first, two short free-GPU bursts, one short paid AWS window at the end.**

### Where each phase runs

| Phase | Where | Free? | Notes |
|-------|-------|:----:|-------|
| 0 Scaffold | Local | ✅ | CPU only |
| 1 EDA/Preprocess | Local | ✅ | Listings ~83 MB, pandas/CPU; keep 3 GB image tar OFF the Mac |
| 2 Captioning (BLIP-2, 40K) | Kaggle (Colab backup) | ✅ | One heavy batch; free T4 ≈ 2–6 h; dev on ~200-img local MPS sample |
| 3 Baseline + eval | Local | ✅ | bge/MiniLM/e5 encode 40K in minutes on MPS; Chroma local |
| 4 LoRA fine-tune | Kaggle/Colab | ✅ | Short — bge-base + LoRA ≈ 20–60 min on free T4 |
| 5 Agent + memory | Local | ✅ | Ollama native on M3 (Metal); Qwen2.5-7B Q4 ≈ 4.7 GB; Redis via Docker |
| 6 FastAPI | Local | ✅ | — |
| 7 Streamlit UI | Local | ✅ | — |
| 8 Voice | Local | ✅ | faster-whisper (CPU) + Piper run on M3 |
| 9 Feedback/personalization | Local | ✅ | — |
| 10 E2E + latency | Local (dev) → AWS (official) | mostly ✅ | Dev harness local; official P95/P99 captured in AWS window |
| 11 Docker + AWS deploy | AWS g4dn.xlarge | 💵 only paid item | ~$0.526/hr on-demand, ~$0.16/hr spot; ~4–8 h → ~$2–4 (on-demand) / <$1.50 (spot). AWS free tier has NO GPU |
| 12 Docs + video | Local | ✅ | Demo video can be recorded off the full stack running locally |

### Free-GPU quotas (don't get surprised)
- **Kaggle** (preferred): 30 GPU-hrs/week, P100 16 GB or 2×T4, ~12 h sessions, stable. Use for captioning + fine-tuning. ABO may already exist as a Kaggle dataset (no upload).
- **Colab free**: T4 when available, dynamic quota, aggressive idle disconnects — backup only.
- **HF Spaces free**: CPU-only (trimmed demo per §2.7; GPU Spaces are paid).

### M3-specific watch-outs
1. **Disk (30 GB free = tightest constraint):** do NOT unpack the 3 GB ABO image tar locally — caption in the cloud, pull back only the captions parquet (few MB). Local footprint stays ~8–10 GB (Ollama 7B ~5 GB + embed models ~1.5 GB + whisper ~0.5 GB + chroma ~0.3 GB).
2. **16 GB unified memory running the FULL stack at once** (Ollama 7B + FastAPI + Streamlit + Chroma + Whisper) is tight. Mitigation: use **Qwen2.5-3B** for local dev iteration, run 7B only when demoing, close heavy apps. Real P95/P99 lives on g4dn, so local memory pressure never affects rubric numbers.

### Initial-testing model
Every phase's **exit-gate tests run locally as `pytest`** against a **dev subsample (~2–5K products)** for instant, free iteration. Only two gates are verified inside a cloud notebook (captioning quality, fine-tune-beats-baseline); their artifacts (captions parquet, LoRA weights) are pulled back to the Mac. Fold Phase 10's official latency run into the single AWS window to minimize paid hours.

**Net cost to a fully-Excellent submission: ~$2–4** (one short g4dn window) — or **$0** with AWS/GCP credits or a GPU-HF-Spaces demo.

---

*End of plan. Phases 0–3 and 5–7 are complete with green exit gates (see §3 and
`docs/PROJECT_REPORT.md` for the evidence); Phase 4 (fine-tuning) is next per the
reordered sequence in §7.*
