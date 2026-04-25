# Rail-Drishti 🛤️

**AI-powered track health monitoring for Indian Railways.**

Rail-Drishti combines a machine learning defect classifier with a RAG pipeline over official maintenance manuals to generate structured, rule-cited maintenance action cards using Sarvam-1, India's own LLM. Every inference is logged to Delta Lake with full provenance — retrieved PWM/SoD page numbers, cosine similarity scores, and the generated action card — and surfaced through a live Flask dashboard that sorts urgent segments by track deterioration severity.

Built entirely on Databricks Free Edition serverless with Delta Lake, Unity Catalog, PySpark, and MLflow.

---

## The problem

Track inspectors currently read TRC (Track Recording Car) output — standard deviations of unevenness, alignment, cross-level, twist, and gauge — and manually cross-reference them against tables in the Permanent Way Manual to decide whether to impose speed restrictions or call a tamping crew. This is slow, inconsistent, and requires the inspector to know the correct paragraph of the correct manual for every defect combination. Rail-Drishti automates the interpretation: sensor values in, action card out, rule cited.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│               Databricks Free Edition — bricksiitm.rail_drishti          │
│                                                                          │
│  ┌─────────────────────────────────────┐                                 │
│  │  Synthetic Data Generator           │                                 │
│  │  IRPWM TQI formula (Para 511-522)   │                                 │
│  │  20 routes × 50 segs × 365 days    │                                 │
│  └───────────────┬─────────────────────┘                                 │
│                  │ ~365k rows                                            │
│  ┌───────────────▼─────────────────────┐  ┌──────────────────────────┐  │
│  │  Delta Lake (Unity Catalog)         │  │  PWM PDF  │  SoD PDF     │  │
│  │  track_health_train                 │  │  IRPWM    │  CS-27       │  │
│  │  track_health_test                  │  └─────┬─────┴──────────────┘  │
│  └───────────────┬─────────────────────┘        │ PyMuPDF extraction    │
│                  │ spark.table()                 │                       │
│  ┌───────────────▼─────────────────────┐  ┌─────▼──────────────────┐   │
│  │  sklearn RandomForestClassifier     │  │  raw_docs (Delta)      │   │
│  │  38 features → 5 defect classes     │  │  chunked_docs (Delta)  │   │
│  │  MLflow: F1, accuracy tracked       │  │  512 tok / 64 overlap  │   │
│  └───────────────┬─────────────────────┘  └─────┬──────────────────┘   │
│                  │ defect_type prediction         │ all-MiniLM-L6-v2    │
│                  │ risk_score_bucket              │ 384-dim embeddings  │
│                  │ urgent_maintenance_72h   ┌─────▼──────────────────┐  │
│                  │                          │  embedded_docs (Delta) │  │
│                  │                          └─────┬──────────────────┘  │
│                  │                                │ FAISS IndexFlatIP   │
│  ┌───────────────▼─────────────────────────────── ▼──────────────────┐  │
│  │  Inference Pipeline (04_inference.py)                              │  │
│  │                                                                    │  │
│  │  1. Build query from defect_type + risk_bucket (template + enrich) │  │
│  │  2. FAISS search → top-4 PWM chunks ≥ sim 0.35                    │  │
│  │  3. SoD targeted retrieval → top-2 SoD chunks ≥ sim 0.30          │  │
│  │  4. Deterministic rule engine (4 rules, fired before LLM)         │  │
│  │  5. Sarvam-1 → structured JSON action card                        │  │
│  │  6. Confidence banner appended (HIGH / MEDIUM / LOW)              │  │
│  └───────────────────────────────────┬────────────────────────────────┘  │
│                                      │ append                            │
│  ┌───────────────────────────────────▼────────────────────────────────┐  │
│  │  inference_log (Delta, Unity Catalog)                              │  │
│  │  inference_id, segment_id, defect_type, risk_bucket, TGI,         │  │
│  │  top_sim_score, chunk_sources, page_numbers, advisory_text        │  │
│  └───────────────────────────────────┬────────────────────────────────┘  │
│                                      │ PySpark read                      │
│  ┌───────────────────────────────────▼────────────────────────────────┐  │
│  │  Flask dashboard (app.py + index.html) — port 8050                 │  │
│  │  /api/dashboard   /api/predictions   /api/segment/<id>             │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Databricks Technologies

| Technology | Role in this project |
|---|---|
| **Delta Lake** | 4 Unity Catalog managed tables persist every pipeline stage: raw PDF pages (`raw_docs`), split text (`chunked_docs`), vector embeddings (`embedded_docs`), and per-inference outputs (`inference_log`). |
| **Unity Catalog** | Schema `rail_drishti` under catalog `bricksiitm`. All tables are Unity Catalog managed — full governance and lineage. |
| **PySpark** | `spark.createDataFrame`, `spark.table`, `write.format("delta").saveAsTable` used at every pipeline stage. The ML training reads from Delta tables via `spark.table().toPandas()`. |
| **MLflow** | Tracks 4 build runs: `pdf_ingestion`, `text_chunking`, `chunk_embedding`, `faiss_index_build`. FAISS index binaries logged as versioned artifacts. One MLflow run per inference with similarity score, token counts, TGI, and urgency flag. |
| **Databricks Serverless** | Fully serverless-compatible — no DBFS root writes, `/tmp/` for ephemeral files, Unity Catalog for persistence. All notebooks run on Databricks Free Edition. |

---

## Open-Source Models

| Model | Role |
|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 384-dim embeddings for chunk indexing and query encoding. Driver-side, batch_size=64, `normalize_embeddings=True`. No GPU required. |
| **FAISS** `IndexFlatIP` | Exact inner-product search on normalised vectors = exact cosine similarity. `index.reconstruct()` used for SoD-targeted retrieval from a filtered subset. |
| **Sarvam-1** (`sarvam-m`) | Indian LLM for action card generation. Chosen for Indian Railways domain terminology. REST API, temperature 0.2, max_tokens 800, JSON-only output. |

---

## Repository Structure

```
rail-drishti/
├── README.md
├── WRITEUP.md
├── DEMO_SCRIPT.md
├── bhashabench_eval.py          # BhashaBench evaluation script
│
├── Synthetic_Data_Generation_TQI.ipynb   # Stage 0: generate training data
├── Prediction_Model.ipynb                # Stage 1: ML classifier
├── 03_build_rag.ipynb                    # Stage 2: PDF ingestion + FAISS index
├── 04_inference.ipynb                    # Stage 3: per-segment inference
│
├── app.py                                # Flask dashboard server
└── templates/
    └── index.html                        # Dashboard frontend
```

---

## How to Run

### Prerequisites

- Databricks Free Edition account (`community.cloud.databricks.com`)
- Catalog `bricksiitm` visible (`SHOW CATALOGS` in a notebook)
- Two PDF files uploaded to your Workspace:
  - `IRPWM 2020 ACS 26-9-24 Final.pdf`
  - `SOD REVISED 2004 CORRECTED UPTO CS-27.pdf`

---

### Step 0 — Generate synthetic training data

Open `Synthetic_Data_Generation_TQI.ipynb` and run all cells. This generates `train_data.csv` and `test_data.csv` in your Workspace Drafts folder (~365k rows, route-based train/test split).

The generator simulates track geometry drift using the IRPWM 2020 TQI formula with NBML thresholds from Para 522. Five defect types are injected probabilistically based on sensor readings.

---

### Step 1 — Train the ML classifier

Open `Prediction_Model.ipynb` and run all cells in order.

```
Cell 1 → Load CSVs from Workspace Drafts
Cell 2 → Write to Delta Lake (track_health_train, track_health_test)
Cell 3 → Verify tables
Cell 4 → Feature prep + label encoding
Cell 5 → Train RandomForestClassifier, log to MLflow
Cell 6 → Save model pickle to Workspace
Cell 7 → Run inference, derive risk_score_bucket + urgency
Cell 8 → Save rag_input_df.csv for notebook 04
```

After Cell 5 you will see F1 and accuracy in the output. After Cell 7 you will see the defect type and risk bucket distribution for the test set.

---

### Step 2 — Build the RAG index

Install dependencies once per session:

```python
%pip install pymupdf sentence-transformers faiss-cpu langchain langchain-text-splitters
dbutils.library.restartPython()
```

Open `03_build_rag.ipynb`. Set your workspace username in Cell 3:

```python
WORKSPACE_USER = "your_username_here"   # visible in sidebar
CATALOG        = "bricksiitm"
SCHEMA         = "rail_drishti"
```

Run cells in order: **Cell 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9**

| Cell | What it does | Output |
|---|---|---|
| 2 | Install deps | — |
| 3 | Config | Paths + table names |
| 4 | Show catalogs | Verify bricksiitm visible |
| 5 | Create schema | `rail_drishti` schema created |
| 6 | Ingest PDFs | `raw_docs` Delta table |
| 7 | Chunk text | `chunked_docs` Delta table |
| 8 | Embed chunks | `embedded_docs` Delta table |
| 9 | Build FAISS | `track_index.bin` + MLflow artifact |

After Cell 9, note the run ID printed — you'll need it if the session restarts.

---

### Step 3 — Run inference

Install dependency:

```python
%pip install faiss-cpu sentence-transformers
```

Open `04_inference.ipynb`.

The notebook auto-recovers the FAISS index from MLflow if `/tmp/` has been wiped (Cell 9). No manual run ID needed — it searches by run name `faiss_index_build`.

Run cells: **9 → 10 → 11 → 12 → 13 → 14**

To run inference on a custom segment, edit the `sample_prediction` dict in Cell 10:

```python
sample_prediction = {
    "segment_id":             "SEG_042",
    "defect_type":            "thermal_stress",     # from ML classifier output
    "risk_score_bucket":      "urgent",             # from post-prediction logic
    "urgent_maintenance_72h": 1,
    "TGI":                    35.0,
    "days_since_tamping":     195,
    "subgrade_type":          "poor",
    "temp_delta":             31.0,
    "bridge_flag":            1
}
```

Each run appends one row to `inference_log` and logs one MLflow run under `rail_drishti_rag`.

---

### Step 4 — Start the dashboard

In a notebook cell:

```python
import subprocess
subprocess.Popen(["python", "/path/to/app.py"])
```

Or from terminal:

```bash
python app.py
```

Open `http://localhost:8050`. The dashboard polls the Delta tables live — reload the page after running new inferences to see updated rows.

---

### Session restart recovery

> ⚠️ If your Databricks session restarts, `/tmp/` is wiped and the FAISS index is lost.

Re-run **Cell 9 of `04_inference.ipynb` only**. It reads from the persistent `embedded_docs` Delta table and rebuilds the FAISS index in under 2 minutes. Do not re-run Cells 6–8.

---

## Metrics

| Metric | Value |
|---|---|
| Training rows | ~365,000 (20 routes × 50 segments × 365 days) |
| ML features | 38 |
| Defect classes | 5 (none, geometry_defect, fastening_sleeper_fault, ballast_subgrade_weakness, thermal_stress) |
| Chunk size | 512 tokens |
| Chunk overlap | 64 tokens |
| Embedding model | all-MiniLM-L6-v2 (384-dim) |
| FAISS index type | IndexFlatIP (exact cosine) |
| PWM chunks per inference | 4 (min sim 0.35) |
| SoD chunks per inference | 2 (min sim 0.30) |
| LLM temperature | 0.2 |
| Confidence thresholds | HIGH ≥ 0.50 / MEDIUM ≥ 0.40 / LOW < 0.40 |
| Delta tables | 4 |
| MLflow runs (build) | 4 |

---

## Design decisions

**Route-based train/test split.** Splitting by `route_id` rather than time means the model must generalise to unseen route profiles — harder than predicting future days on known routes, more realistic for deployment.

**Deterministic rule engine before LLM.** Hard operational rules (tamping overdue, TGI below NBML, thermal patrol triggers from PWM Para 347) run before Sarvam-1 is called. Their output is injected into the prompt verbatim so the LLM cannot contradict them.

**Dual-source retrieval.** PWM and SoD are retrieved with separate queries and similarity thresholds. SoD CS-27 has denser tabular content (tolerance tables, speed restriction triggers) that responds to different query phrasing than PWM narrative text. Merging them in source order gives Sarvam-1 both the procedural rule and the numeric tolerance in context.

**Session-restart-safe FAISS.** Logging the FAISS index as an MLflow artifact and recovering it with `client.download_artifacts` means the RAG pipeline survives Databricks free-tier compute recycling without re-embedding.

**Confidence banner.** If retrieval similarity is below 0.40, the action card shows a red "DO NOT ACT" banner and instructs the engineer to escalate. The system is designed to know its own limits.

---

## License

MIT
