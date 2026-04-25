# Rail-Drishti 🛤️

**Rail-Drishti** is an AI-powered track health monitoring system for Indian Railways that combines a Spark MLlib defect classifier with a RAG pipeline—retrieving rules from official maintenance manuals (PWM + SoD) and generating actionable maintenance cards via Sarvam-1, an Indian LLM.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  Databricks Free Edition  (bricksiitm.rail_drishti)     │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐        ┌───────────────────────┐   │
│  │  PWM PDF     │  │  SoD PDF     │        │  Sensor data (CSV)    │   │
│  │  IRPWM 2020  │  │  CS-27       │        │  Track geometry feat. │   │
│  └──────┬───────┘  └──────┬───────┘        └───────────┬───────────┘   │
│         │                 │                            │               │
│  ┌──────▼─────────────────▼──────┐        ┌───────────▼───────────┐   │
│  │  Cell 5 — PDF ingestion       │        │  GBTClassifier        │   │
│  │  PyMuPDF → raw_docs (Delta)   │        │  (Spark MLlib)        │   │
│  └──────────────┬────────────────┘        └───────────┬───────────┘   │
│                 │                                      │               │
│  ┌──────────────▼────────────────┐        ┌───────────▼───────────┐   │
│  │  Cell 6 — Chunking            │        │  track_predictions    │   │
│  │  512/64 → chunked_docs        │        │  (Unity Catalog Delta)│   │
│  └──────────────┬────────────────┘        └───────────┬───────────┘   │
│                 │                                      │               │
│  ┌──────────────▼────────────────┐        ┌───────────▼───────────┐   │
│  │  Cell 7 — Embedding           │        │  Cell 10 — Query      │   │
│  │  all-MiniLM-L6-v2 384-dim     │        │  builder + enrichment │   │
│  └──────────────┬────────────────┘        └───────────┬───────────┘   │
│                 │                                      │               │
│  ┌──────────────▼────────────────┐  query  ┌──────────▼────────────┐  │
│  │  Cell 8 — FAISS index         ├─────────►  Cell 11 — Retrieval  │  │
│  │  IndexFlatIP /tmp/ + MLflow   │         │  top-4 chunks ≥ 0.35  │  │
│  └───────────────────────────────┘         └──────────┬────────────┘  │
│                                                        │               │
│                                            ┌───────────▼────────────┐  │
│                                            │  Sarvam-1 API          │  │
│                                            │  Cell 13 — action card │  │
│                                            └──────────┬─────────────┘  │
│                                                        │               │
│  ┌─────────────────────────────┐           ┌──────────▼─────────────┐  │
│  │  inference_log (Delta)      ◄───────────┤  Cell 14 — Log output  │  │
│  │  Unity Catalog + MLflow     │           └────────────────────────┘  │
│  └──────────────┬──────────────┘                                       │
│                 │                                                       │
│  ┌──────────────▼─────────────────────────────────────────────────┐   │
│  │        Flask app (app.py) — port 8050                           │   │
│  │  /api/dashboard   /api/predictions   /api/segment/<id>          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Databricks Technologies Used

| Technology | Role |
|---|---|
| **Delta Lake** | 4 Unity Catalog tables: `raw_docs`, `chunked_docs`, `embedded_docs`, `inference_log` |
| **Unity Catalog** | Schema `rail_drishti` under catalog `bricksiitm`; full governance |
| **Spark MLlib** | `GBTClassifier` — defect type + risk bucket classification |
| **PySpark** | DataFrame reads/writes at every pipeline stage |
| **MLflow** | Tracks 4 pipeline runs; FAISS index logged as versioned artifact |
| **Databricks Serverless** | All code serverless-compatible (no SparkContext, no DBFS root) |

## Open-Source Models

| Model | Role |
|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 384-dim embedding, driver-side, no GPU needed |
| **Sarvam-1** (`sarvam-m`) | Indian LLM for maintenance card generation (external API) |
| **FAISS** `IndexFlatIP` | Exact cosine-similarity vector search |

---

## How to Run

### Prerequisites

- Databricks Free Edition account at `community.cloud.databricks.com`
- Workspace: ensure catalog `bricksiitm` is visible (`SHOW CATALOGS`)
- Upload the two PDF manuals to `/Workspace/<your-username>/`

### Step 1 — Install dependencies (run once per session)

```python
%pip install PyMuPDF faiss-cpu sentence-transformers langchain mlflow
dbutils.library.restartPython()
```

### Step 2 — Build the RAG index (`03_build_rag.py`)

Run cells **in order**: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8

```python
# Cell 3 — set your workspace username
WORKSPACE_USER = "bharatbricksiim"   # ← change this
CATALOG        = "bricksiitm"
SCHEMA         = "rail_drishti"
PDF_DIR        = f"/Workspace/{WORKSPACE_USER}/"
FAISS_DIR      = "/tmp/rail_drishti/faiss/"
```

After Cell 8 completes you will see:
- `track_index.bin` + `chunk_metadata.json` in `/tmp/rail_drishti/faiss/`
- MLflow run `faiss_index_build` with both files as artifacts

### Step 3 — Run inference (`04_inference.py`)

```python
# Cell 9 — load FAISS from MLflow artifact (session-restart-safe)
MLFLOW_RUN_ID = "<paste run ID from Cell 8 output>"
```

Run cells 9 → 10 → 11 → 12 → 13 → 14. Each call appends a row to `inference_log`.

### Step 4 — Start the Flask dashboard

```bash
python app.py
# Open the Databricks tunnel URL → port 8050
```

Or in a notebook cell:

```python
import subprocess
subprocess.Popen(["python", "app.py"])
```

Then open `http://localhost:8050` (or the Databricks tunnel URL shown in the cell output).

---

## Demo Steps

### What to click / what to run

1. **Open the dashboard** at `http://localhost:8050`
   - Top bar shows live counts: total inferences, urgent segments, watchlist segments, average similarity score
   - Defect breakdown donut updates in real time

2. **Browse the segment table**
   - Rows are sorted by TGI ascending (worst first)
   - Red `OVERDUE` badge = urgent + ≥ 2 triggered rules
   - Amber `ELEVATED` badge = urgent or ≥ 1 triggered rule

3. **Click any segment row** to expand inference history
   - Shows retrieved chunk sources (PWM page / SoD page)
   - Displays the Sarvam-1 generated maintenance action card
   - Cosine similarity score per retrieval

4. **Trigger a new inference from the notebook**

   ```python
   # In 04_inference.py Cell 10 — change segment and re-run:
   test_segment = {
       "segment_id"        : "SEG_042",
       "defect_type"       : "thermal_stress",
       "risk_score_bucket" : "urgent",
       "TGI"               : 36.5,
       "days_since_tamping": 220,
       "subgrade_type"     : "poor",
       "temp_delta"        : 28.0,
       "bridge_flag"       : 1,
   }
   ```

   Reload the dashboard → new row appears at the top of the urgent list.

5. **MLflow experiment** — open the Databricks MLflow UI, find experiment `rail_drishti_rag`, click `faiss_index_build` to see logged metrics and download the FAISS artifacts.

---

## Session Restart Recovery

> ⚠️ If the Databricks session restarts, `/tmp/` is wiped.

Re-run **Cell 8 only** — it reads from the persistent `embedded_docs` Delta table and rebuilds the FAISS index in under 2 minutes. Do **not** re-run Cells 5–7.

---

## Repository Structure

```
rail-drishti/
├── README.md
├── 03_build_rag.py          # Index building notebook
├── 04_inference.py          # Per-segment inference notebook
├── app.py                   # Flask dashboard server
├── templates/
│   └── index.html           # Dashboard frontend
└── docs/
    └── architecture.png     # Architecture diagram (exported from README)
```

---

## Project Write-Up (≤ 500 characters)

> Rail-Drishti monitors Indian Railways track health by combining a Spark MLlib defect classifier with a RAG pipeline that retrieves rules from official maintenance manuals (PWM & SoD) and generates actionable maintenance cards using Sarvam-1, an Indian LLM. Built entirely on Databricks Free Edition serverless with Delta Lake, Unity Catalog, and MLflow — surfaced through a live Flask dashboard showing urgent and watchlist segments with cited rule sources and cosine similarity scores.

---

## License

MIT
