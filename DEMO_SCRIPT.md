# Demo Script — Rail-Drishti (≤ 2 minutes)

## Before you start — prep checklist

- [ ] Databricks session running, all notebooks already executed
- [ ] Flask app running: `python app.py` → port 8050 open in browser tab
- [ ] MLflow UI open in a second browser tab (Databricks sidebar → Experiments)
- [ ] Data Explorer open in a third tab (sidebar → Catalog → bricksiitm → rail_drishti)
- [ ] Browser zoom at 90% so the dashboard fits without scrolling
- [ ] Pick 2 segments from the dashboard beforehand — one OVERDUE (red), one ELEVATED (amber)
- [ ] Have Cell 10 of `04_inference.ipynb` open and ready with a test segment pre-filled

---

## Scene 1 — Problem statement (0:00–0:12)

**Narrate:**
> "Indian Railways operates 68,000 km of track. Inspectors measure track geometry using recording cars and manually cross-reference standard deviations against tables in the Permanent Way Manual to decide whether to impose a speed restriction or call a tamping crew. Rail-Drishti automates that entire chain — sensor data in, action card out."

**Show:** Nothing yet. Just talk. Keep it punchy.

---

## Scene 2 — Data pipeline in Databricks (0:12–0:32)

**Switch to: Databricks Data Explorer**

**Narrate:**
> "The foundation is Delta Lake. We have four Unity Catalog tables in the `rail_drishti` schema."

**Click:** `bricksiitm` → `rail_drishti`

**Show:** All four tables — `raw_docs`, `chunked_docs`, `embedded_docs`, `inference_log`

**Narrate:**
> "Raw pages from the Permanent Way Manual and Schedule of Dimensions are ingested here. They get chunked, embedded using MiniLM, and indexed in FAISS — all tracked in MLflow."

**Switch to: MLflow Experiments → `rail_drishti_rag`**

**Click:** `faiss_index_build` run

**Show:** Logged params: `chunk_size=512`, `top_k=4`, `min_sim_threshold=0.35`. Artifacts: `track_index.bin`, `chunk_metadata.json`

**Narrate:**
> "The FAISS index is logged as an artifact here so it survives session restarts — just reload from MLflow, no need to re-embed."

---

## Scene 3 — ML model (0:32–0:45)

**Switch to: MLflow Experiments → `track_health_gbt`**

**Click:** `rf_defect_type` run

**Show:** F1 score, accuracy, logged params (`n_estimators=100`, `max_depth=10`, `target=defect_type`)

**Narrate:**
> "The classifier is a Random Forest trained on 38 features — track geometry standard deviations, vibration band energies, temperature readings, and maintenance history — predicting which of five defect types is present. Data is loaded from and written back to Delta tables throughout."

---

## Scene 4 — Dashboard overview (0:45–1:05)

**Switch to: Flask dashboard at localhost:8050**

**Narrate:**
> "This is the live engineer dashboard. It reads directly from the `inference_log` Delta table."

**Point to stat cards:**
> "Total inferences logged, urgent segments, watchlist segments, average retrieval similarity score across all RAG calls."

**Point to defect breakdown grid:**
> "Defect distribution — geometry, thermal stress, ballast weakness, fastening faults — from the ML classifier output."

**Point to segment table:**
> "Segments sorted by TQI ascending — worst track first. Red OVERDUE badges mean the ML model called it urgent and two or more deterministic rules fired. Amber ELEVATED means elevated risk."

---

## Scene 5 — Segment detail (1:05–1:35)

**Click:** An OVERDUE (red) segment row to expand it

**Narrate:**
> "Clicking a segment shows the full inference record."

**Point to triggered rules section:**
> "The deterministic rule engine fired before the LLM call. Here — tamping is 42 days overdue against the PWM 180-day limit, and TGI has dropped to 36.5, below the NBML threshold of 40 from IRPWM Para 522."

**Point to chunk sources:**
> "These are the actual PWM and SoD pages retrieved by FAISS — page number, section tag, cosine similarity score. A judge or engineer can pull up that exact page and verify."

**Point to action card:**
> "Sarvam-1 generated this maintenance action card: recommended action, responsible officer designation, deadline in hours, whether a speed restriction applies, and the PWM paragraph reference. This is structured JSON — it can be piped directly into a work-order system."

**Read one field aloud, e.g.:**
> "Responsible officer: Junior Engineer (P.Way). Deadline: 24 hours. Speed restriction: 30 kmph zone until tamping complete. Reference: Para 347, Para 522."

---

## Scene 6 — Live inference (1:35–2:00)

**Switch to: `04_inference.ipynb`, Cell 10**

**Show the test segment dict already filled in:**
```python
test_segment = {
    "segment_id":          "SEG_088",
    "defect_type":         "ballast_subgrade_weakness",
    "risk_score_bucket":   "urgent",
    "TGI":                 38.2,
    "days_since_tamping":  195,
    "subgrade_type":       "poor",
    "temp_delta":          22.0,
    "bridge_flag":         0,
}
```

**Narrate:**
> "New segment coming in — ballast weakness, TGI at 38, 195 days since last tamping. Running the full pipeline."

**Run the cell** (Cells 10 through 14 chained — should take ~5–8 seconds for Sarvam API call)

**Switch to: Dashboard, refresh**

**Narrate:**
> "New OVERDUE row at the top. Retrieval similarity 0.58 from PWM Section 4.3. Action card generated. Row written to the `inference_log` Delta table — permanent audit trail."

**Optional: Switch to MLflow, show the new `rag_inference` run just logged**

---

## Closing line (at ~2:00)

> "Every inference is grounded in the actual text of the Permanent Way Manual, every action is traceable to a rule, and everything persists in Unity Catalog. Rail-Drishti."

---

## Backup talking points (if judges ask)

**"Why synthetic data?"**
> No labelled Indian Railways track sensor dataset is publicly available. We grounded the synthetic generator in the IRPWM 2020 TQI formula — standard deviation benchmarks are pulled directly from Para 522, pages 268–271. The degradation model mirrors real tamping cycles with hysteresis.

**"Why sklearn RF instead of Spark MLlib GBT?"**
> Databricks Free Edition serverless enforces a 256MB model cache limit that fires during `pipeline.fit()` — not at serialisation time, so you can't work around it with custom saving. We kept Spark as the data layer (Delta reads/writes throughout) and used sklearn for the actual training. F1 and accuracy are still tracked in MLflow.

**"Why Sarvam-1?"**
> Indian LLM trained on Indian domain data. The terminology it uses for officer designations (Junior Engineer P.Way, Permanent Way Inspector, Assistant Divisional Engineer) and rule citations matches what a real Indian Railways engineer would recognise and trust.

**"What's the confidence banner for?"**
> RAG isn't perfect. If the retrieval similarity is below 0.40, the system displays a red banner saying "DO NOT ACT — escalate to Senior Section Engineer for manual review." The system knows when it doesn't know.

**"How does it handle session restarts?"**
> The FAISS index is logged as an MLflow artifact. Cell 9 of the inference notebook checks `/tmp/` first, then falls back to `client.download_artifacts()` from the `faiss_index_build` run. The Delta tables are persistent — only the in-memory index needs rebuilding, which takes under 2 minutes from `embedded_docs`.

---

## Timing guide

| Segment | Target time | Hard limit |
|---|---|---|
| Problem statement | 12s | 15s |
| Data pipeline | 20s | 25s |
| ML model | 13s | 15s |
| Dashboard overview | 20s | 22s |
| Segment detail | 30s | 35s |
| Live inference | 25s | 30s |
| **Total** | **~2:00** | **2:00** |

Practice the Sarvam API call before the demo — it can take 5–10 seconds. Have the result ready to show if the API is slow.
