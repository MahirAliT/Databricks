# Rail-Drishti — Project Write-Up

Rail-Drishti monitors Indian Railways track health by combining a Spark MLlib defect classifier with a RAG pipeline that retrieves rules from official maintenance manuals (PWM & SoD) and generates actionable maintenance cards using Sarvam-1, an Indian LLM. Built on Databricks Free Edition serverless with Delta Lake, Unity Catalog, and MLflow — surfaced through a live Flask dashboard showing urgent and watchlist segments with cited rule sources and cosine similarity scores.

---

## Databricks Technologies

- **Delta Lake** — 4 Unity Catalog tables (raw_docs, chunked_docs, embedded_docs, inference_log)
- **Unity Catalog** — schema `rail_drishti` under catalog `bricksiitm`
- **Spark MLlib** — GBTClassifier for defect type + risk bucket classification
- **PySpark** — DataFrame reads/writes across all pipeline stages
- **MLflow** — tracks 4 pipeline runs; FAISS index logged as versioned artifact
- **Databricks Serverless** — all code serverless-compatible (no SparkContext, no DBFS root)

## Open-Source Models

- `sentence-transformers/all-MiniLM-L6-v2` — 384-dim embedding, driver-side, no GPU
- **FAISS** `IndexFlatIP` — exact cosine-similarity vector search over ~thousands of chunks
- **Sarvam-1** (`sarvam-m`) — Indian LLM for maintenance action card generation

## Quantitative Metrics (reference targets)

| Metric | Target | Notes |
|--------|--------|-------|
| Retrieval similarity (avg) | ≥ 0.45 | FAISS inner product on normalised vectors |
| Retrieval similarity (min threshold) | 0.35 | Chunks below this are filtered |
| Chunks retrieved per inference | 4 | top_k setting |
| Chunk size | 512 tokens | RecursiveCharacterTextSplitter |
| Chunk overlap | 64 tokens | Preserves rule boundaries |
| Embedding dimension | 384 | all-MiniLM-L6-v2 |
| GBT classifier depth | 4 | maxDepth=4, maxIter=50 |
| Action card max tokens | 512 | Sarvam-1 parameter |
| LLM temperature | 0.2 | Deterministic maintenance output |
