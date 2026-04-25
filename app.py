"""
Rail-Drishti — Databricks App Server
Full pipeline: CSV upload (N-day raw signals) → feature derivation →
defect_model.pkl prediction → risk bucket → Sarvam RAG → write to Delta →
serve existing dashboard routes from Delta via Databricks SDK.
"""

import os, io, json, re, uuid, pickle, warnings, threading
from datetime import datetime

import numpy as np
import pandas as pd

import requests
from flask import Flask, jsonify, render_template, request
from databricks.sdk import WorkspaceClient


app = Flask(__name__)

# ── Databricks SDK (auto-authenticates inside Databricks Apps) ─────────────
w = WorkspaceClient()

CATALOG      = "bricksiitm"
SCHEMA       = "rail_drishti"
WAREHOUSE_ID = "660f29549f9a7beb"

# ── Model / artifact paths (co-located with app.py) ───────────────────────
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH          = os.path.join(BASE_DIR, "defect_model.pkl")
FAISS_INDEX_PATH    = os.path.join(BASE_DIR, "artifacts", "track_index.bin")
CHUNK_METADATA_PATH = os.path.join(BASE_DIR, "artifacts", "chunk_metadata.json")

# ── Sarvam API ─────────────────────────────────────────────────────────────
SARVAM_API_URL = "https://api.sarvam.ai/v1/chat/completions"
SARVAM_MODEL   = "sarvam-m"
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "sk_b90cjxut_pnD2L6DFpeu5tE5TY6XweiGy")

# ── RAG config ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K             = 4
MIN_SIM_THRESHOLD = 0.35
SOD_TOP_K         = 2
SOD_MIN_SIM       = 0.30
SOD_SOURCE_NAME   = "SoD_CS27"

# ── Risk thresholds ────────────────────────────────────────────────────────
TAMPING_OVERDUE_DAYS  = 180
TQI_CRITICAL          = 40
TEMP_DELTA_BUCKLE     = 25
TEMP_DELTA_PATROL     = 20
TEMP_DELTA_EMERGENCY  = 30
CONF_HIGH             = 0.50
CONF_MEDIUM           = 0.40

# Indian Railways neutral rail temperature (°C)
IR_NEUTRAL_TEMP = 27.0

# ── Query templates (mirrors 04_inference.ipynb Cell 10) ──────────────────
QUERY_TEMPLATES = {
    ("geometry_defect",           "HIGH"):   "track geometry gauge alignment tolerance urgent corrective action speed restriction",
    ("geometry_defect",           "MEDIUM"): "track geometry gauge alignment inspection schedule monitoring",
    ("ballast_subgrade_weakness", "HIGH"):   "ballast failure tamping schedule speed restriction poor subgrade emergency",
    ("ballast_subgrade_weakness", "MEDIUM"): "ballast tamping interval inspection schedule subgrade monitoring",
    ("thermal_stress",            "HIGH"):   "thermal stress track buckling summer heat emergency patrol speed restriction",
    ("thermal_stress",            "MEDIUM"): "thermal stress heat patrol frequency prevention summer temperature",
    ("fastening_sleeper_fault",   "HIGH"):   "rail fastening failure inspection replacement emergency procedure",
    ("fastening_sleeper_fault",   "MEDIUM"): "rail fastening inspection schedule maintenance routine",
}

SOD_QUERIES = {
    "geometry_defect":           "gauge tolerance alignment speed restriction track geometry index threshold",
    "ballast_subgrade_weakness": "track geometry versine cross level tolerance ballast failure speed restriction",
    "thermal_stress":            "gauge tolerance thermal expansion rail stress speed restriction track class",
    "fastening_sleeper_fault":   "rail joint gap tolerance speed restriction rail defect geometry",
}

SYSTEM_PROMPT = """You are a railway track maintenance assistant for Indian Railways.
Given sensor data and rules retrieved from the Permanent Way Manual (PWM)
and Schedule of Dimensions (SoD CS-27), generate a maintenance action card.

CRITICAL: Respond ONLY with a valid JSON object. No preamble, no markdown
fences, no explanation outside the JSON. The JSON must have exactly these keys:

{
  "recommended_action": "string — specific action the maintainer must take",
  "responsible_officer": "string — exact designation e.g. Junior Engineer (P.Way)",
  "deadline_hours": integer,
  "speed_restriction": "string — 'None' or specific e.g. '30 kmph zone'",
  "pwm_reference": "string — cite relevant paragraph e.g. 'Para 347, Para 522'",
  "risk_summary": "string — concise 1-line summary of why this is urgent"
}

Do NOT wrap in markdown code fences. Return only raw JSON."""

# ── Pipeline state (in-memory, reset per upload) ──────────────────────────
_lock  = threading.Lock()
_state = {
    "status":       "idle",   # idle | processing | done | error
    "progress":     0,
    "progress_msg": "",
    "error":        None,
    "upload_meta":  {},
}

# ── Lazy-loaded globals ────────────────────────────────────────────────────
_model_artifact = None
_embedder       = None
_faiss_index    = None
_chunk_metadata = None


# ══════════════════════════════════════════════════════════════════════════
# HELPERS — Databricks SDK query
# ══════════════════════════════════════════════════════════════════════════

def execute_query(sql):
    result = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=sql,
        wait_timeout="30s"
    )
    columns = []
    if result.manifest and result.manifest.schema:
        columns = [c.name for c in result.manifest.schema.columns]
    rows = []
    if result.result and result.result.data_array:
        rows = result.result.data_array
    return columns, rows


# ══════════════════════════════════════════════════════════════════════════
# STEP 0 — Load model artifact
# ══════════════════════════════════════════════════════════════════════════

def load_model():
    global _model_artifact
    if _model_artifact is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _model_artifact = pickle.load(open(MODEL_PATH, "rb"))
    return _model_artifact


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Aggregate N-day raw signals → 1 row per segment
# ══════════════════════════════════════════════════════════════════════════

# Static cols: identical across all days for a segment — take first value
_STATIC_COLS = [
    "route_id", "km_start", "km_end",
    "curve_flag", "bridge_flag", "turnout_flag",
    "rail_type", "sleeper_type", "subgrade_class",
]

# Sensor cols: vary day-to-day — aggregate as mean
_SENSOR_COLS = [
    "gradient", "ballast_age",
    "train_count_24h", "avg_speed_24h", "max_speed_24h",
    "avg_axle_load", "cumulative_tonnage_30d",
    "gauge_sd", "alignment_sd", "longitudinal_level_sd",
    "cross_level_sd", "twist_sd", "TGI",
    "accel_rms_v", "accel_rms_l", "accel_peak_v",
    "accel_kurtosis", "band_energy_low", "band_energy_mid", "band_energy_high",
    "rail_temp_max", "ambient_temp_max", "temp_delta",
    "days_since_tamping",
    "maintenance_count_total", "pre_maintenance_severity", "degradation_memory",
]

_MAINT_COL = "last_maintenance_type"  # categorical — take mode


def aggregate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse N rows per segment (one per day) into 1 representative row.
    Static infrastructure cols use first value; sensor cols use mean;
    maintenance type uses mode.
    """
    results = []
    for seg_id, grp in df.groupby("segment_id"):
        row = {"segment_id": seg_id}

        for c in _STATIC_COLS:
            if c in grp.columns:
                row[c] = grp[c].iloc[0]

        for c in _SENSOR_COLS:
            if c in grp.columns:
                row[c] = grp[c].mean()

        if _MAINT_COL in grp.columns:
            row[_MAINT_COL] = grp[_MAINT_COL].mode().iloc[0]

        results.append(row)

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Derive the 6 features the model needs but raw CSV lacks
# ══════════════════════════════════════════════════════════════════════════

def derive_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    The training data had 6 features not present in the raw signal CSV.
    All are derivable from columns that do exist:

    unevenness_sd_short  = longitudinal_level_sd
        → short-chord unevenness is captured by longitudinal_level_sd directly

    unevenness_sd_long   = longitudinal_level_sd * 0.85
        → long-chord measurement attenuates the short-chord signal by ~15%
          (standard IRPWM chord-length correction factor)

    alignment_sd_short   = alignment_sd
        → short-chord alignment deviation, same as the raw column

    alignment_sd_long    = alignment_sd * 0.85
        → same 15% long-chord attenuation as above

    temp_above_neutral   = max(0, rail_temp_max - 27)
        → 27°C is Indian Railways standard neutral temperature;
          positive exceedance drives thermal stress risk

    route_max_speed      = max_speed_24h
        → best available proxy; the model uses this to weight geometry
          severity relative to permitted speed
    """
    df = df.copy()

    if "longitudinal_level_sd" in df.columns:
        df["unevenness_sd_short"] = df["longitudinal_level_sd"]
        df["unevenness_sd_long"]  = df["longitudinal_level_sd"] * 0.85
    else:
        df["unevenness_sd_short"] = np.nan
        df["unevenness_sd_long"]  = np.nan

    if "alignment_sd" in df.columns:
        df["alignment_sd_short"] = df["alignment_sd"]
        df["alignment_sd_long"]  = df["alignment_sd"] * 0.85
    else:
        df["alignment_sd_short"] = np.nan
        df["alignment_sd_long"]  = np.nan

    if "rail_temp_max" in df.columns:
        df["temp_above_neutral"] = (df["rail_temp_max"] - IR_NEUTRAL_TEMP).clip(lower=0)
    else:
        df["temp_above_neutral"] = np.nan

    if "max_speed_24h" in df.columns:
        df["route_max_speed"] = df["max_speed_24h"]
    else:
        df["route_max_speed"] = np.nan

    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Run defect_model.pkl predictions
# ══════════════════════════════════════════════════════════════════════════

def run_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply label encoding, imputation, RandomForest prediction, and
    derive risk_score_bucket + urgent_maintenance_72h.
    Returns df with those columns added.
    """
    art       = load_model()
    model     = art["model"]
    imputer   = art["imputer"]
    encoders  = art["encoders"]
    feat_cols = art["feature_cols"]

    df = df.copy()

    # Label-encode categoricals using training-fitted encoders
    for col, le in encoders.items():
        if col in df.columns:
            known = set(le.classes_)
            # Map unseen values to the first known class
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    # Build feature matrix — cols absent from CSV become NaN;
    # the imputer fills them with training-set medians
    X = pd.DataFrame(index=df.index)
    for col in feat_cols:
        X[col] = df[col] if col in df.columns else np.nan

    X_imp = imputer.transform(X)
    df["predicted_defect_type"] = model.predict(X_imp)

    # TQI_primary: use raw TGI column if available
    if "TGI" in df.columns:
        df["TQI_primary"] = df["TGI"]
    else:
        # Approximate from geometry SDs (simplified IRPWM weighting)
        components = sum(
            df[c] for c in ["gauge_sd", "alignment_sd", "longitudinal_level_sd",
                             "cross_level_sd", "twist_sd"]
            if c in df.columns
        )
        df["TQI_primary"] = (100 - components * 10).clip(0, 100)

    # Risk bucket — mirrors Prediction_Model.ipynb exactly
    def derive_risk_bucket(row):
        if row["days_since_tamping"] > TAMPING_OVERDUE_DAYS:
            return "OVERDUE"
        elif row["TQI_primary"] < TQI_CRITICAL or row["temp_delta"] > TEMP_DELTA_EMERGENCY:
            return "HIGH"
        elif row["predicted_defect_type"] != "none" and row["TQI_primary"] < 60:
            return "MEDIUM"
        return "LOW"

    def derive_urgency(row):
        if row["predicted_defect_type"] == "thermal_stress" and row["temp_delta"] > TEMP_DELTA_EMERGENCY:
            return 1
        if row["predicted_defect_type"] == "geometry_defect" and row["TQI_primary"] < 46:
            return 1
        if row["days_since_tamping"] > TAMPING_OVERDUE_DAYS:
            return 1
        return 0

    df["risk_score_bucket"]      = df.apply(derive_risk_bucket, axis=1)
    df["urgent_maintenance_72h"] = df.apply(derive_urgency, axis=1)

    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — RAG: retrieval + Sarvam LLM call
# ══════════════════════════════════════════════════════════════════════════

def load_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer  # ← ADD THIS
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder

def load_faiss():
    global _faiss_index, _chunk_metadata
    if _faiss_index is None:
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNK_METADATA_PATH):
            import faiss  # ← ADD THIS
            _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(CHUNK_METADATA_PATH) as f:
                _chunk_metadata = json.load(f)
    return _faiss_index, _chunk_metadata


def build_retrieval_query(prediction: dict) -> str:
    defect = str(prediction.get("predicted_defect_type", "geometry_defect")).lower()
    bucket = str(prediction.get("risk_score_bucket", "MEDIUM")).upper()
    lookup = "HIGH" if bucket in ("HIGH", "OVERDUE", "URGENT") else "MEDIUM"
    query  = QUERY_TEMPLATES.get((defect, lookup),
             QUERY_TEMPLATES.get((defect, "MEDIUM"),
             "track maintenance inspection schedule"))

    enrichments = []
    if prediction.get("bridge_flag"):
        enrichments.append("bridge section speed restriction girder")
    tdelt = float(prediction.get("temp_delta", 0))
    tqi   = float(prediction.get("TQI_primary", 100))
    if tdelt > TEMP_DELTA_PATROL:
        enrichments.append("thermal stress heat patrol frequency prevention summer temperature")
    if tdelt > TEMP_DELTA_EMERGENCY:
        enrichments.append("track buckling heat emergency creep speed restriction 30 kmph")
    if tqi < TQI_CRITICAL:
        enrichments.append("TQI NBML UML urgent intervention tamping track geometry SD threshold")

    return query + (" " + " ".join(enrichments) if enrichments else "")


def retrieve_chunks(query, index, meta, top_k=TOP_K, min_sim=MIN_SIM_THRESHOLD):
    embedder   = load_embedder()
    qvec       = embedder.encode([query], normalize_embeddings=True).astype("float32")
    dists, ids = index.search(qvec, top_k)
    results    = []
    for dist, idx in zip(dists[0], ids[0]):
        if idx == -1 or dist < min_sim:
            continue
        chunk = dict(meta[str(idx)])
        chunk["similarity_score"] = round(float(dist), 4)
        results.append(chunk)
    return results


def retrieve_sod_chunks(defect_type, index, meta):
    embedder  = load_embedder()
    sod_query = SOD_QUERIES.get(defect_type, "gauge tolerance alignment speed restriction")
    sod_ids   = [int(k) for k, v in meta.items() if v.get("source_name") == SOD_SOURCE_NAME]
    if not sod_ids:
        return []
    qvec     = embedder.encode([sod_query], normalize_embeddings=True).astype("float32")
    sod_vecs = np.array([index.reconstruct(i) for i in sod_ids], dtype="float32")
    scores   = (sod_vecs @ qvec.T).flatten()
    ranked   = sorted(zip(sod_ids, scores.tolist()), key=lambda x: x[1], reverse=True)
    ranked   = [(i, s) for i, s in ranked if s >= SOD_MIN_SIM][:SOD_TOP_K]
    results  = []
    for fid, score in ranked:
        chunk = dict(meta[str(fid)])
        chunk["similarity_score"] = round(score, 4)
        results.append(chunk)
    return results


def evaluate_escalation(days_since_tamping, tqi, temp_delta, bucket, bridge_flag, defect_type):
    triggered = []
    if days_since_tamping > TAMPING_OVERDUE_DAYS:
        triggered.append(
            f"Tamping overdue by {days_since_tamping - TAMPING_OVERDUE_DAYS}d "
            f"(last: {days_since_tamping}d ago, PWM limit: {TAMPING_OVERDUE_DAYS}d)"
        )
    if tqi < TQI_CRITICAL:
        triggered.append(
            f"TQI={tqi:.1f} below critical threshold of {TQI_CRITICAL} (PWM: immediate intervention)"
        )
    if TEMP_DELTA_PATROL < temp_delta <= TEMP_DELTA_EMERGENCY:
        triggered.append(
            f"Temp delta={temp_delta:.1f}°C exceeds {TEMP_DELTA_PATROL}°C — hot weather patrol required (PWM Para 347)"
        )
    if temp_delta > TEMP_DELTA_EMERGENCY:
        triggered.append(
            f"Temp delta={temp_delta:.1f}°C exceeds {TEMP_DELTA_EMERGENCY}°C — buckling risk, emergency patrol (PWM Para 347)"
        )
    if bridge_flag and temp_delta > TEMP_DELTA_PATROL:
        triggered.append(
            f"Bridge section with temp delta={temp_delta:.1f}°C — increased monitoring required"
        )

    b = bucket.upper()
    num = len(triggered)
    if b in ("HIGH", "OVERDUE", "URGENT") and num >= 2:
        tier = "OVERDUE"
    elif b in ("HIGH", "OVERDUE", "URGENT") or num >= 1:
        tier = "ELEVATED"
    else:
        tier = "STANDARD"

    return tier, triggered


def build_confidence_banner(top_sim, num_chunks):
    if top_sim >= CONF_HIGH:
        return "HIGH", ""
    elif top_sim >= CONF_MEDIUM:
        return "MEDIUM", (
            "\n\n───────────────────────────────────────────\n"
            "⚠️  MODERATE RETRIEVAL CONFIDENCE\n"
            f"   Similarity: {top_sim:.4f} | Chunks: {num_chunks}\n"
            "   Cross-check cited clause against source manual before acting.\n"
            "───────────────────────────────────────────"
        )
    return "LOW", (
        "\n\n═══════════════════════════════════════════\n"
        "🔴  LOW RETRIEVAL CONFIDENCE — DO NOT ACT\n"
        f"   Similarity: {top_sim:.4f} | Chunks: {num_chunks}\n"
        "   No strong PWM/SoD match found. Escalate to SSE for manual review.\n"
        "═══════════════════════════════════════════"
    )


def call_sarvam(prediction: dict, chunks: list, rule_injection: str) -> dict:
    seg_id = prediction.get("segment_id", "UNKNOWN")
    context_text = "".join(
        f"\n[{i+1}] {c.get('source_name','?')} p.{c.get('page_number','?')}\n{c.get('chunk_text','')}\n"
        for i, c in enumerate(chunks)
    )
    user_msg = (
        f"Segment ID: {seg_id}\n"
        f"Defect Type: {prediction.get('predicted_defect_type')}\n"
        f"Risk Bucket: {prediction.get('risk_score_bucket')}\n"
        f"Urgent Maintenance (72h): {prediction.get('urgent_maintenance_72h', 0)}\n"
        f"TQI: {prediction.get('TQI_primary')}\n"
        f"Days Since Tamping: {prediction.get('days_since_tamping')}\n"
        f"Subgrade Class: {prediction.get('subgrade_class', 'unknown')}\n"
        f"Temp Delta: {prediction.get('temp_delta')}°C\n"
        f"Bridge Section: {'Yes' if prediction.get('bridge_flag') else 'No'}\n"
        f"\nRETRIEVED PWM/SoD CHUNKS:\n{context_text}"
        f"{rule_injection}"
        f"\nGenerate the maintenance action card as JSON only."
    )
    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": SARVAM_API_KEY,
    }
    payload = {
        "model": SARVAM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": 0.3,
        "max_tokens": 800,
    }
    try:
        resp = requests.post(SARVAM_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$",     "", text, flags=re.MULTILINE).strip()
        card = json.loads(text)
        return {"status": "success", "action_card": card}
    except Exception as e:
        fallback = {
            "recommended_action": f"Manual inspection required — segment {seg_id}",
            "responsible_officer": "Junior Engineer (P.Way)",
            "deadline_hours": 24,
            "speed_restriction": "None",
            "pwm_reference": "Para 522",
            "risk_summary": (
                f"Defect: {prediction.get('predicted_defect_type')} — "
                f"automated advisory unavailable ({e})"
            ),
        }
        return {"status": "error", "action_card": fallback, "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — Write results to Delta tables via Databricks SDK
# ══════════════════════════════════════════════════════════════════════════

def _s(v):
    """Escape a value for SQL string literal."""
    if v is None:
        return "NULL"
    return "'" + str(v).replace("'", "''") + "'"


def write_predictions_to_delta(pred_df: pd.DataFrame):
    """Upsert per-segment predictions into track_predictions Delta table."""
    execute_query(f"""
        CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.track_predictions (
            segment_id STRING, defect_type STRING, risk_score_bucket STRING,
            urgent_maintenance_72h INT, TQI_primary DOUBLE,
            days_since_tamping INT, subgrade_class STRING,
            temp_delta DOUBLE, bridge_flag INT, uploaded_at STRING
        ) USING DELTA
    """)
    ts = datetime.utcnow().isoformat()
    for _, row in pred_df.iterrows():
        execute_query(f"""
            MERGE INTO {CATALOG}.{SCHEMA}.track_predictions t
            USING (SELECT {_s(str(row['segment_id']))} AS segment_id) s
            ON t.segment_id = s.segment_id
            WHEN MATCHED THEN UPDATE SET
                defect_type            = {_s(str(row.get('predicted_defect_type','none')))},
                risk_score_bucket      = {_s(str(row.get('risk_score_bucket','LOW')))},
                urgent_maintenance_72h = {int(row.get('urgent_maintenance_72h', 0))},
                TQI_primary            = {float(row.get('TQI_primary', 100.0))},
                days_since_tamping     = {int(row.get('days_since_tamping', 0))},
                subgrade_class         = {_s(str(row.get('subgrade_class','unknown')))},
                temp_delta             = {float(row.get('temp_delta', 0.0))},
                bridge_flag            = {int(row.get('bridge_flag', 0))},
                uploaded_at            = {_s(ts)}
            WHEN NOT MATCHED THEN INSERT (
                segment_id, defect_type, risk_score_bucket, urgent_maintenance_72h,
                TQI_primary, days_since_tamping, subgrade_class,
                temp_delta, bridge_flag, uploaded_at
            ) VALUES (
                {_s(str(row['segment_id']))},
                {_s(str(row.get('predicted_defect_type','none')))},
                {_s(str(row.get('risk_score_bucket','LOW')))},
                {int(row.get('urgent_maintenance_72h', 0))},
                {float(row.get('TQI_primary', 100.0))},
                {int(row.get('days_since_tamping', 0))},
                {_s(str(row.get('subgrade_class','unknown')))},
                {float(row.get('temp_delta', 0.0))},
                {int(row.get('bridge_flag', 0))},
                {_s(ts)}
            )
        """)


def write_inference_to_delta(seg_id, prediction, query, chunks, action_card, sim_score):
    """Append one RAG inference row to inference_log Delta table."""
    execute_query(f"""
        CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.inference_log (
            inference_id STRING, generated_at STRING, segment_id STRING,
            defect_type STRING, risk_score_bucket STRING,
            urgent_maintenance_72h INT, TGI DOUBLE, days_since_tamping INT,
            subgrade_type STRING, temp_delta DOUBLE, bridge_flag INT,
            retrieval_query STRING, top_chunk_sources STRING,
            top_sim_score DOUBLE, num_chunks_retrieved INT,
            action_card STRING, query_used STRING, parse_status STRING
        ) USING DELTA
    """)
    inference_id  = str(uuid.uuid4())
    generated_at  = datetime.utcnow().isoformat()
    chunk_sources = ", ".join(c.get("source_name", "?") for c in chunks)
    card_str      = json.dumps(action_card)
    parse_status  = "success" if action_card.get("recommended_action") else "failed"

    execute_query(f"""
        INSERT INTO {CATALOG}.{SCHEMA}.inference_log VALUES (
            {_s(inference_id)},
            {_s(generated_at)},
            {_s(str(seg_id))},
            {_s(str(prediction.get('predicted_defect_type','none')))},
            {_s(str(prediction.get('risk_score_bucket','LOW')))},
            {int(prediction.get('urgent_maintenance_72h', 0))},
            {float(prediction.get('TQI_primary', 100.0))},
            {int(prediction.get('days_since_tamping', 0))},
            {_s(str(prediction.get('subgrade_class','unknown')))},
            {float(prediction.get('temp_delta', 0.0))},
            {int(prediction.get('bridge_flag', 0))},
            {_s(query)},
            {_s(chunk_sources)},
            {float(sim_score)},
            {len(chunks)},
            {_s(card_str)},
            {_s(query)},
            {_s(parse_status)}
        )
    """)


# ══════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR — runs in a background daemon thread
# ══════════════════════════════════════════════════════════════════════════

def _set(**kw):
    with _lock:
        _state.update(kw)


def run_pipeline(csv_bytes: bytes, filename: str):
    try:
        # ── 1. Parse ───────────────────────────────────────────────────────
        _set(status="processing", progress=5, progress_msg="Parsing CSV…")
        df       = pd.read_csv(io.BytesIO(csv_bytes))
        n_rows   = len(df)
        n_days   = df["day"].nunique()   if "day"        in df.columns else 1
        n_segs   = df["segment_id"].nunique() if "segment_id" in df.columns else len(df)
        _set(
            progress=10,
            progress_msg=f"Loaded {n_rows:,} rows · {n_days} days · {n_segs} segments",
            upload_meta={
                "filename": filename, "rows": n_rows,
                "days": n_days, "segments": n_segs,
                "uploaded_at": datetime.utcnow().isoformat(),
            }
        )

        # ── 2. Aggregate signals ───────────────────────────────────────────
        _set(progress=15, progress_msg="Aggregating signals across days…")
        agg_df = aggregate_signals(df)

        # ── 3. Derive missing model features ──────────────────────────────
        _set(progress=20, progress_msg="Deriving model features…")
        agg_df = derive_model_features(agg_df)

        # ── 4. Predict defect type + risk bucket ───────────────────────────
        _set(progress=30, progress_msg="Running defect predictions…")
        pred_df = run_predictions(agg_df)

        # ── 5. Write predictions to Delta ──────────────────────────────────
        _set(progress=40, progress_msg="Writing predictions to Delta…")
        write_predictions_to_delta(pred_df)

        # ── 6. RAG for HIGH + OVERDUE + MEDIUM segments ────────────────────
        priority_df = pred_df[
            pred_df["risk_score_bucket"].isin(["HIGH", "OVERDUE", "MEDIUM"])
        ].copy()
        n_hp = len(priority_df)

        _set(progress=45, progress_msg=f"Loading FAISS index — {n_hp} priority segments…")
        faiss_index, chunk_meta = load_faiss()

        if faiss_index is None or chunk_meta is None:
            _set(
                status="done", progress=100,
                progress_msg=(
                    f"✅ Predictions written ({n_segs} segments). "
                    "⚠️ RAG skipped — place track_index.bin and chunk_metadata.json "
                    "in the artifacts/ folder next to app.py."
                )
            )
            return

        for i, (_, row) in enumerate(priority_df.iterrows()):
            seg_id   = str(row["segment_id"])
            pct      = 45 + int((i / max(n_hp, 1)) * 50)
            _set(progress=pct, progress_msg=f"RAG inference {i+1}/{n_hp} — segment {seg_id}…")

            pred_dict = row.to_dict()

            # Retrieval
            query      = build_retrieval_query(pred_dict)
            pwm_chunks = retrieve_chunks(query, faiss_index, chunk_meta)
            sod_chunks = retrieve_sod_chunks(
                str(pred_dict.get("predicted_defect_type", "geometry_defect")),
                faiss_index, chunk_meta
            )
            all_chunks = pwm_chunks + sod_chunks

            # Escalation rule engine
            tier, triggered = evaluate_escalation(
                days_since_tamping=int(pred_dict.get("days_since_tamping", 0)),
                tqi=float(pred_dict.get("TQI_primary", 100)),
                temp_delta=float(pred_dict.get("temp_delta", 0)),
                bucket=str(pred_dict.get("risk_score_bucket", "LOW")),
                bridge_flag=bool(pred_dict.get("bridge_flag", 0)),
                defect_type=str(pred_dict.get("predicted_defect_type", "")),
            )
            rule_injection = (
                "\n\nDETERMINISTIC RULE ENGINE FLAGS (must be reflected in card):\n"
                + "\n".join(f"  • {r}" for r in triggered) + "\n"
            ) if triggered else ""

            # Sarvam LLM call
            sarvam   = call_sarvam(pred_dict, all_chunks, rule_injection)
            card     = sarvam["action_card"]

            # Confidence banner → stored as note in card
            top_sim  = all_chunks[0]["similarity_score"] if all_chunks else 0.0
            _, banner = build_confidence_banner(top_sim, len(all_chunks))
            if banner:
                card["confidence_note"] = banner.strip()

            # Write to inference_log
            write_inference_to_delta(
                seg_id=seg_id,
                prediction=pred_dict,
                query=query,
                chunks=all_chunks,
                action_card=card,
                sim_score=top_sim,
            )

        _set(
            status="done", progress=100,
            progress_msg=(
                f"✅ Pipeline complete — {n_segs} predictions, "
                f"{n_hp} RAG advisories written to Delta."
            )
        )

    except Exception as exc:
        import traceback as tb
        _set(status="error", error=str(exc) + "\n" + tb.format_exc(),
             progress_msg="❌ Pipeline failed — see error field.")


# ══════════════════════════════════════════════════════════════════════════
# EXISTING DASHBOARD ROUTES (read from Delta via SDK — unchanged logic)
# ══════════════════════════════════════════════════════════════════════════

def get_dashboard_stats():
    _, rows = execute_query(
        f"SELECT COUNT(*) FROM {CATALOG}.{SCHEMA}.inference_log")
    total = int(rows[0][0]) if rows else 0

    _, rows = execute_query(f"""
        SELECT COUNT(*) FROM {CATALOG}.{SCHEMA}.inference_log
        WHERE risk_score_bucket IN ('HIGH','OVERDUE','urgent')""")
    urgent = int(rows[0][0]) if rows else 0

    _, rows = execute_query(f"""
        SELECT COUNT(*) FROM {CATALOG}.{SCHEMA}.inference_log
        WHERE risk_score_bucket IN ('MEDIUM','watchlist')""")
    watchlist = int(rows[0][0]) if rows else 0

    _, rows = execute_query(
        f"SELECT AVG(top_sim_score) FROM {CATALOG}.{SCHEMA}.inference_log")
    avg_sim = round(float(rows[0][0]), 4) if rows and rows[0][0] else 0.0

    _, rows = execute_query(f"""
        SELECT COUNT(*) FROM {CATALOG}.{SCHEMA}.inference_log
        WHERE parse_status = 'failed'""")
    parse_failures = int(rows[0][0]) if rows else 0

    _, rows = execute_query(f"""
        SELECT defect_type, COUNT(*) AS cnt
        FROM {CATALOG}.{SCHEMA}.inference_log
        GROUP BY defect_type ORDER BY cnt DESC""")
    defect_breakdown = {r[0]: int(r[1]) for r in rows}

    return {
        "total_inferences": total,
        "urgent_count":     urgent,
        "watchlist_count":  watchlist,
        "normal_count":     max(0, total - urgent - watchlist),
        "avg_sim_score":    avg_sim,
        "parse_failures":   parse_failures,
        "defect_breakdown": defect_breakdown,
    }


def get_live_predictions():
    _, rows = execute_query(f"""
        SELECT segment_id, defect_type, risk_score_bucket,
               TQI_primary, days_since_tamping, subgrade_class,
               temp_delta, bridge_flag
        FROM {CATALOG}.{SCHEMA}.track_predictions
        WHERE risk_score_bucket IN ('HIGH','OVERDUE','MEDIUM','urgent','watchlist')
        ORDER BY TQI_primary ASC
        LIMIT 50""")

    results = []
    for row in rows:
        seg_id, defect, bucket, tqi, days, subgrade, tdelt, bflag = row
        tqi   = float(tqi)
        days  = int(days)
        tdelt = float(tdelt)
        bflag = int(bflag)

        rules = []
        if days > TAMPING_OVERDUE_DAYS:
            rules.append(f"Tamping overdue {days - TAMPING_OVERDUE_DAYS}d")
        if tqi < TQI_CRITICAL:
            rules.append(f"TQI critical ({tqi:.1f})")
        if tdelt > TEMP_DELTA_BUCKLE and bflag:
            rules.append(f"Thermal+bridge risk ({tdelt:.1f}°C)")
        elif tdelt > TEMP_DELTA_BUCKLE:
            rules.append(f"Thermal risk ({tdelt:.1f}°C)")

        b    = (bucket or "").upper()
        tier = ("OVERDUE"  if b in ("HIGH","OVERDUE","URGENT") and len(rules) >= 2 else
                "ELEVATED" if b in ("HIGH","OVERDUE","URGENT") or len(rules) >= 1 else
                "STANDARD")

        results.append({
            "segment_id":         seg_id,
            "defect_type":        defect,
            "risk_score_bucket":  bucket,
            "escalation_tier":    tier,
            "TQI":                tqi,
            "days_since_tamping": days,
            "subgrade_type":      subgrade,
            "temp_delta":         tdelt,
            "bridge_flag":        bflag,
            "triggered_rules":    rules,
        })
    return results


def get_segment_log(segment_id):
    _, rows = execute_query(f"""
        SELECT inference_id, generated_at, risk_score_bucket, defect_type,
               top_sim_score, top_chunk_sources, action_card, query_used
        FROM {CATALOG}.{SCHEMA}.inference_log
        WHERE segment_id = '{segment_id}'
        ORDER BY generated_at DESC
        LIMIT 10""")

    results = []
    for row in rows:
        action_str = row[6] or ""
        try:
            card = (json.loads(action_str) if action_str.startswith("{")
                    else {"action_summary": action_str, "parse_status": "legacy"})
        except Exception:
            card = {"action_summary": action_str, "parse_status": "legacy"}

        results.append({
            "inference_id":      row[0] or "",
            "generated_at":      str(row[1]),
            "risk_score_bucket": row[2],
            "defect_type":       row[3],
            "top_sim_score":     float(row[4]) if row[4] else 0.0,
            "top_chunk_sources": row[5] or "",
            "action_card":       card,
            "query_used":        row[7] or "",
        })
    return results


# ══════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.route("/health")
@app.route("/healthz")
@app.route("/_health")
def health():
    return jsonify({"status": "healthy", "service": "rail-drishti"}), 200


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dashboard")
def api_dashboard():
    try:
        return jsonify(get_dashboard_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predictions")
def api_predictions():
    try:
        return jsonify(get_live_predictions())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/segment/<segment_id>")
def api_segment(segment_id):
    try:
        return jsonify(get_segment_log(segment_id))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    Accept multipart CSV, validate, kick off background pipeline thread.
    Returns 202 immediately; poll /api/pipeline/status for progress.
    """
    with _lock:
        if _state["status"] == "processing":
            return jsonify({"error": "Pipeline already running — please wait."}), 409

    if "file" not in request.files:
        return jsonify({"error": "No file attached."}), 400

    f = request.files["file"]
    if not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only .csv files are accepted."}), 400

    csv_bytes = f.read()

    # Quick structural validation before spawning the thread
    try:
        peek    = pd.read_csv(io.BytesIO(csv_bytes), nrows=3)
        missing = [c for c in ["segment_id", "temp_delta", "days_since_tamping"]
                   if c not in peek.columns]
        if missing:
            return jsonify({"error": f"CSV missing required columns: {missing}"}), 400
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {e}"}), 400

    _set(status="processing", progress=0, progress_msg="Starting pipeline…", error=None)
    threading.Thread(target=run_pipeline, args=(csv_bytes, f.filename), daemon=True).start()
    return jsonify({"message": "Upload accepted — pipeline started.", "filename": f.filename}), 202


@app.route("/api/pipeline/status")
def api_pipeline_status():
    """Poll this endpoint to track pipeline progress (0–100)."""
    with _lock:
        return jsonify({
            "status":       _state["status"],
            "progress":     _state["progress"],
            "progress_msg": _state["progress_msg"],
            "error":        _state["error"],
            "upload_meta":  _state["upload_meta"],
        })


if __name__ == "__main__":
    port = int(os.environ.get("DATABRICKS_APP_PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
