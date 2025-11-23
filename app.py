import os
import io
import time
import json
import re
import mimetypes
from typing import List, Dict

import requests
import streamlit as st
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from rapidfuzz import fuzz
from dotenv import load_dotenv

# ----------------------------------------------------------
# ENV
# ----------------------------------------------------------
load_dotenv()

DOCINT_ENDPOINT = os.getenv("AZURE_DOCINT_ENDPOINT")
DOCINT_KEY = os.getenv("AZURE_DOCINT_KEY")
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
OPENAI_DEPLOY = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # e.g. gpt-5-chat
API_VERSION = os.getenv("API_VERSION") or "2023-07-31"  # DocInt
OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

st.set_page_config(layout="wide", page_title="Simple Field Matcher")
st.title("Document Field Matcher — Azure Doc Intelligence + Azure OpenAI")

# ----------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------
defaults = {
    "fields_a": [],
    "fields_b": [],
    "matches_df": None,
    "has_extracted": False,
    "decision": None,
    "chat_history": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------------------------------------------------
# SIDEBAR CONFIG
# ----------------------------------------------------------
st.sidebar.header("Config")
st.sidebar.write("Set AZURE_DOCINT_* and AZURE_OPENAI_* in .env")

match_threshold = st.sidebar.slider(
    "Match threshold",
    0.5,
    1.0,
    0.75,
    0.01,
    help="Combined name + value similarity needed to mark a field as matched",
)

# ----------------------------------------------------------
# HELPER: Azure Doc Intelligence
# ----------------------------------------------------------
def try_docint_upload(analyze_url: str, headers: dict, file_bytes: bytes):
    try:
        return requests.post(analyze_url, headers=headers, data=file_bytes, timeout=30)
    except Exception as e:
        raise RuntimeError(f"Network error calling Document Intelligence: {e}")

def azure_docint_analyze_bytes(file_bytes: bytes, filename_hint: str = "file.pdf", timeout_sec: int = 60) -> Dict:
    if not DOCINT_ENDPOINT or not DOCINT_KEY:
        raise RuntimeError("Set AZURE_DOCINT_ENDPOINT and AZURE_DOCINT_KEY first.")

    ctype, _ = mimetypes.guess_type(filename_hint)
    if not ctype:
        ctype = "application/octet-stream"

    ep = DOCINT_ENDPOINT.rstrip("/")
    analyze_url = f"{ep}/formrecognizer/documentModels/prebuilt-document:analyze?api-version={API_VERSION}"

    headers = {"Ocp-Apim-Subscription-Key": DOCINT_KEY, "Content-Type": ctype}
    resp = try_docint_upload(analyze_url, headers, file_bytes)

    if resp.status_code not in (200, 202):
        raise RuntimeError(f"DocInt initial call failed: {resp.status_code} {resp.text}")

    op_url = resp.headers.get("operation-location") or resp.headers.get("Operation-Location")
    if not op_url:
        raise RuntimeError(f"No operation-location header. Response: {resp.status_code} {resp.text}")

    last = None
    for _ in range(timeout_sec):
        poll = requests.get(op_url, headers={"Ocp-Apim-Subscription-Key": DOCINT_KEY}, timeout=30)
        last = poll
        if poll.status_code != 200:
            time.sleep(1)
            continue
        j = poll.json()
        status = j.get("status", "").lower()
        if status == "succeeded":
            return j.get("analyzeResult", j)
        if status == "failed":
            raise RuntimeError(f"DocInt analysis failed: {json.dumps(j)}")
        time.sleep(1)

    raise RuntimeError(f"DocInt timed out. Last: {last.status_code if last else 'no response'}")

# ----------------------------------------------------------
# HELPER: extract fields (generic)
# ----------------------------------------------------------
def extract_fields(analyze_result: Dict) -> List[Dict]:
    """
    Returns a list of dicts:
    [
      {"name": ..., "value": ..., "confidence": float, "source": "field" | "kvp"},
      ...
    ]
    Uses both documents[0].fields and keyValuePairs.
    """
    fields = []

    # structured fields
    docs = analyze_result.get("documents") or []
    for doc in docs:
        fdict = doc.get("fields") or {}
        for fname, fval in fdict.items():
            val = (
                fval.get("content")
                or fval.get("valueString")
                or fval.get("value")
                or ""
            )
            conf = fval.get("confidence", 1.0)
            fields.append(
                {
                    "name": str(fname),
                    "value": str(val),
                    "confidence": float(conf),
                    "source": "document_field",
                }
            )

    # key-value pairs
    kvps = analyze_result.get("keyValuePairs") or []
    for i, kv in enumerate(kvps):
        key_text = (kv.get("key") or {}).get("content", "").strip()
        val_text = (kv.get("value") or {}).get("content", "").strip()
        conf = kv.get("confidence", 1.0)
        if not key_text and not val_text:
            continue
        name = key_text or f"KVP_{i}"
        fields.append(
            {
                "name": name,
                "value": val_text,
                "confidence": float(conf),
                "source": "key_value_pair",
            }
        )

    # fallback: if still very few, add page-level content lines
    if len(fields) < 15:
        pages = analyze_result.get("pages") or []
        for p in pages:
            content = p.get("content") or ""
            for idx, line in enumerate(
                [ln.strip() for ln in content.splitlines() if ln.strip()]
            ):
                fields.append(
                    {
                        "name": f"page{p.get('pageNumber', 1)}_line{idx}",
                        "value": line,
                        "confidence": 1.0,
                        "source": "page_line",
                    }
                )
                if len(fields) >= 30:
                    break
            if len(fields) >= 30:
                break

    return fields

# ----------------------------------------------------------
# HELPER: matching
# ----------------------------------------------------------
def normalize(s: str) -> str:
    if not s:
        return ""
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def compute_matches(fields_a: List[Dict], fields_b: List[Dict], threshold: float) -> pd.DataFrame:
    """
    For each field in A, find best B by:
    - name similarity (token_set_ratio)
    - value similarity (token_set_ratio)
    total_score = 0.5 * name_score + 0.5 * value_score
    """
    rows = []
    if not fields_a or not fields_b:
        return pd.DataFrame(rows)

    for fa in fields_a:
        best = None
        best_name = 0.0
        best_val = 0.0
        best_total = -1.0

        a_name = normalize(fa["name"])
        a_val = normalize(fa["value"])

        for fb in fields_b:
            b_name = normalize(fb["name"])
            b_val = normalize(fb["value"])

            if not (a_name or a_val or b_name or b_val):
                continue

            name_score = fuzz.token_set_ratio(a_name, b_name) / 100.0 if (a_name or b_name) else 0.0
            val_score = fuzz.token_set_ratio(a_val, b_val) / 100.0 if (a_val or b_val) else 0.0

            total = 0.5 * name_score + 0.5 * val_score

            if total > best_total:
                best_total = total
                best_name = name_score
                best_val = val_score
                best = fb

        rows.append(
            {
                "fieldA_name": fa["name"],
                "fieldA_value": fa["value"],
                "confA": fa["confidence"],
                "fieldB_name": best["name"] if best else None,
                "fieldB_value": best["value"] if best else None,
                "confB": best["confidence"] if best else None,
                "name_score": best_name,
                "value_score": best_val,
                "match_score": best_total if best is not None else 0.0,
                "matched": bool(best is not None and best_total >= threshold),
            }
        )

    return pd.DataFrame(rows)

# ----------------------------------------------------------
# HELPER: Azure OpenAI
# ----------------------------------------------------------
def azure_openai_completion(prompt: str, max_tokens: int = 300, extra_context: str = "") -> str:
    """
    Chat completion using Azure OpenAI, aligned with your working sample:
    - endpoint: AZURE_OPENAI_ENDPOINT (e.g. https://sache-mibgj0x2-swedencentral.cognitiveservices.azure.com)
    - deployment/model: AZURE_OPENAI_DEPLOYMENT (e.g. gpt-5-chat)
    - api-version: AZURE_OPENAI_API_VERSION (default 2024-12-01-preview)
    """
    if not OPENAI_ENDPOINT or not OPENAI_KEY or not OPENAI_DEPLOY:
        return "Azure OpenAI not configured. Check AZURE_OPENAI_* env vars."

    url = (
        OPENAI_ENDPOINT.rstrip("/")
        + f"/openai/deployments/{OPENAI_DEPLOY}/chat/completions"
        + f"?api-version={OPENAI_API_VERSION}"
    )

    full_user_content = (extra_context + "\n\nUser query:\n" + prompt).strip()

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You help validate entity extraction and matching between two documents. "
                    "Use the provided fields and scores to answer questions clearly."
                ),
            },
            {"role": "user", "content": full_user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 1.0,
        "model": OPENAI_DEPLOY,  # matches your working script
    }

    headers = {"Content-Type": "application/json", "api-key": OPENAI_KEY}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        # Surface any remaining config problem
        return f"OpenAI error: {r.status_code} {r.text}"
    j = r.json()
    return j.get("choices", [{}])[0].get("message", {}).get("content", "")

# ----------------------------------------------------------
# HELPER: PDF report
# ----------------------------------------------------------
def make_pdf_bytes(df: pd.DataFrame, overall_match: float) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    y = h - 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, f"Document Match Report - Overall matched fields: {overall_match:.1f}%")
    y -= 20
    c.setFont("Helvetica", 9)

    max_rows = 60  # limit to avoid huge PDFs
    for idx, row in df.head(max_rows).iterrows():
        line = (
            f"{row['fieldA_name']} = '{row['fieldA_value'][:30]}'  "
            f"|| {row['fieldB_name']} = '{str(row['fieldB_value'])[:30]}'  "
            f"score={row['match_score']:.2f} matched={row['matched']}"
        )
        c.drawString(40, y, line[:140])
        y -= 12
        if y < 40:
            c.showPage()
            y = h - 40
            c.setFont("Helvetica", 9)

    c.save()
    buf.seek(0)
    return buf.read()

# ----------------------------------------------------------
# UI: Upload
# ----------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.header("Document A")
    uploaded_a = st.file_uploader(
        "Upload file A (pdf/png/jpg/docx)",
        type=["pdf", "png", "jpg", "jpeg", "docx"],
        key="file_a",
    )
with col2:
    st.header("Document B")
    uploaded_b = st.file_uploader(
        "Upload file B (pdf/png/jpg/docx)",
        type=["pdf", "png", "jpg", "jpeg", "docx"],
        key="file_b",
    )

colx, coly = st.columns(2)
with colx:
    run_extract = st.button("1️⃣ Run extraction + initial matching")
with coly:
    rerun_match = st.button("2️⃣ Re-run matching with current threshold (no re-extraction)")

# ----------------------------------------------------------
# STEP 1: extraction + initial matching
# ----------------------------------------------------------
if run_extract:
    def read_bytes(u):
        if hasattr(u, "read"):
            b = u.read()
            try:
                u.seek(0)
            except Exception:
                pass
            return b, getattr(u, "name", "file")
        else:
            return None, None

    a_bytes, a_name = read_bytes(uploaded_a) if uploaded_a else (None, None)
    b_bytes, b_name = read_bytes(uploaded_b) if uploaded_b else (None, None)

    if not a_bytes or not b_bytes:
        st.error("Please upload both documents before running.")
    else:
        with st.spinner("Calling Azure Document Intelligence for Document A..."):
            res_a = azure_docint_analyze_bytes(a_bytes, filename_hint=a_name or "a.pdf")
        with st.spinner("Calling Azure Document Intelligence for Document B..."):
            res_b = azure_docint_analyze_bytes(b_bytes, filename_hint=b_name or "b.pdf")

        fields_a = extract_fields(res_a)
        fields_b = extract_fields(res_b)

        st.session_state.fields_a = fields_a
        st.session_state.fields_b = fields_b
        st.session_state.has_extracted = True
        st.session_state.decision = None

        st.success(f"Extraction finished. A: {len(fields_a)} fields, B: {len(fields_b)} fields")

        df_matches = compute_matches(fields_a, fields_b, match_threshold)
        st.session_state.matches_df = df_matches

# ----------------------------------------------------------
# STEP 2: re-match using same extracted fields
# ----------------------------------------------------------
if rerun_match:
    if not st.session_state.has_extracted:
        st.error("No extracted data yet. First click 'Run extraction + initial matching'.")
    else:
        df_matches = compute_matches(
            st.session_state.fields_a,
            st.session_state.fields_b,
            match_threshold,
        )
        st.session_state.matches_df = df_matches
        st.success("Re-matching completed with new threshold.")

# ----------------------------------------------------------
# SHOW MATCH TABLE + APPROVE / REJECT
# ----------------------------------------------------------
df = st.session_state.matches_df

if df is not None:
    st.markdown("## Matching results")

    if len(df) == 0:
        st.write("No fields matched. Try lowering the match threshold and click 'Re-run matching'.")
    else:
        matched_rows = df[df["matched"]]
        overall_percent = (len(matched_rows) / len(df)) * 100 if len(df) > 0 else 0.0
        st.write(f"Matched fields: {len(matched_rows)} / {len(df)}  ({overall_percent:.1f}%)")
        st.dataframe(df)

    colA, colB = st.columns(2)
    with colA:
        if st.button("✅ Approve report"):
            st.session_state.decision = "approved"
    with colB:
        if st.button("❌ Reject report"):
            st.session_state.decision = "rejected"

if st.session_state.decision:
    st.info(f"Current session decision: {st.session_state.decision}")

# ----------------------------------------------------------
# DOWNLOADS AFTER APPROVAL
# ----------------------------------------------------------
if st.session_state.decision == "approved" and st.session_state.matches_df is not None:
    st.markdown("## Download final report")

    df = st.session_state.matches_df.copy()
    matched_rows = df[df["matched"]]
    overall_percent = (len(matched_rows) / len(df)) * 100 if len(df) > 0 else 0.0

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    pdf_bytes = make_pdf_bytes(df, overall_percent)

    st.download_button(
        "⬇️ Download CSV",
        data=csv_bytes,
        file_name="field_match_report.csv",
        mime="text/csv",
    )
    st.download_button(
        "⬇️ Download PDF summary",
        data=pdf_bytes,
        file_name="field_match_report.pdf",
        mime="application/pdf",
    )

# ----------------------------------------------------------
# OPTIONAL LLM SUMMARY
# ----------------------------------------------------------
if st.session_state.matches_df is not None:
    if st.button("✨ Generate LLM summary of matches"):
        df_small = st.session_state.matches_df.head(100)  # avoid huge payload
        context = df_small.to_json(orient="records")
        prompt = (
            "You are given extracted fields from two documents and their match scores. "
            "Summarize the quality of the matching, highlight key strong matches, "
            "and point out suspicious or low-scoring matches that may need manual review."
        )
        out = azure_openai_completion(prompt, max_tokens=500, extra_context=context)
        st.markdown("### LLM Summary")
        st.write(out)

# ----------------------------------------------------------
# LLM CHAT AFTER APPROVAL
# ----------------------------------------------------------
if st.session_state.decision == "approved" and st.session_state.matches_df is not None:
    st.markdown("## LLM chat — ask questions about the matching")

    # show previous chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_msg = st.chat_input(
        "Ask things like: 'Which fields did not match well?', "
        "'Show mismatches for PO numbers', or 'Generate test cases to validate this matching'."
    )

    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.write(user_msg)

        df_small = st.session_state.matches_df.head(150)
        context = df_small.to_json(orient="records")

        reply = azure_openai_completion(
            prompt=user_msg,
            max_tokens=600,
            extra_context=context,
        )

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)
