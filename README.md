# Multi-Agent Local Pipeline (LangGraph-style) for POD/BOL OCR, NER, Matching, Human Review, Report Generation

This project runs entirely locally (no cloud required) and provides:
- OCR using Tesseract (via pytesseract) and pdf2image
- NER using spaCy with rule-based patterns
- Embeddings using sentence-transformers (local)
- Vector store using FAISS
- Matching logic with rapidfuzz + optional Azure OpenAI LLM wrapper (you can call Azure endpoint for LLM decisions)
- A lightweight LangGraph-style orchestrator implemented in Python
- Streamlit UI for upload, review, and export

## What you get
- `app_streamlit.py` — Streamlit front-end to upload PDFs, run the pipeline and review results.
- `pipeline/` — contains the orchestrator and agent implementations.
- `requirements.txt` — Python dependencies.
- `utils/` — additional instructions.

## Quick start (Windows / Linux / macOS)
1. Install Tesseract OCR:
   - Windows: install from https://github.com/UB-Mannheim/tesseract/wiki (select Add to PATH during install).
   - macOS: `brew install tesseract`
   - Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y tesseract-ocr`

2. Create a Python virtual environment and activate it:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install Python requirements:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Run Streamlit app:
   ```bash
   streamlit run app_streamlit.py
   ```

5. Upload your PDF(s) in the UI and click **Run Pipeline**.

## Notes
- If Tesseract isn't found by `pytesseract`, set the path in code:
  ```python
  import pytesseract
  pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
  ```
- The project uses local sentence-transformers embeddings and FAISS. If you want to use Azure OpenAI for LLM decisions, see `utils/azure_instructions.txt` for the minimal wrapper guidance.
