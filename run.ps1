# Starts the FastAPI RAG chatbot server (PowerShell)
$env:PYTHONUTF8 = "1"   # avoids UnicodeEncodeError from emoji in console output on Windows
& ".venv\Scripts\python.exe" -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
