# DiagNova's Prediction (local)

Educational health-screening prototype: FastAPI backend + single-file website. **Not a medical device.**

## Run it (Windows PowerShell)

Requires **Python 3.10+** (`python --version`). From inside this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app:app --reload
```

Open **http://127.0.0.1:8000** (interactive API docs: http://127.0.0.1:8000/docs).
Stop with `Ctrl+C`. Next time, only run the 2nd and 5th commands.

Optional: `Copy-Item .env.example .env` to change settings.

## Folder layout

| Path | What it is |
|---|---|
| `app.py` | The whole backend (auth, screening engine, report/image endpoints, PubMed, records) |
| `static/index.html` | The website. Served at `/` and already wired to the API |
| `data/` | SQLite database and the auto-generated dev secret are created here |
| `.env.example` | Settings template |
| `requirements.txt` | Python packages |
| `LOCAL_FIX_NOTES.md` | Troubleshooting for common Windows errors |
| `MODEL_CARD_NOTES.md` | What the screening/image features really are, and their limits |

## What works

Register / sign in, symptom + lab screening (runs in the browser; same rules on the server at `/api/screen`),
saved history, profile, lab-report upload that auto-fills lab fields, PubMed links, care-request storage.
Image screening (`/api/image-predict`) is a labelled **demo placeholder** until you add real models.

## Reset everything

Stop the server, then delete the files inside `data\` (keep `.gitkeep`).
