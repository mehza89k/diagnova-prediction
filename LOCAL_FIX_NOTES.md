# Local troubleshooting (Windows / PowerShell)

**`Activate.ps1 cannot be loaded because running scripts is disabled`**
Run once in that window: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again.

**`'python' is not recognized`** - reinstall Python 3.10+ from python.org and tick "Add python.exe to PATH". Try `py -m venv .venv`.

**`DiagNova needs Python 3.10 or newer`** - delete `.venv`, install a newer Python, recreate the venv.

**`Address already in use` / port 8000 busy** - `uvicorn app:app --reload --port 8001` and open that port.

**`ModuleNotFoundError`** - the venv isn't active (prompt should start with `(.venv)`), or run `pip install -r requirements.txt` again.

**Page loads but "server not reachable"** - open the site through http://127.0.0.1:8000, not by double-clicking `index.html`.

**Signed out after closing the tab** - by design (token kept in session storage).

**Scanned PDFs / photos of reports aren't read** - they need OCR: `pip install pytesseract` plus the Tesseract installer for Windows. Text PDFs, DOCX, TXT and CSV work without it.

**Image screening says "DEMO MODE"** - expected; see MODEL_CARD_NOTES.md.

**Lost the database / forgot password** - password reset isn't implemented; delete files in `data\` to start fresh.
