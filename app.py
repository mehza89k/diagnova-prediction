"""
DiagNova's Prediction — FastAPI backend (v1.0)
==============================================

Educational screening-support API. NOT a medical device; nothing here is a diagnosis.

Synchronised with `diagnova_home_final_fix.html`
------------------------------------------------
The current frontend is entirely client-side (JS rule engine + Firebase + localStorage) and
makes no HTTP calls, so this contract is derived from its *data model*, not from fetch() calls:

  * Symptom IDs, the 9 conditions, lab rules, weights, match levels (>=70 strong, >=45 moderate,
    else possible), the 12% cut-off and top-5 limit are ported 1:1 from `runScreening()`.
  * Red flags port `detectRedFlags()` and the assistant's "urgent" regex.
  * Lab names/limits (`fasting_glucose`, `hemoglobin`, `ldl`, `systolic_bp`, `diastolic_bp`),
    age 0-120, sex in {male, female, other}, password >= 8 chars mirror the forms.
  * A saved record has the same shape the UI keeps in `AppState.records`:
        {id, created_at, input, results}  (+ kind/title/red_flags)
  * Profile fields (fullName, phone, ... medications) and care-request fields mirror the forms.

Endpoints
---------
  Public      GET  /healthz            GET /api/symptoms         GET /api/evidence
              POST /api/screen   (auth optional; `save: true` persists when signed in)
  Auth        POST /api/auth/register  POST /api/auth/login      GET /api/auth/me
              POST /api/auth/delete-account
  Uploads     POST /api/parse-report   POST /api/image-predict?category=skin|eye|oncology
  Records     POST/GET/DELETE /api/records ... (+ /bulk, /export, /{id})
  Account     GET/PUT /api/profile     POST/GET/DELETE /api/care-requests

Run:  pip install -r requirements.txt && python app.py      (or: uvicorn app:app --reload)
"""
from __future__ import annotations

import sys

if sys.version_info < (3, 10):
    raise SystemExit("DiagNova needs Python 3.10 or newer (you have %d.%d). Install it from python.org and recreate .venv." % sys.version_info[:2])

import base64
import datetime as dt
import hashlib
import hmac
import html as html_lib
import io
import json
import logging
import math
import os
import re
import secrets
import shutil
import sqlite3
import threading
import time
import zipfile
from collections import OrderedDict, defaultdict, deque
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import quote_plus

import jwt  # PyJWT
import requests
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

log = logging.getLogger("diagnova")
logging.basicConfig(level=os.getenv("DIAGNOVA_LOG_LEVEL", "INFO"))
# NOTE: never log request bodies, extracted report text, or lab values — they are health data.

# ============================================================================================
# 0. CONFIGURATION
# ============================================================================================
APP_DIR = Path(__file__).resolve().parent
try:  # optional: load settings from a .env file next to app.py (see .env.example)
    from dotenv import load_dotenv
    load_dotenv(APP_DIR / ".env")
except ImportError:
    pass
ENV = os.getenv("DIAGNOVA_ENV", "development").lower()
IS_PROD = ENV == "production"

DB_PATH = Path(os.getenv("DIAGNOVA_DB_PATH", str(APP_DIR / "data" / "diagnova.sqlite3")))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

TOKEN_TTL_SECONDS = int(os.getenv("DIAGNOVA_TOKEN_TTL", str(60 * 60 * 12)))
JWT_ISSUER = "diagnova"
PBKDF2_ITERATIONS = 600_000          # OWASP 2023 guidance for PBKDF2-HMAC-SHA256
LEGACY_PBKDF2_ITERATIONS = 310_000   # iteration count used by the previous app.py hashes

MAX_DOC_BYTES = 10 * 1024 * 1024     # frontend: "Each file must be 10 MB or smaller"
MAX_DOC_FILES = 5                    # frontend: "no more than 5 files"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_PDF_PAGES = 40
MAX_RECORDS_PER_USER = 5000
MAX_RECORD_JSON_BYTES = 256 * 1024

NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")      # NCBI asks API clients to identify themselves
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")  # optional; raises the rate limit from 3 to 10 req/s


def _load_secret_key() -> str:
    """JWT signing key: env var wins; in dev we generate one and persist it beside the DB."""
    env_key = os.getenv("DIAGNOVA_SECRET_KEY")
    if env_key:
        if len(env_key) < 32:
            raise RuntimeError("DIAGNOVA_SECRET_KEY must be at least 32 characters.")
        return env_key
    if IS_PROD:
        raise RuntimeError("Set DIAGNOVA_SECRET_KEY (>=32 chars) when DIAGNOVA_ENV=production.")
    key_file = DB_PATH.parent / ".dev_secret_key"
    if key_file.exists():
        return key_file.read_text().strip()
    key = secrets.token_urlsafe(48)
    key_file.write_text(key)
    try:
        key_file.chmod(0o600)
    except OSError:
        pass
    log.warning("Generated a development JWT secret at %s (do NOT use this in production).", key_file)
    return key


SECRET_KEY = _load_secret_key()


def _cors_origins() -> list[str]:
    """Allowed browser origins. Dev defaults cover common local servers and file:// ('null')."""
    env = os.getenv("DIAGNOVA_CORS_ORIGINS")
    if env:
        return [o.strip() for o in env.split(",") if o.strip()]
    if IS_PROD:
        return []  # same-origin only unless explicitly configured
    ports = (3000, 5173, 5500, 8000, 8080)
    origins = [f"http://{host}:{p}" for host in ("localhost", "127.0.0.1") for p in ports]
    origins.append("null")  # an HTML file opened by double-click sends `Origin: null`
    return origins


# ============================================================================================
# 1. SMALL UTILITIES
# ============================================================================================
UTC = timezone.utc


def iso_now() -> str:
    """UTC timestamp in the same format as JS `toISOString()` so string sorting == time sorting."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def normalize_ts(value: str | None) -> str:
    """Accept a client-supplied ISO timestamp; fall back to 'now' if missing/invalid/future."""
    if not value:
        return iso_now()
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return iso_now()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    dt = dt.astimezone(UTC)
    if dt > datetime.now(UTC) + timedelta(minutes=5):
        return iso_now()
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def fmt_num(v: float) -> str:
    """Format like JS number->string (126.0 -> '126', 126.5 -> '126.5')."""
    return f"{v:g}"


class RateLimiter:
    """Tiny in-memory sliding-window limiter. Per-process only: use Redis/nginx if you scale out."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int, window: int) -> bool:
        """Record a hit; return False if the caller is over the limit."""
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            if len(self._hits) > 20_000:  # crude memory guard
                self._hits = defaultdict(deque, {k: v for k, v in self._hits.items() if v})
            return True


limiter = RateLimiter()


def client_ip(request: Request) -> str:
    # Deliberately ignores X-Forwarded-For (spoofable). Behind a trusted proxy, run uvicorn with
    # --proxy-headers --forwarded-allow-ips=<proxy ip> so request.client is already correct.
    return request.client.host if request.client else "unknown"


def rate_limit(name: str, limit: int, window: int):
    """Dependency factory: `Depends(rate_limit('screen', 30, 60))` -> 30 requests / 60 s / IP."""

    def dep(request: Request) -> None:
        if not limiter.hit(f"{name}:{client_ip(request)}", limit, window):
            raise HTTPException(status_code=429, detail="Too many requests. Please slow down and retry shortly.")

    return dep


# ============================================================================================
# 2. DATABASE (raw sqlite3)
# ============================================================================================
@contextmanager
def db():
    """Open a short-lived connection; commit on success, roll back on error, always close."""
    con = sqlite3.connect(DB_PATH, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db() -> None:
    with db() as con:
        con.execute("PRAGMA journal_mode=WAL")
        # `users` keeps the previous schema so existing accounts keep working.
        con.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL)""")
        con.execute("""CREATE TABLE IF NOT EXISTS profiles (
            user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            data_json TEXT NOT NULL,
            updated_at TEXT NOT NULL)""")
        # Records are keyed (user_id, id) so client-generated ids (the UI uses
        # "<Date.now()>-<rand>") can never collide across users and upserts are idempotent.
        # Denormalised columns exist purely to make list/filter queries cheap.
        con.execute("""CREATE TABLE IF NOT EXISTS health_records (
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            id TEXT NOT NULL,
            kind TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            top_condition TEXT,
            top_category TEXT,
            top_confidence INTEGER,
            symptom_count INTEGER NOT NULL DEFAULT 0,
            has_red_flags INTEGER NOT NULL DEFAULT 0,
            categories TEXT NOT NULL DEFAULT '',
            search_text TEXT NOT NULL DEFAULT '',
            payload_json TEXT NOT NULL,
            PRIMARY KEY (user_id, id))""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_hr_user_created ON health_records(user_id, created_at DESC)")
        con.execute("""CREATE TABLE IF NOT EXISTS care_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL,
            data_json TEXT NOT NULL)""")
        con.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        _migrate_legacy_records(con)


def _migrate_legacy_records(con: sqlite3.Connection) -> None:
    """One-time copy of rows from the old `records` table (integer ids, free-form payloads)."""
    if con.execute("SELECT 1 FROM meta WHERE key='legacy_records_migrated'").fetchone():
        return
    has_legacy = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='records'").fetchone()
    if has_legacy:
        kind_map = {"screening": "screening", "skin": "image", "image_multimodal": "image", "report_ocr": "report"}
        for r in con.execute("SELECT id, user_id, kind, title, created_at, payload_json FROM records").fetchall():
            try:
                old = json.loads(r["payload_json"])
            except ValueError:
                old = {}
            ts = normalize_ts(r["created_at"])
            con.execute(
                """INSERT OR IGNORE INTO health_records
                   (user_id,id,kind,title,created_at,updated_at,search_text,payload_json)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (r["user_id"], f"legacy-{r['id']}", kind_map.get(r["kind"], "screening"), r["title"], ts, ts,
                 (r["title"] or "").lower(), json.dumps({"input": None, "results": [], "red_flags": [], "payload": old})),
            )
    con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('legacy_records_migrated','1')")


# ============================================================================================
# 3. AUTHENTICATION (PBKDF2 password hashing + HS256 JWT bearer tokens)
# ============================================================================================
# Hash format: pbkdf2_sha256$<iterations>$<salt_b64url>$<digest_b64url>
# The previous 3-part format (no iteration field, 310k rounds) is still verified and is
# transparently upgraded on the user's next successful login.
def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s.encode())


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> tuple[bool, bool]:
    """Return (is_valid, needs_rehash)."""
    try:
        parts = stored.split("$")
        if parts[0] != "pbkdf2_sha256":
            return False, False
        if len(parts) == 4:
            iterations, salt, expected = int(parts[1]), _unb64(parts[2]), _unb64(parts[3])
        elif len(parts) == 3:  # legacy
            iterations, salt, expected = LEGACY_PBKDF2_ITERATIONS, _unb64(parts[1]), _unb64(parts[2])
        else:
            return False, False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
        ok = hmac.compare_digest(actual, expected)
        return ok, ok and iterations < PBKDF2_ITERATIONS
    except Exception:
        return False, False


_DUMMY_HASH: str | None = None


def _dummy_hash() -> str:
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password(secrets.token_urlsafe(16))
    return _DUMMY_HASH


def create_token(user_id: int) -> str:
    now = int(time.time())
    return jwt.encode(
        {"sub": str(user_id), "iat": now, "exp": now + TOKEN_TTL_SECONDS, "iss": JWT_ISSUER, "jti": secrets.token_hex(8)},
        SECRET_KEY, algorithm="HS256",
    )


def decode_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"], issuer=JWT_ISSUER, options={"require": ["exp", "sub"]})
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Your session expired. Please sign in again.",
                            headers={"WWW-Authenticate": "Bearer"})


bearer = HTTPBearer(auto_error=False)


def _user_from_credentials(credentials: HTTPAuthorizationCredentials) -> dict[str, Any]:
    payload = decode_token(credentials.credentials)
    try:
        uid = int(payload["sub"])
    except (KeyError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid session.")
    with db() as con:
        row = con.execute("SELECT id, email, display_name FROM users WHERE id=?", (uid,)).fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="Account not found.")
    return dict(row)


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict[str, Any]:
    """Dependency: a valid bearer token is REQUIRED."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Sign in to use this feature.", headers={"WWW-Authenticate": "Bearer"})
    return _user_from_credentials(credentials)


def optional_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict[str, Any] | None:
    """Dependency: anonymous allowed (the UI is 'privacy-first'). A bad/expired token == anonymous."""
    if not credentials:
        return None
    try:
        return _user_from_credentials(credentials)
    except HTTPException:
        return None


CurrentUser = Annotated[dict, Depends(current_user)]
OptionalUser = Annotated[dict | None, Depends(optional_user)]


class LoginThrottle:
    """Lock an (ip,email) pair for 15 min after 5 failed logins. In-memory, per process."""

    MAX_FAILS, WINDOW = 5, 15 * 60

    def __init__(self) -> None:
        self._fails: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, q: deque[float], now: float) -> None:
        while q and now - q[0] > self.WINDOW:
            q.popleft()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            q = self._fails[key]
            self._prune(q, now)
            if len(q) >= self.MAX_FAILS:
                raise HTTPException(status_code=429, detail="Too many failed sign-in attempts. Try again in 15 minutes.")

    def fail(self, key: str) -> None:
        with self._lock:
            self._fails[key].append(time.monotonic())

    def clear(self, key: str) -> None:
        with self._lock:
            self._fails.pop(key, None)


throttle = LoginThrottle()

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ============================================================================================
# 4. PYDANTIC v2 MODELS (field names/limits mirror the frontend forms)
# ============================================================================================
# Frontend `labLimits`: glucose 0-2000, hemoglobin 0-100, LDL 0-2000, SBP 0-400, DBP 0-300.
# We require > 0 (a literal 0 is never a real measurement).
LAB_LIMITS = {"fasting_glucose": 2000, "hemoglobin": 100, "ldl": 2000, "systolic_bp": 400, "diastolic_bp": 300}


class RegisterIn(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    # The UI field is "Full name"; accept the common spellings.
    display_name: str = Field(min_length=1, max_length=80,
                              validation_alias=AliasChoices("display_name", "name", "displayName", "fullName"))
    email: str = Field(max_length=254)
    password: str = Field(min_length=8, max_length=128)  # UI: "At least 8 characters"; max stops PBKDF2 DoS

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        v = v.strip().lower()
        if not EMAIL_RE.fullmatch(v):
            raise ValueError("Enter a valid email address.")
        return v

    @field_validator("display_name")
    @classmethod
    def _name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Name is required.")
        return v


class LoginIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    email: str = Field(max_length=254)
    password: str = Field(max_length=128)


class DeleteAccountIn(BaseModel):
    password: str = Field(max_length=128)


class LabValues(BaseModel):
    model_config = ConfigDict(extra="ignore")
    fasting_glucose: float | None = Field(default=None, gt=0, le=LAB_LIMITS["fasting_glucose"], allow_inf_nan=False)
    hemoglobin: float | None = Field(default=None, gt=0, le=LAB_LIMITS["hemoglobin"], allow_inf_nan=False)
    ldl: float | None = Field(default=None, gt=0, le=LAB_LIMITS["ldl"], allow_inf_nan=False)
    systolic_bp: float | None = Field(default=None, gt=0, le=LAB_LIMITS["systolic_bp"], allow_inf_nan=False)
    diastolic_bp: float | None = Field(default=None, gt=0, le=LAB_LIMITS["diastolic_bp"], allow_inf_nan=False)


# Symptom ids used by the previous backend -> ids used by the current UI.
LEGACY_SYMPTOM_ALIASES = {
    "thirst": "increased_thirst", "urination": "frequent_urination", "weight_loss": "unintentional_weight_change",
    "shortness_breath": "shortness_of_breath", "chest_pain": "chest_discomfort", "cough": "persistent_cough",
}


class ScreenIn(BaseModel):
    """Body of POST /api/screen — same fields the UI's `handleScreeningSubmit()` builds, plus extras."""
    model_config = ConfigDict(extra="ignore")
    age: float | None = Field(default=None, ge=0, le=120, allow_inf_nan=False)
    sex: Literal["male", "female", "other"] | None = None
    symptoms: list[str] = Field(default_factory=list, max_length=40)
    labs: LabValues = Field(default_factory=LabValues)
    # Free text (the "AI Health Assistant" box). Keyword-matched into symptoms + red flags.
    description: str | None = Field(default=None, max_length=2000)
    save: bool = False                       # mirrors the "Save this screening" checkbox
    title: str | None = Field(default=None, max_length=160)

    @field_validator("sex", mode="before")
    @classmethod
    def _sex(cls, v: Any) -> Any:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None  # the <select> sends "" for "Prefer not to say"
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("symptoms")
    @classmethod
    def _symptoms(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for s in v:
            s = LEGACY_SYMPTOM_ALIASES.get(s.strip().lower(), s.strip().lower())
            if s not in SYMPTOM_MAP:
                raise ValueError(f"Unknown symptom id '{s}'. Valid ids: {', '.join(SYMPTOM_MAP)}")
            if s not in out:
                out.append(s)
        return out


class ResultItem(BaseModel):
    """One condition match as stored in a record. Extra keys (evidence, matchLevel...) are preserved."""
    model_config = ConfigDict(extra="allow")
    id: str | None = Field(default=None, max_length=64)
    name: str = Field(min_length=1, max_length=200)
    category: str = Field(default="", max_length=200)
    specialist: str = Field(default="", max_length=200)
    confidence: float = Field(default=0, ge=0, le=100)


class RecordIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_\-]{1,64}$")  # optional client id => idempotent upsert
    kind: Literal["screening", "report", "image"] = "screening"
    title: str | None = Field(default=None, max_length=160)
    created_at: str | None = Field(default=None, max_length=40)
    input: dict[str, Any] | None = None
    results: list[ResultItem] = Field(default_factory=list, max_length=20)
    red_flags: list[str] = Field(default_factory=list, max_length=10)
    payload: dict[str, Any] | None = None   # free-form body for report/image records

    @field_validator("kind", mode="before")
    @classmethod
    def _legacy_kind(cls, v: Any) -> Any:
        # Kinds used by the previous backend.
        return {"skin": "image", "image_multimodal": "image", "report_ocr": "report"}.get(v, v) if isinstance(v, str) else v


class BulkRecordsIn(BaseModel):
    records: list[RecordIn] = Field(max_length=400)  # frontend syncs in batches of <= 400


class ProfileIn(BaseModel):
    """Mirrors `profileFormData()`. camelCase on purpose: identical keys to the frontend."""
    model_config = ConfigDict(extra="ignore")
    fullName: str = Field(min_length=1, max_length=100)
    phone: str = Field(default="", max_length=30)
    email: str = Field(default="", max_length=160)
    dob: str = Field(default="", max_length=10)          # YYYY-MM-DD from <input type=date>
    gender: Literal["", "Female", "Male", "Non-binary", "Self-describe"] = ""
    bloodGroup: Literal["", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"] = ""
    emergencyContact: str = Field(default="", max_length=100)
    conditions: str = Field(default="", max_length=1500)
    allergies: str = Field(default="", max_length=1500)
    medications: str = Field(default="", max_length=1500)

    @field_validator("dob")
    @classmethod
    def _dob(cls, v: str) -> str:
        if v:
            d = date.fromisoformat(v)  # ValueError -> 422
            if d > date.today():
                raise ValueError("Date of birth cannot be in the future.")
        return v


CARE_SPECIALTIES = Literal["General Medicine", "Cardiology", "Dermatology", "Gynecology", "Pediatrics",
                           "Neurology", "Orthopedics", "Mental Health", "Other / not sure"]


class CareRequestIn(BaseModel):
    """Mirrors the 'care request' modal. Stored only; NOT forwarded to any provider."""
    model_config = ConfigDict(extra="ignore")
    name: str = Field(min_length=1, max_length=100)
    phone: str = Field(pattern=r"^[0-9+()\-\s.]{5,30}$")
    specialty: CARE_SPECIALTIES
    type: Literal["Online consultation", "Clinic visit", "Home care inquiry"] = "Online consultation"
    date: dt.date | None = None  # NB: annotated via `dt.` because the field name shadows `date`
    area: str = Field(default="", max_length=100)
    reason: str = Field(default="", max_length=1000)


# ============================================================================================
# 5. CLINICAL KNOWLEDGE BASE  (ported 1:1 from the frontend's SYMPTOMS / CONDITIONS)
# ============================================================================================
SYMPTOMS = [
    ("fatigue", "Fatigue / Excessive Tiredness"), ("fever", "Fever / Elevated Body Temperature"),
    ("headache", "Headache"), ("dizziness", "Dizziness / Lightheadedness"),
    ("increased_thirst", "Increased Thirst (Polydipsia)"), ("frequent_urination", "Frequent Urination (Polyuria)"),
    ("unintentional_weight_change", "Unintentional Weight Change"), ("shortness_of_breath", "Shortness of Breath"),
    ("chest_discomfort", "Chest Discomfort / Chest Pain"), ("severe_chest_pain", "Severe Chest Pain"),
    ("abdominal_pain", "Abdominal Pain"), ("nausea", "Nausea or Vomiting"), ("joint_pain", "Joint Pain (Arthralgia)"),
    ("blurred_vision", "Blurred Vision"), ("persistent_cough", "Persistent Cough"),
]
SYMPTOM_MAP = dict(SYMPTOMS)


def _lab(key: str, weight: int, label: str, min: float | None = None, below: float | None = None) -> dict:
    """Lab rule: matches when  min <= value < below  (either bound optional).
    The frontend used inclusive `max` values like 125 / 139 / 11.9 / 159, which leave gaps for
    decimals (e.g. glucose 125.5 matched nothing). Half-open ranges give identical results on
    whole numbers and no gaps on decimals."""
    return {"key": key, "min": min, "below": below, "weight": weight, "label": label}


CONDITIONS: list[dict[str, Any]] = [
    {"id": "t2dm", "name": "Type 2 Diabetes Mellitus", "category": "Endocrine & Metabolic", "specialist": "Endocrinologist",
     "keySymptoms": ["increased_thirst", "frequent_urination"],
     "supportSymptoms": ["fatigue", "blurred_vision", "unintentional_weight_change", "dizziness"],
     "labs": [_lab("fasting_glucose", 4, "Fasting glucose ≥126 mg/dL suggests diabetes range", min=126),
              _lab("fasting_glucose", 2, "Fasting glucose 100–125 mg/dL suggests prediabetes range", min=100, below=126)],
     "explanation": "The combination of thirst, frequent urination and elevated fasting glucose may suggest dysregulated blood sugar control. Formal testing is required.",
     "limitations": ["Requires confirmatory HbA1c or OGTT", "A single fasting glucose reading is not diagnostic"],
     "pubmedQuery": "type 2 diabetes mellitus screening diagnosis"},
    {"id": "htn", "name": "Hypertension (High Blood Pressure)", "category": "Cardiovascular", "specialist": "Cardiologist / Internal Medicine",
     "keySymptoms": ["headache", "dizziness"], "supportSymptoms": ["chest_discomfort", "blurred_vision", "fatigue"],
     "labs": [_lab("systolic_bp", 4, "Systolic BP ≥140 mmHg (stage 2 hypertension range)", min=140),
              _lab("systolic_bp", 2, "Systolic BP 130–139 mmHg (stage 1 range)", min=130, below=140),
              _lab("diastolic_bp", 4, "Diastolic BP ≥90 mmHg", min=90)],
     "explanation": "Elevated blood pressure readings combined with headache or dizziness may suggest hypertension. Multiple readings on separate occasions are required for diagnosis.",
     "limitations": ["Single BP reading insufficient", "White-coat hypertension must be excluded"],
     "pubmedQuery": "hypertension diagnosis blood pressure guidelines"},
    {"id": "ida", "name": "Iron Deficiency Anemia", "category": "Hematologic", "specialist": "Hematologist / General Physician",
     "keySymptoms": ["fatigue", "dizziness", "shortness_of_breath"],
     "supportSymptoms": ["headache", "blurred_vision", "unintentional_weight_change"],
     "labs": [_lab("hemoglobin", 3, "Hemoglobin 10–11.9 g/dL (mild anemia range)", min=10, below=12),
              _lab("hemoglobin", 4, "Hemoglobin <10 g/dL (moderate anemia range)", below=10)],
     "explanation": "Low hemoglobin combined with fatigue and shortness of breath may suggest anemia. Iron studies (ferritin, TIBC) are needed to confirm iron deficiency.",
     "limitations": ["Hemoglobin alone cannot distinguish anemia types", "Requires ferritin and peripheral smear"],
     "pubmedQuery": "iron deficiency anemia diagnosis ferritin"},
    {"id": "cad", "name": "Cardiovascular Disease / Dyslipidemia", "category": "Cardiovascular", "specialist": "Cardiologist",
     "keySymptoms": ["chest_discomfort", "shortness_of_breath"], "supportSymptoms": ["fatigue", "dizziness", "joint_pain"],
     "labs": [_lab("ldl", 4, "LDL ≥160 mg/dL (high)", min=160),
              _lab("ldl", 2, "LDL 130–159 mg/dL (borderline high)", min=130, below=160)],
     "explanation": "Chest discomfort, shortness of breath, and elevated LDL may be relevant to cardiovascular risk. Formal cardiac evaluation is recommended for persistent symptoms.",
     "limitations": ["Cannot rule in/out acute coronary syndrome", "Requires ECG, troponin and formal risk scoring"],
     "pubmedQuery": "cardiovascular risk LDL cholesterol chest pain"},
    {"id": "hypothyroid", "name": "Hypothyroidism", "category": "Endocrine & Metabolic", "specialist": "Endocrinologist",
     "keySymptoms": ["fatigue", "unintentional_weight_change"], "supportSymptoms": ["joint_pain", "dizziness", "headache"],
     "labs": [],
     "explanation": "Fatigue and weight change can have many causes including thyroid dysfunction. TSH and free T4 testing are required to evaluate this possibility.",
     "limitations": ["Requires TSH/free T4", "Symptoms overlap with many other conditions"],
     "pubmedQuery": "hypothyroidism symptoms diagnosis TSH"},
    {"id": "migraine", "name": "Migraine / Tension-Type Headache", "category": "Neurologic", "specialist": "Neurologist",
     "keySymptoms": ["headache"], "supportSymptoms": ["nausea", "blurred_vision", "dizziness"], "labs": [],
     "explanation": "Recurrent headaches with nausea or visual changes may be consistent with migraine. Red-flag features (sudden onset, focal deficits) require urgent evaluation.",
     "limitations": ["Cannot exclude secondary headache causes", "Requires clinical history and neurological exam"],
     "pubmedQuery": "migraine diagnosis criteria headache"},
    {"id": "uri", "name": "Respiratory Infection (e.g. URI)", "category": "Respiratory", "specialist": "General Physician / Pulmonologist",
     "keySymptoms": ["fever", "persistent_cough"], "supportSymptoms": ["fatigue", "shortness_of_breath", "headache"], "labs": [],
     "explanation": "Fever with cough may reflect a respiratory infection. Most are self-limited; persistent symptoms or breathlessness require in-person evaluation.",
     "limitations": ["Cannot distinguish viral vs bacterial", "Requires clinical exam, possible imaging"],
     "pubmedQuery": "respiratory tract infection diagnosis cough fever"},
    {"id": "gerd", "name": "Gastroesophageal Reflux / Dyspepsia", "category": "Gastrointestinal", "specialist": "Gastroenterologist",
     "keySymptoms": ["abdominal_pain", "nausea"], "supportSymptoms": ["chest_discomfort", "persistent_cough"], "labs": [],
     "explanation": "Upper abdominal discomfort with nausea may relate to reflux or dyspepsia. Persistent or alarm symptoms warrant gastroenterology review.",
     "limitations": ["Overlaps with cardiac causes of chest pain", "Requires clinical assessment"],
     "pubmedQuery": "GERD dyspepsia diagnosis management"},
    {"id": "anxiety", "name": "Anxiety / Stress-Related Symptoms", "category": "Mental Health", "specialist": "Psychiatrist / Clinical Psychologist",
     "keySymptoms": ["fatigue", "dizziness"], "supportSymptoms": ["chest_discomfort", "shortness_of_breath", "headache", "nausea"], "labs": [],
     "explanation": "Overlapping physical symptoms without clear laboratory explanation may relate to anxiety. A thorough medical and psychological assessment is recommended.",
     "limitations": ["Diagnosis of exclusion", "Requires clinical interview"],
     "pubmedQuery": "anxiety disorder physical symptoms assessment"},
]
CONDITION_BY_ID = {c["id"]: c for c in CONDITIONS}

# Tailwind classes are a UI concern; they are included ONLY so a saved result can be dropped straight
# back into the existing `openRecordDetail()` renderer. Remove if you restyle the frontend.
MATCH_LEVELS = {
    "strong": ("Strong match", "bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/60 dark:text-emerald-300 dark:border-emerald-900"),
    "moderate": ("Moderate match", "bg-teal-50 text-teal-700 border-teal-200 dark:bg-teal-950/60 dark:text-teal-300 dark:border-teal-900"),
    "possible": ("Possible match", "bg-slate-100 text-slate-700 border-slate-200 dark:bg-slate-800 dark:text-slate-300 dark:border-slate-700"),
}
MIN_CONFIDENCE = 12   # results below this % are dropped (frontend: `if(confidence < 12) continue`)
MAX_RESULTS = 5
ENGINE_NAME = "DiagNova's Prediction Rule Engine v1.0 (frontend-synchronised)"
DISCLAIMER = ("Educational screening support only. Scores show how well your inputs fit a rule pattern; they are "
              "not probabilities and not a diagnosis. Consult a qualified clinician.")


def match_level(confidence: int) -> dict[str, str]:
    key = "strong" if confidence >= 70 else "moderate" if confidence >= 45 else "possible"
    label, cls = MATCH_LEVELS[key]
    return {"level": key, "label": label, "cls": cls}


# ============================================================================================
# 6. RULES ENGINE
# ============================================================================================
def run_screening(symptoms: list[str], labs: dict[str, float | None]) -> list[dict[str, Any]]:
    """Score every condition. Identical arithmetic to the frontend's `runScreening()`:
       key symptom = 3 pts, supporting symptom = 1 pt, lab rule = its weight;
       confidence = round(score / maxScore * 100) where maxScore sums ALL possible points
       (including mutually-exclusive lab tiers — this is why confidence rarely reaches 100)."""
    selected = set(symptoms)
    results: list[dict[str, Any]] = []
    for cond in CONDITIONS:
        score, evidence, matched = 0, [], []
        for s in cond["keySymptoms"]:
            if s in selected:
                score += 3
                matched.append(s)
                evidence.append("Key symptom present: " + SYMPTOM_MAP.get(s, s))
        for s in cond["supportSymptoms"]:
            if s in selected:
                score += 1
                matched.append(s)
                evidence.append("Supporting symptom: " + SYMPTOM_MAP.get(s, s))
        for rule in cond["labs"]:
            val = labs.get(rule["key"])
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            if (rule["min"] is None or val >= rule["min"]) and (rule["below"] is None or val < rule["below"]):
                score += rule["weight"]
                evidence.append(f"{rule['label']} ({fmt_num(val)})")
        if score == 0:
            continue
        max_score = len(cond["keySymptoms"]) * 3 + len(cond["supportSymptoms"]) + sum(r["weight"] for r in cond["labs"])
        # JS Math.round rounds halves UP (12.5 -> 13); Python's round() is banker's rounding (12.5 -> 12).
        confidence = min(100, math.floor(score / max_score * 100 + 0.5))
        if confidence < MIN_CONFIDENCE:
            continue
        results.append({
            "id": cond["id"], "name": cond["name"], "category": cond["category"], "specialist": cond["specialist"],
            "explanation": cond["explanation"], "limitations": list(cond["limitations"]), "pubmedQuery": cond["pubmedQuery"],
            "confidence": confidence, "evidence": evidence, "matchedSymptoms": matched,
            "matchLevel": match_level(confidence),
            "pubmedUrl": "https://pubmed.ncbi.nlm.nih.gov/?term=" + quote_plus(cond["pubmedQuery"]),
        })
    results.sort(key=lambda r: r["confidence"], reverse=True)
    return results[:MAX_RESULTS]


def group_specialists(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """'Specialist matches': one row per specialist with its best confidence and the conditions behind it."""
    grouped: dict[str, dict[str, Any]] = {}
    for r in results:
        g = grouped.setdefault(r["specialist"], {"specialist": r["specialist"], "top_confidence": 0, "conditions": []})
        g["top_confidence"] = max(g["top_confidence"], r["confidence"])
        g["conditions"].append(r["name"])
    return sorted(grouped.values(), key=lambda g: g["top_confidence"], reverse=True)


# ---- free-text description -> symptoms / red flags -------------------------------------------
# Same keyword list as the UI's "AI Health Assistant" (`analyzeSymptomText()`), with two tweaks:
# bare "temperature" no longer implies fever, and simple negation ("no fever") is respected.
_SYMPTOM_PATTERNS: list[tuple[str, re.Pattern[str]]] = [(sid, re.compile(rx, re.I)) for sid, rx in [
    ("severe_chest_pain", r"(?:severe|crushing|intense|worst)\s+(?:chest\s+pain|pain\s+in\s+(?:my\s+)?chest)"),
    ("chest_discomfort", r"chest\s+(?:discomfort|tightness|pressure|ache|pain)"),
    ("shortness_of_breath", r"shortness of breath|difficulty breathing|trouble breathing|breathless|can'?t breathe|cannot breathe"),
    ("fever", r"fever|feverish|high temperature"),
    ("headache", r"headache|head pain|migraine"),
    ("dizziness", r"dizz(?:y|iness)|lightheaded|vertigo"),
    ("fatigue", r"fatigue|tired|exhausted|low energy|weakness"),
    ("increased_thirst", r"increased thirst|very thirsty|excessive thirst"),
    ("frequent_urination", r"frequent urination|urinating frequently|pee often|passing urine often"),
    ("unintentional_weight_change", r"weight loss|losing weight|weight gain|gained weight|weight change"),
    ("abdominal_pain", r"abdominal pain|stomach ache|stomach pain|belly pain"),
    ("nausea", r"nausea|nauseous|vomit(?:ing)?|feeling sick"),
    ("joint_pain", r"joint pain|painful joints|arthritis pain"),
    ("blurred_vision", r"blurred vision|blurry vision|vision is blurry"),
    ("persistent_cough", r"persistent cough|coughing|cough"),
]]
# Symptoms that drive emergency flags are NEVER negated away ("no relief from chest pain" must not
# be silenced by a naive negation rule). Only low-acuity symptoms honour negation.
_NEVER_NEGATE = {"severe_chest_pain", "chest_discomfort", "shortness_of_breath"}
_NEGATION_TAIL = re.compile(r"\b(?:no|not|without|denies|denied|never)\s+(?:(?!but\b|however\b|although\b|though\b|yet\b)\w+\s+){0,2}$", re.I)

_URGENT_PATTERNS: list[tuple[re.Pattern[str], str]] = [(re.compile(rx, re.I), msg) for rx, msg in [
    (r"severe chest pain|crushing chest pain|intense chest pain|chest pain.{0,40}(?:spreading|radiat)",
     "Your description mentions severe chest pain, which can indicate a heart attack. Call emergency services now."),
    (r"can'?t breathe|cannot breathe|difficulty breathing|trouble breathing",
     "Your description mentions difficulty breathing, which needs urgent in-person evaluation."),
    (r"face droop|one side (?:of (?:my )?)?(?:body|face).{0,30}(?:weak|numb)|sudden (?:confusion|vision loss|weakness|slurred)",
     "Your description mentions possible stroke warning signs. Call emergency services immediately."),
    (r"uncontrolled bleeding|coughing up blood|vomiting blood",
     "Your description mentions serious bleeding, which needs emergency care."),
    (r"suicid(?:al|e)|kill myself|end my life|harm myself",
     "If you are thinking about harming yourself, contact your local emergency number or a crisis line right now."),
]]


def extract_symptoms_from_text(text: str) -> list[str]:
    found: list[str] = []
    for sid, rx in _SYMPTOM_PATTERNS:
        for m in rx.finditer(text):
            negated = sid not in _NEVER_NEGATE and _NEGATION_TAIL.search(text[max(0, m.start() - 40):m.start()])
            if not negated:
                found.append(sid)
                break
    return found


def detect_red_flags(symptoms: list[str], description: str | None = None) -> list[str]:
    """Port of `detectRedFlags()` + urgent phrases from the free-text description."""
    flags: list[str] = []
    s = set(symptoms)
    if "severe_chest_pain" in s:
        flags.append("Severe chest pain can indicate a heart attack or other life-threatening event. Call emergency services now.")
    if "shortness_of_breath" in s and "chest_discomfort" in s:
        flags.append("Chest discomfort with shortness of breath requires urgent evaluation.")
    for rx, msg in _URGENT_PATTERNS:
        if description and rx.search(description) and msg not in flags:
            flags.append(msg)
    return flags


def critical_lab_alerts(labs: dict[str, float | None]) -> list[str]:
    """Conservative 'seek care promptly' thresholds for the UI's `critical-result-warning` banner."""
    out: list[str] = []
    g, hb, sbp, dbp = labs.get("fasting_glucose"), labs.get("hemoglobin"), labs.get("systolic_bp"), labs.get("diastolic_bp")
    if g is not None and (g >= 400 or g < 54):
        out.append(f"Glucose of {fmt_num(g)} mg/dL is in a range that needs prompt medical attention.")
    if hb is not None and hb < 7:
        out.append(f"Hemoglobin of {fmt_num(hb)} g/dL is very low and needs prompt medical attention.")
    if (sbp is not None and sbp >= 180) or (dbp is not None and dbp >= 120):
        out.append("Blood pressure at or above 180/120 mmHg needs prompt medical attention, "
                   "and emergency care if you also have chest pain, breathlessness, weakness or vision changes.")
    return out


def build_screening(age: float | None, sex: str | None, symptoms: list[str], labs: dict[str, float | None],
                    description: str | None) -> dict[str, Any]:
    """Pure function (no I/O): everything /api/screen returns except persistence info."""
    from_text = extract_symptoms_from_text(description) if description else []
    merged = list(dict.fromkeys([*symptoms, *from_text]))
    if not merged and not any(v is not None for v in labs.values()):
        raise HTTPException(status_code=422, detail="Provide at least one symptom, lab value, or a description.")
    results = run_screening(merged, labs)
    flags = detect_red_flags(merged, description)
    return {
        "created_at": iso_now(),
        "engine": ENGINE_NAME,
        "input": {"age": age, "sex": sex, "symptoms": merged, "labs": dict(labs)},
        "detected_from_description": [s for s in from_text if s not in symptoms],
        "red_flags": flags,
        "urgent_notice": bool(flags),
        "critical_lab_alerts": critical_lab_alerts(labs),
        "results": results,
        "specialists": group_specialists(results),
        "top_specialist": results[0]["specialist"] if results else None,
        "no_match": not results,
        "disclaimer": DISCLAIMER,
    }


# ============================================================================================
# 7. RECORDS (persistence helpers)
# ============================================================================================
def _record_columns(rec: RecordIn, results: list[dict[str, Any]]) -> dict[str, Any]:
    top = results[0] if results else None
    title = (rec.title or "").strip()
    if not title:
        title = (f"{top['name']} ({int(top['confidence'])}%)" if top else f"{rec.kind.capitalize()} record")
    cats = sorted({r.get("category", "") for r in results if r.get("category")})
    names = [r.get("name", "") for r in results]
    return {
        "title": title[:160],
        "top_condition": top["name"] if top else None,
        "top_category": top.get("category") if top else None,
        "top_confidence": int(top["confidence"]) if top else None,
        "symptom_count": len((rec.input or {}).get("symptoms", []) or []),
        "has_red_flags": 1 if rec.red_flags else 0,
        "categories": "|" + "|".join(c.lower() for c in cats) + "|" if cats else "",
        "search_text": " ".join([title, *names, *cats]).lower()[:2000],
    }


def upsert_record(con: sqlite3.Connection, user_id: int, rec: RecordIn) -> dict[str, Any]:
    """Insert or replace a record owned by `user_id` (idempotent when the client supplies an id)."""
    rid = rec.id or secrets.token_hex(8)
    results = [r.model_dump() for r in rec.results]
    body = {"input": rec.input, "results": results, "red_flags": rec.red_flags, "payload": rec.payload}
    payload_json = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    if len(payload_json.encode()) > MAX_RECORD_JSON_BYTES:
        raise HTTPException(status_code=413, detail="Record is too large.")
    exists = con.execute("SELECT created_at FROM health_records WHERE user_id=? AND id=?", (user_id, rid)).fetchone()
    if not exists:
        count = con.execute("SELECT COUNT(*) FROM health_records WHERE user_id=?", (user_id,)).fetchone()[0]
        if count >= MAX_RECORDS_PER_USER:
            raise HTTPException(status_code=409, detail="Record limit reached. Delete old records first.")
    created = exists["created_at"] if exists else normalize_ts(rec.created_at)
    c = _record_columns(rec, results)
    con.execute(
        """INSERT INTO health_records
           (user_id,id,kind,title,created_at,updated_at,top_condition,top_category,top_confidence,
            symptom_count,has_red_flags,categories,search_text,payload_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(user_id,id) DO UPDATE SET
             kind=excluded.kind,title=excluded.title,updated_at=excluded.updated_at,
             top_condition=excluded.top_condition,top_category=excluded.top_category,
             top_confidence=excluded.top_confidence,symptom_count=excluded.symptom_count,
             has_red_flags=excluded.has_red_flags,categories=excluded.categories,
             search_text=excluded.search_text,payload_json=excluded.payload_json""",
        (user_id, rid, rec.kind, c["title"], created, iso_now(), c["top_condition"], c["top_category"],
         c["top_confidence"], c["symptom_count"], c["has_red_flags"], c["categories"], c["search_text"], payload_json),
    )
    return {"id": rid, "kind": rec.kind, "title": c["title"], "created_at": created}


def row_to_record(row: sqlite3.Row) -> dict[str, Any]:
    """Shape matches the UI's AppState.records entries ({id, created_at, input, results}) plus extras."""
    body = json.loads(row["payload_json"])
    return {
        "id": row["id"], "kind": row["kind"], "title": row["title"], "created_at": row["created_at"],
        "input": body.get("input"), "results": body.get("results", []), "red_flags": body.get("red_flags", []),
        "payload": body.get("payload"), "top_condition": row["top_condition"], "top_confidence": row["top_confidence"],
    }


def _like(s: str) -> str:
    return "%" + s.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


# ============================================================================================
# 8. DOCUMENT PARSING (/api/parse-report)
# ============================================================================================
_NUM = r"(\d{1,4}(?:[.,]\d{1,2})?)"
_UNIT_RE = re.compile(r"(mg\s*/\s*dl|mmol\s*/\s*l|g\s*/\s*dl|g\s*/\s*l)", re.I)


def _lab_candidates(label_rx: str, text: str, gap: int = 25):
    """Yield (match, number, unit, gap_text) for `LABEL <non-digits> NUMBER [unit]`, skipping reference ranges."""
    rx = re.compile(rf"(?P<label>{label_rx})(?P<gap>[^\d\n]{{0,{gap}}}?)(?P<num>{_NUM})(?P<unit>\s*(?:mg\s*/\s*dl|mmol\s*/\s*l|g\s*/\s*dl|g\s*/\s*l)?)", re.I)
    for m in rx.finditer(text):
        if re.match(r"\s*[-–]\s*\d", text[m.end("num"):m.end("num") + 6]):
            continue  # "70-99" style reference range, not a result
        unit_m = _UNIT_RE.search(m.group("unit") or "") or _UNIT_RE.search(m.group("gap") or "")
        unit = re.sub(r"\s+", "", unit_m.group(1).lower()) if unit_m else None
        yield m, float(m.group("num").replace(",", ".")), unit, m.group("gap")


def _bounded(key: str, v: float) -> bool:
    return 0 < v <= LAB_LIMITS[key]


def extract_labs(text: str) -> tuple[dict[str, float | None], list[dict[str, Any]], list[str]]:
    """Regex lab extraction with unit conversion. Returns (labs, extraction details, warnings).
    Order of preference for glucose: explicitly *fasting* values first; a generic 'glucose' is used
    only if it isn't flagged random/post-prandial, and is reported as lower confidence."""
    labs: dict[str, float | None] = {k: None for k in LAB_LIMITS}
    details: list[dict[str, Any]] = []
    warnings: list[str] = []

    def record(key, value, unit, raw, confidence="high", note=None):
        labs[key] = round(value, 1)
        details.append({"key": key, "value": labs[key], "source_unit": unit, "matched_text": raw.strip()[:80],
                        "confidence": confidence, **({"note": note} if note else {})})

    # --- glucose (target unit mg/dL) ---
    fasting_rx = r"fasting\s+(?:plasma\s+|blood\s+)?(?:glucose|sugar)|\bf[bp]g\b|\bfbs\b|glucose\s*[,(]?\s*fasting"
    for rx, conf in ((fasting_rx, "high"), (r"\bglucose\b|blood\s+sugar", "low")):
        done = False
        for m, v, unit, _ in _lab_candidates(rx, text):
            before = text[max(0, m.start() - 25):m.start()].lower()
            if conf == "low" and re.search(r"random|post[\s-]?prandial|\bpp\b|ogtt|2[\s-]?h(?:ou)?r|after\s+meal", before + m.group("gap").lower()):
                continue
            note = None
            if unit == "mmol/l" or (unit is None and v < 30):
                v, note = v * 18.016, "converted from mmol/L" + ("" if unit else " (unit assumed)")
            if _bounded("fasting_glucose", v):
                record("fasting_glucose", v, unit, m.group(0), conf,
                       note or ("not labelled 'fasting' in the report; confirm it was a fasting sample" if conf == "low" else None))
                done = True
                break
        if done:
            break

    # --- hemoglobin (target g/dL); skip HbA1c / glycated hemoglobin ---
    hb_rx = r"\bhaemoglobin\b|\bhemoglobin\b(?!\s*a1c)|\bhgb\b|\bhb\b(?!\s*a1c)"
    for m, v, unit, _ in _lab_candidates(hb_rx, text):
        if re.search(r"glyc|a1c", text[max(0, m.start() - 14):m.start()].lower()):
            continue
        note = None
        if unit == "g/l" or (unit is None and v > 30):
            v, note = v / 10, "converted from g/L"
        if _bounded("hemoglobin", v) and v <= 25:
            record("hemoglobin", v, unit, m.group(0), note=note)
            break

    # --- LDL (target mg/dL) ---
    ldl_rx = r"\bldl(?:[\s\-]*(?:c|cholesterol|chol))*\b|low[\s\-]density\s+lipoprotein(?:\s+cholesterol)?"
    for m, v, unit, gap in _lab_candidates(ldl_rx, text):
        if re.search(r"ratio|hdl", gap, re.I):
            continue
        note = None
        if unit == "mmol/l" or (unit is None and v < 15):
            v, note = v * 38.67, "converted from mmol/L" + ("" if unit else " (unit assumed)")
        if _bounded("ldl", v):
            record("ldl", v, unit, m.group(0), note=note)
            break

    # --- blood pressure "120/80" ---
    bp_rx = re.compile(r"(?:\bbp\b|blood\s+pressure)[^\d\n]{0,20}(\d{2,3})\s*/\s*(\d{2,3})", re.I)
    for m in bp_rx.finditer(text):
        sys_, dia = float(m.group(1)), float(m.group(2))
        if 60 <= sys_ <= 300 and 30 <= dia <= 200 and sys_ > dia:
            record("systolic_bp", sys_, "mmHg", m.group(0))
            record("diastolic_bp", dia, "mmHg", m.group(0))
            break

    if not any(v is not None for v in labs.values()):
        warnings.append("No supported lab values were recognised. Enter them manually.")
    return labs, details, warnings


def _read_capped(upload: UploadFile, limit: int) -> bytes:
    """Read an upload in chunks; abort once it exceeds `limit` (never trust Content-Length)."""
    buf, total = io.BytesIO(), 0
    while chunk := upload.file.read(1024 * 1024):
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail=f"'{upload.filename}' exceeds the {limit // (1024 * 1024)} MB limit.")
        buf.write(chunk)
    return buf.getvalue()


def _ocr_available() -> bool:
    try:
        import pytesseract  # noqa: F401
        return shutil.which("tesseract") is not None
    except ImportError:
        return False


def _ocr_image(img) -> str:
    import pytesseract
    return pytesseract.image_to_string(img.convert("L"))


def _pdf_text(data: bytes) -> tuple[str, list[str]]:
    notes: list[str] = []
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = pdf.pages[:MAX_PDF_PAGES]
            if len(pdf.pages) > MAX_PDF_PAGES:
                notes.append(f"Only the first {MAX_PDF_PAGES} pages were read.")
            text = "\n".join((p.extract_text() or "") for p in pages)
            if not text.strip() and _ocr_available():  # scanned PDF -> OCR first few pages
                notes.append("No text layer found; used OCR on the first pages.")
                text = "\n".join(_ocr_image(p.to_image(resolution=200).original) for p in pages[:5])
    except ImportError:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((p.extract_text() or "") for p in reader.pages[:MAX_PDF_PAGES])
    return text, notes


def _docx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        info = z.getinfo("word/document.xml")
        if info.file_size > 20 * 1024 * 1024:  # zip-bomb guard
            raise ValueError("document.xml too large")
        xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"</w:tc>|<w:tab/>", " \t ", xml)  # keep table cells apart: "Glucose | 132"
    return html_lib.unescape(re.sub(r"<[^>]+>", "", xml))


def parse_one_file(name: str, data: bytes) -> dict[str, Any]:
    """Extract text from one uploaded file. Never raises: returns a per-file status instead."""
    ext = Path(name).suffix.lower()
    out: dict[str, Any] = {"filename": Path(name).name[:120], "status": "parsed", "notes": [], "text": ""}
    try:
        if ext == ".pdf":
            if not data.startswith(b"%PDF"):
                raise ValueError("not a valid PDF")
            out["text"], out["notes"] = _pdf_text(data)
        elif ext in (".txt", ".csv"):
            out["text"] = data.decode("utf-8-sig", errors="ignore")
        elif ext == ".docx":
            out["text"] = _docx_text(data)
        elif ext in (".png", ".jpg", ".jpeg", ".webp"):
            from PIL import Image
            Image.open(io.BytesIO(data)).verify()
            if _ocr_available():
                out["text"] = _ocr_image(Image.open(io.BytesIO(data)))
                out["notes"].append("Text read with OCR; double-check the values.")
            else:
                out["status"] = "unsupported"
                out["notes"].append("OCR is not installed on the server (pip install pytesseract + tesseract binary).")
        elif ext == ".doc":
            out["status"] = "unsupported"
            out["notes"].append("Legacy .doc is not supported. Save as .docx or PDF.")
        else:
            out["status"] = "unsupported"
            out["notes"].append(f"Unsupported file type '{ext or 'unknown'}'.")
    except Exception as exc:  # corrupt/encrypted file etc.
        log.info("Report parse failed (%s): %s", ext, type(exc).__name__)
        out["status"], out["notes"] = "error", [*out["notes"], "This file could not be read (corrupt or password-protected?)."]
    out["text"] = out["text"][:300_000]
    if out["status"] == "parsed" and not out["text"].strip():
        out["status"] = "no_text"
        out["notes"].append("No readable text found (scanned image?).")
    return out


# ============================================================================================
# 9. IMAGE ANALYSIS (/api/image-predict)
# ============================================================================================
# TWO MODES
#  * "model": real ONNX classifiers placed in $DIAGNOVA_MODEL_DIR as <category>.onnx + <category>.labels.json
#             (a JSON list of label strings in output order). Requires `pip install onnxruntime numpy`.
#  * "demo" : deterministic PLACEHOLDER scores derived from the image bytes. They have NO clinical
#             meaning and exist only so the UI can be built/tested. Enabled by default in development,
#             disabled in production unless DIAGNOVA_ALLOW_DEMO_IMAGE=true.
IMAGE_CATEGORIES: dict[str, dict[str, Any]] = {
    "skin": {"specialist": "Dermatologist", "labels": [
        "Eczema / dermatitis pattern", "Psoriasis-like pattern", "Urticaria / allergic reaction pattern",
        "Benign-appearing lesion pattern", "Features warranting dermatology review"]},
    "eye": {"specialist": "Ophthalmologist", "labels": [
        "Retinal changes that may relate to diabetic eye disease", "Normal-appearing retinal vasculature",
        "Surface inflammation (conjunctivitis-like) pattern"]},
    "oncology": {"specialist": "Oncologist / Radiologist", "labels": [
        "Findings that warrant further imaging review", "Benign-appearing tissue pattern"]},
}
MODEL_DIR = Path(os.getenv("DIAGNOVA_MODEL_DIR", str(APP_DIR / "models")))
ALLOW_DEMO_IMAGE = os.getenv("DIAGNOVA_ALLOW_DEMO_IMAGE", "false" if IS_PROD else "true").lower() == "true"
_onnx_sessions: dict[str, Any] = {}


def _demo_scores(img, n: int) -> list[float]:
    """Deterministic softmax over hash-derived logits (stable across re-encodes of the same pixels)."""
    digest = hashlib.sha256(img.convert("L").resize((32, 32)).tobytes()).digest()
    logits = [digest[i] / 255 * 3.0 for i in range(n)]
    exps = [math.exp(x) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _onnx_scores(category: str, img) -> list[float]:
    """UNTESTED reference integration: generic 224x224 RGB classifier with ImageNet normalisation.
    Adapt pre/post-processing to however your model was trained."""
    import numpy as np
    import onnxruntime as ort
    if category not in _onnx_sessions:
        _onnx_sessions[category] = ort.InferenceSession(str(MODEL_DIR / f"{category}.onnx"), providers=["CPUExecutionProvider"])
    sess = _onnx_sessions[category]
    arr = np.asarray(img.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    logits = sess.run(None, {sess.get_inputs()[0].name: arr.transpose(2, 0, 1)[None]})[0][0]
    e = np.exp(logits - logits.max())
    return (e / e.sum()).tolist()


def predict_image(category: str, data: bytes, filename: str) -> dict[str, Any]:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 50_000_000  # decompression-bomb guard
    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()                       # integrity check (also rejects non-images)
        img = Image.open(io.BytesIO(data))   # verify() invalidates the handle, so reopen
        img.load()
    except Exception:
        raise HTTPException(status_code=415, detail="The file is not a valid JPG, PNG or WebP image.")
    if img.format not in {"JPEG", "PNG", "WEBP"}:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or WebP image.")
    if min(img.size) < 64:
        raise HTTPException(status_code=422, detail="Image is too small (minimum 64×64 pixels).")

    cfg = IMAGE_CATEGORIES[category]
    labels: list[str] = cfg["labels"]
    model_file, labels_file = MODEL_DIR / f"{category}.onnx", MODEL_DIR / f"{category}.labels.json"
    if model_file.exists() and labels_file.exists():
        mode = "model"
        try:
            labels = json.loads(labels_file.read_text())
            scores = _onnx_scores(category, img)
            if len(scores) != len(labels):
                raise ValueError("label/output size mismatch")
        except Exception as exc:  # a configured-but-broken model must NOT silently fall back to fake scores
            log.error("Image model for '%s' failed: %s", category, exc)
            raise HTTPException(status_code=503, detail="The image model is unavailable right now.")
        warning = None
    elif ALLOW_DEMO_IMAGE:
        mode, scores = "demo", _demo_scores(img, len(labels))
        warning = ("DEMO MODE: these scores are generated from the file's pixels by a placeholder and carry NO "
                   "clinical meaning. Do not use them to make health decisions.")
    else:
        raise HTTPException(status_code=503, detail="Image analysis is not configured on this server.")

    ranked = sorted(({"label": l, "score": round(float(s), 4)} for l, s in zip(labels, scores)), key=lambda r: r["score"], reverse=True)
    return {
        "created_at": iso_now(), "category": category, "filename": Path(filename or "image").name[:100],
        "mode": mode, "warning": warning, "is_diagnostic": False,
        "image_info": {"width": img.width, "height": img.height, "format": img.format},
        "results": ranked, "top_label": ranked[0]["label"], "suggested_specialist": cfg["specialist"],
        "reasoning": "Exploratory pattern scores from an image model. They indicate similarity to learned patterns, not a diagnosis. "
                     "Images are analysed in memory and are not stored.",
        "disclaimer": DISCLAIMER,
    }


# ============================================================================================
# 10. PUBMED EVIDENCE (NCBI E-utilities esearch + esummary)
# ============================================================================================
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_pubmed_cache: OrderedDict[tuple[str, int], tuple[float, dict]] = OrderedDict()
_pubmed_lock = threading.Lock()
PUBMED_TTL, PUBMED_CACHE_MAX = 6 * 3600, 256


class PubMedError(Exception):
    pass


def _eutils(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    params = {**params, "retmode": "json", "tool": "diagnova"}
    if NCBI_EMAIL:
        params["email"] = NCBI_EMAIL
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    try:
        r = requests.get(f"{EUTILS}/{endpoint}.fcgi", params=params, timeout=(3, 7), headers={"User-Agent": "DiagNova/1.0"})
        if r.status_code == 429:
            raise PubMedError("rate limited by NCBI")
        r.raise_for_status()
        return r.json()
    except PubMedError:
        raise
    except (requests.RequestException, ValueError) as exc:  # timeouts, DNS, 5xx, bad JSON
        raise PubMedError(type(exc).__name__)


def fetch_pubmed(query: str, retmax: int) -> dict[str, Any]:
    ids = _eutils("esearch", {"db": "pubmed", "term": query, "retmax": retmax, "sort": "relevance"}) \
        .get("esearchresult", {}).get("idlist", [])
    if not ids:
        return {"articles": []}
    result = _eutils("esummary", {"db": "pubmed", "id": ",".join(ids)}).get("result", {})
    articles = []
    for pmid in ids:
        item = result.get(pmid)
        if not isinstance(item, dict) or not item.get("title"):
            continue  # skip anything NCBI didn't describe; never invent citations
        authors = [a.get("name", "") for a in item.get("authors", [])[:3] if isinstance(a, dict)]
        articles.append({
            "pmid": pmid, "title": str(item["title"]).strip(), "authors": [a for a in authors if a],
            "journal": item.get("fulljournalname") or item.get("source") or "",
            "year": (item.get("pubdate") or "")[:4], "pubtypes": item.get("pubtype", [])[:3],
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return {"articles": articles}


# ============================================================================================
# 11. APPLICATION + MIDDLEWARE
# ============================================================================================
init_db()  # idempotent; runs at import so it also works under TestClient / multiple workers

app = FastAPI(
    title="DiagNova's Prediction API", version="1.0.0",
    description="Educational screening-support API: rule engine, report parsing, image screening, PubMed evidence, records.",
)

# CORS: auth uses an Authorization bearer header (no cookies), so credentials stay OFF — this means a
# wildcard-free explicit allow-list is sufficient and CSRF via cookies is not a concern.
app.add_middleware(
    CORSMiddleware, allow_origins=_cors_origins(), allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"], allow_headers=["Authorization", "Content-Type"],
    max_age=600,
)


@app.middleware("http")
async def no_store_for_api(request: Request, call_next):
    """Health data must not be cached by browsers or intermediaries."""
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
    return response


# ============================================================================================
# 12. ROUTES
# ============================================================================================
FRONTEND_CANDIDATES = [APP_DIR / "diagnova_home_final_fix.html", APP_DIR / "static" / "index.html", APP_DIR / "index.html"]


@app.get("/", include_in_schema=False)
def home():
    for p in FRONTEND_CANDIDATES:
        if p.exists():
            return FileResponse(p)
    return JSONResponse({"name": "DiagNova's Prediction API", "docs": "/docs"})


@app.get("/healthz")
def healthz():
    return {"status": "ok", "env": ENV, "ocr": _ocr_available(), "image_mode": "demo" if ALLOW_DEMO_IMAGE else "disabled/model"}


@app.get("/api/symptoms")
def list_symptoms():
    """Symptom catalogue (same ids/labels the UI renders as chips)."""
    return {"symptoms": [{"id": i, "name": n} for i, n in SYMPTOMS]}


# ---- auth -------------------------------------------------------------------------------------
def _auth_payload(uid: int, email: str, name: str) -> dict[str, Any]:
    token = create_token(uid)
    # `token` (old contract) and `access_token`/`token_type` (OAuth2 convention) are both returned.
    return {"token": token, "access_token": token, "token_type": "bearer", "expires_in": TOKEN_TTL_SECONDS,
            "user": {"id": uid, "email": email, "display_name": name}}


@app.post("/api/auth/register", status_code=201)
def register(data: RegisterIn, _rl=Depends(rate_limit("register", 10, 3600))):
    try:
        with db() as con:
            cur = con.execute("INSERT INTO users(email, display_name, password_hash, created_at) VALUES(?,?,?,?)",
                              (data.email, data.display_name, hash_password(data.password), iso_now()))
            uid = cur.lastrowid
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    return _auth_payload(uid, data.email, data.display_name)


@app.post("/api/auth/login")
def login(data: LoginIn, request: Request):
    email = data.email.strip().lower()
    key = f"{client_ip(request)}|{email}"
    throttle.check(key)
    with db() as con:
        row = con.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        # Always run a hash verification so response time doesn't reveal whether the email exists.
        ok, rehash = verify_password(data.password, row["password_hash"] if row else _dummy_hash())
        if not row or not ok:
            throttle.fail(key)
            raise HTTPException(status_code=401, detail="Email or password is incorrect.")
        if rehash:  # silently upgrade legacy hashes
            con.execute("UPDATE users SET password_hash=? WHERE id=?", (hash_password(data.password), row["id"]))
    throttle.clear(key)
    return _auth_payload(row["id"], row["email"], row["display_name"])


@app.get("/api/auth/me")
def me(user: CurrentUser):
    return {"user": user}


@app.post("/api/auth/delete-account")
def delete_account(data: DeleteAccountIn, user: CurrentUser):
    """Permanently delete the account and (via ON DELETE CASCADE) all profile, records and requests."""
    with db() as con:
        row = con.execute("SELECT password_hash FROM users WHERE id=?", (user["id"],)).fetchone()
        if not row or not verify_password(data.password, row["password_hash"])[0]:
            raise HTTPException(status_code=401, detail="Password is incorrect.")
        con.execute("DELETE FROM users WHERE id=?", (user["id"],))
    return {"deleted": True}


# ---- screening --------------------------------------------------------------------------------
@app.post("/api/screen")
def screen(data: ScreenIn, user: OptionalUser, _rl=Depends(rate_limit("screen", 40, 60))):
    """Evaluate symptoms + optional labs + optional description against the rule base.
    Works anonymously. With `save: true` AND a valid bearer token, the screening is also stored and
    its id returned (mirrors the UI's 'Save this screening' checkbox)."""
    labs = data.labs.model_dump()
    out = build_screening(data.age, data.sex, data.symptoms, labs, data.description)
    out["saved"], out["record_id"] = False, None
    if data.save:
        if not user:
            out["save_error"] = "Sign in to save screenings."
        else:
            body = dict(out["input"], description=data.description) if data.description else out["input"]
            rec = RecordIn(kind="screening", title=data.title, input=body, results=out["results"], red_flags=out["red_flags"])
            with db() as con:
                saved = upsert_record(con, user["id"], rec)
            out["saved"], out["record_id"] = True, saved["id"]
    return out


# ---- documents --------------------------------------------------------------------------------
@app.post("/api/parse-report")
def parse_report(user: CurrentUser, files: list[UploadFile] = File(default=[]), file: UploadFile | None = File(default=None),
                 _rl=Depends(rate_limit("upload", 20, 60))):
    """Upload up to 5 lab reports (PDF / TXT / CSV / DOCX / image-with-OCR, <=10 MB each) and get back
    lab values in the same keys /api/screen accepts. Accepts `files` (multi) or legacy `file` (single)."""
    uploads = [u for u in [*(files or []), file] if u is not None and u.filename]
    if not uploads:
        raise HTTPException(status_code=422, detail="Attach at least one file.")
    if len(uploads) > MAX_DOC_FILES:
        raise HTTPException(status_code=413, detail=f"Please upload no more than {MAX_DOC_FILES} files.")

    combined: dict[str, float | None] = {k: None for k in LAB_LIMITS}
    extractions: list[dict[str, Any]] = []
    file_reports: list[dict[str, Any]] = []
    warnings: list[str] = []
    for up in uploads:
        info = parse_one_file(up.filename, _read_capped(up, MAX_DOC_BYTES))
        labs, details, w = extract_labs(info["text"]) if info["text"] else ({}, [], [])
        for d in details:  # first file to supply a value wins; later conflicts are surfaced, not silently merged
            if combined[d["key"]] is None:
                combined[d["key"]] = d["value"]
                extractions.append({**d, "filename": info["filename"]})
            elif combined[d["key"]] != d["value"]:
                warnings.append(f"{d['key']} differs between files ({combined[d['key']]} vs {d['value']}); kept the first.")
        file_reports.append({"filename": info["filename"], "status": info["status"], "notes": info["notes"] + w,
                             "text_preview": info["text"].strip()[:500], "labs_found": [d["key"] for d in details]})
    if not any(v is not None for v in combined.values()):
        warnings.append("No supported lab values were recognised in the uploaded files.")
    return {
        "created_at": iso_now(),
        "parsed_labs": combined,          # drop straight into ScreenIn.labs
        "extractions": extractions,
        "files": file_reports,
        "warnings": warnings,
        "reasoning": "Values were located by pattern matching and unit-converted where needed. Always verify them against the original report.",
    }


@app.post("/api/image-predict")
def image_predict(user: CurrentUser, category: Annotated[str, Query(description="skin | eye | oncology")],
                  file: UploadFile = File(...), _rl=Depends(rate_limit("image", 20, 60))):
    category = category.strip().lower()
    if category not in IMAGE_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category. Choose one of: {', '.join(IMAGE_CATEGORIES)}.")
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or WebP image.")
    return predict_image(category, _read_capped(file, MAX_IMAGE_BYTES), file.filename or "image")


# ---- PubMed -----------------------------------------------------------------------------------
@app.get("/api/evidence")
def evidence(condition: Annotated[str, Query(max_length=200, description="A condition id (e.g. 't2dm') or free-text query")] = "",
             retmax: Annotated[int, Query(ge=1, le=10)] = 5, _rl=Depends(rate_limit("evidence", 30, 60))):
    """Related PubMed reading. Always HTTP 200: if NCBI is down the body says so (status='unavailable')
    with a search_url fallback, and no citations are ever fabricated."""
    cond = CONDITION_BY_ID.get(condition.strip().lower())
    query = cond["pubmedQuery"] if cond else re.sub(r"[\x00-\x1f]+", " ", condition).strip() or "general health screening"
    base = {"query": query, "source": "PubMed (NCBI E-utilities)", "search_url": "https://pubmed.ncbi.nlm.nih.gov/?term=" + quote_plus(query)}
    key = (query.lower(), retmax)
    with _pubmed_lock:
        hit = _pubmed_cache.get(key)
        if hit and hit[0] > time.time():
            _pubmed_cache.move_to_end(key)
            return {**base, **hit[1], "cached": True}
    try:
        data = fetch_pubmed(query, retmax)
    except PubMedError as exc:
        log.warning("PubMed unavailable: %s", exc)
        return {**base, "status": "unavailable", "articles": [], "cached": False,
                "note": "PubMed could not be reached right now. No citations were fabricated; use the search link instead."}
    data.update(status="ok" if data["articles"] else "empty",
                note="Related reading only; not proof of a diagnosis." if data["articles"] else "No matching articles were returned.")
    with _pubmed_lock:
        _pubmed_cache[key] = (time.time() + PUBMED_TTL, data)
        while len(_pubmed_cache) > PUBMED_CACHE_MAX:
            _pubmed_cache.popitem(last=False)
    return {**base, **data, "cached": False}


# ---- records ----------------------------------------------------------------------------------
@app.post("/api/records", status_code=201)
def save_record(data: RecordIn, user: CurrentUser):
    """Create — or, if `id` already exists for this user, replace — a record."""
    with db() as con:
        return upsert_record(con, user["id"], data)


@app.post("/api/records/bulk")
def bulk_save(data: BulkRecordsIn, user: CurrentUser):
    """Batch upsert (equivalent of the UI's `syncRecordsToCloud()`)."""
    with db() as con:
        ids = [upsert_record(con, user["id"], r)["id"] for r in data.records]
    return {"saved": len(ids), "ids": ids}


@app.get("/api/records")
def list_records(
    user: CurrentUser,
    q: Annotated[str | None, Query(max_length=100, description="Search condition names, categories, title")] = None,
    kind: Literal["screening", "report", "image"] | None = None,
    category: Annotated[str | None, Query(max_length=100)] = None,
    min_confidence: Annotated[int | None, Query(ge=0, le=100)] = None,
    has_red_flags: bool | None = None,
    date_from: date | None = None, date_to: date | None = None,
    order: Literal["asc", "desc"] = "desc",
    limit: Annotated[int, Query(ge=1, le=500)] = 100, offset: Annotated[int, Query(ge=0)] = 0,
):
    where, args = ["user_id=?"], [user["id"]]
    if q:
        where.append("search_text LIKE ? ESCAPE '\\'"); args.append(_like(q.strip()))
    if kind:
        where.append("kind=?"); args.append(kind)
    if category:
        where.append("categories LIKE ? ESCAPE '\\'"); args.append("%|" + category.strip().lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "|%")
    if min_confidence is not None:
        where.append("top_confidence >= ?"); args.append(min_confidence)
    if has_red_flags is not None:
        where.append("has_red_flags=?"); args.append(int(has_red_flags))
    if date_from:
        where.append("created_at >= ?"); args.append(date_from.isoformat())
    if date_to:
        where.append("created_at < ?"); args.append((date_to + timedelta(days=1)).isoformat())
    clause = " AND ".join(where)
    with db() as con:
        total = con.execute(f"SELECT COUNT(*) FROM health_records WHERE {clause}", args).fetchone()[0]
        rows = con.execute(f"SELECT * FROM health_records WHERE {clause} ORDER BY created_at {order.upper()} LIMIT ? OFFSET ?",
                           [*args, limit, offset]).fetchall()
    return {"records": [row_to_record(r) for r in rows], "total": total, "limit": limit, "offset": offset}


@app.get("/api/records/export")  # declared before /{record_id} so "export" isn't parsed as an id
def export_records(user: CurrentUser):
    """Full JSON download (the server-side counterpart of the UI's 'Export records' button)."""
    with db() as con:
        rows = con.execute("SELECT * FROM health_records WHERE user_id=? ORDER BY created_at DESC", (user["id"],)).fetchall()
    return JSONResponse([row_to_record(r) for r in rows],
                        headers={"Content-Disposition": f'attachment; filename="diagnova-records-{date.today().isoformat()}.json"'})


@app.get("/api/records/{record_id}")
def get_record(record_id: str, user: CurrentUser):
    with db() as con:
        row = con.execute("SELECT * FROM health_records WHERE user_id=? AND id=?", (user["id"], record_id)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Record not found.")
    return row_to_record(row)


@app.delete("/api/records/{record_id}")
def delete_record(record_id: str, user: CurrentUser):
    with db() as con:  # user_id in the WHERE clause = ownership check
        cur = con.execute("DELETE FROM health_records WHERE user_id=? AND id=?", (user["id"], record_id))
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Record not found.")
    return {"deleted": True, "id": record_id}


@app.delete("/api/records")
def clear_records(user: CurrentUser, confirm: bool = False):
    """Delete ALL of the caller's records (UI: 'Clear all'). Requires ?confirm=true."""
    if not confirm:
        raise HTTPException(status_code=400, detail="Pass confirm=true to delete all records.")
    with db() as con:
        cur = con.execute("DELETE FROM health_records WHERE user_id=?", (user["id"],))
    return {"deleted": cur.rowcount}


# ---- profile & care requests ------------------------------------------------------------------
@app.get("/api/profile")
def get_profile(user: CurrentUser):
    with db() as con:
        row = con.execute("SELECT data_json, updated_at FROM profiles WHERE user_id=?", (user["id"],)).fetchone()
    if not row:  # same default the UI uses on first sign-in
        return {"profile": {"fullName": user["display_name"], "email": user["email"]}, "updated_at": None}
    return {"profile": json.loads(row["data_json"]), "updated_at": row["updated_at"]}


@app.put("/api/profile")
def put_profile(data: ProfileIn, user: CurrentUser):
    profile = data.model_dump()
    profile["email"] = user["email"]  # the UI treats email as read-only (taken from the account)
    now = iso_now()
    with db() as con:
        con.execute("""INSERT INTO profiles(user_id,data_json,updated_at) VALUES(?,?,?)
                       ON CONFLICT(user_id) DO UPDATE SET data_json=excluded.data_json, updated_at=excluded.updated_at""",
                    (user["id"], json.dumps(profile, ensure_ascii=False), now))
        con.execute("UPDATE users SET display_name=? WHERE id=?", (profile["fullName"].strip(), user["id"]))
    return {"profile": profile, "updated_at": now}


@app.post("/api/care-requests", status_code=201)
def create_care_request(data: CareRequestIn, user: CurrentUser, _rl=Depends(rate_limit("care", 10, 3600))):
    body = data.model_dump(mode="json")
    now = iso_now()
    with db() as con:
        cur = con.execute("INSERT INTO care_requests(user_id,created_at,data_json) VALUES(?,?,?)",
                          (user["id"], now, json.dumps(body, ensure_ascii=False)))
    return {"id": cur.lastrowid, "created_at": now, **body, "status": "Stored — not forwarded to any provider"}


@app.get("/api/care-requests")
def list_care_requests(user: CurrentUser):
    with db() as con:
        rows = con.execute("SELECT id,created_at,data_json FROM care_requests WHERE user_id=? ORDER BY id DESC", (user["id"],)).fetchall()
    return {"requests": [{"id": r["id"], "created_at": r["created_at"], **json.loads(r["data_json"])} for r in rows]}


@app.delete("/api/care-requests/{request_id}")
def delete_care_request(request_id: int, user: CurrentUser):
    with db() as con:
        cur = con.execute("DELETE FROM care_requests WHERE id=? AND user_id=?", (request_id, user["id"]))
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Request not found.")
    return {"deleted": True, "id": request_id}


# ============================================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
