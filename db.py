"""SQLite persistence layer for SatQuery AI.
Adds three genuinely new capabilities on top of the original stateless Q&A tool:
1. Query History   - every analysis is saved and can be revisited.
2. Watchlist        - a saved 'baseline' image for a location, re-checked later.
3. Report generation reads from this store (see report.py).
"""
import sqlite3, json, time, os

DB_PATH = os.getenv("SATQUERY_DB", "satquery.db")


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init():
    c = _conn()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL UNIQUE,
        salt TEXT NOT NULL,
        pass_hash TEXT NOT NULL,
        created_ts REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS sessions (
        token TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        created_ts REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        ts REAL NOT NULL,
        prompt TEXT NOT NULL,
        intent TEXT,
        answer TEXT,
        confidence TEXT,
        thumb_b64 TEXT,
        observations TEXT
    );
    CREATE TABLE IF NOT EXISTS watchlist (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        name TEXT NOT NULL,
        created_ts REAL NOT NULL,
        baseline_b64 TEXT NOT NULL,
        baseline_meas TEXT NOT NULL,
        last_check_ts REAL,
        last_changed_pct REAL,
        last_alert INTEGER
    );
    """)
    # best-effort migration for pre-existing DBs created before user_id existed
    for tbl in ("history", "watchlist"):
        try:
            c.execute(f"ALTER TABLE {tbl} ADD COLUMN user_id INTEGER")
        except sqlite3.OperationalError:
            pass
    c.commit(); c.close()


# ---------------- Users & Sessions ----------------

def create_user(email, salt, pass_hash):
    c = _conn()
    cur = c.execute("INSERT INTO users (email, salt, pass_hash, created_ts) VALUES (?,?,?,?)",
                     (email.lower().strip(), salt, pass_hash, time.time()))
    c.commit(); uid = cur.lastrowid; c.close()
    return uid


def get_user_by_email(email):
    c = _conn()
    r = c.execute("SELECT * FROM users WHERE email=?", (email.lower().strip(),)).fetchone()
    c.close()
    return dict(r) if r else None


def create_session(user_id, token):
    c = _conn()
    c.execute("INSERT INTO sessions (token, user_id, created_ts) VALUES (?,?,?)", (token, user_id, time.time()))
    c.commit(); c.close()


def get_user_id_by_token(token):
    c = _conn()
    r = c.execute("SELECT user_id FROM sessions WHERE token=?", (token,)).fetchone()
    c.close()
    return r["user_id"] if r else None


def delete_session(token):
    c = _conn()
    c.execute("DELETE FROM sessions WHERE token=?", (token,))
    c.commit(); c.close()


# ---------------- History ----------------

def add_history(user_id, prompt, intent, answer, confidence, thumb_b64, observations):
    c = _conn()
    cur = c.execute(
        "INSERT INTO history (user_id, ts, prompt, intent, answer, confidence, thumb_b64, observations) VALUES (?,?,?,?,?,?,?,?)",
        (user_id, time.time(), prompt, intent, answer, confidence, thumb_b64, json.dumps(observations or [])))
    c.commit(); rid = cur.lastrowid; c.close()
    return rid


def list_history(user_id, limit=30):
    c = _conn()
    rows = c.execute("""SELECT id, ts, prompt, intent, confidence, thumb_b64 FROM history
                         WHERE user_id=? ORDER BY id DESC LIMIT ?""", (user_id, limit)).fetchall()
    c.close()
    return [dict(r) for r in rows]


def get_history(hid, user_id=None, public=False):
    c = _conn()
    r = c.execute("SELECT * FROM history WHERE id=?", (hid,)).fetchone()
    c.close()
    if not r: return None
    d = dict(r); d["observations"] = json.loads(d.get("observations") or "[]")
    # ownership check: only the owner can fetch/report a logged-in history entry;
    # entries saved without a login (user_id NULL) stay publicly viewable
    # public=True is used only for the shareable PDF report link ("anyone with the link can view",
    # like a Google Doc link) so a QR code scanned on another device works without logging in
    if not public and d.get("user_id") is not None and user_id != d.get("user_id"):
        return None
    return d


# ---------------- Watchlist ----------------

def add_watchlist(user_id, name, baseline_b64, baseline_meas):
    c = _conn()
    cur = c.execute(
        "INSERT INTO watchlist (user_id, name, created_ts, baseline_b64, baseline_meas) VALUES (?,?,?,?,?)",
        (user_id, name, time.time(), baseline_b64, json.dumps(baseline_meas)))
    c.commit(); wid = cur.lastrowid; c.close()
    return wid


def list_watchlist(user_id):
    c = _conn()
    rows = c.execute("""SELECT id, name, created_ts, last_check_ts, last_changed_pct, last_alert
                         FROM watchlist WHERE user_id=? ORDER BY id DESC""", (user_id,)).fetchall()
    c.close()
    return [dict(r) for r in rows]


def get_watchlist(wid, user_id):
    c = _conn()
    r = c.execute("SELECT * FROM watchlist WHERE id=? AND user_id=?", (wid, user_id)).fetchone()
    c.close()
    if not r: return None
    d = dict(r); d["baseline_meas"] = json.loads(d["baseline_meas"])
    return d


def update_watchlist_check(wid, changed_pct, alert):
    c = _conn()
    c.execute("UPDATE watchlist SET last_check_ts=?, last_changed_pct=?, last_alert=? WHERE id=?",
              (time.time(), changed_pct, int(alert), wid))
    c.commit(); c.close()
