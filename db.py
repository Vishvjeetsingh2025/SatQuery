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
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
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
        name TEXT NOT NULL,
        created_ts REAL NOT NULL,
        baseline_b64 TEXT NOT NULL,
        baseline_meas TEXT NOT NULL,
        last_check_ts REAL,
        last_changed_pct REAL,
        last_alert INTEGER
    );
    """)
    c.commit(); c.close()


# ---------------- History ----------------

def add_history(prompt, intent, answer, confidence, thumb_b64, observations):
    c = _conn()
    cur = c.execute(
        "INSERT INTO history (ts, prompt, intent, answer, confidence, thumb_b64, observations) VALUES (?,?,?,?,?,?,?)",
        (time.time(), prompt, intent, answer, confidence, thumb_b64, json.dumps(observations or [])))
    c.commit(); rid = cur.lastrowid; c.close()
    return rid


def list_history(limit=30):
    c = _conn()
    rows = c.execute("SELECT id, ts, prompt, intent, confidence, thumb_b64 FROM history ORDER BY id DESC LIMIT ?",
                      (limit,)).fetchall()
    c.close()
    return [dict(r) for r in rows]


def get_history(hid):
    c = _conn()
    r = c.execute("SELECT * FROM history WHERE id=?", (hid,)).fetchone()
    c.close()
    if not r: return None
    d = dict(r); d["observations"] = json.loads(d.get("observations") or "[]")
    return d


# ---------------- Watchlist ----------------

def add_watchlist(name, baseline_b64, baseline_meas):
    c = _conn()
    cur = c.execute(
        "INSERT INTO watchlist (name, created_ts, baseline_b64, baseline_meas) VALUES (?,?,?,?)",
        (name, time.time(), baseline_b64, json.dumps(baseline_meas)))
    c.commit(); wid = cur.lastrowid; c.close()
    return wid


def list_watchlist():
    c = _conn()
    rows = c.execute("""SELECT id, name, created_ts, last_check_ts, last_changed_pct, last_alert
                         FROM watchlist ORDER BY id DESC""").fetchall()
    c.close()
    return [dict(r) for r in rows]


def get_watchlist(wid):
    c = _conn()
    r = c.execute("SELECT * FROM watchlist WHERE id=?", (wid,)).fetchone()
    c.close()
    if not r: return None
    d = dict(r); d["baseline_meas"] = json.loads(d["baseline_meas"])
    return d


def update_watchlist_check(wid, changed_pct, alert):
    c = _conn()
    c.execute("UPDATE watchlist SET last_check_ts=?, last_changed_pct=?, last_alert=? WHERE id=?",
              (time.time(), changed_pct, int(alert), wid))
    c.commit(); c.close()
