import base64, hashlib, hmac, io, json, os, re, secrets
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image
from google import genai
from google.genai import types
import analytics as an
import db
import report as rpt

app = FastAPI(title="SatQuery AI")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])  # set in terminal, never in code
db.init()
ALERT_THRESHOLD_PCT = 8.0  # watchlist auto-alert fires above this % changed area
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ---------------- Auth helpers ----------------

def hash_password(password: str, salt: str | None = None):
    salt = salt or secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 100_000).hex()
    return salt, h


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    _, h = hash_password(password, salt)
    return hmac.compare_digest(h, expected_hash)


def current_user_id(authorization: str = Header(None)) -> int:
    """Required auth: raises 401 if no valid session. Used for History/Watchlist endpoints."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Login required")
    uid = db.get_user_id_by_token(authorization.split(" ", 1)[1])
    if not uid:
        raise HTTPException(401, "Session expired, please log in again")
    return uid


def optional_user_id(authorization: str = Header(None)):
    """Best-effort auth: returns None instead of raising, for endpoints usable without login."""
    if authorization and authorization.startswith("Bearer "):
        return db.get_user_id_by_token(authorization.split(" ", 1)[1])
    return None


@app.post("/api/auth/signup")
def auth_signup(email: str = Form(...), password: str = Form(...)):
    email = email.strip().lower()
    if not EMAIL_RE.match(email):
        return JSONResponse({"error": "Enter a valid email address"}, status_code=400)
    if len(password) < 6:
        return JSONResponse({"error": "Password must be at least 6 characters"}, status_code=400)
    if db.get_user_by_email(email):
        return JSONResponse({"error": "An account with this email already exists"}, status_code=400)
    salt, h = hash_password(password)
    uid = db.create_user(email, salt, h)
    token = secrets.token_hex(24)
    db.create_session(uid, token)
    return {"token": token, "email": email}


@app.post("/api/auth/login")
def auth_login(email: str = Form(...), password: str = Form(...)):
    user = db.get_user_by_email(email)
    if not user or not verify_password(password, user["salt"], user["pass_hash"]):
        return JSONResponse({"error": "Incorrect email or password"}, status_code=401)
    token = secrets.token_hex(24)
    db.create_session(user["id"], token)
    return {"token": token, "email": user["email"]}


@app.post("/api/auth/logout")
def auth_logout(authorization: str = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        db.delete_session(authorization.split(" ", 1)[1])
    return {"ok": True}

ROUTER = """Classify a user's query about N satellite images. Return ONLY JSON:
{"intent": "describe|retrieve|measure|count|locate|compare|change|other",
 "change_pairs": [[a,b], ...]}
change_pairs = 1-based (before, after) pairs needing change detection; [] if the query is not about
change/compare/before-after/growth/loss/damage/difference. For a time series or 'all changes' use
consecutive pairs (1,2),(2,3)... Max 4 pairs."""

SYSTEM = """You are SatQuery AI, an expert remote-sensing analyst.
Inputs: images labelled 'Image N'; MEASUREMENTS from classical image analysis (RGB-proxy, approximate,
not true NDVI/NDWI); optionally CHANGE MAPS (red = changed pixels, with class gain/loss % of image area).
Rules:
- Answer only from the images and measurements. Prefer measured numbers over guessing; quote them as approximate.
- Change queries: say WHAT changed and WHERE (quadrant/landmark), classify each change (urban growth/construction,
  deforestation or vegetation loss, vegetation gain, flooding/water change, agriculture/seasonal, burn/damage,
  or an artifact: cloud, shadow, lighting, misalignment), give magnitude, and say if it looks real or an artifact.
- Images are assumed chronological (Image 1 oldest) unless the user says otherwise; mention if that matters.
- If the query cannot be answered from this data, say so and say what data is needed.
Return ONLY JSON: {"answer": str (use **bold**), "relevant_images": [int], "confidence": "low|medium|high",
"observations": [short evidence strings]}"""


def ask_json(contents, system, temp=0.2):
    r = client.models.generate_content(model=MODEL, contents=contents, config=types.GenerateContentConfig(
        system_instruction=system, response_mime_type="application/json", temperature=temp))
    return json.loads(r.text)


def b64(im, max_side=None, quality=80):
    im2 = im.convert("RGB")
    if max_side:
        im2 = im2.copy(); im2.thumbnail((max_side, max_side))
    buf = io.BytesIO(); im2.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_image(s):
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")


@app.post("/api/query")
async def query(prompt: str = Form(...), history: str = Form("[]"), files: list[UploadFile] = File(...),
                 authorization: str = Header(None)):
    user_id = optional_user_id(authorization)  # querying stays usable without login; history is tagged if logged in
    try:
        imgs = []
        for f in files[:16]:
            im = Image.open(io.BytesIO(await f.read())).convert("RGB"); im.thumbnail((1024, 1024)); imgs.append(im)
        n = len(imgs)

        try:  # 1) route the query
            route = ask_json(f"N={n}\nQuery: {prompt}", ROUTER, 0)
            pairs = [p for p in route.get("change_pairs", []) if len(p) == 2 and all(1 <= x <= n for x in p) and p[0] != p[1]][:4]
        except Exception:
            route, pairs = {"intent": "other"}, ([[1, 2]] if n == 2 else [])

        # 2) measure (classical CV) + detect change
        meas = [an.stats(im) for im in imgs]
        changes = []
        for a, b in pairs:
            c = an.change(imgs[a - 1], imgs[b - 1]); c["pair"] = [a, b]; changes.append(c)

        # 3) VLM reasons over images + numbers + change maps
        contents = []
        for i, im in enumerate(imgs):
            contents += [f"Image {i + 1}:", im]
        contents.append("MEASUREMENTS (% of image area):\n" + "\n".join(f"Image {i + 1}: {m}" for i, m in enumerate(meas)))
        for c in changes:
            contents += [f"CHANGE MAP Image {c['pair'][0]} -> Image {c['pair'][1]}: changed_area={c['changed_pct']}%, "
                         f"class transitions={c['transitions']}", c["overlay"]]
        past = "\n".join(f"{m['role']}: {m['text']}" for m in json.loads(history)[-6:])
        contents.append(f"Previous conversation:\n{past}\n\nUser query: {prompt}")
        data = ask_json(contents, SYSTEM)

        data["intent"] = route.get("intent")
        data["measurements"] = meas
        data["changes"] = [{"pair": c["pair"], "changed_pct": c["changed_pct"],
                            "transitions": c["transitions"], "overlay": b64(c["overlay"])} for c in changes]

        # 4) persist to history (new: every analysis becomes revisitable + report-able)
        thumb = b64(imgs[0], max_side=360, quality=70) if imgs else None
        hid = db.add_history(user_id, prompt, data.get("intent"), data.get("answer"),
                              data.get("confidence"), thumb, data.get("observations"))
        data["history_id"] = hid
        return data
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------- History (requires login: each account sees only its own analyses) ----------------

@app.get("/api/history")
def api_history_list(user_id: int = Depends(current_user_id)):
    return db.list_history(user_id)


@app.get("/api/report/{history_id}")
def api_report(history_id: int, user_id: int | None = Depends(optional_user_id)):
    record = db.get_history(history_id, user_id)
    if not record:
        return JSONResponse({"error": "Not found, or this report belongs to a different account"}, status_code=404)
    pdf_bytes = rpt.build_report(record)
    return Response(content=pdf_bytes, media_type="application/pdf",
                     headers={"Content-Disposition": f'attachment; filename="satquery_report_{history_id}.pdf"'})


# ---------------- Watchlist (requires login: each account has its own private watchlist) ----------------

@app.post("/api/watchlist/save")
async def watchlist_save(name: str = Form(...), file: UploadFile = File(...), user_id: int = Depends(current_user_id)):
    im = Image.open(io.BytesIO(await file.read())).convert("RGB"); im.thumbnail((1024, 1024))
    meas = an.stats(im)
    wid = db.add_watchlist(user_id, name, b64(im, max_side=800), meas)
    return {"id": wid, "name": name, "baseline_meas": meas}


@app.get("/api/watchlist")
def watchlist_list(user_id: int = Depends(current_user_id)):
    return db.list_watchlist(user_id)


@app.post("/api/watchlist/{watchlist_id}/check")
async def watchlist_check(watchlist_id: int, file: UploadFile = File(...), user_id: int = Depends(current_user_id)):
    wl = db.get_watchlist(watchlist_id, user_id)
    if not wl:
        return JSONResponse({"error": "Not found, or this watchlist entry belongs to a different account"}, status_code=404)
    baseline_im = b64_to_image(wl["baseline_b64"])
    new_im = Image.open(io.BytesIO(await file.read())).convert("RGB"); new_im.thumbnail((1024, 1024))
    c = an.change(baseline_im, new_im)
    alert = c["changed_pct"] >= ALERT_THRESHOLD_PCT
    db.update_watchlist_check(watchlist_id, c["changed_pct"], alert)
    return {"changed_pct": c["changed_pct"], "transitions": c["transitions"],
            "overlay": b64(c["overlay"]), "alert": alert, "threshold": ALERT_THRESHOLD_PCT}


@app.get("/")
def index():
    # no-store: this is a single-file SPA, so the browser must always fetch the latest
    # index.html after every deploy instead of silently reusing a stale cached copy
    return FileResponse("static/index.html", headers={"Cache-Control": "no-store, must-revalidate"})
