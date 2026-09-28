import base64, hashlib, hmac, io, json, os, re, secrets  # hashlib also used by the demo-response cache below
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image
from google import genai
from google.genai import types
import analytics as an
import trust
import qrcode
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
- If a CHANGE MAP line says location_match=different, the two images do not show the same place: say the
  comparison is not meaningful and do NOT describe changed_area as real change. If it says uncertain, caveat it.
- Some images may state a real ground area in hectares (from geo-referenced capture); use those figures when quoting areas.
Return ONLY JSON: {"answer": str (use **bold**), "relevant_images": [int], "confidence": "low|medium|high",
"observations": [short evidence strings],
"claims": [{"image": int, "kind": "vegetation|water|built_up|cloud", "level": "none|low|moderate|high"}]}
"claims" = at most 6 statements about the OVERALL amount of vegetation / water / built-up / cloud in a specific
image that your answer relies on (they are automatically checked against pixel measurements). Use [] if none."""


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


# ---------------- Demo-safety response cache ----------------
# Identical (prompt + exact same images) queries are served instantly from memory instead of
# re-running CV + two Gemini calls. This protects a live demo from a slow/rate-limited API call
# on a question you've already asked once — it never fabricates anything, it just reuses a
# genuinely-computed prior result for an *exact* repeat of the same input.
_response_cache: dict[str, dict] = {}
_CACHE_MAX = 200


def _cache_key(prompt: str, files_bytes: list[bytes]) -> str:
    h = hashlib.sha256()
    h.update(prompt.strip().lower().encode())
    for b in files_bytes:
        h.update(hashlib.sha256(b).digest())
    return h.hexdigest()


# Carbon/economic figure. Hectares now come from REAL scene geometry (Map Explorer bounding box, or an
# area the user typed in) - never an assumed footprint. Carbon density & price are still standard averages,
# so the tons/USD stay an order-of-magnitude estimate (labelled), but the hectares are measured.
CARBON_TONS_PER_HECTARE = 180      # approx. above-ground biomass carbon stock, tropical/subtropical forest
CARBON_PRICE_USD_PER_TON = 8       # approx. voluntary carbon market price (varies 3-15 USD/ton)


def _carbon_estimate(transitions: dict, area_ha, area_source: str) -> dict | None:
    veg_loss_pct = transitions.get("vegetation", {}).get("loss_pct", 0)
    if veg_loss_pct <= 0 or not area_ha:
        return None
    hectares_lost = round(area_ha * veg_loss_pct / 100, 2)
    tons_co2 = round(hectares_lost * CARBON_TONS_PER_HECTARE, 1)
    usd = round(tons_co2 * CARBON_PRICE_USD_PER_TON)
    return {"hectares_lost_est": hectares_lost, "carbon_tons_est": tons_co2, "value_usd_est": usd,
            "area_source": area_source,
            "note": f"Hectares come from the real scene area ({area_source}). CO2 tons and USD apply standard "
                    "tropical-forest averages, so treat them as an order-of-magnitude estimate (not a certified audit)."}


# Reference context from a real, checkable government source. Deliberately NOT turned into a
# "N times the national average" multiplier: the two metrics differ (share of a scene vs. % of a
# state's forest cover), so we present it as context with the source named, not as a fake-precise ratio.
ISFR_CONTEXT = {
    "source": "India State of Forest Report (ISFR) 2021, Forest Survey of India",
    "range_pct": [0.39, 1.88],
    "text": "ISFR 2021 reported state-wise forest-cover declines of about 0.39%–1.88% in North-East Indian states "
            "between assessment cycles.",
    "caveat": "Context only — different metric and region from this scene; not a like-for-like comparison.",
}


def _benchmark_context(transitions: dict) -> dict | None:
    veg_loss = transitions.get("vegetation", {}).get("loss_pct", 0)
    if veg_loss <= 0:
        return None
    return {**ISFR_CONTEXT, "scene_veg_loss_pct": round(veg_loss, 1)}


@app.post("/api/query")
async def query(prompt: str = Form(...), history: str = Form("[]"), files: list[UploadFile] = File(...),
                 scene_km2: float | None = Form(None), authorization: str = Header(None)):
    user_id = optional_user_id(authorization)  # querying stays usable without login; history is tagged if logged in
    try:
        raw = [await f.read() for f in files[:16]]
        names = [f.filename or "" for f in files[:16]]
        km2 = scene_km2 if (scene_km2 and 0 < scene_km2 < 100000) else None
        imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in raw]
        for im in imgs:
            im.thumbnail((1024, 1024))
        n = len(imgs)

        ckey = _cache_key(prompt + "|" + "|".join(names) + f"|{km2}", raw)
        cached = _response_cache.get(ckey)
        if cached:
            data = json.loads(json.dumps(cached))  # deep copy so history_id below doesn't mutate the cached entry
        else:
            try:  # 1) route the query
                route = ask_json(f"N={n}\nQuery: {prompt}", ROUTER, 0)
                pairs = [p for p in route.get("change_pairs", []) if len(p) == 2 and all(1 <= x <= n for x in p) and p[0] != p[1]][:4]
            except Exception:
                route, pairs = {"intent": "other"}, ([[1, 2]] if n == 2 else [])

            # 2) measure (classical CV) + detect change
            meas = [an.stats(im) for im in imgs]
            # real ground area per image: Map Explorer bbox (exact) > user-typed km^2 > unknown
            area = []
            for nm in names[:n]:
                g = trust.parse_geo(nm)
                if g:
                    area.append({"ha": trust.geo_area_ha(g), "source": "Map Explorer bounding box"})
                elif km2:
                    area.append({"ha": km2 * 100.0, "source": "area entered by user"})
                else:
                    area.append(None)
            changes = []
            for a, b in pairs:
                c = an.change(imgs[a - 1], imgs[b - 1]); c["pair"] = [a, b]; changes.append(c)

            # 3) VLM reasons over images + numbers + change maps
            contents = []
            for i, im in enumerate(imgs):
                contents += [f"Image {i + 1}:", im]
            contents.append("MEASUREMENTS (% of image area):\n" + "\n".join(f"Image {i + 1}: {m}" for i, m in enumerate(meas)))
            for i, ar in enumerate(area):
                if ar:
                    ha = ar["ha"]
                    contents.append(f"Image {i + 1} real ground area = {ha:,.0f} ha ({ar['source']}); by the pixel classification "
                                    f"vegetation ~{ha * meas[i]['vegetation'] / 100:,.0f} ha, water ~{ha * meas[i]['water'] / 100:,.0f} ha, "
                                    f"built/bare ~{ha * meas[i]['built_or_bare'] / 100:,.0f} ha (approximate).")
            for c in changes:
                L = c["location"]
                warn = (" WARNING: these images appear to show DIFFERENT places; changed_area is not meaningful."
                        if L["status"] == "different" else
                        (" (same-place could not be verified; caveat the result)" if L["status"] == "uncertain" else ""))
                contents += [f"CHANGE MAP Image {c['pair'][0]} -> Image {c['pair'][1]}: location_match={L['status']} "
                             f"(SIFT inliers={L['inliers']}, ratio={L['ratio']}, feature-aligned={L['aligned']}), "
                             f"changed_area={c['changed_pct']}% (threshold-sensitivity range {c['changed_pct_low']}-{c['changed_pct_high']}%), "
                             f"class transitions={c['transitions']}.{warn}", c["overlay"]]
            past = "\n".join(f"{m['role']}: {m['text']}" for m in json.loads(history)[-6:])
            contents.append(f"Previous conversation:\n{past}\n\nUser query: {prompt}")
            data = ask_json(contents, SYSTEM)

            data["intent"] = route.get("intent")
            data["measurements"] = meas
            data["changes"] = []
            for c in changes:
                ok = c["location"]["status"] != "different"          # never quote carbon/benchmark for non-comparable images
                ar = area[c["pair"][0] - 1]
                data["changes"].append({
                    "pair": c["pair"], "changed_pct": c["changed_pct"],
                    "changed_pct_low": c["changed_pct_low"], "changed_pct_high": c["changed_pct_high"],
                    "location": c["location"], "transitions": c["transitions"], "overlay": b64(c["overlay"]),
                    "carbon_estimate": _carbon_estimate(c["transitions"], ar["ha"], ar["source"]) if (ok and ar) else None,
                    "benchmark": _benchmark_context(c["transitions"]) if ok else None})
            data["area"] = [({"area_ha": round(ar["ha"]), "source": ar["source"],
                              "vegetation_ha": round(ar["ha"] * meas[i]["vegetation"] / 100),
                              "water_ha": round(ar["ha"] * meas[i]["water"] / 100)} if ar else None)
                             for i, ar in enumerate(area)]

            # ---- Trust Engine: audit the AI's own answer ----
            claims = data.get("claims") if isinstance(data.get("claims"), list) else []
            px = trust.verify_claims(claims, meas)
            locs = [{"pair": c["pair"], **c["location"]} for c in changes]
            conf0 = data.get("confidence")
            conf1, reasons = trust.adjust_confidence(conf0, px["status"], [l["status"] for l in locs])
            data["confidence"] = conf1
            data["trust"] = {"pixel_check": px, "locations": locs, "confidence_original": conf0,
                             "confidence_adjusted": conf1, "reasons": reasons}

            if len(_response_cache) >= _CACHE_MAX:
                _response_cache.pop(next(iter(_response_cache)))  # drop oldest, simple FIFO cap
            _response_cache[ckey] = json.loads(json.dumps(data))

        # 4) persist to history (every analysis becomes revisitable + report-able)
        thumb = b64(imgs[0], max_side=360, quality=70) if imgs else None
        hist_obs = list(data.get("observations") or []) + trust.summary_lines(data.get("trust", {}))
        hid = db.add_history(user_id, prompt, data.get("intent"), data.get("answer"),
                              data.get("confidence"), thumb, hist_obs)
        data["history_id"] = hid
        return data
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/mask")
async def api_mask(file: UploadFile = File(...)):
    """Visual proof-of-work: colour-codes the actual classical-CV land-cover classification
    (vegetation / water / built-up / cloud) computed for this image, so it can be shown
    side-by-side with the AI's text answer as evidence it isn't just a guess."""
    im = Image.open(io.BytesIO(await file.read())).convert("RGB"); im.thumbnail((1024, 1024))
    overlay = an.mask_overlay(im)
    buf = io.BytesIO(); overlay.save(buf, "JPEG", quality=85)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


# ---------------- History (requires login: each account sees only its own analyses) ----------------

@app.get("/api/history")
def api_history_list(user_id: int = Depends(current_user_id)):
    return db.list_history(user_id)


@app.get("/api/report/{history_id}")
def api_report(history_id: int):
    # Shareable by link (history list itself stays private per account) so QR scans on another device work
    record = db.get_history(history_id, public=True)
    if not record:
        return JSONResponse({"error": "Report not found"}, status_code=404)
    pdf_bytes = rpt.build_report(record)
    # "inline" (not "attachment") so clicking the link opens the PDF directly in the browser tab's
    # built-in viewer, instead of silently triggering a background download with no visible result
    return Response(content=pdf_bytes, media_type="application/pdf",
                     headers={"Content-Disposition": f'inline; filename="satquery_report_{history_id}.pdf"'})


def _public_base_url(request: Request) -> str:
    """Behind Render's proxy request.base_url can come back as http://; prefer forwarded headers
    so the QR code always encodes the real public https URL a phone can open."""
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
    return f"{proto}://{host}"


@app.get("/api/report/{history_id}/qr")
def api_report_qr(history_id: int, request: Request):
    if not db.get_history(history_id, public=True):
        return JSONResponse({"error": "Report not found"}, status_code=404)
    url = f"{_public_base_url(request)}/api/report/{history_id}"
    img = qrcode.make(url, box_size=8, border=2)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})


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
    if c["location"]["status"] == "different":
        # Don't record a meaningless number or raise a false alert: the new image isn't this watchlist area.
        return {"not_comparable": True, "location": c["location"],
                "message": "This image doesn't appear to show the same place as the saved baseline, so it was not compared."}
    alert = c["changed_pct"] >= ALERT_THRESHOLD_PCT
    db.update_watchlist_check(watchlist_id, c["changed_pct"], alert)
    return {"changed_pct": c["changed_pct"], "changed_pct_low": c["changed_pct_low"], "changed_pct_high": c["changed_pct_high"],
            "location": c["location"], "transitions": c["transitions"],
            "overlay": b64(c["overlay"]), "alert": alert, "threshold": ALERT_THRESHOLD_PCT}


@app.get("/")
def index():
    # no-store: this is a single-file SPA, so the browser must always fetch the latest
    # index.html after every deploy instead of silently reusing a stale cached copy
    return FileResponse("static/index.html", headers={"Cache-Control": "no-store, must-revalidate"})
