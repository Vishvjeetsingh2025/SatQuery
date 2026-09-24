import base64, io, json, os
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from google import genai
from google.genai import types
import analytics as an

app = FastAPI(title="SatQuery AI")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])  # set in terminal, never in code

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


def b64(im):
    buf = io.BytesIO(); im.convert("RGB").save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


@app.post("/api/query")
async def query(prompt: str = Form(...), history: str = Form("[]"), files: list[UploadFile] = File(...)):
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
        return data
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
def index():
    return FileResponse("static/index.html")
