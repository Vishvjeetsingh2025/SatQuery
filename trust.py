"""Trust Engine helpers: things that let SatQuery audit its own answers.
 - real scene area from Map Explorer geo-metadata (or a user-supplied area)
 - AI-vs-pixels cross-check: does the model's claim agree with the classical-CV measurement?
 - confidence adjustment driven by those checks
All logic here is deterministic and unit-tested; nothing is estimated by an LLM."""
import math
import re

GEO_RE = re.compile(r"^map_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_r(\d+(?:\.\d+)?)\.png$")


def parse_geo(filename):
    """Map Explorer names its captures map_<lat>_<lon>_r<halfwidth-deg>.png -> exact bounding box."""
    m = GEO_RE.match(filename or "")
    if not m:
        return None
    lat, lon, half = float(m.group(1)), float(m.group(2)), float(m.group(3))
    if not (-90 <= lat <= 90 and -180 <= lon <= 180 and 0 < half <= 1):
        return None
    return {"lat": lat, "lon": lon, "half_deg": half}


def geo_area_ha(geo):
    h_m = 2 * geo["half_deg"] * 110574.0
    w_m = 2 * geo["half_deg"] * 111320.0 * math.cos(math.radians(geo["lat"]))
    return w_m * h_m / 10000.0


LEVELS = ("none", "low", "moderate", "high")
CHECKABLE = {"vegetation": "vegetation", "water": "water", "cloud": "cloud_or_bright"}
# built-up is deliberately NOT hard-checked: the RGB proxy under-detects dense urban areas, so a
# "mismatch" there would often be the proxy's fault, not the AI's. We report it as not verifiable.


def level_of(pct):
    return 0 if pct < 2 else 1 if pct < 15 else 2 if pct < 40 else 3


def verify_claims(claims, meas):
    checked = agreed = 0
    contradicted, skipped = [], 0
    for cl in (claims or [])[:8]:
        if not isinstance(cl, dict):
            continue
        i, kind, lvl = cl.get("image"), cl.get("kind"), cl.get("level")
        if not isinstance(i, int) or not (1 <= i <= len(meas)) or lvl not in LEVELS:
            continue
        if kind not in CHECKABLE:
            skipped += 1
            continue
        measured = meas[i - 1].get(CHECKABLE[kind], 0)
        checked += 1
        if abs(LEVELS.index(lvl) - level_of(measured)) >= 2:      # lenient: adjacent bands still agree
            contradicted.append({"image": i, "kind": kind, "claimed": lvl, "measured_pct": measured})
        else:
            agreed += 1
    status = "not_checked" if checked == 0 else ("mismatch" if contradicted else "verified")
    return {"status": status, "checked": checked, "agreed": agreed, "contradicted": contradicted, "skipped": skipped}


def adjust_confidence(conf, pixel_status, loc_statuses):
    order = ["low", "medium", "high"]
    reasons = []
    if conf not in order:
        return conf, reasons
    i = order.index(conf)
    if pixel_status == "mismatch":
        i = max(0, i - 1)
        reasons.append("AI claim disagreed with pixel measurement")
    if "different" in loc_statuses:
        i = 0
        reasons.append("compared images do not appear to show the same place")
    elif "uncertain" in loc_statuses:
        i = min(i, 1)
        reasons.append("could not verify the compared images show the same place")
    return order[i], reasons


def summary_lines(trust):
    """Short strings appended to the saved history record so the PDF report carries the audit trail."""
    out = []
    px = trust.get("pixel_check", {})
    if px.get("status") == "verified":
        out.append(f"Trust check: {px['agreed']}/{px['checked']} AI claims agree with pixel measurements")
    elif px.get("status") == "mismatch":
        for c in px["contradicted"]:
            out.append(f"Trust check MISMATCH: AI said image {c['image']} has {c['claimed']} {c['kind']}, "
                       f"pixels measure {c['measured_pct']}%")
    for lc in trust.get("locations", []):
        out.append(f"Same-location check image {lc['pair'][0]}->{lc['pair'][1]}: {lc['status']} "
                   f"(SIFT inlier ratio {lc['ratio']}, aligned={lc['aligned']})")
    if trust.get("reasons"):
        out.append("Confidence adjusted "
                   f"{trust.get('confidence_original')} -> {trust.get('confidence_adjusted')}: " + "; ".join(trust["reasons"]))
    return out
