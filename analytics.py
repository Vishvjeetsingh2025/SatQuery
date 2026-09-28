"""Classical image analysis used alongside the VLM.
NOTE: RGB-only proxies (approximate). True NDVI/NDWI needs multispectral (NIR) bands."""
import cv2
import numpy as np
from PIL import Image


def _arr(im, size=512):
    a = np.array(im.convert("RGB"))
    s = size / max(a.shape[:2])
    return cv2.resize(a, (int(a.shape[1] * s), int(a.shape[0] * s)), interpolation=cv2.INTER_AREA)


def masks(a):
    f = a.astype(np.float32) / 255
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    cloud = (v > 215) & (s < 35)
    veg = ((2 * g - r - b) > 0.06) & (g > r) & (g > b) & ~cloud      # Excess-Green index
    water = ~veg & ~cloud & (b >= r) & (v < 150) & (s > 25)          # dark + bluish
    built = ~veg & ~water & ~cloud & (s < 60) & (v > 80)             # grey/low-saturation
    other = ~(veg | water | built | cloud)                           # soil, mixed
    return dict(vegetation=veg, water=water, built_or_bare=built, other=other, cloud_or_bright=cloud)


def mask_overlay(im, alpha=0.45):
    """Colour-codes the same classical-CV land-cover mask used for the % measurements, so a
    person can visually verify the numbers came from real pixel classification, not a guess.
    green=vegetation, blue=water, grey=built/bare, white=cloud/bright."""
    a = _arr(im, size=768)
    m = masks(a)
    out = a.astype(np.float32)
    colors = {"vegetation": (40, 200, 80), "water": (60, 140, 255),
              "built_or_bare": (170, 170, 180), "cloud_or_bright": (255, 255, 255)}
    for k, color in colors.items():
        sel = m[k]
        out[sel] = out[sel] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    return Image.fromarray(out.astype(np.uint8)).resize(im.size)


def stats(im):
    a = _arr(im)
    out = {k: round(100 * float(m.mean()), 1) for k, m in masks(a).items()}
    edges = cv2.Canny(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), 80, 160)
    out["edge_density"] = round(100 * float((edges > 0).mean()), 1)  # high = urban/structured
    out["brightness"] = int(a.mean())
    return out


def _match(a, ref, valid=None):  # radiometric normalisation: reduces lighting/season colour shift
    a = a.astype(np.float32)
    sel = valid if valid is not None else np.ones(a.shape[:2], bool)
    for c in range(3):
        av, rv = a[..., c][sel], ref[..., c][sel]
        a[..., c] = (a[..., c] - av.mean()) / (av.std() + 1e-6) * rv.std() + rv.mean()
    return np.clip(a, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------
# Trust Engine, part 1: "are these two images even the same place?"  (+ real feature-based alignment)
# SIFT keypoints -> Lowe ratio test -> RANSAC similarity transform. Thresholds were calibrated on real
# Google Earth screenshots of one area at different dates (same-place inlier ratio 0.76-1.00) versus
# far-apart crops / unrelated images (ratio <= 0.24).
# ---------------------------------------------------------------------------------------------

def _sane(M):
    s = float(np.hypot(M[0, 0], M[1, 0]))
    return 0.3 < s < 3.0


def locate(A, B):
    """A, B: RGB arrays. Returns dict(status same|different|uncertain, good, inliers, ratio, M) with M mapping B->A."""
    res = {"status": "uncertain", "good": 0, "inliers": 0, "ratio": 0.0, "M": None}
    try:
        def prep(x):
            return cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY))
        sift = cv2.SIFT_create(nfeatures=1500)
        ka, da = sift.detectAndCompute(prep(A), None)
        kb, db = sift.detectAndCompute(prep(B), None)
        if da is None or db is None or len(ka) < 8 or len(kb) < 8:
            return res                                   # too little texture to judge -> honest "uncertain"
        knn = cv2.BFMatcher().knnMatch(db, da, k=2)      # query=B, train=A  => transform maps B into A's frame
        good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        res["good"] = len(good)
        if len(good) < 8:
            return res
        pb = np.float32([kb[g.queryIdx].pt for g in good])
        pa = np.float32([ka[g.trainIdx].pt for g in good])
        M, inl = cv2.estimateAffinePartial2D(pb, pa, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        n = int(inl.sum()) if inl is not None else 0
        ratio = n / len(good)
        res.update(inliers=n, ratio=round(ratio, 2))
        if ratio >= 0.6 and n >= 15 and M is not None and _sane(M):
            res.update(status="same", M=M)
        elif ratio < 0.4:
            res["status"] = "different"
        return res
    except Exception:                                    # never let the guard itself break an analysis
        return res


def _mask_at(d, thr, valid):
    m = ((d > thr) & valid).astype(np.uint8)
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    n, lb, st, _ = cv2.connectedComponentsWithStats(m)
    big = st[:, cv2.CC_STAT_AREA] > 0.0005 * m.size      # drop speckle noise
    big[0] = False
    return big[lb] & valid


def change(im_a, im_b, align=True):
    A = _arr(im_a)
    Braw = _arr(im_b)
    H, W = A.shape[:2]
    loc = locate(A, Braw) if align else {"status": "uncertain", "good": 0, "inliers": 0, "ratio": 0.0, "M": None}
    aligned, overlap = False, 100.0
    if loc["M"] is not None:
        B = cv2.warpAffine(Braw, loc["M"], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = cv2.warpAffine(np.full(Braw.shape[:2], 255, np.uint8), loc["M"], (W, H), flags=cv2.INTER_NEAREST) > 0
        valid = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)   # drop interpolated border
        overlap = round(100 * float(valid.mean()), 1)
        aligned = True
        if overlap < 25:
            loc["status"] = "uncertain"                  # same place, but barely any overlap to compare
    else:
        B = cv2.resize(Braw, (W, H))
        valid = np.ones((H, W), bool)
    Bn = _match(B, A, valid)
    lab = lambda x: cv2.cvtColor(cv2.GaussianBlur(x, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    d = np.linalg.norm(lab(A) - lab(Bn), axis=2)
    d[~valid] = 0
    dv = d[valid]
    dmin, dmax = float(dv.min()), float(dv.max())
    d8 = ((dv - dmin) / max(dmax - dmin, 1e-6) * 255).astype(np.uint8).reshape(-1, 1)
    t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(t / 255 * dmax, 18)                        # absolute floor: identical images => ~0% change
    nvalid = float(valid.sum())
    ch = _mask_at(d, thr, valid)
    pct = lambda mk: round(100 * float(mk.sum()) / nvalid, 1)
    central = pct(ch)
    # Uncertainty: re-run at a stricter / looser threshold. This is a *threshold-sensitivity range*,
    # not a formal confidence interval - labelled that way in the UI.
    hi_thr, lo_thr = pct(_mask_at(d, thr * 1.25, valid)), pct(_mask_at(d, thr * 0.8, valid))
    low, high = min(hi_thr, central), max(lo_thr, central)
    ma, mb = masks(A), masks(B)
    tr = {}
    for k_ in ("vegetation", "water", "built_or_bare"):
        tr[k_] = {"gain_pct": round(100 * float((mb[k_] & ~ma[k_] & ch).sum()) / nvalid, 1),
                  "loss_pct": round(100 * float((ma[k_] & ~mb[k_] & ch).sum()) / nvalid, 1)}
    ov = A.copy()
    ov[ch] = (0.45 * ov[ch] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)
    cnts, _ = cv2.findContours(ch.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnts, -1, (255, 255, 0), 1)
    return {"changed_pct": central, "changed_pct_low": low, "changed_pct_high": high, "transitions": tr,
            "overlay": Image.fromarray(ov),
            "location": {"status": loc["status"], "inliers": loc["inliers"], "ratio": loc["ratio"],
                         "aligned": aligned, "overlap_pct": overlap}}
