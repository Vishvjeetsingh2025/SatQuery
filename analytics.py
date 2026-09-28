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


def _match(a, ref):  # radiometric normalisation: reduces lighting/season colour shift
    a = a.astype(np.float32)
    for c in range(3):
        a[..., c] = (a[..., c] - a[..., c].mean()) / (a[..., c].std() + 1e-6) * ref[..., c].std() + ref[..., c].mean()
    return np.clip(a, 0, 255).astype(np.uint8)


def change(im_a, im_b):
    A = _arr(im_a)
    B = cv2.resize(_arr(im_b), (A.shape[1], A.shape[0]))
    Bn = _match(B, A)
    lab = lambda x: cv2.cvtColor(cv2.GaussianBlur(x, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    d = np.linalg.norm(lab(A) - lab(Bn), axis=2)
    d8 = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(t / 255 * float(d.max()), 18)          # absolute floor: identical images => ~0% change

    def _changed(th):
        m = (d > th).astype(np.uint8)
        k = np.ones((5, 5), np.uint8)
        m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        n, lb, st, _ = cv2.connectedComponentsWithStats(m)
        keep = np.zeros_like(m)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] > 0.0005 * m.size:  # drop speckle noise
                keep[lb == i] = 1
        return keep

    keep = _changed(thr)
    ch = keep.astype(bool)
    # uncertainty range: how much the answer moves if the threshold is 15% stricter / 13% looser
    lo_pct = round(100 * float(_changed(thr * 1.15).astype(bool).mean()), 1)
    hi_pct = round(100 * float(_changed(max(thr * 0.87, 12)).astype(bool).mean()), 1)
    ma, mb = masks(A), masks(B)
    tr = {}
    for k_ in ("vegetation", "water", "built_or_bare"):
        tr[k_] = {"gain_pct": round(100 * float((mb[k_] & ~ma[k_] & ch).mean()), 1),
                  "loss_pct": round(100 * float((ma[k_] & ~mb[k_] & ch).mean()), 1)}
    ov = A.copy()
    ov[ch] = (0.45 * ov[ch] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)
    cnts, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnts, -1, (255, 255, 0), 1)
    pct = round(100 * float(ch.mean()), 1)
    return {"changed_pct": pct, "changed_range": [min(lo_pct, pct), max(hi_pct, pct)], "transitions": tr,
            "overlay": Image.fromarray(ov), "location_check": same_location(im_a, im_b)}


def same_location(im_a, im_b):
    """Feature-matching sanity check (ORB + RANSAC homography): do these two images actually show the
    same place? Guards against comparing e.g. two different cities and reporting a fake 'change'."""
    ga = cv2.cvtColor(_arr(im_a, 640), cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(_arr(im_b, 640), cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(2000)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db_ = orb.detectAndCompute(gb, None)
    if da is None or db_ is None or len(ka) < 10 or len(kb) < 10:
        return {"status": "uncertain", "inliers": 0, "message": "Too little texture to verify the two images show the same place."}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db_, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    inl = 0
    if len(good) >= 8:
        src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        inl = int(mask.sum()) if mask is not None else 0
    if inl >= 20:
        return {"status": "same", "inliers": inl, "message": "Images match the same location."}
    if inl >= 10:
        return {"status": "uncertain", "inliers": inl, "message": "Images only weakly match — verify they show the same area."}
    return {"status": "different", "inliers": inl,
            "message": "These images do not appear to show the same location — the change result is not reliable."}
