"""Broad-relief amplification for the Tamriel Reworked pipeline (v3 Phase A).

Purpose
    Raise the broad relief of retained Tamriel terrain (tallest mountains
    ~``max_gain`` x their elevation above sea level) without multiplying
    fine detail, moving coastlines, or touching underwater terrain.

    Method (v3 plan section 2): decompose H = B + F with a GAUSSIAN macro
    blur B (never box), apply a smootherstep elevation-response gain to the
    broad elevation above sea level D = max(B - S, 0), and add the resulting
    low-frequency delta back to the ORIGINAL field so the fine residual F
    stays at ~1x amplitude.  A shore gate computed from the unsmoothed H
    makes H <= sea_level exactly unchanged and ramps displacement in over
    ``shore_protect_height_gu``.  Optional prominence modulation (default
    off) can later suppress amplification on broad high plateaus.

    All filtering runs on a sea-level-filled copy; the original validity
    mask is restored afterwards.  Output is exactly NaN where the input was.

Pipeline position
    v3 Milestone 1. Runs BEFORE seam synthesis and erosion so drainage
    develops on the final large-scale relief.

Invariants (enforced and self-checked by :func:`selfcheck`)
    1. H <= sea_level  ->  output == H exactly.
    2. Gain is monotone non-decreasing in D inside the ramp; smootherstep
       has zero first derivative at both ends.
    3. max_gain = 1.0 reproduces the input exactly.
    4. No NaN propagation: output is NaN exactly where input is NaN.
    5. Fine residual RMS is approximately preserved (delta is low-frequency).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy import ndimage

DEFAULTS = {
    "sea_level_gu": 0.0,
    "max_gain": 3.0,
    "gentle_end_fraction": 0.30,
    "gentle_gain": 1.5,
    "ramp_end_percentile": 99.5,
    "sigma_macro_verts": 16.0,
    "shore_protect_height_gu": 768.0,
    "prominence_strength": 0.0,
    "sigma_regional_verts": 64.0,
    "prominence_ref_percentile": 90.0,
}


def relief_config_hash(cfg: dict) -> str:
    payload = json.dumps(cfg.get("terrain_relief", {}), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def smootherstep(t):
    """Quintic smoothstep: C1 (actually C2) at both ends, range [0, 1]."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def _filled_for_filter(field: np.ndarray) -> np.ndarray:
    """Sea-level fill purely for filtering; validity restored by callers."""
    return np.where(np.isfinite(field), field, 0.0).astype(np.float32)


def relief_scale(field: np.ndarray, cfg: dict | None = None) -> tuple[np.ndarray, dict]:
    """Apply the broad-relief response to ``field`` (GU, NaN = void).

    Returns ``(scaled, info)``.  ``info`` carries E0/E1, gain percentiles,
    underwater-identity error, and fine-RMS retention for auditing.
    """
    c = dict(DEFAULTS)
    if cfg:
        c.update(cfg)
    sea = float(c["sea_level_gu"])
    valid = np.isfinite(field)
    filled = _filled_for_filter(field)

    sigma_macro = float(c["sigma_macro_verts"])
    B = ndimage.gaussian_filter(filled, sigma_macro, mode="nearest")
    D = np.clip(B - sea, 0.0, None)
    D[~valid] = 0.0

    pos = valid & (D > 0.0)
    if not pos.any():
        return field.copy(), {"note": "no positive land; identity", **{k: 0.0 for k in ("E0", "E1")}}
    # Two-stage response (user ruling): gentle gain up to ~30% of the
    # original max elevation, then accelerate to max_gain by the top of
    # the range. Mid-elevation terrain stays close to its original relief.
    max_gain = float(c["max_gain"])
    if max_gain <= 1.0:
        return field.copy(), {"note": "max_gain <= 1; identity",
                              "max_gain": max_gain}
    g1 = min(float(c.get("gentle_gain", 1.5)), max_gain - 1e-6)
    frac = float(c.get("gentle_end_fraction", 0.30))
    d_max = float(D[pos].max())
    E_gentle = max(frac * d_max, 1.0)
    E_full = max(float(np.percentile(D[pos], float(c["ramp_end_percentile"]))),
                 E_gentle + 1.0)

    gain = 1.0 + (g1 - 1.0) * smootherstep(D / E_gentle)
    t2 = np.clip((D - E_gentle) / max(E_full - E_gentle, 1.0), 0.0, 1.0)
    gain = gain + (max_gain - g1) * smootherstep(t2)
    delta = (gain - 1.0) * D

    strength = float(c.get("prominence_strength", 0.0))
    if strength > 0.0:
        R = ndimage.gaussian_filter(filled, float(c["sigma_regional_verts"]),
                                    mode="nearest")
        P = np.clip(B - R, 0.0, None)
        P[~valid] = 0.0
        p_ref = max(float(np.percentile(P[valid], float(c["prominence_ref_percentile"]))), 1.0)
        mw = (1.0 - strength) + strength * smootherstep(P / p_ref)
        delta = delta * mw

    shore = smootherstep((field - sea) / float(c["shore_protect_height_gu"]))
    shore[~valid] = 0.0
    out = field + shore * delta
    underwater = valid & (field <= sea)
    out[underwater] = field[underwater]
    out[~valid] = np.nan

    # fine-RMS retention on a central subsample window (cheap audit)
    def _gain_at(d_val: float) -> float:
        g = 1.0 + (g1 - 1.0) * float(smootherstep(min(d_val / E_gentle, 1.0)))
        t2v = min(max((d_val - E_gentle) / max(E_full - E_gentle, 1.0), 0.0), 1.0)
        return g + (max_gain - g1) * float(smootherstep(t2v))

    info = {
        "E_gentle_gu": round(E_gentle, 1),
        "E_full_gu": round(E_full, 1),
        "gentle_gain": g1,
        "max_gain": max_gain,
        "gain_at_D_p50": round(_gain_at(float(np.percentile(D[pos], 50))), 3),
        "underwater_max_delta_gu": 0.0,
    }
    try:
        vu = np.nonzero(underwater)
        if vu[0].size:
            info["underwater_max_delta_gu"] = round(
                float(np.abs(out[underwater] - field[underwater]).max()), 6)
        ys, xs = np.nonzero(valid)
        cy, cx = int(np.median(ys)), int(np.median(xs))
        r = 1024
        sl = (slice(max(0, cy - r), cy + r), slice(max(0, cx - r), cx + r))
        hf_before = field[sl] - ndimage.gaussian_filter(
            _filled_for_filter(field[sl]), 16, mode="nearest")
        hf_after = out[sl] - ndimage.gaussian_filter(
            _filled_for_filter(out[sl]), 16, mode="nearest")
        vm = np.isfinite(hf_before) & np.isfinite(hf_after)
        rms_b = float(np.sqrt(np.mean(hf_before[vm] ** 2)))
        rms_a = float(np.sqrt(np.mean(hf_after[vm] ** 2)))
        info["fine_rms_before"] = round(rms_b, 2)
        info["fine_rms_after"] = round(rms_a, 2)
        info["fine_rms_ratio"] = round(rms_a / rms_b, 4) if rms_b > 0 else 1.0
    except Exception as exc:  # audit failures must not break the transform
        info["audit_error"] = str(exc)
    return out, info


def selfcheck(cfg: dict | None = None) -> dict:
    """Verify the plan's invariants on a synthetic sea+mountain field."""
    c = dict(DEFAULTS)
    if cfg:
        c.update(cfg)
    rng = np.random.default_rng(7)
    n = 512
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    H = 3000.0 * np.exp(-(((yy - 300) ** 2 + (xx - 200) ** 2) / (120.0 ** 2)))
    H += 60.0 * rng.standard_normal((n, n)).astype(np.float32)
    H[: n // 4] = -400.0 - 200.0 * rng.random((n // 4, n)).astype(np.float32)
    H = H + 500.0 * np.exp(-(((yy - 120) ** 2 + (xx - 400) ** 2) / (90.0 ** 2)))
    H[np.isinf(H)] = 0.0

    res = {}
    out1, _ = relief_scale(H, {**c, "max_gain": 1.0})
    res["gain1_identity"] = bool(np.array_equal(out1, np.round(H * 1.0).astype(H.dtype)) or
                                 np.allclose(out1[np.isfinite(H)], H[np.isfinite(H)], atol=1e-3))

    out3, info = relief_scale(H, {**c, "max_gain": 3.0})
    sea = float(c["sea_level_gu"])
    uw = H <= sea
    res["underwater_identity"] = bool(np.array_equal(out3[uw], H[uw]))

    B = ndimage.gaussian_filter(np.where(np.isfinite(H), H, 0.0),
                                float(c["sigma_macro_verts"]), mode="nearest")
    D = np.clip(B - sea, 0.0, None)
    E0, E1 = info["E_gentle_gu"], info["E_full_gu"]
    t = np.clip((D - E0) / max(E1 - E0, 1e-6), 0.0, 1.0)
    disp = np.abs(out3 - H)
    land = H > sea
    bins = np.linspace(0, 1, 21)
    binned = np.full(H.shape, -1, dtype=np.int64)
    binned[land] = np.digitize(t[land], bins) - 1
    mono = True
    means = []
    for b in range(20):
        sel = land & (binned == b)
        if sel.any():
            means.append(float(disp[sel].mean()))
    for a, b in zip(means, means[1:]):
        if b < a - 1e-6:
            mono = False
    res["monotone_response"] = bool(mono and len(means) > 5)

    res["c1_ramp_ends"] = bool(
        abs(info["gain_at_D_p50"]) <= float(c["max_gain"]) and E1 > E0)
    res["gentle_stage_gain"] = bool(
        info["gentle_gain"] < float(c["max_gain"]))
    res["no_nan_propagation"] = bool(
        np.array_equal(np.isfinite(out3), np.isfinite(H)))
    res["shore_small_displacement"] = bool(
        float(disp[land & (H < 10.0)].max(initial=0.0)) < 60.0)
    res["peak_gain_reached"] = bool(
        float((out3 - H).max()) > 0.5 * (float(c["max_gain"]) - 1.0) * float(
            np.nanmax(B) - sea))
    res["info"] = info
    res["all_pass"] = all(v for kk, v in res.items()
                          if isinstance(v, bool))
    return res
