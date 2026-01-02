# src/ifaman/met/aman_windy_pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import bisect
import math
import time

import requests

# -----------------------------
# Constants / units
# -----------------------------
EARTH_RADIUS_M = 6371008.8
FT_PER_NM = 6076.12
NM_TO_M = 1852.0
KT_TO_MPS = 0.514444
R_DRY_AIR = 287.05287  # J/(kg*K)
RHO0 = 1.225           # kg/m^3 at sea level ISA

DEFAULT_LEVELS = (
    "surface", "950h", "925h", "900h", "875h", "850h", "825h", "800h",
    "750h", "700h", "650h", "600h", "550h", "500h"
)


def nm_to_m(nm: float) -> float: return nm * NM_TO_M
def m_to_nm(m: float) -> float: return m / NM_TO_M
def kt_to_mps(kt: float) -> float: return kt * KT_TO_MPS
def mps_to_kt(mps: float) -> float: return mps / KT_TO_MPS
def ft_to_m(ft: float) -> float: return ft * 0.3048
def m_to_ft(m: float) -> float: return m / 0.3048
def deg2rad(d: float) -> float: return d * math.pi / 180.0
def rad2deg(r: float) -> float: return r * 180.0 / math.pi
def wrap_lon_deg(lon: float) -> float: return (lon + 180.0) % 360.0 - 180.0
def wrap360(x: float) -> float: return (x % 360.0 + 360.0) % 360.0


# -----------------------------
# Data models
# -----------------------------
@dataclass(frozen=True)
class GeoPoint:
    lat_deg: float
    lon_deg: float

@dataclass(frozen=True)
class FixPoint:
    fix_id: str
    point: GeoPoint
    # "published" altitude constraints from procedure (if any, in ft)
    constraint_alt_ft: Optional[float] = None
    # Optional inbound course from procedure leg (deg)
    inbound_crs_deg: Optional[float] = None

@dataclass(frozen=True)
class SegmentGeom:
    a_id: str
    a_pt: GeoPoint
    b_id: str
    b_pt: GeoPoint
    dist_nm: float
    track_deg: float   # initial bearing a->b
    s0_nm: float       # along-route distance from entry at start of segment
    s1_nm: float       # along-route distance from entry at end of segment

@dataclass(frozen=True)
class SegmentTiming:
    seg: SegmentGeom
    # sampled values at start/mid/end
    ias0_kt: float
    iasm_kt: float
    ias1_kt: float
    alt0_ft: float
    altm_ft: float
    alt1_ft: float
    time_s: float
    t0_ms: int
    t1_ms: int


def runway_threshold_alt_ft(doc: dict, runway_id: str) -> float:
    for r in doc.get("runways", []):
        if r.get("id") == runway_id:
            pt = GeoPoint(float(r["lat"]), float(r["lon"]))
            elev_ft = float(r.get("alt", 0.0))  # already feet
            return pt, elev_ft
    raise KeyError(f"Runway '{runway_id}' not found in runways[]")

# -----------------------------
# Geometry
# -----------------------------
def gc_inverse_nm_bearing(p0: GeoPoint, p1: GeoPoint) -> Tuple[float, float]:
    """Great-circle distance (NM) and initial bearing (deg)."""
    lat0, lon0 = deg2rad(p0.lat_deg), deg2rad(p0.lon_deg)
    lat1, lon1 = deg2rad(p1.lat_deg), deg2rad(p1.lon_deg)
    dlat = lat1 - lat0
    dlon = lon1 - lon0

    a = math.sin(dlat/2)**2 + math.cos(lat0)*math.cos(lat1)*math.sin(dlon/2)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    dist_m = EARTH_RADIUS_M * c
    dist_nm = m_to_nm(dist_m)

    y = math.sin(dlon) * math.cos(lat1)
    x = math.cos(lat0)*math.sin(lat1) - math.sin(lat0)*math.cos(lat1)*math.cos(dlon)
    brng = wrap360(rad2deg(math.atan2(y, x)))
    return dist_nm, brng

def forward_spherical(start: GeoPoint, crs_deg: float, dist_nm: float) -> GeoPoint:
    """Great-circle forward from start with initial course and distance."""
    lat1 = deg2rad(start.lat_deg)
    lon1 = deg2rad(start.lon_deg)
    brng = deg2rad(crs_deg)
    d = nm_to_m(dist_nm) / EARTH_RADIUS_M

    sin_lat1, cos_lat1 = math.sin(lat1), math.cos(lat1)
    sin_d, cos_d = math.sin(d), math.cos(d)

    lat2 = math.asin(sin_lat1 * cos_d + cos_lat1 * sin_d * math.cos(brng))
    lon2 = lon1 + math.atan2(
        math.sin(brng) * sin_d * cos_lat1,
        cos_d - sin_lat1 * math.sin(lat2),
    )
    return GeoPoint(lat_deg=rad2deg(lat2), lon_deg=wrap_lon_deg(rad2deg(lon2)))

def interpolate_gc(p0: GeoPoint, p1: GeoPoint, frac: float) -> GeoPoint:
    """Slerp great-circle interpolation."""
    if frac <= 0.0: return p0
    if frac >= 1.0: return p1

    lat0, lon0 = deg2rad(p0.lat_deg), deg2rad(p0.lon_deg)
    lat1, lon1 = deg2rad(p1.lat_deg), deg2rad(p1.lon_deg)

    def to_vec(lat: float, lon: float) -> Tuple[float, float, float]:
        cl = math.cos(lat)
        return (cl * math.cos(lon), cl * math.sin(lon), math.sin(lat))

    v0 = to_vec(lat0, lon0)
    v1 = to_vec(lat1, lon1)

    dot = max(-1.0, min(1.0, v0[0]*v1[0] + v0[1]*v1[1] + v0[2]*v1[2]))
    omega = math.acos(dot)
    if omega < 1e-12:
        return p0

    sin_omega = math.sin(omega)
    a = math.sin((1.0 - frac) * omega) / sin_omega
    b = math.sin(frac * omega) / sin_omega

    x = a*v0[0] + b*v1[0]
    y = a*v0[1] + b*v1[1]
    z = a*v0[2] + b*v1[2]

    lat = math.atan2(z, math.hypot(x, y))
    lon = math.atan2(y, x)
    return GeoPoint(lat_deg=rad2deg(lat), lon_deg=wrap_lon_deg(rad2deg(lon)))


# -----------------------------
# Windy client (wind + gh + temp)
# -----------------------------
class WindyError(RuntimeError):
    pass

class WindyClient:
    ENDPOINT = "https://api.windy.com/api/point-forecast/v2"

    def __init__(
        self,
        api_key: str,
        model: str = "gfs",
        levels: Sequence[str] = DEFAULT_LEVELS,
        timeout_s: float = 12.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.levels = tuple(levels)
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._cache: Dict[Tuple[float, float, str, Tuple[str, ...]], dict] = {}

    @staticmethod
    def _round2(x: float) -> float:
        return round(x, 2)

    def fetch(self, lat_deg: float, lon_deg: float) -> dict:
        lat2, lon2 = self._round2(lat_deg), self._round2(lon_deg)
        key = (lat2, lon2, self.model, self.levels)
        if key in self._cache:
            return self._cache[key]

        payload = {
            "lat": lat2,
            "lon": lon2,
            "model": self.model,
            "parameters": ["wind", "gh", "temp"],
            "levels": list(self.levels),
            "key": self.api_key,
        }

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(self.ENDPOINT, json=payload, timeout=self.timeout_s)
                if r.status_code == 200:
                    data = r.json()
                    if "ts" not in data:
                        raise WindyError(f"Malformed response: missing ts, keys={list(data.keys())}")
                    self._cache[key] = data
                    return data
                raise WindyError(f"Windy HTTP {r.status_code}: {r.text[:300]}")
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(0.4 * (2 ** attempt))
        raise WindyError(f"Windy request failed after retries: {last_err}") from last_err

    @staticmethod
    def _interp_time(ts: List[int], ys: List[Optional[float]], t_ms: int) -> float:
        """Linear interpolation in time (clamped)."""
        n = len(ts)
        if n == 0:
            raise WindyError("Empty time series")

        if t_ms <= ts[0]:
            return float(ys[0] or 0.0)
        if t_ms >= ts[-1]:
            return float(ys[-1] or 0.0)

        lo, hi = 0, n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ts[mid] <= t_ms:
                lo = mid
            else:
                hi = mid

        y0, y1 = ys[lo], ys[hi]
        if y0 is None and y1 is None:
            return 0.0
        if y0 is None:
            return float(y1)  # type: ignore[arg-type]
        if y1 is None:
            return float(y0)

        t0, t1 = ts[lo], ts[hi]
        a = (t_ms - t0) / (t1 - t0)
        return float(y0) * (1.0 - a) + float(y1) * a

    def _samples_by_height(
        self, data: dict, t_ms: int
    ) -> List[Tuple[float, float, float, float, float]]:
        """
        Returns list of tuples for each level:
            (gh_m, p_pa, u_mps, v_mps, temp_k)
        sorted by gh.
        Pressure p_pa is derived from level label: '850h' -> 85000 Pa.
        """
        ts = data["ts"]
        out: List[Tuple[float, float, float, float, float]] = []

        for lvl in self.levels:
            ku, kv, kg, kt = f"wind_u-{lvl}", f"wind_v-{lvl}", f"gh-{lvl}", f"temp-{lvl}"
            if ku not in data or kv not in data or kg not in data or kt not in data:
                continue

            gh = self._interp_time(ts, data[kg], t_ms)  # meters
            u = self._interp_time(ts, data[ku], t_ms)   # m/s
            v = self._interp_time(ts, data[kv], t_ms)   # m/s
            temp = self._interp_time(ts, data[kt], t_ms)  # K

            # derive pressure from level string
            if lvl == "surface":
                # not a real pressure level. We keep it as "near-sfc" but can't use it for p interpolation reliably.
                # We'll include it for wind/temperature if you fly very low; pressure will be NaN and ignored.
                p = float("nan")
            else:
                # e.g. "850h" -> 850 hPa -> 85000 Pa
                hpa = float(lvl.replace("h", ""))
                p = hpa * 100.0

            out.append((gh, p, u, v, temp))

        out.sort(key=lambda x: x[0])
        return out

    def wind_temp_density(
        self, point: GeoPoint, alt_ft: float, t_ms: int
    ) -> Tuple[float, float, float, float]:
        """
        Returns (u_mps, v_mps, temp_k, rho_kgm3) at (lat,lon,alt,t),
        using vertical interpolation by geopotential height (gh).

        Pressure is interpolated between bracketing pressure levels using log-linear in height:
            ln p(h) linear between (gh0, p0) and (gh1, p1).
        """
        data = self.fetch(point.lat_deg, point.lon_deg)
        levels = self._samples_by_height(data, t_ms)
        if len(levels) < 2:
            raise WindyError("Not enough level samples for vertical interpolation")

        h_m = ft_to_m(alt_ft)

        # clamp to range
        if h_m <= levels[0][0]:
            gh, p, u, v, temp = levels[0]
            rho = _rho_from_levels(levels, h_m, temp)  # uses bracketed p if possible
            return u, v, temp, rho
        if h_m >= levels[-1][0]:
            gh, p, u, v, temp = levels[-1]
            rho = _rho_from_levels(levels, h_m, temp)
            return u, v, temp, rho

        # bracket
        for i in range(len(levels) - 1):
            gh0, p0, u0, v0, t0 = levels[i]
            gh1, p1, u1, v1, t1 = levels[i + 1]
            if gh0 <= h_m <= gh1:
                a = (h_m - gh0) / (gh1 - gh0) if gh1 != gh0 else 0.0
                u = u0*(1-a) + u1*a
                v = v0*(1-a) + v1*a
                temp = t0*(1-a) + t1*a
                rho = _rho_from_bracket(gh0, p0, gh1, p1, h_m, temp)
                return u, v, temp, rho

        # fallback
        gh, p, u, v, temp = levels[-1]
        rho = _rho_from_levels(levels, h_m, temp)
        return u, v, temp, rho


def _rho_from_bracket(gh0: float, p0: float, gh1: float, p1: float, h_m: float, temp_k: float) -> float:
    """
    Compute pressure at height by log-linear interpolation of p vs gh,
    then rho = p/(R*T). If p0/p1 invalid (NaN), fall back to NaN-safe behavior.
    """
    # if one is NaN (surface), try to use the other
    if not math.isfinite(p0) and math.isfinite(p1):
        p = p1
    elif math.isfinite(p0) and not math.isfinite(p1):
        p = p0
    elif not math.isfinite(p0) and not math.isfinite(p1):
        # no usable pressure; fall back to ISA sea-level density scaling (weak fallback)
        return max(0.1, min(2.0, RHO0))

    else:
        # log-linear: ln p interpolated in height
        if p0 <= 0.0 or p1 <= 0.0:
            p = max(p0, p1)
        else:
            a = (h_m - gh0) / (gh1 - gh0) if gh1 != gh0 else 0.0
            ln_p = math.log(p0)*(1-a) + math.log(p1)*a
            p = math.exp(ln_p)

    rho = p / (R_DRY_AIR * max(1.0, temp_k))
    return rho

def _rho_from_levels(levels: List[Tuple[float, float, float, float, float]], h_m: float, temp_k: float) -> float:
    # find nearest bracket with finite p
    for i in range(len(levels) - 1):
        gh0, p0, *_ = levels[i]
        gh1, p1, *_ = levels[i + 1]
        if gh0 <= h_m <= gh1:
            return _rho_from_bracket(gh0, p0, gh1, p1, h_m, temp_k)
    # fallback
    # choose nearest finite p
    finite_ps = [p for (_, p, *_ ) in levels if math.isfinite(p)]
    if not finite_ps:
        return RHO0
    p = finite_ps[-1]
    return p / (R_DRY_AIR * max(1.0, temp_k))


# -----------------------------
# Wind projection + Simpson time
# -----------------------------
def headwind_along_track(u_mps: float, v_mps: float, track_deg: float) -> float:
    """Along-track wind component (m/s), positive = tailwind."""
    chi = deg2rad(track_deg)
    return u_mps * math.sin(chi) + v_mps * math.cos(chi)

def ias_to_tas_mps(ias_kt: float, rho_kgm3: float) -> float:
    """IAS≈EAS -> TAS = IAS / sqrt(rho/rho0)."""
    sigma = max(1e-9, rho_kgm3 / RHO0)
    return kt_to_mps(ias_kt) / math.sqrt(sigma)

def simpson_time_seconds(L_m: float, gs0: float, gsm: float, gs1: float, min_gs_mps: float = 30.0) -> float:
    gs0 = max(min_gs_mps, gs0)
    gsm = max(min_gs_mps, gsm)
    gs1 = max(min_gs_mps, gs1)
    return (L_m / 6.0) * (1.0/gs0 + 4.0/gsm + 1.0/gs1)

def segment_time_simpson_windy(
    windy: WindyClient,
    start: GeoPoint,
    end: GeoPoint,
    track_deg: float,
    dist_nm: float,
    ias0_kt: float,
    iasm_kt: float,
    ias1_kt: float,
    alt0_ft: float,
    altm_ft: float,
    alt1_ft: float,
    t0_ms: int,
    iters: int = 2,
) -> Tuple[float, int, int]:
    """
    Samples start/mid/end with their own IAS + ALT + (time) and returns (T_s, tm_ms, t1_ms).
    Fixed-point iterates because wind/temp depend on time.
    """
    mid = interpolate_gc(start, end, 0.5)
    L_m = nm_to_m(dist_nm)

    # initial guess: mid point, no wind
    um, vm, tmK, rhom = windy.wind_temp_density(mid, altm_ft, t0_ms)
    tasm = ias_to_tas_mps(iasm_kt, rhom)
    T_s = L_m / max(1e-6, tasm)
    t1_ms = t0_ms + int(1000.0 * T_s)
    tm_ms = t0_ms + int(500.0 * T_s)

    for _ in range(max(1, iters)):
        u0, v0, _, rho0 = windy.wind_temp_density(start, alt0_ft, t0_ms)
        um, vm, _, rhom = windy.wind_temp_density(mid,   altm_ft, tm_ms)
        u1, v1, _, rho1 = windy.wind_temp_density(end,   alt1_ft, t1_ms)

        tas0 = ias_to_tas_mps(ias0_kt, rho0)
        tasm = ias_to_tas_mps(iasm_kt, rhom)
        tas1 = ias_to_tas_mps(ias1_kt, rho1)

        w0 = headwind_along_track(u0, v0, track_deg)
        wm = headwind_along_track(um, vm, track_deg)
        w1 = headwind_along_track(u1, v1, track_deg)

        gs0 = tas0 + w0
        gsm = tasm + wm
        gs1 = tas1 + w1

        T_s = simpson_time_seconds(L_m, gs0, gsm, gs1)
        t1_ms = t0_ms + int(1000.0 * T_s)
        tm_ms = t0_ms + int(500.0 * T_s)

    return T_s, tm_ms, t1_ms


# -----------------------------
# JSON -> route resolution
# -----------------------------
def waypoint_map(doc: dict) -> Dict[str, GeoPoint]:
    out: Dict[str, GeoPoint] = {}
    for w in doc.get("waypoints", []):
        lat = w.get("lat_deg", w.get("lat"))
        lon = w.get("lon_deg", w.get("lon"))
        out[w["id"]] = GeoPoint(float(lat), float(lon))
    return out

def runway_threshold(doc: dict, runway_id: str) -> Tuple[GeoPoint, float]:
    """
    Returns (threshold point, elevation_ft).
    Converts runway alt from meters->ft if it looks like meters.
    """
    for r in doc.get("runways", []):
        if r.get("id") == runway_id:
            pt = GeoPoint(float(r["lat"]), float(r["lon"]))
            alt_val = float(r.get("alt", 0.0))
            return pt, alt_val
    raise KeyError(f"Runway '{runway_id}' not found in runways[]")

def parse_itc_nm(fix_id: str) -> float:
    # "ITC_D0.4" -> 0.4 NM
    return float(fix_id.split("ITC_D", 1)[1])

def resolve_iap_transition_points(
    doc: dict,
    wps: Dict[str, GeoPoint],
    iap_id: str,
    transition: str,
) -> Tuple[dict, List[FixPoint]]:
    """
    Resolve an IAP transition into an ordered list of FixPoint.
    Supports fix ids:
      - normal fixes in waypoints[]
      - RWxx (runway threshold)
      - ITC_Dx (x NM from threshold along final inbound)
    """
    iap = next((x for x in doc.get("iaps", []) if x.get("id") == iap_id), None)
    if iap is None:
        raise KeyError(f"IAP '{iap_id}' not found")

    trans = iap.get("transitions", {})
    if transition not in trans:
        raise KeyError(f"Transition '{transition}' not in IAP '{iap_id}'. Available: {list(trans.keys())}")

    rwy_id = iap["rwy"]
    rw_pt, rw_elev_ft = runway_threshold(doc, rwy_id)

    legs = trans[transition]
    last_crs: Optional[float] = None
    last_alt: Optional[float] = None

    out: List[FixPoint] = []

    for leg in legs:
        fix_id = leg["id"]
        if leg.get("crs") is not None:
            last_crs = float(leg["crs"])
        if leg.get("alt") is not None:
            last_alt = float(leg["alt"])

        if fix_id in wps:
            out.append(FixPoint(fix_id, wps[fix_id], constraint_alt_ft=last_alt, inbound_crs_deg=last_crs))
            continue

        if fix_id.startswith("RW"):
            out.append(FixPoint(fix_id, rw_pt, constraint_alt_ft=rw_elev_ft, inbound_crs_deg=last_crs))
            continue

        if fix_id.startswith("ITC_D"):
            if last_crs is None:
                raise RuntimeError(f"ITC point '{fix_id}' appears before any inbound crs is known")
            d_nm = parse_itc_nm(fix_id)
            # From threshold, go outbound on reciprocal to place a point at DME d
            reciprocal = wrap360(last_crs + 180.0)
            pt = forward_spherical(rw_pt, reciprocal, d_nm)
            out.append(FixPoint(fix_id, pt, constraint_alt_ft=last_alt, inbound_crs_deg=last_crs))
            continue

        raise KeyError(f"IAP fix '{fix_id}' not resolvable (not waypoint, RW*, or ITC_D*)")

    return iap, out

def _reindex_segment_s(segs: List[SegmentGeom]) -> List[SegmentGeom]:
    """Recompute s0_nm/s1_nm cumulatively (needed after splitting)."""
    out: List[SegmentGeom] = []
    s = 0.0
    for seg in segs:
        out.append(SegmentGeom(
            a_id=seg.a_id, a_pt=seg.a_pt,
            b_id=seg.b_id, b_pt=seg.b_pt,
            dist_nm=seg.dist_nm, track_deg=seg.track_deg,
            s0_nm=s, s1_nm=s + seg.dist_nm
        ))
        s += seg.dist_nm
    return out

def _split_segment_gc(
    seg: SegmentGeom,
    *,
    split_if_gt_nm: float = 1.0,
    max_subseg_nm: float = 0.75,
) -> List[SegmentGeom]:
    """
    Split one segment along the great-circle if it's longer than split_if_gt_nm.
    Creates n = ceil(dist/max_subseg_nm) subsegments.
    """
    if seg.dist_nm <= split_if_gt_nm:
        return [seg]

    n = max(2, int(math.ceil(seg.dist_nm / max_subseg_nm)))

    # Generate interpolated points
    pts = [interpolate_gc(seg.a_pt, seg.b_pt, i / n) for i in range(n + 1)]

    # Generate synthetic ids for intermediate points (purely for reporting/debug)
    ids = [seg.a_id] + [f"{seg.b_id}_SUB{i}" for i in range(1, n)] + [seg.b_id]

    out: List[SegmentGeom] = []
    for i in range(n):
        d_nm, trk = gc_inverse_nm_bearing(pts[i], pts[i + 1])
        out.append(SegmentGeom(
            a_id=ids[i], a_pt=pts[i],
            b_id=ids[i + 1], b_pt=pts[i + 1],
            dist_nm=d_nm, track_deg=trk,
            s0_nm=0.0, s1_nm=0.0,  # will be reindexed later
        ))
    return out


def split_final_leg_to_threshold(
    segs: List[SegmentGeom],
    *,
    split_if_gt_nm: float = 1.0,
    max_subseg_nm: float = 0.75,
) -> List[SegmentGeom]:
    """
    If the last segment ends at a runway fix (b_id startswith 'RW') and is long,
    split it into subsegments to improve Simpson stability.
    """
    if not segs:
        return segs

    last = segs[-1]
    if not last.b_id.startswith("RW"):
        return segs

    if last.dist_nm <= split_if_gt_nm:
        return segs

    refined = segs[:-1] + _split_segment_gc(
        last, split_if_gt_nm=split_if_gt_nm, max_subseg_nm=max_subseg_nm
    )
    return _reindex_segment_s(refined)


def build_segments(points: List[FixPoint]) -> List[SegmentGeom]:
    """Turn ordered FixPoint list into SegmentGeom list with cumulative s."""
    if len(points) < 2:
        return []
    segs: List[SegmentGeom] = []
    s = 0.0
    for i in range(len(points) - 1):
        a = points[i]
        b = points[i + 1]
        dist_nm, track = gc_inverse_nm_bearing(a.point, b.point)
        segs.append(SegmentGeom(
            a_id=a.fix_id, a_pt=a.point,
            b_id=b.fix_id, b_pt=b.point,
            dist_nm=dist_nm, track_deg=track,
            s0_nm=s, s1_nm=s + dist_nm
        ))
        s += dist_nm
    return segs


# -----------------------------
# Profiles: IAS (decel) and ALT (constraint-hold then GP)
# -----------------------------
def make_speed_profile_ias(
    total_nm: float,
    v_entry_kt: float = 210.0,
    v_6dme_kt: float = 180.0,
    v_app_kt: float = 140.0,
    decel_kps: float = 0.5,     # knots per second, magnitude
    final_decel_from_dme_nm: float = 6.0,
):
    """
    IAS(s) with:
      - start at v_entry
      - reach v_6dme by distance-to-go = final_decel_from_dme_nm
      - from there, decelerate immediately toward v_app with constant decel
      - once v_app reached, hold v_app

    s is along-route distance from entry (NM).
    """
    a = -abs(decel_kps) * KT_TO_MPS  # m/s^2 (negative)

    total_m = total_nm * NM_TO_M
    s6_m = max(0.0, total_m - final_decel_from_dme_nm * NM_TO_M)  # position where DME-to-go hits 6

    v0 = v_entry_kt * KT_TO_MPS
    v1 = v_6dme_kt * KT_TO_MPS
    v2 = v_app_kt * KT_TO_MPS

    # Feasibility flags (optional, for debugging/logging)
    # Can we reach v_6dme by s6 if we decel immediately from entry?
    v_at_s6_if_immediate = math.sqrt(max(0.0, v0*v0 + 2.0*a*s6_m))
    reach_6dme = (v_at_s6_if_immediate <= v1 + 1e-9)

    # Can we reach v_app by threshold if we start decel at 6DME?
    v_at_thr_if_from6 = math.sqrt(max(0.0, v1*v1 + 2.0*a*(final_decel_from_dme_nm * NM_TO_M)))
    reach_vapp = (v_at_thr_if_from6 <= v2 + 1e-9)

    # Distance required to go v0 -> v1 at decel a
    def dist_needed(v_from: float, v_to: float) -> float:
        if v_to >= v_from:
            return 0.0
        return (v_to*v_to - v_from*v_from) / (2.0 * a)  # positive since a<0

    d01 = dist_needed(v0, v1)
    cruise_before6 = max(0.0, s6_m - d01)  # cruise at v_entry, then decel to hit v_6dme at s6 if possible

    def ias_at_s(s_nm: float) -> float:
        s_m = max(0.0, min(total_m, s_nm * NM_TO_M))

        # Before 6DME point: manage 210 -> 180 by s6 (cruise then decel)
        if s_m <= s6_m:
            if s_m <= cruise_before6:
                return v_entry_kt
            x = s_m - cruise_before6
            vv = v0*v0 + 2.0*a*x
            # Don’t allow going below v_6dme before the 6DME point:
            vv = max(v1*v1, vv)
            return math.sqrt(max(0.0, vv)) / KT_TO_MPS

        # After 6DME point: decelerate immediately from v_6dme toward v_app
        x = s_m - s6_m
        vv = v1*v1 + 2.0*a*x

        if reach_vapp:
            vv = max(v2*v2, vv)  # once you hit v_app, hold it
        # if not reachable, you’ll still be above v_app at threshold (no clamping)
        return math.sqrt(max(0.0, vv)) / KT_TO_MPS

    # attach debug info
    ias_at_s.reach_6dme = reach_6dme
    ias_at_s.reach_vapp = reach_vapp
    ias_at_s.v_at_s6_kt = v_at_s6_if_immediate / KT_TO_MPS
    ias_at_s.v_at_thr_from6_kt = v_at_thr_if_from6 / KT_TO_MPS
    return ias_at_s

def split_at_s_nm(segs: List[SegmentGeom], s_star_nm: float) -> List[SegmentGeom]:
    if not segs:
        return segs

    out: List[SegmentGeom] = []
    for seg in segs:
        if not (seg.s0_nm < s_star_nm < seg.s1_nm):
            out.append(seg)
            continue

        frac = (s_star_nm - seg.s0_nm) / max(1e-9, seg.dist_nm)
        mid_pt = interpolate_gc(seg.a_pt, seg.b_pt, frac)

        d1, trk1 = gc_inverse_nm_bearing(seg.a_pt, mid_pt)
        d2, trk2 = gc_inverse_nm_bearing(mid_pt, seg.b_pt)

        out.append(SegmentGeom(seg.a_id, seg.a_pt, f"{seg.b_id}_SPLIT", mid_pt, d1, trk1, 0.0, 0.0))
        out.append(SegmentGeom(f"{seg.b_id}_SPLIT", mid_pt, seg.b_id, seg.b_pt, d2, trk2, 0.0, 0.0))

    return _reindex_segment_s(out)

def glide_alt_ft(d_to_go_nm: float, gp_deg: float, rw_elev_ft: float) -> float:
    return rw_elev_ft + math.tan(deg2rad(gp_deg)) * d_to_go_nm * FT_PER_NM

def d_hit_nm_for_constraint(alt_ft: float, gp_deg: float, rw_elev_ft: float) -> float:
    den = max(1e-9, math.tan(deg2rad(gp_deg)) * FT_PER_NM)
    return max(0.0, (alt_ft - rw_elev_ft) / den)

def make_alt_profile_hold_then_gp(
    total_nm: float,
    gp_deg: float,
    rw_elev_ft: float,
    constraint_points: List[Tuple[float, float]],
) -> Callable[[float], float]:
    """
    constraint_points: list of (s_nm, alt_ft) in increasing s (along-route).
    At position s:
      d = total_nm - s
      h_c = last constraint altitude at/before s (if none, follow GP)
      d_hit = distance-to-go where GP equals h_c
      if d > d_hit: stay at h_c
      else: follow GP
    """
    constraint_points = sorted(constraint_points, key=lambda x: x[0])
    s_list = [s for s, _ in constraint_points]
    a_list = [a for _, a in constraint_points]

    def last_constraint_alt(s_nm: float) -> Optional[float]:
        j = bisect.bisect_right(s_list, s_nm) - 1
        return a_list[j] if j >= 0 else None

    def alt_at_s(s_nm: float) -> float:
        s_nm = max(0.0, min(total_nm, s_nm))
        d = total_nm - s_nm
        h_gp = glide_alt_ft(d, gp_deg, rw_elev_ft)

        h_c = last_constraint_alt(s_nm)
        if h_c is None:
            return h_gp

        d_hit = d_hit_nm_for_constraint(h_c, gp_deg, rw_elev_ft)
        return h_c if d > d_hit else h_gp

    return alt_at_s

def extract_constraint_points(points: List[FixPoint], segs: List[SegmentGeom]) -> List[Tuple[float, float]]:
    """
    Produce constraint points (s_nm at fix, alt_ft) from FixPoint.constraint_alt_ft.
    """
    # s at each fix: s of segment start for fix i, last fix is total_nm
    s_by_fix: Dict[str, float] = {}
    if not segs:
        return []
    s_by_fix[segs[0].a_id] = segs[0].s0_nm
    for seg in segs:
        s_by_fix[seg.b_id] = seg.s1_nm

    out: List[Tuple[float, float]] = []
    for fp in points:
        if fp.constraint_alt_ft is not None and fp.fix_id in s_by_fix:
            out.append((s_by_fix[fp.fix_id], float(fp.constraint_alt_ft)))
    out.sort(key=lambda x: x[0])
    return out


# -----------------------------
# End-to-end: IAP -> segment times
# -----------------------------
def split_at_s_nm(segs: List[SegmentGeom], s_star_nm: float) -> List[SegmentGeom]:
    if not segs:
        return segs

    out: List[SegmentGeom] = []
    for seg in segs:
        if not (seg.s0_nm < s_star_nm < seg.s1_nm):
            out.append(seg)
            continue

        frac = (s_star_nm - seg.s0_nm) / max(1e-9, seg.dist_nm)
        mid_pt = interpolate_gc(seg.a_pt, seg.b_pt, frac)

        d1, trk1 = gc_inverse_nm_bearing(seg.a_pt, mid_pt)
        d2, trk2 = gc_inverse_nm_bearing(mid_pt, seg.b_pt)

        out.append(SegmentGeom(seg.a_id, seg.a_pt, f"{seg.b_id}_SPLIT", mid_pt, d1, trk1, 0.0, 0.0))
        out.append(SegmentGeom(f"{seg.b_id}_SPLIT", mid_pt, seg.b_id, seg.b_pt, d2, trk2, 0.0, 0.0))

    return _reindex_segment_s(out)

def compute_iap_segment_times(
    doc: dict,
    windy: WindyClient,
    iap_id: str,
    transition: str,
    v_entry_kt: float = 210.0,
    v_6dme_kt: float = 180.0,
    v_app_kt: float = 140.0,
    decel_mps2: float = 0.05,
    t0_ms: Optional[int] = None,
    iters: int = 2,
    final_split_if_gt_nm: float = 1.0,   # NEW
    final_max_subseg_nm: float = 0.75,   # NEW (pick 0.5–1.0)
) -> List[SegmentTiming]:
    wps = waypoint_map(doc)
    iap, points = resolve_iap_transition_points(doc, wps, iap_id, transition)

    segs = build_segments(points)
    segs = split_final_leg_to_threshold(
        segs,
        split_if_gt_nm=final_split_if_gt_nm,
        max_subseg_nm=final_max_subseg_nm,
    )
    if not segs:
        return []

    total_nm = segs[-1].s1_nm
    segs = split_at_s_nm(segs, s_star_nm=total_nm - 6.0)
    gp_deg = float(iap.get("gp", 3.0))
    _, rw_elev_ft = runway_threshold(doc, iap["rwy"])

    ias_at_s = make_speed_profile_ias(total_nm, v_entry_kt, v_6dme_kt, v_app_kt, decel_mps2)
    constraints = extract_constraint_points(points, segs)
    alt_at_s = make_alt_profile_hold_then_gp(total_nm, gp_deg, rw_elev_ft, constraints)

    if t0_ms is None:
        t0_ms = int(time.time() * 1000)

    out: List[SegmentTiming] = []
    cur_t0 = t0_ms

    for seg in segs:
        s0, s1 = seg.s0_nm, seg.s1_nm
        sm = 0.5 * (s0 + s1)

        ias0 = float(ias_at_s(s0))
        iasm = float(ias_at_s(sm))
        ias1 = float(ias_at_s(s1))

        alt0 = float(alt_at_s(s0))
        altm = float(alt_at_s(sm))
        alt1 = float(alt_at_s(s1))

        T_s, _, _ = segment_time_simpson_windy(
            windy=windy,
            start=seg.a_pt,
            end=seg.b_pt,
            track_deg=seg.track_deg,
            dist_nm=seg.dist_nm,
            ias0_kt=ias0, iasm_kt=iasm, ias1_kt=ias1,
            alt0_ft=alt0, altm_ft=altm, alt1_ft=alt1,
            t0_ms=cur_t0,
            iters=iters,
        )

        out.append(SegmentTiming(
            seg=seg,
            ias0_kt=ias0, iasm_kt=iasm, ias1_kt=ias1,
            alt0_ft=alt0, altm_ft=altm, alt1_ft=alt1,
            time_s=float(T_s),
            t0_ms=int(cur_t0),
            t1_ms=int(cur_t0 + int(1000.0 * T_s)),
        ))
        cur_t0 = int(cur_t0 + int(1000.0 * T_s))

    return out

# -----------------------------
# Convenience runner
# -----------------------------
def print_iap_report(times: List[SegmentTiming]) -> None:
    if not times:
        print("No segments.")
        return

    total_s = sum(x.time_s for x in times)
    total_nm = times[-1].seg.s1_nm

    print(f"Segments: {len(times)}, total distance: {total_nm:.2f} NM, total time: {total_s:.1f}s ({total_s/60:.2f} min)")
    for x in times:
        seg = x.seg
        d0 = total_nm - seg.s0_nm
        d1 = total_nm - seg.s1_nm
        print(
            f"{seg.a_id:>10} -> {seg.b_id:<10} {seg.dist_nm:5.2f}NM hdg {seg.track_deg:6.1f}° "
            f"DME {d0:5.2f}->{d1:5.2f} "
            f"IAS {x.ias0_kt:5.0f}/{x.iasm_kt:5.0f}/{x.ias1_kt:5.0f} "
            f"ALT {x.alt0_ft:5.0f}/{x.altm_ft:5.0f}/{x.alt1_ft:5.0f} "
            f"{x.time_s:6.1f}s"
        )
