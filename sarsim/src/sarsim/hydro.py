"""Hydrology-lite: bucket-model recession + acquisition-pair sampling (D1).

Urban floods live on hours; coherent Sentinel-1 pairs live on 6/12-day
repeat cycles. A corpus item is therefore an ACQUISITION PAIR whose flood
states come from a reduced-order hydrograph — the standard linear-reservoir
("bucket") abstraction, morphology-conditioned, closed-form. Full 2D
hydrodynamics (MIKE+/SWMM) is explicitly out of scope; the parameter ranges
below are order-of-magnitude anchors from drainage-engineering practice.

Class balance: scenario bins are STRATIFIED (training-friendly mix), and
each draw carries its `natural_weight` — the probability a uniformly-timed
acquisition would catch that state given the drawn hydrograph — so natural
frequencies can be recovered by reweighting. The stratification mirrors
UrbanSARFloods' own selection bias (events enter the dataset only when SAR
caught water); the natural weights quantify how often that actually happens
(a deployment-relevant number in itself).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# drainage capacity k_drain (mm/hr) by morphology class: engineered storm
# systems in dense cores vs partial drainage vs flat/deltaic ponding
K_DRAIN_MM_HR = {'dense': (20.0, 80.0),
                 'fragmented': (10.0, 50.0),
                 'flat': (2.0, 15.0)}
K_INFIL_MM_HR = (5.0, 20.0)        # pervious-soil infiltration
BETA_SLOPE = (20.0, 60.0)          # surface-drainage mm/hr per % slope
W_PEAK_M = (0.15, 1.0)             # street-flood peak depths
T_DRY_HR = (12.0, 72.0)            # wet-surface persistence after w -> 0
FLUVIAL_PROB = 0.3
TAU_FLUVIAL_DAYS = (0.5, 4.0)      # exponential recession constants

BIN_PROBS = {'near_peak': 0.30, 'mid_recession': 0.30,
             'receded_wet': 0.20, 'non_flood': 0.20}
BIN_RATIO = {'near_peak': (0.7, 1.0), 'mid_recession': (0.2, 0.7)}
FLOODED_T1_PROB = 0.02             # flooded->flooded (rare in urban)


def classify_morphology(height_mean_m: float, fill_prob: float,
                        slope_pct: float) -> str:
    if slope_pct < 0.2:
        return 'flat'
    if height_mean_m > 12.0 and fill_prob > 0.75:
        return 'dense'
    return 'fragmented'


@dataclass
class HydroParams:
    morphology: str
    event_type: str            # pluvial | fluvial
    k_total_mm_hr: float       # pluvial linear-bucket total drain rate
    tau_days: float            # fluvial exponential constant
    w_peak_m: float
    t_dry_hr: float

    @classmethod
    def draw(cls, rng: np.random.Generator, height_mean_m: float,
             fill_prob: float, slope_pct: float,
             pervious_frac: float) -> 'HydroParams':
        morph = classify_morphology(height_mean_m, fill_prob, slope_pct)
        k = (rng.uniform(*K_DRAIN_MM_HR[morph])
             + rng.uniform(*K_INFIL_MM_HR) * pervious_frac
             + rng.uniform(*BETA_SLOPE) * slope_pct)
        return cls(
            morphology=morph,
            event_type='fluvial' if rng.uniform() < FLUVIAL_PROB
                       else 'pluvial',
            k_total_mm_hr=float(k),
            tau_days=float(np.exp(rng.uniform(*np.log(TAU_FLUVIAL_DAYS)))),
            w_peak_m=float(rng.uniform(*W_PEAK_M)),
            t_dry_hr=float(rng.uniform(*T_DRY_HR)))

    def water_level(self, t_hr: float) -> float:
        """Depth above the peak-flood waterline datum, t hours after peak."""
        if self.event_type == 'fluvial':
            return self.w_peak_m * float(np.exp(-t_hr / (self.tau_days * 24)))
        return max(self.w_peak_m - self.k_total_mm_hr * t_hr / 1000.0, 0.0)

    def time_at_ratio(self, r: float) -> float:
        """Hours after peak when w/w_peak = r (inverse of water_level)."""
        if self.event_type == 'fluvial':
            return -float(np.log(max(r, 1e-9))) * self.tau_days * 24
        return self.w_peak_m * (1.0 - r) * 1000.0 / self.k_total_mm_hr


@dataclass
class PairSpec:
    """One acquisition pair. States: dry | wet | flooded. w_* are water
    levels RELATIVE to the peak-flood waterline datum (w=0 means no
    standing water); w2_frac = w2/w_peak for mask construction."""
    state_t1: str
    state_t2: str
    w1_frac: float
    w2_frac: float
    has_flood: bool            # any standing water at t2
    scenario: str
    dt_hr: float               # peak -> t2 offset (flood bins)
    revisit_days: int
    natural_weight: float
    hydro: dict

    def to_meta(self) -> dict:
        return asdict(self)


def sample_pair(rng: np.random.Generator, hydro: HydroParams) -> PairSpec:
    revisit = int(rng.choice([6, 12]))
    window_hr = revisit * 24.0
    # prior surface wetness at t1 (monsoon-ish climates rain before floods)
    wet_t1 = bool(rng.uniform() < rng.uniform(0.1, 0.6))

    bin_name = str(rng.choice(list(BIN_PROBS), p=list(BIN_PROBS.values())))

    if bin_name == 'non_flood':
        s1, s2 = {0: ('dry', 'wet'), 1: ('wet', 'wet'), 2: ('dry', 'dry')}[
            int(rng.choice(3, p=[0.4, 0.3, 0.3]))]
        return PairSpec(state_t1=s1, state_t2=s2, w1_frac=0.0, w2_frac=0.0,
                        has_flood=False, scenario=f'{s1}->{s2}',
                        dt_hr=float('nan'), revisit_days=revisit,
                        natural_weight=1.0, hydro=asdict(hydro))

    if bin_name == 'receded_wet':
        # standing water gone at t2; previously-flooded area is saturated
        t0 = hydro.time_at_ratio(0.02)          # ~fully receded
        dt = float(t0 + rng.uniform(0, hydro.t_dry_hr))
        w = min(hydro.t_dry_hr / window_hr, 1.0)
        return PairSpec(state_t1='wet' if wet_t1 else 'dry', state_t2='wet',
                        w1_frac=0.0, w2_frac=0.0, has_flood=False,
                        scenario='receded_wet', dt_hr=dt,
                        revisit_days=revisit, natural_weight=float(w),
                        hydro=asdict(hydro))

    r_lo, r_hi = BIN_RATIO[bin_name]
    r = float(rng.uniform(r_lo, r_hi))
    dt = hydro.time_at_ratio(r)
    # natural weight: fraction of the revisit window during which the
    # hydrograph sits inside this bin's depth-ratio band
    t_hi, t_lo = hydro.time_at_ratio(r_lo), hydro.time_at_ratio(r_hi)
    w = float(np.clip((t_hi - t_lo) / window_hr, 0.0, 1.0))
    w1 = 0.0
    s1 = 'wet' if wet_t1 else 'dry'
    if rng.uniform() < FLOODED_T1_PROB:
        s1, w1 = 'flooded', float(rng.uniform(0.1, 0.5))
    return PairSpec(state_t1=s1, state_t2='flooded', w1_frac=w1,
                    w2_frac=r, has_flood=True,
                    scenario=f'{s1}->flooded@{bin_name}', dt_hr=float(dt),
                    revisit_days=revisit, natural_weight=w,
                    hydro=asdict(hydro))


def natural_catch_rate(rng: np.random.Generator, n: int = 4000,
                       height_mean_m: float = 15.0, fill_prob: float = 0.85,
                       slope_pct: float = 0.4,
                       pervious_frac: float = 0.15) -> float:
    """Deployment number: P(a uniformly-timed acquisition within one revisit
    window after peak sees ANY standing water) for a morphology draw."""
    hits = 0
    for _ in range(n):
        h = HydroParams.draw(rng, height_mean_m, fill_prob, slope_pct,
                             pervious_frac)
        revisit = rng.choice([6, 12])
        t = rng.uniform(0, revisit * 24.0)
        hits += h.water_level(t) > 0.01
    return hits / n
