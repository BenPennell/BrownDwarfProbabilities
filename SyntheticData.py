# =============================
# Silence noisy deprecation warnings
# =============================
import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API"
)

warnings.filterwarnings(
    "ignore",
    message=".*unable to import Axes3D.*",
    category=UserWarning,
)

# =============================
# Standard imports
# =============================
import numpy as np
from functools import lru_cache
from joblib import Parallel, delayed, parallel
from contextlib import contextmanager
from tqdm.auto import tqdm
import json
import sys
import os

# =============================
# joblib + tqdm integration
# =============================

@contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar.
    """
    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

# =============================
# Configuration / Imports
# =============================

with open("config.json") as f:
    d = json.load(f)

folder_a_path = os.path.abspath(os.path.join(os.getcwd(), d["wrapperlitepath"]))
if folder_a_path not in sys.path:
    sys.path.append(folder_a_path)

import GaiamockWrapperLite as gw

# =============================
# Random utility functions
# =============================

def random_angle(n=1):
    x = np.random.rand(n) * 2 * np.pi
    return float(x[0]) if n == 1 else x


def random_inc(n=1):
    x = np.arccos(2 * np.random.rand(n) - 1)
    return float(x[0]) if n == 1 else x


def random_Tp(n=1):
    x = np.random.rand(n) - 0.5
    return float(x[0]) if n == 1 else x


# =============================
# Distributions
# =============================

def gaussian(x, mu, sigma):
    return np.exp(-(mu - x) ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)


def pexp(val, exp, val_range=(0, 1), ignore_a=False):
    a = 1
    if not ignore_a:
        a = (exp + 1) / (val_range[1] ** (exp + 1) - val_range[0] ** (exp + 1))
    return a * (val ** exp)


# =============================
# Eccentricity distributions
# =============================

es = np.linspace(0, 1, 100)

e_pdf = np.zeros_like(es)
e_pdf[1:] = pexp(es[1:], 1)
e_cdf = np.cumsum(e_pdf / np.sum(e_pdf))

p_therm = pexp(es, 1, val_range=(es[0], es[-1]))
p_therm /= np.sum(p_therm)

rayleigh_params = (0.38, 0.2)
p_gaus = gaussian(es, *rayleigh_params)
p_gaus /= np.sum(p_gaus)

periods_grid = np.linspace(1, 8, 100)
turnover_params = (3.5, 1)
turnover_pdf = gaussian(periods_grid, *turnover_params)
turnover_weight = np.cumsum(turnover_pdf / np.sum(turnover_pdf))


# =============================
# Eccentricity models
# =============================

def circular_e(*args):
    return 0.0


def thermal_e(*args):
    return np.interp(np.random.rand(), e_cdf, es)


def turnover_e(logP):
    w = turnover_weight[np.argmin(np.abs(periods_grid - logP))]
    dist = (1 - w) * p_gaus + w * p_therm
    cdf = np.cumsum(dist / np.sum(dist))
    return np.interp(np.random.rand(), cdf, es)


# =============================
# Gaia helpers
# =============================

c_funcs = gw.generate_cfuncs()

_GOST_NDP = 5  # RA/Dec rounding


@lru_cache(maxsize=None)
def _get_gost_cached(ra_r, dec_r):
    return gw.gaiamock.get_gost_one_position(
        ra_r, dec_r, data_release="dr3"
    )


def get_gost(ra, dec):
    return _get_gost_cached(
        round(float(ra), _GOST_NDP),
        round(float(dec), _GOST_NDP),
    )


# =============================
# Sampling helper
# =============================

def choose_value(cdf, grid, size):
    u = np.random.uniform(cdf.min(), cdf.max(), size)
    return np.interp(u, cdf, grid)


# =============================
# Parallel worker
# =============================

def solve_binary(
    period, q, ecc, inc, w, omega, Tp,
    ra, dec, pmra, pmdec, plx, mass, gmag
):
    t = get_gost(ra, dec)
    return gw.rapid_solution_type(
        period, q, plx, mass,
        gmag, 1e-10, ecc,
        inc, w, omega, Tp,
        ra, dec, pmra, pmdec,
        t, c_funcs
    )


# =============================
# Main generator
# =============================

def create_synthetic_data(
    object_count,
    catalogue,
    binary_fraction,
    mass_model=None,
    period_model=None,
    ecc_type="circular",
    m_lim=(0.017, 0.2),
    p_lim=(1, 8),
    p_resolution=100,
    verbose=True,
    n_jobs=-1,
    g=None
):

    ecc_func = {
        "circular": circular_e,
        "thermal": thermal_e,
        "turnover": turnover_e,
    }.get(ecc_type, circular_e)

    # --- binary mask ---
    binary_count = int(np.floor(object_count * binary_fraction))
    binary_mask = np.zeros(object_count, dtype=bool)
    binary_mask[:binary_count] = True
    np.random.shuffle(binary_mask)

    # --- select catalogue rows ---
    idx = np.random.choice(len(catalogue), object_count)

    ra = catalogue["ra"][idx].astype(float)
    dec = catalogue["dec"][idx].astype(float)
    pmra = catalogue["pmra"][idx].astype(float)
    pmdec = catalogue["pmdec"][idx].astype(float)
    plx = catalogue["parallax"][idx].astype(float)
    mass = catalogue["mass_single"][idx].astype(float)
    if g is None:
        gmag = catalogue["phot_g_mean_mag"][idx].astype(float)
    else:
        gmag = np.ones(object_count)*g # constant magnitude
    bprp = catalogue["bp_rp"][idx].astype(float)

    bin_idx = np.where(binary_mask)[0]
    nb = len(bin_idx)

    # --- periods ---
    if period_model is not None:
        mu, si = period_model
        ps = np.linspace(*p_lim, p_resolution)
        pdf = gaussian(ps, mu, si)
        cdf = np.cumsum(pdf / pdf.sum())
        logP = choose_value(cdf, ps, nb)
    else:
        logP = np.random.uniform(p_lim[0], p_lim[1], nb)

    period = 10 ** logP

    # --- mass ratios ---
    if mass_model is None:
        q = np.random.uniform(0.05, 0.5, nb)
    m2 = q * mass[bin_idx]
    bad = (m2 < m_lim[0]) | (m2 > m_lim[1])
    while np.any(bad):
        q[bad] = np.random.uniform(0.05, 0.5, bad.sum())
        m2[bad] = q[bad] * mass[bin_idx][bad]
        bad = (m2 < m_lim[0]) | (m2 > m_lim[1])

    ecc = np.array([ecc_func(lp) for lp in logP])
    inc = random_inc(nb)
    w = random_angle(nb)
    omega = random_angle(nb)
    Tp = random_Tp(nb) * period

    # =============================
    # Parallel solve with progress bar
    # =============================

    if verbose:
        pbar = tqdm(total=nb, desc="Computing Binaries")

    with tqdm_joblib(pbar if verbose else tqdm(disable=True)):
        results = Parallel(
            n_jobs=n_jobs,
            backend="loky"
        )(
            delayed(solve_binary)(
                period[b], q[b], ecc[b], inc[b], w[b], omega[b], Tp[b],
                ra[i], dec[i], pmra[i], pmdec[i],
                plx[i], mass[i], gmag[i]
            )
            for b, i in enumerate(bin_idx)
        )

    # =============================
    # Assemble output
    # =============================

    outdata = []
    b = 0
    for i in range(object_count):
        out = {
            "ra": ra[i],
            "dec": dec[i],
            "pmra": pmra[i],
            "pmdec": pmdec[i],
            "parallax": plx[i],
            "mass": mass[i],
            "phot_g_mean_mag": gmag[i],
            "bp_rp": bprp[i],
            "is_binary": bool(binary_mask[i]),
            "solution_type": 0,
        }

        if binary_mask[i]:
            out.update({
                "period": period[b],
                "m2": m2[b],
                "q": q[b],
                "ecc": ecc[b],
                "inc": inc[b],
                "w": w[b],
                "omega": omega[b],
                "Tp": Tp[b],
                "solution_type": results[b],
            })
            b += 1

        outdata.append(out)

    return np.array(outdata)
