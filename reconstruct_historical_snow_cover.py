# -*- coding: utf-8 -*-
"""Reconstruct monthly historical snow cover fraction at 0.25° resolution.

For each target grid cell and calendar month, STIM-Snow identifies same-month
donor observations from 2001–2023 using monthly mean temperature similarity.
The temperature tolerance starts at ±2 °C and increases by a factor of 1.5
when fewer than 250 donors are available, up to a maximum of ±5 °C. Donors
within the final tolerance are weighted by composite climate distance, mean
temperature difference and latitude difference. The weighted samples are used
to fit a local ridge regression for the target grid cell.

The workflow performs year-stratified training, validation and testing,
rebuilds the final donor libraries from the complete 2001–2023 observation
period, reconstructs monthly snow cover fraction for 1901–2000, and exports a
continuous 1700–2023 product. Monthly 1901 values are repeated for 1700–1900,
consistent with the reconstruction protocol described in the manuscript.

Predictions remain missing when no same-month donor satisfies the maximum
temperature tolerance.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import re
import glob
import time
import gc
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.model_selection import StratifiedShuffleSplit
from joblib import Parallel, delayed, dump

warnings.filterwarnings("ignore")

# =========================================================
# Execution modes
# =========================================================
PREDICT_FULL_FIELDS = False         # Full-field predictions for evaluation years
PREDICT_HISTORICAL_FIELDS = True    # Reconstruct 1901–2000 using all observations
SAVE_LOCAL_INFO = False             # Save donor count and selected temperature tolerance

# Only monthly snow cover fraction (areaFrac) is required by the manuscript.
HISTORICAL_TARGETS = ("areaFrac",)
HIST_START_YEAR = 1901
HIST_END_YEAR = 2000
HIST_MONTHS = tuple(range(1, 13))
SKIP_EXISTING_HISTORICAL = True
TARGETS_TO_PROCESS = ("areaFrac",)
EXPORT_UNIFIED_YEARLY_MAT = True
UNIFIED_START_YEAR = 1700
UNIFIED_END_YEAR = 2023
PRE_1901_REFERENCE_YEAR = 1901

# =========================================================
# Paths and core settings
# =========================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("HISTSNOW_DATA_DIR", os.path.join(PROJECT_DIR, "data"))
CRU_FILES = {
    key: os.path.join(
        DATA_DIR,
        "cru_ts_v4.08_0p25deg",
        f"cru_ts4.08.1901.2023.{key}.dat_0p25deg.nc",
    )
    for key in ("pre", "tmn", "tmp", "tmx", "wet")
}

MODIS_MONTHLY_MAT_DIR = os.path.join(DATA_DIR, "modis_monthly_scf_0p25deg")

OUT_BASE = os.environ.get(
    "HISTSNOW_OUTPUT_DIR",
    os.path.join(PROJECT_DIR, "outputs", "historical_snow_reconstruction"),
)

os.makedirs(OUT_BASE, exist_ok=True)
OUT_METRICS_DIR = os.path.join(OUT_BASE, "metrics")
OUT_PLOTS_DIR = os.path.join(OUT_BASE, "plots_hexbin")
OUT_DONOR_DIR = os.path.join(OUT_BASE, "donor_libraries")
OUT_FULL_DIR = os.path.join(OUT_BASE, "fullfield_predictions")
OUT_HIST_DIR = os.path.join(OUT_BASE, "historical_predictions_1901_2000")
OUT_UNIFIED_SCF_DIR = os.path.join(OUT_BASE, "snowfrac_025_1700_2023")
os.makedirs(OUT_METRICS_DIR, exist_ok=True)
os.makedirs(OUT_PLOTS_DIR, exist_ok=True)
os.makedirs(OUT_DONOR_DIR, exist_ok=True)
os.makedirs(OUT_FULL_DIR, exist_ok=True)
os.makedirs(OUT_HIST_DIR, exist_ok=True)
os.makedirs(OUT_UNIFIED_SCF_DIR, exist_ok=True)

TARGET_KEYS = {
    "daysFrac": "snow_days_frac_025",
    "areaFrac": "snow_cover_frac_025",
}

TRAIN_START_YEAR = 2001
TRAIN_END_YEAR = 2023
SUP_YEARS = list(range(TRAIN_START_YEAR, TRAIN_END_YEAR + 1))

N_JOBS = int(os.environ.get("HISTSNOW_N_JOBS", "12"))
RANDOM_SEED = 42
EPS_POS = 1e-4
CLIP_01 = True

BASE_FEATS = ["pre", "tmn", "tmp", "tmx", "wet"]
FEAT_MODE = "cur"                         # Five climate variables in the current month

LAT_ARR = np.linspace(90 - 0.125, -90 + 0.125, 720).astype(np.float32)
LON_ARR = np.linspace(-180 + 0.125, 180 - 0.125, 1440).astype(np.float32)

# =========================================================
# Donor-library settings
# =========================================================
def _optional_positive_int_env(name):
    value = int(os.environ.get(name, "0"))
    return value if value > 0 else None


# The manuscript method uses all available same-month observations. Optional
# caps are disabled by default and are provided only for constrained systems.
MAX_DONORS_PER_YEAR_MONTH = _optional_positive_int_env(
    "HISTSNOW_MAX_DONORS_PER_YEAR_MONTH"
)
MAX_TOTAL_DONORS_PER_MONTH = _optional_positive_int_env(
    "HISTSNOW_MAX_TOTAL_DONORS_PER_MONTH"
)
SAVE_DONOR_LIBRARY = True

# =========================================================
# Local donor-selection and regression settings
# =========================================================
TEMP_THRESH_INIT_C = 2.0
TEMP_THRESH_MAX_C = 5.0
TEMP_THRESH_GROW = 1.5
MIN_DONORS = 250
RIDGE_LAMBDA = 2.0
MIN_WEIGHT_SUM = 1e-8

SIGMA_CLIM = 2.5
SIGMA_TEMP_C = 2.0
SIGMA_LAT_DEG = 6.0

# Composite climate distance in Equation S3 excludes monthly mean temperature,
# which enters the weighting function as a separate term.
CLIM_DISTANCE_FEATURES = ("pre", "tmn", "tmx", "wet")
CLIM_DISTANCE_INDICES = np.array(
    [BASE_FEATS.index(name) for name in CLIM_DISTANCE_FEATURES],
    dtype=np.int64,
)

# =========================================================
# Evaluation settings
# =========================================================
SCATTER_PER_YEAR_MONTH = 3000
SCATTER_MAX_TOTAL = 200000
EVAL_METRICS_USE_SAMPLED_POINTS = True

# Full-field block-parallel settings
FULLFIELD_BLOCK_SIZE = 4000
FULLFIELD_N_JOBS = int(os.environ.get("HISTSNOW_FULLFIELD_N_JOBS", str(N_JOBS)))
FULLFIELD_VERBOSE = 10

# =========================================================
# Plot style
# =========================================================
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.serif": ["Times New Roman"],
    "axes.unicode_minus": False,
    "font.size": 16,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

# =========================================================
# Donor libraries use plain dictionaries for portable joblib serialization.
# =========================================================
def make_donor_library(X, y, tmp_cur, lat, feat_mean, feat_std, tmp_sorted_idx, tmp_sorted_vals):
    return {
        "X": X,
        "y": y,
        "tmp_cur": tmp_cur,
        "lat": lat,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "tmp_sorted_idx": tmp_sorted_idx,
        "tmp_sorted_vals": tmp_sorted_vals,
    }

# =========================================================
# Utilities
# =========================================================
def resolve_monthly_mat_fp(ym_to_fp, year, month, mat_dir):
    fp = ym_to_fp.get((year, month))
    if fp is not None:
        return fp
    raise RuntimeError(
        f"Missing monthly MODIS SCF MAT file for {year}-{month:02d} in {mat_dir}. "
        "The manuscript workflow requires a complete 2001–2023 monthly record."
    )


def sel_year_month(ds_or_da, year, month):
    t = np.datetime64(f"{year}-{month:02d}-15")
    try:
        out = ds_or_da.sel(time=t, method="nearest")
    except Exception:
        return None
    dt = np.abs(out["time"].values - t)
    if dt > np.timedelta64(20, "D"):
        return None
    return out


def prev_year_month(year, month):
    if month == 1:
        return year - 1, 12
    return year, month - 1


def get_cur_and_lag1(cru, year, month):
    cur = sel_year_month(cru, year, month)
    py, pm = prev_year_month(year, month)
    lag = sel_year_month(cru, py, pm)
    if lag is None:
        lag = cur
    return cur, lag


def _nearest_index_1d(src, tgt, tol=1e-3):
    src = np.asarray(src, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    idx = np.empty(tgt.shape[0], dtype=np.int64)
    for i, v in enumerate(tgt):
        j = int(np.argmin(np.abs(src - v)))
        idx[i] = j
    max_abs = float(np.max(np.abs(src[idx] - tgt)))
    if max_abs > tol:
        raise RuntimeError(
            f"Grid mismatch too large: max|src-tgt|={max_abs} > tol={tol}. "
            f"Please check CRU lat/lon definition vs LAT_ARR/LON_ARR."
        )
    return idx


def _standardize_lon(lon):
    lon = np.asarray(lon, dtype=np.float64)
    if np.nanmax(lon) > 180.0 + 1e-6:
        lon2 = ((lon + 180.0) % 360.0) - 180.0
        return lon2
    return lon


def read_cru_to_target_grid():
    das = {}
    for var, path in CRU_FILES.items():
        ds = xr.open_dataset(path)
        da = ds[list(ds.data_vars)[0]]

        lat_name = "lat" if "lat" in da.coords else ("latitude" if "latitude" in da.coords else None)
        lon_name = "lon" if "lon" in da.coords else ("longitude" if "longitude" in da.coords else None)
        if lat_name is None or lon_name is None:
            raise RuntimeError(f"Cannot find lat/lon coords in {path}. coords={list(da.coords)}")

        lat_src = da[lat_name].values.astype(np.float64)
        lon_src = da[lon_name].values.astype(np.float64)

        lon_std = _standardize_lon(lon_src)
        lon_order = np.argsort(lon_std)
        lon_std_sorted = lon_std[lon_order]

        lat_desc = lat_src
        lat_flip = False
        if lat_src[0] < lat_src[-1]:
            lat_desc = lat_src[::-1]
            lat_flip = True

        vals = da.values.astype(np.float32)

        if lat_flip:
            vals = vals[:, ::-1, :]
        vals = vals[:, :, lon_order]

        lat_idx = _nearest_index_1d(lat_desc, LAT_ARR, tol=1e-3)
        lon_idx = _nearest_index_1d(lon_std_sorted, LON_ARR, tol=1e-3)

        vals_final = vals[:, lat_idx, :][:, :, lon_idx]

        das[var] = xr.DataArray(
            vals_final,
            dims=("time", "lat", "lon"),
            coords={"time": da["time"].values, "lat": LAT_ARR, "lon": LON_ARR},
        )
        ds.close()

    cru = xr.Dataset(das)
    if cru.sizes.get("lat", None) != 720 or cru.sizes.get("lon", None) != 1440:
        raise RuntimeError(f"CRU grid unexpected: lat={cru.sizes.get('lat')} lon={cru.sizes.get('lon')}")
    return cru


def _read_mat_array(fp, key):
    try:
        with h5py.File(fp, "r") as f:
            if key in f:
                return np.array(f[key])
            raise KeyError(f"Key '{key}' not found in {fp}. Available keys: {list(f.keys())}")
    except (OSError, KeyError):
        pass

    try:
        from scipy.io import loadmat
        m = loadmat(fp)
        if key in m:
            return m[key]
        raise KeyError(f"Key '{key}' not found in {fp}. Available keys: {list(m.keys())}")
    except Exception as e:
        raise RuntimeError(f"Failed to read MAT '{fp}' key='{key}': {e}")


def _fix_to_720_1440(arr):
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expect 2D after squeeze, got shape={arr.shape}")
    if arr.shape == (720, 1440):
        return arr.astype(np.float32, copy=False)
    if arr.shape == (1440, 720):
        return arr.T.astype(np.float32, copy=False)
    raise ValueError(f"Cannot reshape to (720,1440), got {arr.shape}")


def build_year_month_to_matfile_map(mat_dir):
    files = sorted(glob.glob(os.path.join(mat_dir, "*.mat")))
    mp = {}
    for fp in files:
        bn = os.path.basename(fp)
        m = re.search(r"(\d{4})[_-](\d{2})", bn)
        if not m:
            continue
        y = int(m.group(1))
        mm = int(m.group(2))
        if 1 <= mm <= 12:
            mp[(y, mm)] = fp
    return mp


def load_modis_targets_as_da_monthly(target_key, ym_to_fp, years):
    all_data, all_times = [], []
    for y in years:
        for mm in range(1, 13):
            fp = resolve_monthly_mat_fp(ym_to_fp, y, mm, MODIS_MONTHLY_MAT_DIR)
            arr = _read_mat_array(fp, target_key)
            arr = _fix_to_720_1440(arr)
            arr = np.clip(arr, 0.0, 1.0)
            all_data.append(arr)
            all_times.append(np.datetime64(f"{y}-{mm:02d}-15"))

    return xr.DataArray(
        np.stack(all_data, axis=0),
        dims=("time", "lat", "lon"),
        coords={"time": all_times, "lat": LAT_ARR, "lon": LON_ARR},
        name=target_key,
    )


# =========================================================
# Year-stratified training, validation and testing
# =========================================================
STRAT_N_PIX = 20000
STRAT_BINS = 5
RATIO_TEST = 0.2
RATIO_VAL = 0.1

def _compute_year_strata(y_da, years, n_pix=20000, n_bins=5, seed=42):
    rng = np.random.default_rng(seed)
    years_in_data = np.unique(pd.to_datetime(y_da["time"].values).year).astype(int).tolist()
    years = np.array(sorted([y for y in years if y in years_in_data]), dtype=int)
    if years.size == 0:
        raise RuntimeError("No valid years in y_da for stratification.")

    n_total_pix = int(y_da.sizes["lat"] * y_da.sizes["lon"])
    n_pix = int(min(n_pix, n_total_pix))
    pix_idx = rng.choice(n_total_pix, size=n_pix, replace=False)

    scores = []
    for y in years:
        da_y = y_da.sel(time=str(y))
        arr = da_y.values.reshape(da_y.sizes["time"], -1)[:, pix_idx]
        valid = np.isfinite(arr)
        denom = int(valid.sum())
        score = float(((arr > EPS_POS) & valid).sum() / denom) if denom > 0 else 0.0
        scores.append(score)
    scores = np.array(scores, dtype=np.float32)

    edges = np.unique(np.quantile(scores, np.linspace(0, 1, n_bins + 1)))
    if len(edges) <= 2:
        strata = np.zeros_like(scores, dtype=int)
    else:
        strata = np.digitize(scores, edges[1:-1], right=True).astype(int)
    return years, strata


def _safe_stratified_split(X, y, test_size, random_state):
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(X)
    if n < 3:
        idx = np.arange(n)
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)
        n_test = max(1, int(round(test_size * n))) if isinstance(test_size, float) else int(test_size)
        n_test = min(max(1, n_test), n - 1)
        return idx[n_test:], idx[:n_test]

    if isinstance(test_size, float):
        n_test = int(np.ceil(test_size * n))
    else:
        n_test = int(test_size)
    n_test = min(max(1, n_test), n - 1)

    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    if n_test < n_classes:
        idx = np.arange(n)
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)
        return idx[n_test:], idx[:n_test]

    if np.any(counts < 2):
        most_common = classes[np.argmax(counts)]
        y2 = y.copy()
        for c, ct in zip(classes, counts):
            if ct < 2:
                y2[y2 == c] = most_common
        y = y2
        classes, counts = np.unique(y, return_counts=True)
        n_classes = len(classes)
        if n_test < n_classes:
            idx = np.arange(n)
            rng = np.random.default_rng(random_state)
            rng.shuffle(idx)
            return idx[n_test:], idx[:n_test]

    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=random_state)
        idx_train, idx_test = next(sss.split(X, y))
        return idx_train, idx_test
    except Exception:
        idx = np.arange(n)
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)
        return idx[n_test:], idx[:n_test]


def split_years_stratified_train_val_test(y_da, years, seed=42):
    years_arr, strata = _compute_year_strata(y_da, years, n_pix=STRAT_N_PIX, n_bins=STRAT_BINS, seed=seed)

    n_years = len(years_arr)
    n_test = max(1, int(round(RATIO_TEST * n_years)))
    n_val = max(1, int(round(RATIO_VAL * n_years)))
    if n_test + n_val >= n_years:
        raise RuntimeError(
            "Not enough years to form non-empty training, validation and test sets."
        )

    idx_trainval, idx_test = _safe_stratified_split(
        X=years_arr, y=strata, test_size=n_test, random_state=seed
    )
    years_trainval = years_arr[idx_trainval]
    strata_trainval = strata[idx_trainval]
    years_test = years_arr[idx_test]

    idx_train, idx_val = _safe_stratified_split(
        X=years_trainval,
        y=strata_trainval,
        test_size=n_val,
        random_state=seed + 1,
    )
    years_train = years_trainval[idx_train]
    years_val = years_trainval[idx_val]

    return sorted(years_train.tolist()), sorted(years_val.tolist()), sorted(years_test.tolist())


# =========================================================
# Feature construction
# =========================================================
def build_flat_feats(cur_time, lag_time):
    if FEAT_MODE == "cur":
        sources = [cur_time]
    elif FEAT_MODE == "lag1":
        sources = [lag_time]
    elif FEAT_MODE == "cur_lag1":
        sources = [cur_time, lag_time]
    else:
        raise ValueError(f"Unknown FEAT_MODE={FEAT_MODE}")

    flat = []
    for src in sources:
        for v in BASE_FEATS:
            flat.append(src[v].values.astype(np.float32).reshape(-1))
    return flat


def feature_names():
    names = []
    if FEAT_MODE == "cur":
        sources = ["cur"]
    elif FEAT_MODE == "lag1":
        sources = ["lag1"]
    elif FEAT_MODE == "cur_lag1":
        sources = ["cur", "lag1"]
    else:
        raise ValueError(FEAT_MODE)
    for s in sources:
        for v in BASE_FEATS:
            names.append(f"{s}_{v}")
    return names


def stacked_feature_matrix_from_flat(flat_feats, idx):
    return np.stack([f[idx] for f in flat_feats], axis=1).astype(np.float32, copy=False)


# =========================================================
# Donor-library construction
# =========================================================
def build_single_month_donor_library(month, cru, y_da, train_years, target_name):
    rng = np.random.default_rng(RANDOM_SEED + 1000 + month)

    X_list, y_list, tmp_list, lat_list = [], [], [], []

    lat_grid = np.repeat(LAT_ARR[:, None], 1440, axis=1).reshape(-1).astype(np.float32)

    for year in train_years:
        cur_time, lag_time = get_cur_and_lag1(cru, year, month)
        y_time = sel_year_month(y_da, year, month)
        if cur_time is None or y_time is None:
            continue
        if lag_time is None:
            lag_time = cur_time

        y_full = y_time.values.astype(np.float32).reshape(-1)
        flat_feats = build_flat_feats(cur_time, lag_time)
        tmp_cur = cur_time["tmp"].values.astype(np.float32).reshape(-1)

        valid = np.isfinite(y_full) & np.isfinite(tmp_cur)
        for f in flat_feats:
            valid &= np.isfinite(f)

        idx = np.where(valid)[0]
        if idx.size == 0:
            continue

        if (
            MAX_DONORS_PER_YEAR_MONTH is not None
            and idx.size > MAX_DONORS_PER_YEAR_MONTH
        ):
            idx = rng.choice(idx, size=MAX_DONORS_PER_YEAR_MONTH, replace=False)

        X = stacked_feature_matrix_from_flat(flat_feats, idx)
        y = y_full[idx]
        t = tmp_cur[idx]
        latv = lat_grid[idx]

        X_list.append(X)
        y_list.append(y)
        tmp_list.append(t)
        lat_list.append(latv)

    if not X_list:
        raise RuntimeError(f"[{target_name}] month={month} donor library empty.")

    X = np.vstack(X_list).astype(np.float32, copy=False)
    y = np.hstack(y_list).astype(np.float32, copy=False)
    tmp_cur = np.hstack(tmp_list).astype(np.float32, copy=False)
    latv = np.hstack(lat_list).astype(np.float32, copy=False)

    if (
        MAX_TOTAL_DONORS_PER_MONTH is not None
        and X.shape[0] > MAX_TOTAL_DONORS_PER_MONTH
    ):
        keep = rng.choice(X.shape[0], size=MAX_TOTAL_DONORS_PER_MONTH, replace=False)
        X = X[keep]
        y = y[keep]
        tmp_cur = tmp_cur[keep]
        latv = latv[keep]

    feat_mean = np.nanmean(X, axis=0).astype(np.float32)
    feat_std = np.nanstd(X, axis=0).astype(np.float32)
    feat_std[feat_std < 1e-6] = 1.0

    sort_idx = np.argsort(tmp_cur)
    tmp_sorted_vals = tmp_cur[sort_idx]

    lib = make_donor_library(
        X=X,
        y=y,
        tmp_cur=tmp_cur,
        lat=latv,
        feat_mean=feat_mean,
        feat_std=feat_std,
        tmp_sorted_idx=sort_idx,
        tmp_sorted_vals=tmp_sorted_vals
    )

    pos_ratio = float((y > EPS_POS).mean())
    print(f"[Donor] {target_name} month={month:02d} n={X.shape[0]} pos_ratio={pos_ratio:.4f}")
    return month, lib


def build_all_month_donor_libraries(cru, y_da, train_years, target_name):
    res = Parallel(n_jobs=min(12, N_JOBS), backend="loky", verbose=10)(
        delayed(build_single_month_donor_library)(m, cru, y_da, train_years, target_name)
        for m in range(1, 13)
    )
    libs = {m: lib for (m, lib) in res}

    if SAVE_DONOR_LIBRARY:
        for m, lib in libs.items():
            dump(lib, os.path.join(OUT_DONOR_DIR, f"donor_library_{target_name}_month{m:02d}.joblib"))

    return libs


# =========================================================
# Local donor search, weighting and ridge regression
# =========================================================
def _sorted_tmp_candidate_idx(lib, tmp_q, temp_thr):
    left = np.searchsorted(lib["tmp_sorted_vals"], tmp_q - temp_thr, side="left")
    right = np.searchsorted(lib["tmp_sorted_vals"], tmp_q + temp_thr, side="right")
    if right <= left:
        return np.empty((0,), dtype=np.int64)
    return lib["tmp_sorted_idx"][left:right]


def _select_local_donors(lib, tmp_q):
    """Select same-month donors using the adaptive temperature threshold."""
    temp_thr = TEMP_THRESH_INIT_C
    cand = np.empty((0,), dtype=np.int64)

    while True:
        cand = _sorted_tmp_candidate_idx(lib, tmp_q, temp_thr)
        if cand.size >= MIN_DONORS or temp_thr >= TEMP_THRESH_MAX_C:
            break

        temp_thr = min(TEMP_THRESH_MAX_C, temp_thr * TEMP_THRESH_GROW)

    # If fewer than MIN_DONORS remain at ±5 °C, all available donors within
    # that maximum threshold are retained, as described in Supplementary S1.3.
    return cand, float(temp_thr)


def _kernel_weights(lib, cand_idx, xq, tmp_q, lat_q):
    Xc = lib["X"][cand_idx]
    tc = lib["tmp_cur"][cand_idx]
    lc = lib["lat"][cand_idx]

    clim_idx = CLIM_DISTANCE_INDICES
    clim_mean = lib["feat_mean"][clim_idx]
    clim_std = lib["feat_std"][clim_idx]
    xqz = (xq[clim_idx] - clim_mean) / clim_std
    Xcz = (Xc[:, clim_idx] - clim_mean[None, :]) / clim_std[None, :]

    # Equation S3 defines d_clim as the Euclidean distance across the four
    # standardized climate variables: pre, tmn, tmx and wet.
    dclim = np.sqrt(np.sum((Xcz - xqz[None, :]) ** 2, axis=1))
    dtmp = np.abs(tc - tmp_q)
    dlat = np.abs(lc - lat_q)

    w_clim = np.exp(-0.5 * (dclim / SIGMA_CLIM) ** 2)
    w_temp = np.exp(-0.5 * (dtmp / SIGMA_TEMP_C) ** 2)
    w_lat = np.exp(-0.5 * (dlat / SIGMA_LAT_DEG) ** 2)

    return (w_clim * w_temp * w_lat).astype(np.float64)


def _fit_weighted_ridge_predict(X_train, y_train, w_train, xq, lam=RIDGE_LAMBDA):
    mu = np.nanmean(X_train, axis=0)
    sd = np.nanstd(X_train, axis=0)
    sd[sd < 1e-6] = 1.0

    Xs = (X_train - mu) / sd
    xq_s = (xq - mu) / sd

    Xd = np.concatenate([np.ones((Xs.shape[0], 1), dtype=np.float64), Xs.astype(np.float64)], axis=1)
    xqd = np.concatenate([[1.0], xq_s.astype(np.float64)])

    w = np.asarray(w_train, dtype=np.float64)
    if np.sum(w) < MIN_WEIGHT_SUM:
        return float(np.average(y_train))

    WX = Xd * w[:, None]
    A = Xd.T @ WX
    b = Xd.T @ (w * y_train.astype(np.float64))

    reg = np.eye(A.shape[0], dtype=np.float64) * lam
    reg[0, 0] = 0.0
    A = A + reg

    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(A, b, rcond=None)[0]

    return float(xqd @ beta)


def predict_one_query_local_ridge(lib, xq, tmp_q, lat_q):
    """Predict SCF using all selected donors in one weighted ridge model."""
    cand_idx, temp_thr_used = _select_local_donors(lib, tmp_q)
    if cand_idx.size == 0:
        return np.nan, 0, temp_thr_used

    w = _kernel_weights(lib, cand_idx, xq, tmp_q, lat_q)
    yc = lib["y"][cand_idx]
    Xc = lib["X"][cand_idx]

    if yc.size == 0 or np.sum(w) < MIN_WEIGHT_SUM:
        return np.nan, int(cand_idx.size), temp_thr_used

    pred = _fit_weighted_ridge_predict(
        X_train=Xc,
        y_train=yc,
        w_train=w,
        xq=xq,
        lam=RIDGE_LAMBDA,
    )
    if CLIP_01:
        pred = float(np.clip(pred, 0.0, 1.0))
    return pred, int(cand_idx.size), temp_thr_used


# =========================================================
# Query construction and prediction
# =========================================================
def _build_query_arrays(cru, y_da, year, month):
    cur_time, lag_time = get_cur_and_lag1(cru, year, month)
    y_time = sel_year_month(y_da, year, month)
    if cur_time is None or y_time is None:
        return None
    if lag_time is None:
        lag_time = cur_time

    y_full = y_time.values.astype(np.float32).reshape(-1)
    flat_feats = build_flat_feats(cur_time, lag_time)
    tmp_cur = cur_time["tmp"].values.astype(np.float32).reshape(-1)
    lat_grid = np.repeat(LAT_ARR[:, None], 1440, axis=1).reshape(-1).astype(np.float32)

    valid = np.isfinite(y_full) & np.isfinite(tmp_cur)
    for f in flat_feats:
        valid &= np.isfinite(f)

    idx_valid = np.where(valid)[0]
    if idx_valid.size == 0:
        return None

    X_valid = stacked_feature_matrix_from_flat(flat_feats, idx_valid)
    return {
        "idx_valid": idx_valid,
        "X_valid": X_valid,
        "y_valid": y_full[idx_valid],
        "tmp_valid": tmp_cur[idx_valid],
        "lat_valid": lat_grid[idx_valid],
    }


def _build_historical_query_arrays(cru, year, month):
    """Construct historical predictors without requiring a MODIS target."""
    cur_time, lag_time = get_cur_and_lag1(cru, year, month)
    if cur_time is None:
        return None
    if lag_time is None:
        lag_time = cur_time

    flat_feats = build_flat_feats(cur_time, lag_time)
    tmp_cur = cur_time["tmp"].values.astype(np.float32).reshape(-1)
    lat_grid = np.repeat(LAT_ARR[:, None], 1440, axis=1).reshape(-1).astype(np.float32)

    # The finite CRU domain defines valid historical query cells. Ocean cells
    # and cells missing any climate predictor retain NaN.
    valid = np.isfinite(tmp_cur)
    for f in flat_feats:
        valid &= np.isfinite(f)

    idx_valid = np.where(valid)[0]
    if idx_valid.size == 0:
        return None

    X_valid = stacked_feature_matrix_from_flat(flat_feats, idx_valid)
    return {
        "idx_valid": idx_valid,
        "X_valid": X_valid,
        "tmp_valid": tmp_cur[idx_valid],
        "lat_valid": lat_grid[idx_valid],
    }


def sample_query_points(cru, y_da, years, seed=1234):
    rng = np.random.default_rng(seed)
    chunks = []
    total = 0

    for year in years:
        for month in range(1, 13):
            q = _build_query_arrays(cru, y_da, year, month)
            if q is None:
                continue
            idx_valid = q["idx_valid"]
            k = min(SCATTER_PER_YEAR_MONTH, idx_valid.size)
            if k <= 0:
                continue

            sel = rng.choice(np.arange(idx_valid.size), size=k, replace=False)
            chunks.append({
                "year": year,
                "month": month,
                "X": q["X_valid"][sel],
                "y": q["y_valid"][sel],
                "tmp": q["tmp_valid"][sel],
                "lat": q["lat_valid"][sel],
            })
            total += k
            if total >= SCATTER_MAX_TOTAL:
                break
        if total >= SCATTER_MAX_TOTAL:
            break
    return chunks


def predict_query_chunk(lib, qchunk):
    Xq = qchunk["X"]
    yq = qchunk["y"]
    tq = qchunk["tmp"]
    lq = qchunk["lat"]

    preds = np.full((Xq.shape[0],), np.nan, dtype=np.float32)
    ndon = np.zeros((Xq.shape[0],), dtype=np.int32)
    tthr = np.zeros((Xq.shape[0],), dtype=np.float32)

    for i in range(Xq.shape[0]):
        pred, n_use, tt = predict_one_query_local_ridge(
            lib, Xq[i], float(tq[i]), float(lq[i])
        )
        preds[i] = pred
        ndon[i] = n_use
        tthr[i] = tt

    out = {
        "year": qchunk["year"],
        "month": qchunk["month"],
        "y_true": yq.astype(np.float32, copy=False),
        "y_pred": preds.astype(np.float32, copy=False),
    }
    if SAVE_LOCAL_INFO:
        out["n_donors"] = ndon
        out["temp_thr"] = tthr
    return out


def run_sampled_predictions(libs, cru, y_da, years, tag, seed):
    qchunks = sample_query_points(cru, y_da, years, seed=seed)
    if len(qchunks) == 0:
        return None

    res = Parallel(n_jobs=min(N_JOBS, len(qchunks)), backend="loky", verbose=10)(
        delayed(predict_query_chunk)(libs[ch["month"]], ch) for ch in qchunks
    )

    rows = []
    for r in res:
        for yt, yp in zip(r["y_true"], r["y_pred"]):
            rows.append([tag, r["year"], r["month"], float(yt), float(yp)])

    df = pd.DataFrame(rows, columns=["tag", "year", "month", "y_true", "y_pred"])
    return df


# =========================================================
# Block-parallel full-field prediction
# =========================================================
def _predict_full_block(lib, idx_slice, X_slice, tmp_slice, lat_slice):
    n = idx_slice.size
    pred = np.full((n,), np.nan, dtype=np.float32)
    ndon = np.full((n,), -1, dtype=np.int32)
    tthr = np.full((n,), np.nan, dtype=np.float32)

    for i in range(n):
        p, n_use, tt = predict_one_query_local_ridge(
            lib,
            X_slice[i],
            float(tmp_slice[i]),
            float(lat_slice[i])
        )
        pred[i] = p
        ndon[i] = n_use
        tthr[i] = tt

    return idx_slice, pred, ndon, tthr


def predict_full_single_year_month(lib, cru, y_da, year, month, target_name):
    q = _build_query_arrays(cru, y_da, year, month)
    if q is None:
        return None

    idx_valid = q["idx_valid"]
    X_valid = q["X_valid"]
    y_valid = q["y_valid"]
    tmp_valid = q["tmp_valid"]
    lat_valid = q["lat_valid"]

    pred_full = np.full((720 * 1440,), np.nan, dtype=np.float32)
    donor_n_full = np.full((720 * 1440,), -1, dtype=np.int32)
    temp_thr_full = np.full((720 * 1440,), np.nan, dtype=np.float32)

    n = idx_valid.size
    block_ranges = [(st, min(n, st + FULLFIELD_BLOCK_SIZE)) for st in range(0, n, FULLFIELD_BLOCK_SIZE)]
    print(f"[FullPred] {target_name} {year}-{month:02d} valid_pixels={n}, n_blocks={len(block_ranges)}, n_jobs={min(FULLFIELD_N_JOBS, len(block_ranges))}")

    results = Parallel(
        n_jobs=min(FULLFIELD_N_JOBS, len(block_ranges)),
        backend="loky",
        verbose=FULLFIELD_VERBOSE
    )(
        delayed(_predict_full_block)(
            lib,
            idx_valid[st:ed],
            X_valid[st:ed],
            tmp_valid[st:ed],
            lat_valid[st:ed]
        )
        for st, ed in block_ranges
    )

    for idx_slice, pred, ndon, tthr in results:
        pred_full[idx_slice] = pred
        donor_n_full[idx_slice] = ndon
        temp_thr_full[idx_slice] = tthr

    arr_pred = pred_full.reshape(720, 1440)
    arr_ndon = donor_n_full.reshape(720, 1440)
    arr_tthr = temp_thr_full.reshape(720, 1440)

    truth_full = np.full((720 * 1440,), np.nan, dtype=np.float32)
    truth_full[idx_valid] = y_valid
    arr_truth = truth_full.reshape(720, 1440)

    ds = xr.Dataset(
        {
            f"{target_name}_pred": (("lat", "lon"), arr_pred),
            f"{target_name}_donor_n": (("lat", "lon"), arr_ndon),
            f"{target_name}_truth": (("lat", "lon"), arr_truth),
            f"{target_name}_temp_thr_used": (("lat", "lon"), arr_tthr),
        },
        coords={"lat": LAT_ARR, "lon": LON_ARR}
    )

    out_nc = os.path.join(
        OUT_FULL_DIR,
        f"{target_name}_{year}_{month:02d}_local_pixel_model.nc",
    )
    ds.to_netcdf(out_nc)
    print(f"saved full field: {out_nc}")
    return out_nc


def predict_historical_single_year_month(lib, cru, year, month, target_name):
    """
    Predict one historical monthly global land SCF field using donor libraries
    built from all available 2001–2023 MODIS observations.
    """
    out_nc = os.path.join(
        OUT_HIST_DIR,
        f"{target_name}_{year}_{month:02d}_historical_reconstruction.nc"
    )
    if SKIP_EXISTING_HISTORICAL and os.path.exists(out_nc):
        print(f"[HistPred] skip existing: {out_nc}")
        return out_nc

    q = _build_historical_query_arrays(cru, year, month)
    if q is None:
        print(f"[HistPred] no valid CRU pixels: {target_name} {year}-{month:02d}")
        return None

    idx_valid = q["idx_valid"]
    X_valid = q["X_valid"]
    tmp_valid = q["tmp_valid"]
    lat_valid = q["lat_valid"]

    pred_full = np.full((720 * 1440,), np.nan, dtype=np.float32)
    donor_n_full = np.full((720 * 1440,), -1, dtype=np.int32)
    temp_thr_full = np.full((720 * 1440,), np.nan, dtype=np.float32)

    n = idx_valid.size
    block_ranges = [
        (st, min(n, st + FULLFIELD_BLOCK_SIZE))
        for st in range(0, n, FULLFIELD_BLOCK_SIZE)
    ]
    n_jobs = min(FULLFIELD_N_JOBS, len(block_ranges))
    print(
        f"[HistPred] {target_name} {year}-{month:02d} "
        f"valid_pixels={n}, n_blocks={len(block_ranges)}, n_jobs={n_jobs}"
    )

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=FULLFIELD_VERBOSE
    )(
        delayed(_predict_full_block)(
            lib,
            idx_valid[st:ed],
            X_valid[st:ed],
            tmp_valid[st:ed],
            lat_valid[st:ed]
        )
        for st, ed in block_ranges
    )

    for idx_slice, pred, ndon, tthr in results:
        pred_full[idx_slice] = pred
        donor_n_full[idx_slice] = ndon
        temp_thr_full[idx_slice] = tthr

    arr_pred = pred_full.reshape(720, 1440)
    arr_ndon = donor_n_full.reshape(720, 1440)
    arr_tthr = temp_thr_full.reshape(720, 1440)
    time_value = np.array([np.datetime64(f"{year}-{month:02d}-15")])

    data_vars = {
        f"{target_name}_pred": (
            ("time", "lat", "lon"),
            arr_pred[None, :, :],
            {
                "long_name": "historically reconstructed monthly snow cover fraction",
                "units": "1",
                "valid_min": 0.0,
                "valid_max": 1.0,
            },
        )
    }
    if SAVE_LOCAL_INFO:
        data_vars.update({
            f"{target_name}_donor_n": (
                ("time", "lat", "lon"), arr_ndon[None, :, :]
            ),
            f"{target_name}_temp_thr_used": (
                ("time", "lat", "lon"), arr_tthr[None, :, :], {"units": "degree_Celsius"}
            ),
        })

    ds = xr.Dataset(
        data_vars,
        coords={"time": time_value, "lat": LAT_ARR, "lon": LON_ARR},
        attrs={
            "title": "Historical monthly snow-cover reconstruction",
            "training_period": f"{TRAIN_START_YEAR}-{TRAIN_END_YEAR}",
            "climate_predictors": ",".join(BASE_FEATS),
            "feature_mode": FEAT_MODE,
            "maximum_temperature_difference_degree_C": TEMP_THRESH_MAX_C,
            "temperature_threshold_initial_degree_C": TEMP_THRESH_INIT_C,
            "temperature_threshold_growth_factor": TEMP_THRESH_GROW,
            "minimum_candidate_count": MIN_DONORS,
            "sigma_climate_distance": SIGMA_CLIM,
            "sigma_temperature_degree_C": SIGMA_TEMP_C,
            "sigma_latitude_degree": SIGMA_LAT_DEG,
            "ridge_lambda": RIDGE_LAMBDA,
        },
    )
    ds.to_netcdf(out_nc)
    ds.close()
    print(f"saved historical reconstruction: {out_nc}")
    return out_nc


def _read_historical_area_fraction(year, month):
    """Read one reconstructed monthly SCF field on the target grid."""
    path = os.path.join(
        OUT_HIST_DIR,
        f"areaFrac_{year}_{month:02d}_historical_reconstruction.nc"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing historical SCF prediction: {path}. "
            "Complete the 1901-2000 reconstruction before exporting."
        )
    with xr.open_dataset(path) as ds:
        var = "areaFrac_pred"
        if var not in ds:
            raise KeyError(f"Variable {var!r} is absent from {path}")
        da = ds[var]
        if "time" in da.dims:
            da = da.isel(time=0)
        arr = np.asarray(da.values, dtype=np.float32)
    if arr.shape != (720, 1440):
        raise RuntimeError(f"Unexpected reconstructed SCF shape in {path}: {arr.shape}")
    return np.clip(arr, 0.0, 1.0)


def _read_observed_area_fraction(y_area, year, month):
    """Read one observed MODIS monthly SCF field already aligned by this script."""
    da = sel_year_month(y_area, year, month)
    if da is None:
        raise RuntimeError(f"Observed MODIS SCF is unavailable for {year}-{month:02d}")
    arr = np.asarray(da.values, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.shape != (720, 1440):
        raise RuntimeError(
            f"Unexpected observed SCF shape for {year}-{month:02d}: {arr.shape}"
        )
    invalid = ~np.isfinite(arr)
    arr = np.clip(arr, 0.0, 1.0)
    arr[invalid] = np.nan
    return arr


def export_unified_yearly_scf_mat(y_area):
    """Export yearly HDF5-MAT files used directly by the albedo synthesis.

    1700-1900: repeat the corresponding 1901 monthly reconstructed fields.
    1901-2000: use the historical reconstruction.
    2001-2023: use the monthly MODIS observations.
    """
    os.makedirs(OUT_UNIFIED_SCF_DIR, exist_ok=True)
    for year in range(UNIFIED_START_YEAR, UNIFIED_END_YEAR + 1):
        out_path = os.path.join(OUT_UNIFIED_SCF_DIR, f"snowfrac_025_{year}.mat")
        if SKIP_EXISTING_HISTORICAL and os.path.exists(out_path):
            print(f"[SCF export] skip existing: {out_path}")
            continue

        fields = []
        if year < HIST_START_YEAR:
            source_year = PRE_1901_REFERENCE_YEAR
            for month in range(1, 13):
                fields.append(_read_historical_area_fraction(source_year, month))
            source = f"monthly reconstructed SCF from {source_year} repeated for {year}"
        elif year <= HIST_END_YEAR:
            for month in range(1, 13):
                fields.append(_read_historical_area_fraction(year, month))
            source = "historical SCF reconstruction"
        else:
            for month in range(1, 13):
                fields.append(_read_observed_area_fraction(y_area, year, month))
            source = "MODIS monthly SCF observation"

        stack = np.stack(fields, axis=0).astype(np.float32)
        if stack.shape != (12, 720, 1440):
            raise RuntimeError(f"Unexpected unified SCF shape for {year}: {stack.shape}")

        tmp_path = out_path + ".tmp"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        with h5py.File(tmp_path, "w") as handle:
            ds = handle.create_dataset(
                TARGET_KEYS["areaFrac"],
                data=stack,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                chunks=(1, 180, 360),
            )
            ds.attrs["units"] = "1"
            ds.attrs["dimensions"] = "month,lat,lon"
            handle.attrs["year"] = year
            handle.attrs["source"] = source
            handle.attrs["latitude_order"] = "north_to_south"
            handle.attrs["longitude_order"] = "west_to_east"
        os.replace(tmp_path, out_path)
        print(f"[SCF export] saved: {out_path}")


# =========================================================
# Metrics and plots
# =========================================================
def calc_r2_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]
    y_pred = y_pred[m]
    if y_true.size == 0:
        return np.nan, np.nan, 0

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    sse = float(np.sum((y_true - y_pred) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 1e-12 else np.nan
    return r2, rmse, int(y_true.size)


def summarize_metrics(df, tag):
    rows = []
    for m in range(1, 13):
        sub = df[df["month"] == m]
        r2, rmse, n = calc_r2_rmse(sub["y_true"].values, sub["y_pred"].values)
        rows.append({"tag": tag, "month": m, "r2": r2, "rmse": rmse, "n_samples": n})

    r2, rmse, n = calc_r2_rmse(df["y_true"].values, df["y_pred"].values)
    rows.append({"tag": tag, "month": "ALL", "r2": r2, "rmse": rmse, "n_samples": n})
    return pd.DataFrame(rows)


def plot_hexbin_dense(y_true, y_pred, out_fp, gridsize=90, mincnt=5, bins="log", max_points=180000, seed=42):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]
    y_pred = y_pred[m]

    if y_true.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(y_true.size, size=max_points, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    r2, rmse, n = calc_r2_rmse(y_true, y_pred)

    fig = plt.figure(figsize=(7.8, 7.8), dpi=220)
    ax = plt.gca()

    hb = ax.hexbin(
        y_true, y_pred,
        gridsize=int(gridsize),
        extent=(0, 1, 0, 1),
        mincnt=int(mincnt),
        bins=bins
    )
    ax.plot([0, 1], [0, 1], linewidth=2.0)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")

    txt = f"$R^2$={r2:.4f}\nRMSE={rmse:.5f}\n$n$={n}"
    ax.text(
        0.03, 0.97, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=18,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=4.0)
    )

    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=14)
    cb.set_label("log10(count)" if bins == "log" else "count", fontsize=16)

    plt.tight_layout()
    plt.savefig(out_fp)
    plt.close(fig)


# =========================================================
# Main workflow
# =========================================================
def main():
    t0 = time.time()
    print("=" * 120)
    print("STIM-Snow local weighted-ridge SCF reconstruction")
    print(f"FEAT_MODE={FEAT_MODE}")
    print(f"TRAIN={TRAIN_START_YEAR}-{TRAIN_END_YEAR}")
    print(f"N_JOBS={N_JOBS}")
    print(f"FULLFIELD_N_JOBS={FULLFIELD_N_JOBS}")
    print(f"MAX_DONORS_PER_YEAR_MONTH={MAX_DONORS_PER_YEAR_MONTH}")
    print(f"TEMP_THRESH_INIT/MAX={TEMP_THRESH_INIT_C}/{TEMP_THRESH_MAX_C}")
    print(
        f"MIN_DONORS={MIN_DONORS}, RIDGE_LAMBDA={RIDGE_LAMBDA}, "
        f"SIGMAS(clim,temp,lat)={SIGMA_CLIM},{SIGMA_TEMP_C},{SIGMA_LAT_DEG}"
    )
    print(f"PREDICT_FULL_FIELDS={PREDICT_FULL_FIELDS}")
    print(
        f"PREDICT_HISTORICAL_FIELDS={PREDICT_HISTORICAL_FIELDS}, "
        f"HISTORICAL_TARGETS={HISTORICAL_TARGETS}, "
        f"HIST_YEARS={HIST_START_YEAR}-{HIST_END_YEAR}, "
        f"HIST_MONTHS={HIST_MONTHS}"
    )
    print("=" * 120)

    ym_to_modis_fp = build_year_month_to_matfile_map(MODIS_MONTHLY_MAT_DIR)
    if not ym_to_modis_fp:
        raise RuntimeError(f"No monthly MODIS MAT files found in {MODIS_MONTHLY_MAT_DIR}")

    print("\n1) Loading CRU predictors on the target grid ...")
    cru = read_cru_to_target_grid()
    print(f"CRU grid: lat={cru.sizes['lat']} lon={cru.sizes['lon']}")

    print("\n2) Loading monthly MODIS snow cover fraction ...")
    target_data = {}
    if "daysFrac" in TARGETS_TO_PROCESS:
        target_data["daysFrac"] = load_modis_targets_as_da_monthly(
            TARGET_KEYS["daysFrac"], ym_to_modis_fp, SUP_YEARS
        )
    if "areaFrac" in TARGETS_TO_PROCESS or EXPORT_UNIFIED_YEARLY_MAT:
        target_data["areaFrac"] = load_modis_targets_as_da_monthly(
            TARGET_KEYS["areaFrac"], ym_to_modis_fp, SUP_YEARS
        )

    metrics_all = []

    for target_name in TARGETS_TO_PROCESS:
        y_da = target_data[target_name]
        print("\n" + "#" * 120)
        print(f"# TARGET: {target_name} ({TARGET_KEYS[target_name]})")
        print("#" * 120)

        train_years, val_years, test_years = split_years_stratified_train_val_test(y_da, SUP_YEARS, seed=RANDOM_SEED)
        print(f"TRAIN({len(train_years)}): {train_years}")
        print(f"VAL  ({len(val_years)}): {val_years}")
        print(f"TEST ({len(test_years)}): {test_years}")

        print("\n3) Building 12 monthly donor libraries ...")
        libs = build_all_month_donor_libraries(cru, y_da, train_years, target_name)

        print("\n4) Running sampled training, validation and test evaluation ...")
        df_tr = run_sampled_predictions(libs, cru, y_da, train_years, tag=f"{target_name}_TRAIN", seed=RANDOM_SEED + 11)
        df_va = run_sampled_predictions(libs, cru, y_da, val_years, tag=f"{target_name}_VAL", seed=RANDOM_SEED + 22)
        df_te = run_sampled_predictions(libs, cru, y_da, test_years, tag=f"{target_name}_TEST", seed=RANDOM_SEED + 33)

        dfs = [x for x in [df_tr, df_va, df_te] if x is not None]
        if len(dfs) == 0:
            raise RuntimeError(f"{target_name}: no sampled prediction results.")
        df_pred = pd.concat(dfs, ignore_index=True)

        out_pred_csv = os.path.join(OUT_METRICS_DIR, f"sampled_predictions_{target_name}.csv")
        df_pred.to_csv(out_pred_csv, index=False, encoding="utf-8-sig")
        print(f"saved sampled predictions: {out_pred_csv}")

        metric_parts = []
        if df_tr is not None:
            metric_parts.append(summarize_metrics(df_tr, f"{target_name}_TRAIN"))
        if df_va is not None:
            metric_parts.append(summarize_metrics(df_va, f"{target_name}_VAL"))
        if df_te is not None:
            metric_parts.append(summarize_metrics(df_te, f"{target_name}_TEST"))
        dfm = pd.concat(metric_parts, ignore_index=True)

        out_met_csv = os.path.join(OUT_METRICS_DIR, f"metrics_train_val_test_{target_name}.csv")
        dfm.to_csv(out_met_csv, index=False, encoding="utf-8-sig")
        print(f"saved metrics: {out_met_csv}")
        metrics_all.append(dfm)

        if df_tr is not None:
            plot_hexbin_dense(
                df_tr["y_true"].values, df_tr["y_pred"].values,
                os.path.join(OUT_PLOTS_DIR, f"hexbin_{target_name}_TRAIN.png")
            )
        if df_va is not None:
            plot_hexbin_dense(
                df_va["y_true"].values, df_va["y_pred"].values,
                os.path.join(OUT_PLOTS_DIR, f"hexbin_{target_name}_VAL.png")
            )
        if df_te is not None:
            plot_hexbin_dense(
                df_te["y_true"].values, df_te["y_pred"].values,
                os.path.join(OUT_PLOTS_DIR, f"hexbin_{target_name}_TEST.png")
            )
        print(f"saved hexbin plots for {target_name}")

        if PREDICT_FULL_FIELDS:
            for yy in test_years:
                for mm in range(1, 13):
                    predict_full_single_year_month(libs[mm], cru, y_da, yy, mm, target_name)

        if PREDICT_HISTORICAL_FIELDS and target_name in HISTORICAL_TARGETS:
            print(
                "\n5) Rebuilding final donor libraries from all 2001–2023 "
                f"observations and reconstructing {HIST_START_YEAR}–{HIST_END_YEAR} ..."
            )

            # Evaluation libraries contain training years only. Historical
            # reconstruction requires new libraries built from the full
            # 2001–2023 observation period.
            del libs
            gc.collect()
            final_libs = build_all_month_donor_libraries(
                cru,
                y_da,
                SUP_YEARS,
                f"{target_name}_final_all_{TRAIN_START_YEAR}_{TRAIN_END_YEAR}"
            )

            for yy in range(HIST_START_YEAR, HIST_END_YEAR + 1):
                for mm in HIST_MONTHS:
                    predict_historical_single_year_month(
                        final_libs[mm], cru, yy, mm, target_name
                    )

            del final_libs
            gc.collect()

    if EXPORT_UNIFIED_YEARLY_MAT:
        if not PREDICT_HISTORICAL_FIELDS:
            raise RuntimeError(
                "EXPORT_UNIFIED_YEARLY_MAT=True requires "
                "PREDICT_HISTORICAL_FIELDS=True."
            )
        print(
            f"\n6) Exporting unified yearly SCF MAT files for "
            f"{UNIFIED_START_YEAR}–{UNIFIED_END_YEAR} ..."
        )
        export_unified_yearly_scf_mat(target_data["areaFrac"])

    if len(metrics_all) > 0:
        df_all = pd.concat(metrics_all, ignore_index=True)
        out_all = os.path.join(OUT_METRICS_DIR, "metrics_ALLTARGETS.csv")
        df_all.to_csv(out_all, index=False, encoding="utf-8-sig")
        print(f"saved summary metrics: {out_all}")

    print(f"\nCompleted in {(time.time() - t0)/60:.1f} min")
    print(f"OUT_BASE = {OUT_BASE}")


if __name__ == "__main__":
    main()
