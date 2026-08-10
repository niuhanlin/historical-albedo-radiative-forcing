# -*- coding: utf-8 -*-
"""
Calculate top-of-atmosphere shortwave radiative forcing (RF) induced by
historical land-surface albedo change, following Equation (4) of the manuscript.

This script calculates three components in one run:
  1. total: land cover and snow cover both vary through time;
  2. landcover: snow cover fraction is fixed at its 1700 monthly values;
  3. snow: total minus landcover.

For each radiative kernel r, year y and month m:

    RF_r(y,m,x) = 100 * [alpha(y,m,x) - alpha(1700,m,x)]
                  * K_r(m,x) * M85(m,x)

The monthly global mean is calculated first using the area of the whole Earth
as the denominator. Monthly global means are then averaged using normal-year
month lengths. Months excluded by M85 contribute zero and are NOT renormalized.

Required inputs
---------------
1. Monthly total-albedo GeoTIFFs for 1700-2023.
2. Monthly land-cover-driven albedo GeoTIFFs for 1700-2023, constructed with
   snow cover fraction fixed at its 1700 monthly values.
3. Three monthly all-sky shortwave albedo kernels:
   CESM-CAM5 (FSNT), HadGEM3-GA7.1 (albedo_sw), HadGEM2 (albedo).

The albedo GeoTIFFs are produced by ``synthesize_surface_albedo_5LC.py``.
They must contain fractions in the range 0-1 and follow:
    albedo_weighted_<year>_<month>_0p25deg_global_wgs84.tif
"""

import csv
import glob
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio
import xarray as xr


# =============================================================================
# CONFIGURATION: edit paths before running
# =============================================================================
REF_YEAR = 1700
YEAR_START = 1700
YEAR_END = 2023
MONTHS = tuple(range(1, 13))

# Albedo is stored as a fraction, not percent or scaled integer.
ALBEDO_IS_0_1 = True

# M85 definition used by this script:
#   any_day: M85=1 if local-noon SZA <=85 degrees on at least one day in month.
#   all_days: M85=1 only if local-noon SZA <=85 degrees on every day in month.
SZA_THRESH_DEG = 85.0
M85_MONTH_RULE = "any_day"  # "any_day" or "all_days"

# Monthly weights use the number of days in a normal year.
MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("RF_DATA_DIR", os.path.join(PROJECT_DIR, "data"))

ALBEDO_ROOT = os.environ.get(
    "ALBEDO_ROOT",
    os.path.join(PROJECT_DIR, "outputs", "albedo_synthesis_5LC_1700_2023"),
)

TEMPLATE_TIF = os.path.join(
    ALBEDO_ROOT,
    "dynamic",
    "monthly",
    "albedo_weighted_1700_01_0p25deg_global_wgs84.tif",
)

# Total albedo: land cover and snow cover both vary through time.
TOTAL_ALBEDO_DIR = os.path.join(ALBEDO_ROOT, "dynamic")

# Land-cover-driven albedo: monthly SCF is fixed at its 1700 values.
LANDCOVER_ALBEDO_DIR = os.path.join(ALBEDO_ROOT, "fixed1700")

SCENARIO_DIRS = {
    "total": TOTAL_ALBEDO_DIR,
    "landcover": LANDCOVER_ALBEDO_DIR,
}
COMPONENTS = ("total", "landcover", "snow")

KERNEL_DIR = os.environ.get(
    "RADIATIVE_KERNEL_DIR",
    os.path.join(DATA_DIR, "radiative_kernels"),
)
PENDER_NC = os.path.join(KERNEL_DIR, "alb.kernel.nc")
HADGEM3_NC = os.path.join(KERNEL_DIR, "HadGEM3-GA7.1_TOA_kernel_L19.nc")
HADGEM2_NET_TOA_L38 = os.path.join(KERNEL_DIR, "HadGEM2_net_TOA_L38.nc")
HADGEM2_NET_TOA_L17 = os.path.join(KERNEL_DIR, "HadGEM2_net_TOA_L17.nc")

KERNEL_NAMES = (
    "CESM_CAM5_FSNT",
    "HadGEM3_albedo_sw",
    "HadGEM2_albedo",
)

OUT_DIR = os.environ.get(
    "RF_OUTPUT_DIR",
    os.path.join(PROJECT_DIR, "outputs", "radiative_forcing_1700_2023"),
)
OUT_TIF_DIR = os.path.join(OUT_DIR, "annual_RF_tif")
CACHE_DIR = os.path.join(OUT_DIR, "_cache_v2_allsky_FSNT")
CSV_OUT = os.path.join(
    OUT_DIR,
    f"RF_EQ4_EarthMean_{YEAR_START}_{YEAR_END}_minus_{REF_YEAR}.csv",
)

# Each worker keeps the three 12-month kernels and two reference-albedo stacks
# in memory. Start conservatively; set RF_MAX_WORKERS in the environment if the
# machine has enough RAM, for example: set RF_MAX_WORKERS=60
MAX_WORKERS = max(
    1,
    min(int(os.environ.get("RF_MAX_WORKERS", "16")), os.cpu_count() or 1),
)

SKIP_IF_ALL_OUTPUTS_EXIST = True
REBUILD_CACHE = False
MULT_01 = 100.0
TIF_NODATA = -9999.0


# Worker-local cache, loaded once by the ProcessPool initializer.
_WORKER = None


# =============================================================================
# BASIC UTILITIES
# =============================================================================
def month_day_weights_normal_year():
    total_days = float(sum(MONTH_DAYS))
    return {m: MONTH_DAYS[m - 1] / total_days for m in MONTHS}


def get_grid_1d_from_template(template_path):
    with rasterio.open(template_path) as ds:
        transform = ds.transform
        height, width = ds.height, ds.width
        profile = ds.profile.copy()
        crs = ds.crs

    cols = np.arange(width, dtype=np.float64) + 0.5
    rows = np.arange(height, dtype=np.float64) + 0.5
    lon1d = transform.c + cols * transform.a
    lat1d = transform.f + rows * transform.e
    return lat1d, lon1d, profile, crs, (height, width)


def assert_same_grid(path, profile_tpl, crs_tpl, shape_hw):
    with rasterio.open(path) as ds:
        if (ds.height, ds.width) != tuple(shape_hw):
            raise ValueError(
                f"Grid shape mismatch: {path}; got {(ds.height, ds.width)}, "
                f"expected {shape_hw}."
            )
        if ds.crs != crs_tpl:
            raise ValueError(
                f"CRS mismatch: {path}; got {ds.crs}, expected {crs_tpl}."
            )
        if not ds.transform.almost_equals(profile_tpl["transform"]):
            raise ValueError(f"Grid transform mismatch: {path}")


def read_tif_as_masked(path, profile_tpl=None, crs_tpl=None, shape_hw=None):
    if profile_tpl is not None:
        assert_same_grid(path, profile_tpl, crs_tpl, shape_hw)
    with rasterio.open(path) as ds:
        return ds.read(1, masked=True).astype(np.float32)


def postprocess_alpha(alpha_ma, source_path=""):
    data = np.asarray(alpha_ma.data, dtype=np.float32).copy()
    mask = np.ma.getmaskarray(alpha_ma).copy()
    valid = (~mask) & np.isfinite(data)

    if np.any(valid):
        vmin = float(np.nanmin(data[valid]))
        vmax = float(np.nanmax(data[valid]))
        if not ALBEDO_IS_0_1:
            raise ValueError(
                "This version requires albedo fractions in 0-1. "
                "Convert the input explicitly before running."
            )
        if vmin < -0.01 or vmax > 1.01:
            raise ValueError(
                f"Albedo outside expected 0-1 range in {source_path}: "
                f"min={vmin}, max={vmax}. Check scale and nodata."
            )

    bad = (~mask) & (~np.isfinite(data))
    return np.ma.array(data, mask=(mask | bad), dtype=np.float32)


def resolve_albedo_dir(path):
    monthly = os.path.join(path, "monthly")
    resolved = monthly if os.path.isdir(monthly) else path
    if not os.path.isdir(resolved):
        raise FileNotFoundError(f"Albedo directory not found: {resolved}")
    return resolved


def find_alpha_file(albedo_dir, year, month):
    exact = os.path.join(
        albedo_dir,
        f"albedo_weighted_{year}_{month:02d}_0p25deg_global_wgs84.tif",
    )
    if os.path.exists(exact):
        return exact

    hits = sorted(
        glob.glob(os.path.join(albedo_dir, f"*{year}*{month:02d}*0p25deg*.tif"))
    )
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise RuntimeError(
            f"Multiple candidate albedo files for {year}-{month:02d}: {hits}"
        )
    raise FileNotFoundError(exact)


def load_alpha(albedo_dir, year, month, grid_info=None):
    path = find_alpha_file(albedo_dir, year, month)
    if grid_info is None:
        ma = read_tif_as_masked(path)
    else:
        profile_tpl, crs_tpl, shape_hw = grid_info
        ma = read_tif_as_masked(path, profile_tpl, crs_tpl, shape_hw)
    return postprocess_alpha(ma, source_path=path)


def area_lat_weights(lat1d, width):
    row_weights = np.cos(np.deg2rad(lat1d)).astype(np.float64)
    if np.any(row_weights < 0):
        raise ValueError("Latitude centers must lie within -90 to 90 degrees.")
    return np.repeat(row_weights[:, None], width, axis=1)


def write_float_tif(out_path, arr_float32, profile_tpl):
    profile = profile_tpl.copy()
    profile.update(
        dtype="float32",
        count=1,
        compress="deflate",
        predictor=2,
        nodata=TIF_NODATA,
        BIGTIFF="IF_SAFER",
    )
    out = np.where(
        np.isfinite(arr_float32),
        np.asarray(arr_float32, dtype=np.float32),
        np.float32(TIF_NODATA),
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as ds:
        ds.write(out, 1)


# =============================================================================
# RADIATIVE KERNELS
# =============================================================================
def _to_lon_minus180_180(lon):
    return ((lon + 180.0) % 360.0) - 180.0


def _infer_month_dim(da, month_dim_hint=None):
    if month_dim_hint and month_dim_hint in da.dims:
        return month_dim_hint
    for candidate in ("time", "month", "mons", "mon"):
        if candidate in da.dims:
            return candidate
    first_dim = da.dims[0]
    if da.sizes[first_dim] == 12:
        return first_dim
    raise RuntimeError(
        f"Cannot identify the 12-month dimension: dims={da.dims}, shape={da.shape}"
    )


def _add_cyclic_longitude(da):
    """Add one cyclic column on each side before longitude interpolation."""
    da = da.sortby("lon")
    left = da.isel(lon=-1).expand_dims("lon")
    right = da.isel(lon=0).expand_dims("lon")
    left = left.assign_coords(lon=[float(da.lon.values[-1]) - 360.0])
    right = right.assign_coords(lon=[float(da.lon.values[0]) + 360.0])
    return xr.concat([left, da, right], dim="lon")


def load_and_regrid_kernel(
    kernel_nc,
    varname,
    lat_tpl,
    lon_tpl,
    month_dim_hint=None,
    require_all_sky=True,
):
    if not os.path.exists(kernel_nc):
        raise FileNotFoundError(kernel_nc)

    with xr.open_dataset(kernel_nc, decode_times=False) as ds:
        if varname not in ds.data_vars:
            raise KeyError(
                f"Kernel variable '{varname}' not found in {kernel_nc}. "
                f"Available variables: {list(ds.data_vars)[:80]}"
            )

        da = ds[varname]
        month_dim = _infer_month_dim(da, month_dim_hint)
        if da.sizes[month_dim] != 12:
            raise RuntimeError(
                f"Kernel must contain 12 months: {kernel_nc}, "
                f"{month_dim}={da.sizes[month_dim]}"
            )
        if "lat" not in da.coords or "lon" not in da.coords:
            raise RuntimeError(
                f"Kernel requires lat/lon coordinates: {kernel_nc}, var={varname}"
            )

        units = str(da.attrs.get("units", "")).strip()
        long_name = str(da.attrs.get("long_name", "")).strip()
        descriptor = f"{varname} {long_name}".lower()
        if require_all_sky and "clear" in descriptor:
            raise ValueError(
                f"A clear-sky kernel was supplied where an all-sky kernel is "
                f"required: {kernel_nc}, variable={varname}, long_name={long_name}"
            )

        lon_src = _to_lon_minus180_180(da["lon"].values.astype(np.float64))
        da = da.assign_coords(lon=lon_src).sortby("lon").sortby("lat")
        da = da.assign_coords({month_dim: np.arange(1, 13)})
        da = _add_cyclic_longitude(da)

        lat_increasing = bool(np.all(np.diff(lat_tpl) > 0))
        lon_increasing = bool(np.all(np.diff(lon_tpl) > 0))
        lat_query = lat_tpl if lat_increasing else lat_tpl[::-1]
        lon_query = lon_tpl if lon_increasing else lon_tpl[::-1]

        interpolated = da.interp(
            lat=xr.DataArray(lat_query, dims="lat"),
            lon=xr.DataArray(lon_query, dims="lon"),
            method="linear",
            kwargs={"fill_value": "extrapolate"},
        ).transpose(month_dim, "lat", "lon")

        array = interpolated.values.astype(np.float32)

    if not lat_increasing:
        array = array[:, ::-1, :]
    if not lon_increasing:
        array = array[:, :, ::-1]

    finite_fraction = float(np.mean(np.isfinite(array)))
    if finite_fraction < 0.999:
        raise ValueError(
            f"Regridded kernel contains too many non-finite cells: "
            f"{kernel_nc}, {varname}, finite_fraction={finite_fraction:.6f}"
        )

    return array, units, long_name


def resolve_hadgem2_kernel_file():
    for path in (HADGEM2_NET_TOA_L38, HADGEM2_NET_TOA_L17):
        if not os.path.exists(path):
            continue
        with xr.open_dataset(path, decode_times=False) as ds:
            if "albedo" not in ds.data_vars:
                raise KeyError(
                    f"HadGEM2 file exists but lacks variable 'albedo': {path}"
                )
        return path
    raise FileNotFoundError(
        "HadGEM2 net-TOA kernel not found. Expected L38 or L17 file."
    )


# =============================================================================
# SOLAR GEOMETRY AND M85
# =============================================================================
def solar_declination_rad(day_of_year):
    return np.deg2rad(23.44) * np.sin(
        2.0 * np.pi * (284.0 + float(day_of_year)) / 365.0
    )


def local_noon_sza_deg(lat_deg, day_of_year):
    latitude = np.deg2rad(lat_deg)
    declination = solar_declination_rad(day_of_year)
    cos_theta = (
        np.sin(latitude) * np.sin(declination)
        + np.cos(latitude) * np.cos(declination)
    )
    return np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def build_m85_rowmask(lat1d, month):
    if M85_MONTH_RULE not in {"any_day", "all_days"}:
        raise ValueError(
            f"M85_MONTH_RULE must be 'any_day' or 'all_days', got {M85_MONTH_RULE}"
        )

    first_doy = int(sum(MONTH_DAYS[: month - 1]) + 1)
    number_of_days = MONTH_DAYS[month - 1]
    valid_any = np.zeros(lat1d.shape, dtype=bool)
    valid_all = np.ones(lat1d.shape, dtype=bool)

    for offset in range(number_of_days):
        sza = local_noon_sza_deg(lat1d, first_doy + offset)
        valid_day = sza <= float(SZA_THRESH_DEG)
        valid_any |= valid_day
        valid_all &= valid_day

    return valid_any if M85_MONTH_RULE == "any_day" else valid_all


def build_m85_3d(lat1d, shape_hw):
    height, width = shape_hw
    output = np.zeros((12, height, width), dtype=bool)
    for index, month in enumerate(MONTHS):
        row_mask = build_m85_rowmask(lat1d, month)
        output[index] = np.repeat(row_mask[:, None], width, axis=1)
    return output


# =============================================================================
# CACHE
# =============================================================================
def cache_paths():
    return {
        "m85": os.path.join(
            CACHE_DIR,
            f"M85_local_noon_SZA_le_{int(SZA_THRESH_DEG)}_{M85_MONTH_RULE}.npz",
        ),
        "area": os.path.join(CACHE_DIR, "earth_area_coslat_weights.npz"),
        "alpha_ref_total": os.path.join(
            CACHE_DIR, f"alpha_ref_total_{REF_YEAR}_12months.npz"
        ),
        "alpha_ref_landcover": os.path.join(
            CACHE_DIR, f"alpha_ref_landcover_{REF_YEAR}_12months.npz"
        ),
        "kernel_cesm": os.path.join(
            CACHE_DIR, "kernel_CESM_CAM5_FSNT_allsky_0p25deg.npz"
        ),
        "kernel_hadgem3": os.path.join(
            CACHE_DIR, "kernel_HadGEM3_albedo_sw_allsky_0p25deg.npz"
        ),
        "kernel_hadgem2": os.path.join(
            CACHE_DIR, "kernel_HadGEM2_albedo_allsky_0p25deg.npz"
        ),
    }


def _remove_cache_if_requested(path):
    if REBUILD_CACHE and os.path.exists(path):
        os.remove(path)


def _reference_cache_matches(path, source_dir):
    if not os.path.exists(path):
        return False
    try:
        with np.load(path, allow_pickle=True) as data:
            cached_source = str(data["source_dir"].item())
            return os.path.normcase(cached_source) == os.path.normcase(source_dir)
    except Exception:
        return False


def _kernel_cache_matches(path, source_file, variable):
    if not os.path.exists(path):
        return False
    try:
        with np.load(path, allow_pickle=True) as data:
            cached_source = str(data["source_file"].item())
            cached_variable = str(data["variable"].item())
            return (
                os.path.normcase(cached_source) == os.path.normcase(source_file)
                and cached_variable == variable
            )
    except Exception:
        return False


def prepare_all_cache():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUT_TIF_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    scenario_dirs = {
        name: resolve_albedo_dir(path) for name, path in SCENARIO_DIRS.items()
    }
    lat1d, lon1d, profile_tpl, crs_tpl, shape_hw = get_grid_1d_from_template(
        TEMPLATE_TIF
    )
    height, width = shape_hw
    grid_info = (profile_tpl, crs_tpl, shape_hw)
    paths = cache_paths()

    for path in paths.values():
        _remove_cache_if_requested(path)

    if not os.path.exists(paths["area"]):
        weights = area_lat_weights(lat1d, width)
        np.savez_compressed(paths["area"], weights=weights)

    if not os.path.exists(paths["m85"]):
        m85 = build_m85_3d(lat1d, shape_hw)
        np.savez_compressed(paths["m85"], m85=m85)

    for scenario in ("total", "landcover"):
        cache_key = f"alpha_ref_{scenario}"
        cache_path = paths[cache_key]
        source_dir = scenario_dirs[scenario]
        if not _reference_cache_matches(cache_path, source_dir):
            reference = np.full((12, height, width), np.nan, dtype=np.float32)
            valid = np.zeros((12, height, width), dtype=bool)
            for index, month in enumerate(MONTHS):
                alpha = load_alpha(
                    source_dir, REF_YEAR, month, grid_info=grid_info
                )
                alpha_data = np.asarray(alpha.data, dtype=np.float32)
                alpha_valid = (~np.ma.getmaskarray(alpha)) & np.isfinite(alpha_data)
                reference[index] = np.where(alpha_valid, alpha_data, np.nan)
                valid[index] = alpha_valid
            np.savez_compressed(
                cache_path,
                alpha_ref=reference,
                alpha_ref_valid=valid,
                source_dir=np.array(source_dir),
            )

    kernel_specs = (
        (
            "kernel_cesm",
            PENDER_NC,
            "FSNT",
            "time",
        ),
        (
            "kernel_hadgem3",
            HADGEM3_NC,
            "albedo_sw",
            "month",
        ),
        (
            "kernel_hadgem2",
            resolve_hadgem2_kernel_file(),
            "albedo",
            None,
        ),
    )

    for cache_key, source_file, variable, month_hint in kernel_specs:
        cache_path = paths[cache_key]
        if not _kernel_cache_matches(cache_path, source_file, variable):
            kernel, units, long_name = load_and_regrid_kernel(
                source_file,
                variable,
                lat1d,
                lon1d,
                month_dim_hint=month_hint,
                require_all_sky=True,
            )
            np.savez_compressed(
                cache_path,
                kernel=kernel,
                units=np.array(units),
                long_name=np.array(long_name),
                source_file=np.array(source_file),
                variable=np.array(variable),
            )

    # Check that the two 1700 albedo products define the same baseline wherever
    # both are valid. Small floating-point differences are acceptable.
    with np.load(paths["alpha_ref_total"]) as total_data, np.load(
        paths["alpha_ref_landcover"]
    ) as land_data:
        total_ref = total_data["alpha_ref"]
        land_ref = land_data["alpha_ref"]
        common = np.isfinite(total_ref) & np.isfinite(land_ref)
        if np.any(common):
            maximum_difference = float(
                np.nanmax(np.abs(total_ref[common] - land_ref[common]))
            )
            if maximum_difference > 1.0e-5:
                print(
                    "[WARNING] The total and land-cover scenarios do not share "
                    f"an identical 1700 albedo baseline. max_abs_diff="
                    f"{maximum_difference:.8f}. Each scenario will still be "
                    "calculated relative to its own 1700 baseline."
                )

    return scenario_dirs, profile_tpl, crs_tpl, shape_hw


def load_worker_cache():
    paths = cache_paths()
    with np.load(paths["area"]) as data:
        area_weights = data["weights"].astype(np.float64)
    with np.load(paths["m85"]) as data:
        m85 = data["m85"].astype(bool)

    alpha_ref = {}
    alpha_ref_valid = {}
    for scenario in ("total", "landcover"):
        with np.load(paths[f"alpha_ref_{scenario}"]) as data:
            alpha_ref[scenario] = data["alpha_ref"].astype(np.float32)
            alpha_ref_valid[scenario] = data["alpha_ref_valid"].astype(bool)

    kernels = {}
    for kernel_name, cache_key in zip(
        KERNEL_NAMES,
        ("kernel_cesm", "kernel_hadgem3", "kernel_hadgem2"),
    ):
        with np.load(paths[cache_key], allow_pickle=True) as data:
            kernels[kernel_name] = data["kernel"].astype(np.float32)

    return area_weights, m85, alpha_ref, alpha_ref_valid, kernels


def init_worker(scenario_dirs, profile_tpl, crs_tpl, shape_hw):
    global _WORKER
    area_weights, m85, alpha_ref, alpha_ref_valid, kernels = load_worker_cache()
    _WORKER = {
        "scenario_dirs": scenario_dirs,
        "profile_tpl": profile_tpl,
        "crs_tpl": crs_tpl,
        "shape_hw": shape_hw,
        "area_weights": area_weights,
        "global_area_weight": float(np.sum(area_weights)),
        "m85": m85,
        "alpha_ref": alpha_ref,
        "alpha_ref_valid": alpha_ref_valid,
        "kernels": kernels,
        "month_weights": month_day_weights_normal_year(),
    }


# =============================================================================
# OUTPUT HELPERS
# =============================================================================
def annual_tif_path(component, kernel_name, year):
    return os.path.join(
        OUT_TIF_DIR,
        component,
        kernel_name,
        f"RF_EQ4_{component}_{year}_minus_{REF_YEAR}_{kernel_name}.tif",
    )


def mean3_tif_path(component, year):
    return os.path.join(
        OUT_TIF_DIR,
        component,
        "MEAN3",
        f"RF_EQ4_{component}_{year}_minus_{REF_YEAR}_MEAN3.tif",
    )


def expected_outputs_for_year(year):
    paths = []
    for component in COMPONENTS:
        paths.extend(
            annual_tif_path(component, kernel_name, year)
            for kernel_name in KERNEL_NAMES
        )
        paths.append(mean3_tif_path(component, year))
    return paths


def component_column_prefix(component):
    return component.capitalize()


def empty_result(year):
    row = {"year": year}
    for component in COMPONENTS:
        prefix = component_column_prefix(component)
        for kernel_name in KERNEL_NAMES:
            row[f"{prefix}_{kernel_name}_Wm2"] = np.nan
        row[f"{prefix}_MEAN3_Wm2"] = np.nan
        row[f"{prefix}_Envelope_min_Wm2"] = np.nan
        row[f"{prefix}_Envelope_max_Wm2"] = np.nan
        row[f"{prefix}_Envelope_min_kernel"] = ""
        row[f"{prefix}_Envelope_max_kernel"] = ""
    row["error"] = ""
    return row


def add_component_statistics(row, component, earthmeans):
    prefix = component_column_prefix(component)
    values = np.array(
        [earthmeans[kernel_name] for kernel_name in KERNEL_NAMES],
        dtype=np.float64,
    )
    for kernel_name, value in zip(KERNEL_NAMES, values):
        row[f"{prefix}_{kernel_name}_Wm2"] = float(value)

    row[f"{prefix}_MEAN3_Wm2"] = float(np.mean(values))
    index_min = int(np.argmin(values))
    index_max = int(np.argmax(values))
    row[f"{prefix}_Envelope_min_Wm2"] = float(values[index_min])
    row[f"{prefix}_Envelope_max_Wm2"] = float(values[index_max])
    row[f"{prefix}_Envelope_min_kernel"] = KERNEL_NAMES[index_min]
    row[f"{prefix}_Envelope_max_kernel"] = KERNEL_NAMES[index_max]


def earth_mean_from_tif(path, area_weights, global_area_weight):
    array = read_tif_as_masked(path)
    values = np.asarray(array.filled(0.0), dtype=np.float64)
    return float(np.sum(values * area_weights) / global_area_weight)


def load_existing_year_result(year):
    worker = _WORKER
    row = empty_result(year)
    for component in COMPONENTS:
        earthmeans = {
            kernel_name: earth_mean_from_tif(
                annual_tif_path(component, kernel_name, year),
                worker["area_weights"],
                worker["global_area_weight"],
            )
            for kernel_name in KERNEL_NAMES
        }
        add_component_statistics(row, component, earthmeans)
    return row


# =============================================================================
# YEARLY RF CALCULATION
# =============================================================================
def compute_one_year(year):
    try:
        if _WORKER is None:
            raise RuntimeError("Worker cache was not initialized.")

        if SKIP_IF_ALL_OUTPUTS_EXIST and all(
            os.path.exists(path) for path in expected_outputs_for_year(year)
        ):
            return load_existing_year_result(year)

        worker = _WORKER
        height, width = worker["shape_hw"]
        grid_info = (
            worker["profile_tpl"],
            worker["crs_tpl"],
            worker["shape_hw"],
        )

        # Full-grid annual accumulators. Values outside land, outside M85, or
        # outside a valid kernel are zero in the global integral.
        annual_acc = {
            scenario: {
                kernel_name: np.zeros((height, width), dtype=np.float64)
                for kernel_name in KERNEL_NAMES
            }
            for scenario in ("total", "landcover")
        }
        valid_any = {
            scenario: {
                kernel_name: np.zeros((height, width), dtype=bool)
                for kernel_name in KERNEL_NAMES
            }
            for scenario in ("total", "landcover")
        }

        for month_index, month in enumerate(MONTHS):
            month_weight = float(worker["month_weights"][month])
            m85 = worker["m85"][month_index]

            for scenario in ("total", "landcover"):
                alpha = load_alpha(
                    worker["scenario_dirs"][scenario],
                    year,
                    month,
                    grid_info=grid_info,
                )
                alpha_y = np.asarray(alpha.data, dtype=np.float32)
                alpha_y_valid = (
                    (~np.ma.getmaskarray(alpha)) & np.isfinite(alpha_y)
                )

                alpha_0 = worker["alpha_ref"][scenario][month_index]
                alpha_0_valid = worker["alpha_ref_valid"][scenario][month_index]
                albedo_valid = alpha_y_valid & alpha_0_valid
                delta_alpha = alpha_y - alpha_0

                for kernel_name in KERNEL_NAMES:
                    kernel = worker["kernels"][kernel_name][month_index]
                    valid = albedo_valid & m85 & np.isfinite(kernel)
                    if not np.any(valid):
                        continue

                    rf_month = np.zeros((height, width), dtype=np.float64)
                    rf_month[valid] = (
                        MULT_01
                        * delta_alpha[valid].astype(np.float64)
                        * kernel[valid].astype(np.float64)
                    )

                    # Equation (4), followed by normal-year month weighting.
                    # Do not divide by the sum of valid-month weights.
                    annual_acc[scenario][kernel_name] += rf_month * month_weight
                    valid_any[scenario][kernel_name] |= valid

        component_acc = {
            "total": annual_acc["total"],
            "landcover": annual_acc["landcover"],
            "snow": {
                kernel_name: (
                    annual_acc["total"][kernel_name]
                    - annual_acc["landcover"][kernel_name]
                )
                for kernel_name in KERNEL_NAMES
            },
        }
        component_valid = {
            "total": valid_any["total"],
            "landcover": valid_any["landcover"],
            "snow": {
                kernel_name: (
                    valid_any["total"][kernel_name]
                    & valid_any["landcover"][kernel_name]
                )
                for kernel_name in KERNEL_NAMES
            },
        }

        row = empty_result(year)
        for component in COMPONENTS:
            earthmeans = {}
            output_maps = {}

            for kernel_name in KERNEL_NAMES:
                full_grid = component_acc[component][kernel_name]
                earthmeans[kernel_name] = float(
                    np.sum(full_grid * worker["area_weights"])
                    / worker["global_area_weight"]
                )
                output_maps[kernel_name] = np.where(
                    component_valid[component][kernel_name],
                    full_grid,
                    np.nan,
                ).astype(np.float32)
                write_float_tif(
                    annual_tif_path(component, kernel_name, year),
                    output_maps[kernel_name],
                    worker["profile_tpl"],
                )

            # Require all three kernels to be valid for the displayed mean map.
            common_valid = np.logical_and.reduce(
                [component_valid[component][name] for name in KERNEL_NAMES]
            )
            mean3_full = np.mean(
                np.stack(
                    [component_acc[component][name] for name in KERNEL_NAMES],
                    axis=0,
                ),
                axis=0,
            )
            mean3_map = np.where(common_valid, mean3_full, np.nan).astype(np.float32)
            write_float_tif(
                mean3_tif_path(component, year),
                mean3_map,
                worker["profile_tpl"],
            )

            add_component_statistics(row, component, earthmeans)

        # Numerical closure checks.
        for kernel_name in KERNEL_NAMES:
            total_value = row[f"Total_{kernel_name}_Wm2"]
            land_value = row[f"Landcover_{kernel_name}_Wm2"]
            snow_value = row[f"Snow_{kernel_name}_Wm2"]
            if not np.isclose(
                total_value,
                land_value + snow_value,
                rtol=0.0,
                atol=1.0e-10,
            ):
                raise RuntimeError(
                    f"RF closure failed for {year}, {kernel_name}: "
                    f"total={total_value}, landcover={land_value}, snow={snow_value}"
                )

        return row

    except Exception as error:
        row = empty_result(year)
        row["error"] = (
            f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
        )
        return row


# =============================================================================
# MAIN
# =============================================================================
def csv_fieldnames():
    fields = ["year"]
    for component in COMPONENTS:
        prefix = component_column_prefix(component)
        fields.extend(
            f"{prefix}_{kernel_name}_Wm2" for kernel_name in KERNEL_NAMES
        )
        fields.extend(
            [
                f"{prefix}_MEAN3_Wm2",
                f"{prefix}_Envelope_min_Wm2",
                f"{prefix}_Envelope_max_Wm2",
                f"{prefix}_Envelope_min_kernel",
                f"{prefix}_Envelope_max_kernel",
            ]
        )
    fields.append("error")
    return fields


def main():
    scenario_dirs, profile_tpl, crs_tpl, shape_hw = prepare_all_cache()
    years = list(range(YEAR_START, YEAR_END + 1))
    rows = []

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=init_worker,
        initargs=(scenario_dirs, profile_tpl, crs_tpl, shape_hw),
    ) as executor:
        futures = {executor.submit(compute_one_year, year): year for year in years}
        for future in as_completed(futures):
            result = future.result()
            rows.append(result)
            if result["error"]:
                print(f"[ERROR] year={result['year']}: {result['error'].splitlines()[0]}")
            else:
                print(
                    f"[OK] year={result['year']} | "
                    f"total={result['Total_MEAN3_Wm2']:.6f} | "
                    f"landcover={result['Landcover_MEAN3_Wm2']:.6f} | "
                    f"snow={result['Snow_MEAN3_Wm2']:.6f} W m-2"
                )

    rows.sort(key=lambda item: item["year"])
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fieldnames())
        writer.writeheader()
        writer.writerows(rows)

    failures = sum(bool(row["error"]) for row in rows)
    last = rows[-1]
    print("=" * 100)
    print(
        f"[DONE | manuscript Eq. (4) | ALL-SKY] {YEAR_START}-{YEAR_END} "
        f"relative to {REF_YEAR}"
    )
    print(f"M85: local-noon SZA <= {SZA_THRESH_DEG} deg | {M85_MONTH_RULE}")
    print("Annual RF: monthly global mean first, then normal-year day weighting")
    print(f"Workers: {MAX_WORKERS}")
    print(f"CSV: {CSV_OUT}")
    print(f"GeoTIFF directory: {OUT_TIF_DIR}")
    if failures:
        print(f"WARNING: {failures} years failed. Check the CSV error column.")
    print(
        f"[LAST YEAR {last['year']}] "
        f"total={last['Total_MEAN3_Wm2']:.6f} | "
        f"landcover={last['Landcover_MEAN3_Wm2']:.6f} | "
        f"snow={last['Snow_MEAN3_Wm2']:.6f} W m-2"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
