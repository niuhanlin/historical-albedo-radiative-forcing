# -*- coding: utf-8 -*-
"""Synthesize monthly 0.25° global land surface albedo for 1700–2023.

The calculation follows the manuscript equations:

    alpha_sf = sum_c(f_c * alpha_sf,c)
    alpha_sn = sum_c(f_c * alpha_sn,c)
    alpha    = (1 - SCF) * alpha_sf + SCF * alpha_sn

The five input classes are forest, shrubland, grassland, cropland and
non-vegetated land.

Two products are written in one run:
    dynamic  : annual land cover + annual/monthly SCF (total albedo)
    fixed1700: annual land cover + 1700 monthly SCF (land-cover-only albedo)
"""

from __future__ import annotations

import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from osgeo import gdal, osr


gdal.UseExceptions()
gdal.SetConfigOption("GDAL_CACHEMAX", "2048")


# =============================================================================
# USER CONFIGURATION
# =============================================================================
YEAR_START, YEAR_END = 1700, 2023
FIXED_SCF_YEAR = 1700
LOOKUP_YEAR = 2020

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("ALBEDO_DATA_DIR", PROJECT_DIR / "data"))

NOSNOW_ALBEDO_DIR = Path(
    os.environ.get(
        "SNOWFREE_ALBEDO_DIR",
        DATA_DIR / "albedo_lookup" / "snow_free",
    )
)
SNOW_ALBEDO_DIR = Path(
    os.environ.get(
        "SNOWCOVERED_ALBEDO_DIR",
        DATA_DIR / "albedo_lookup" / "snow_covered",
    )
)
FRACTION_DIR = Path(
    os.environ.get(
        "LANDCOVER_FRACTION_DIR",
        DATA_DIR / "land_cover_fractions",
    )
)

# Produced by reconstruct_historical_snow_cover.py.
SCF_DIR = Path(
    os.environ.get(
        "SCF_DIR",
        PROJECT_DIR
        / "outputs"
        / "historical_snow_reconstruction"
        / "snowfrac_025_1700_2023",
    )
)
SCF_VARIABLE = "snow_cover_frac_025"

OUTPUT_ROOT = Path(
    os.environ.get(
        "ALBEDO_OUTPUT_DIR",
        PROJECT_DIR / "outputs" / "albedo_synthesis_5LC_1700_2023",
    )
)

NODATA = -9999.0
FRACTION_SCALE = 1.0e6
FRACTION_SUM_TOLERANCE = 0.02
FRACTION_EPS = 1.0e-8

# Set to 1.0 for physical albedo or 0.001 for scaled integers. None detects
# 0.001 scaling when the 99th percentile is greater than 2.
ALBEDO_SCALE: float | None = None

SCF_NORTH_TO_SOUTH = True
SKIP_COMPLETE_YEARS = True
WRITE_ANNUAL_MEAN = True
MAX_WORKERS = int(os.environ.get("ALBEDO_WORKERS", "8"))

# One-based band order in each annual five-band land-cover fraction stack.
# The same names are used for the lookup-table subdirectories and filenames.
LAND_COVER_CLASSES = (
    ("forest", 1),
    ("shrubland", 2),
    ("grassland", 3),
    ("cropland", 4),
    ("nonvegetated", 5),
)


# =============================================================================
# GRID AND FILE HELPERS
# =============================================================================
def choose_first_match(patterns: list[str]) -> Path:
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return Path(matches[0])
    raise FileNotFoundError("No file matched:\n" + "\n".join(patterns))


def open_raster(path: Path):
    ds = gdal.Open(str(path))
    if ds is None:
        raise RuntimeError(f"Cannot open raster: {path}")
    return ds


def get_reference_grid():
    path = choose_first_match(
        [
            str(NOSNOW_ALBEDO_DIR / "*" / "*albedoNoSnow*0p25deg_global_wgs84*.tif"),
            str(NOSNOW_ALBEDO_DIR / "*" / "*albedoNosnow*0p25deg_global_wgs84*.tif"),
        ]
    )
    ds = open_raster(path)
    result = (
        ds.GetGeoTransform(),
        ds.GetProjection(),
        ds.RasterYSize,
        ds.RasterXSize,
    )
    ds = None
    return result


def projections_equal(wkt_a: str, wkt_b: str) -> bool:
    if not wkt_a or not wkt_b:
        return wkt_a == wkt_b
    a = osr.SpatialReference()
    b = osr.SpatialReference()
    a.ImportFromWkt(wkt_a)
    b.ImportFromWkt(wkt_b)
    return bool(a.IsSame(b))


def assert_grid(ds, ref_gt, ref_projection, rows, cols, label: str) -> None:
    if (ds.RasterYSize, ds.RasterXSize) != (rows, cols):
        raise RuntimeError(
            f"Grid size mismatch for {label}: "
            f"{(ds.RasterYSize, ds.RasterXSize)} != {(rows, cols)}"
        )
    if np.max(np.abs(np.asarray(ds.GetGeoTransform()) - np.asarray(ref_gt))) > 1.0e-8:
        raise RuntimeError(f"Geotransform mismatch for {label}")
    if not projections_equal(ds.GetProjection(), ref_projection):
        raise RuntimeError(f"Projection mismatch for {label}")


def postprocess_albedo(raw: np.ndarray, nodata) -> np.ndarray:
    arr = raw.astype(np.float32)
    if nodata is not None:
        arr[arr == nodata] = np.nan
    if ALBEDO_SCALE is not None:
        arr *= np.float32(ALBEDO_SCALE)
    elif np.isfinite(arr).any() and np.nanpercentile(arr, 99) > 2.0:
        arr *= np.float32(0.001)
    arr[(arr < 0.0) | (arr > 1.0)] = np.nan
    return arr


def find_endmember_file(condition: str, source_name: str, month: int) -> Path:
    """Return the fixed 2020 monthly albedo lookup map.

    The manuscript holds the lookup-table albedos fixed across 1700–2023.
    Consequently, target-year lookup files must never be selected here.
    """
    if condition == "snowfree":
        root = NOSNOW_ALBEDO_DIR
        labels = ("albedoNoSnow", "albedoNosnow")
    elif condition == "snowcovered":
        root = SNOW_ALBEDO_DIR
        labels = ("albedoWithSnow",)
    else:
        raise ValueError(condition)

    folder = root / source_name
    patterns = []
    for label in labels:
        patterns.extend(
            [
                str(folder / f"{source_name}_{label}_{LOOKUP_YEAR}_{month:02d}_0p25deg_global_wgs84*_from2020.tif"),
                str(folder / f"{source_name}_{label}_{LOOKUP_YEAR}_{month:02d}_0p25deg_global_wgs84*.tif"),
            ]
        )
    return choose_first_match(patterns)


def read_endmember(
    condition: str,
    source_name: str,
    month: int,
    ref_gt,
    ref_projection,
    rows: int,
    cols: int,
) -> np.ndarray:
    path = find_endmember_file(condition, source_name, month)
    ds = open_raster(path)
    assert_grid(ds, ref_gt, ref_projection, rows, cols, str(path))
    band = ds.GetRasterBand(1)
    raw = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    ds = None
    return postprocess_albedo(raw, nodata)


def read_fraction_stack(year, ref_gt, ref_projection, rows, cols):
    path = FRACTION_DIR / f"LUH2_HYDE_fused_{year}_global0.25D.tif"
    ds = open_raster(path)
    assert_grid(ds, ref_gt, ref_projection, rows, cols, str(path))
    expected_bands = len(LAND_COVER_CLASSES)
    if ds.RasterCount != expected_bands:
        raise RuntimeError(
            f"Expected {expected_bands} land-cover fraction bands in {path}; "
            f"found {ds.RasterCount}"
        )
    raw = ds.ReadAsArray().astype(np.float32)
    ds = None
    if raw.ndim != 3 or raw.shape[0] != expected_bands:
        raise RuntimeError(f"Unexpected fraction-stack shape in {path}: {raw.shape}")

    fractions = raw / np.float32(FRACTION_SCALE)
    fractions[~np.isfinite(fractions)] = 0.0
    fractions = np.clip(fractions, 0.0, 1.0)
    fraction_sum = np.sum(fractions, axis=0)
    land = fraction_sum > FRACTION_EPS
    bad = land & (np.abs(fraction_sum - 1.0) > FRACTION_SUM_TOLERANCE)
    if np.any(bad):
        values = fraction_sum[bad]
        raise RuntimeError(
            f"Land-cover fractions fail closure in {bad.sum()} cells for {year}; "
            f"invalid range={values.min():.4f}–{values.max():.4f}."
        )
    return fractions, land


def read_scf(year: int, rows: int, cols: int) -> np.ndarray:
    path = SCF_DIR / f"snowfrac_025_{year}.mat"
    if not path.exists():
        raise FileNotFoundError(f"Missing SCF file: {path}")
    with h5py.File(path, "r") as handle:
        if SCF_VARIABLE not in handle:
            raise KeyError(f"Variable {SCF_VARIABLE!r} is absent from {path}")
        scf = np.asarray(handle[SCF_VARIABLE][()], dtype=np.float32)
    if scf.shape != (12, rows, cols):
        raise RuntimeError(f"Unexpected SCF shape in {path}: {scf.shape}")
    if not SCF_NORTH_TO_SOUTH:
        scf = scf[:, ::-1, :]
    invalid = ~np.isfinite(scf)
    scf = np.clip(scf, 0.0, 1.0)
    scf[invalid] = np.nan
    return scf


def write_tif_atomic(path: Path, arr: np.ndarray, gt, projection) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    if tmp.exists():
        tmp.unlink()
    rows, cols = arr.shape
    ds = gdal.GetDriverByName("GTiff").Create(
        str(tmp),
        cols,
        rows,
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    ds.SetGeoTransform(gt)
    ds.SetProjection(projection)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(NODATA)
    band.WriteArray(np.where(np.isfinite(arr), arr, NODATA).astype(np.float32))
    band.FlushCache()
    ds.FlushCache()
    ds = None
    os.replace(tmp, path)


# =============================================================================
# OUTPUTS
# =============================================================================
def monthly_output(scenario: str, year: int, month: int) -> Path:
    return (
        OUTPUT_ROOT
        / scenario
        / "monthly"
        / f"albedo_weighted_{year}_{month:02d}_0p25deg_global_wgs84.tif"
    )


def annual_output(scenario: str, year: int) -> Path:
    return (
        OUTPUT_ROOT
        / scenario
        / "annual_mean"
        / f"albedo_weighted_{year}_ANN_0p25deg_global_wgs84.tif"
    )


def year_complete(year: int) -> bool:
    paths = []
    for scenario in ("dynamic", "fixed1700"):
        paths.extend(monthly_output(scenario, year, month) for month in range(1, 13))
        if WRITE_ANNUAL_MEAN:
            paths.append(annual_output(scenario, year))
    return all(path.exists() and path.stat().st_size > 0 for path in paths)


# =============================================================================
# ALBEDO SYNTHESIS
# =============================================================================
def synthesize_state_albedo(
    condition,
    year,
    month,
    fractions,
    ref_gt,
    ref_projection,
    rows,
    cols,
):
    """Return one snow-state albedo and a missing-required-input mask."""
    total = np.zeros((rows, cols), dtype=np.float32)
    missing_required = np.zeros((rows, cols), dtype=bool)

    for source_name, one_based_band in LAND_COVER_CLASSES:
        fraction = fractions[one_based_band - 1]
        endmember = read_endmember(
            condition,
            source_name,
            month,
            ref_gt,
            ref_projection,
            rows,
            cols,
        )
        missing_required |= (fraction > FRACTION_EPS) & (~np.isfinite(endmember))
        total += np.where(
            np.isfinite(endmember), fraction * endmember, 0.0
        ).astype(np.float32)
    return total, missing_required


def process_one_year(year, ref_gt, ref_projection, rows, cols):
    if SKIP_COMPLETE_YEARS and year_complete(year):
        return f"{year}: skipped (complete)"

    fractions, land = read_fraction_stack(
        year, ref_gt, ref_projection, rows, cols
    )
    scf_dynamic = read_scf(year, rows, cols)
    scf_fixed = read_scf(FIXED_SCF_YEAR, rows, cols)
    scf_by_scenario = {"dynamic": scf_dynamic, "fixed1700": scf_fixed}

    annual_sum = {
        scenario: np.zeros((rows, cols), dtype=np.float64)
        for scenario in scf_by_scenario
    }
    annual_count = {
        scenario: np.zeros((rows, cols), dtype=np.uint8)
        for scenario in scf_by_scenario
    }

    for month in range(1, 13):
        alpha_sf, missing_sf = synthesize_state_albedo(
            "snowfree",
            year,
            month,
            fractions,
            ref_gt,
            ref_projection,
            rows,
            cols,
        )
        alpha_sn, missing_sn = synthesize_state_albedo(
            "snowcovered",
            year,
            month,
            fractions,
            ref_gt,
            ref_projection,
            rows,
            cols,
        )
        missing_required = missing_sf | missing_sn

        for scenario, scf12 in scf_by_scenario.items():
            scf = scf12[month - 1]
            valid = land & np.isfinite(scf) & (~missing_required)
            alpha = np.full((rows, cols), np.nan, dtype=np.float32)
            alpha[valid] = (
                (1.0 - scf[valid]) * alpha_sf[valid]
                + scf[valid] * alpha_sn[valid]
            )
            if np.any(valid & ((alpha < 0.0) | (alpha > 1.0))):
                raise RuntimeError(
                    f"Composite albedo outside [0,1]: {scenario}, {year}-{month:02d}"
                )

            write_tif_atomic(
                monthly_output(scenario, year, month),
                alpha,
                ref_gt,
                ref_projection,
            )
            annual_sum[scenario] += np.where(valid, alpha, 0.0)
            annual_count[scenario] += valid.astype(np.uint8)

    if WRITE_ANNUAL_MEAN:
        for scenario in scf_by_scenario:
            full_year = annual_count[scenario] == 12
            annual = np.full((rows, cols), np.nan, dtype=np.float32)
            annual[full_year] = (
                annual_sum[scenario][full_year] / 12.0
            ).astype(np.float32)
            write_tif_atomic(
                annual_output(scenario, year), annual, ref_gt, ref_projection
            )

    return f"{year}: complete"


def main() -> None:
    ref_gt, ref_projection, rows, cols = get_reference_grid()
    years = list(range(YEAR_START, YEAR_END + 1))
    workers = max(1, min(MAX_WORKERS, len(years)))
    print(
        f"Synthesizing dynamic and fixed-1700-SCF albedo for "
        f"{YEAR_START}–{YEAR_END} with {workers} workers"
    )

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_one_year,
                year,
                ref_gt,
                ref_projection,
                rows,
                cols,
            ): year
            for year in years
        }
        for future in as_completed(futures):
            year = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                raise RuntimeError(f"Albedo synthesis failed for {year}") from exc

    print("Surface-albedo synthesis is complete.")


if __name__ == "__main__":
    main()
