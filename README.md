# Historical surface albedo and radiative forcing (1700–2023)

This repository contains the three principal analysis scripts supporting the
historical snow-cover reconstruction, surface-albedo synthesis, and
top-of-atmosphere shortwave radiative-forcing calculation.

## Code overview

Run the scripts in the following order.

### 1. `reconstruct_historical_snow_cover.py`

Reconstructs monthly global snow cover fraction (SCF) for 1901–2000 using
monthly CRU TS v4.08 climate variables and MODIS SCF observations from
2001–2023.

For every target grid cell and calendar month, same-month observations from
2001–2023 are screened using an initial monthly mean temperature tolerance of
±2 °C. If fewer than 250 candidates are available, the tolerance is increased
by a factor of 1.5 up to a maximum of ±5 °C. All candidates within the final
tolerance are weighted by composite climate distance, mean temperature
difference and latitude difference. The weighted samples are then used to fit a
local ridge regression for the target grid cell. The climate, temperature and
latitude decay scales are 2.5, 2.0 °C and 6.0°, respectively.

The script then creates a unified set of yearly SCF files:

- 1700–1900: monthly SCF fixed at the reconstructed 1901 values;
- 1901–2000: reconstructed monthly SCF;
- 2001–2023: observed monthly MODIS SCF.

Output:

```text
snowfrac_025_1700_2023/snowfrac_025_<YEAR>.mat
```

Each file contains:

```text
snow_cover_frac_025   # dimensions: month, lat, lon = 12, 720, 1440
```

The script uses monthly MODIS SCF and CRU TS v4.08 variables harmonized to the
0.25° analysis grid.

### 2. `synthesize_surface_albedo_5LC.py`

Combines annual land-cover fractions, fixed monthly snow-free and snow-covered
albedo lookup maps, and monthly SCF:

```text
alpha_sf = sum_c(f_c * alpha_sf,c)
alpha_sn = sum_c(f_c * alpha_sn,c)
alpha    = (1 - SCF) * alpha_sf + SCF * alpha_sn
```

The five reported land-cover classes are:

1. Forest
2. Shrubland
3. Grassland
4. Cropland
5. Non-vegetated land

The source fraction stack contains eight bands. The four forest
subtypes (ENF, EBF, DNF and DBF) are grouped into the single forest class by
summing their contributions. This aggregation preserves their combined albedo
contribution exactly.

The script produces two internally consistent albedo experiments:

| Output | Land-cover fractions | SCF |
|---|---|---|
| `dynamic` | Vary annually | Varies monthly and annually |
| `fixed1700` | Vary annually | Fixed at monthly 1700 values |

Monthly output naming:

```text
albedo_synthesis_5LC_1700_2023/
├── dynamic/monthly/albedo_weighted_<YEAR>_<MONTH>_0p25deg_global_wgs84.tif
└── fixed1700/monthly/albedo_weighted_<YEAR>_<MONTH>_0p25deg_global_wgs84.tif
```

Source band 7 is defined as total cropland, including rice. The script verifies
that all land-cover fractions sum to approximately one.

### 3. `calculate_albedo_radiative_forcing.py`

Calculates annual Earth-mean top-of-atmosphere shortwave radiative forcing from
the two monthly albedo experiments using three all-sky albedo kernels:

- CESM-CAM5 (`FSNT`);
- HadGEM3-GA7.1 (`albedo_sw`);
- HadGEM2 (`albedo`).

For kernel `r`, year `y`, month `m`, and grid cell `x`:

```text
RF_r(y,m,x) = 100 * [alpha(y,m,x) - alpha(1700,m,x)]
              * K_r(m,x) * M85(m,x)
```

The factor 100 converts fractional albedo change to percentage-point albedo
change because the kernels are expressed per 1% albedo perturbation. `M85`
excludes months/grid cells with local-noon solar zenith angles outside the
selected threshold. Excluded contributions are set to zero and are not
renormalized.

Annual values use normal-year month-length weights. Earth-mean forcing is
calculated using the whole-Earth area as the denominator, with ocean and masked
land cells contributing zero to the global integral.

The script calculates all three components in one run:

```text
RF_total      = RF(dynamic albedo)
RF_landcover  = RF(fixed-1700-SCF albedo)
RF_snow       = RF_total - RF_landcover
```

It writes annual GeoTIFFs for each kernel and their three-kernel mean, together
with a CSV containing the annual Earth-mean time series and kernel envelope.

## Input data

The scripts use harmonized inputs derived from the datasets listed under
**Data access**, including monthly snow cover and climate variables, annual
land-cover fractions, land-cover-specific snow-free and snow-covered albedo
lookup tables, and monthly all-sky shortwave albedo kernels. Intermediate SCF
and albedo products are generated sequentially by the first two scripts. Data
files are not duplicated in this repository because the source datasets are
large and are maintained by their respective providers.

## Data access

The third-party datasets used to prepare the model inputs are available from
their original repositories:

| Dataset | Official access |
|---|---|
| MODIS/Terra+Aqua BRDF/Albedo Daily L3 Global 500 m, Collection 6.1 (`MCD43A3`) | [NASA LP DAAC](https://doi.org/10.5067/MODIS/MCD43A3.061) |
| MODIS/Terra+Aqua BRDF/Albedo Quality Daily L3 Global 500 m, Collection 6.1 (`MCD43A2`) | [NASA LP DAAC](https://doi.org/10.5067/MODIS/MCD43A2.061) |
| MODIS/Terra+Aqua Land Cover Type Yearly L3 Global 500 m, Collection 6.1 (`MCD12Q1`) | [NASA LP DAAC](https://doi.org/10.5067/MODIS/MCD12Q1.061) |
| MODIS/Terra Snow Cover Monthly L3 Global 0.05° CMG, Collection 6.1 (`MOD10CM`) | [NASA NSIDC DAAC](https://doi.org/10.5067/MODIS/MOD10CM.061) |
| CRU TS v4.08 | [Climatic Research Unit, University of East Anglia](https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.08/) |
| Land-Use Harmonization v2 (`LUH2`) | [University of Maryland LUH2 data portal](https://luh.umd.edu/data.shtml) |
| NCEP/NCAR Reanalysis 1 | [NOAA Physical Sciences Laboratory](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html) |
| CESM-CAM5 radiative kernels | [UCAR/NCAR Research Data Archive](https://doi.org/10.5065/D6F47MT6) |
| HadGEM3-GA7.1 radiative kernels | [Zenodo](https://doi.org/10.5281/zenodo.3594673) |
| HadGEM2 radiative kernels | [Research Data Leeds Repository](https://doi.org/10.5518/406) |

A NASA Earthdata Login is required to access the MODIS products. Users should
follow the licences and citation requirements specified by each data provider.
The scripts use harmonized inputs derived from these source datasets.

## Configuration

The scripts use repository-relative `data/` and `outputs/` directories by
default. Input and output locations can be overridden with environment
variables without modifying the source code:

```text
HISTSNOW_DATA_DIR
HISTSNOW_OUTPUT_DIR
HISTSNOW_MAX_DONORS_PER_YEAR_MONTH
HISTSNOW_MAX_TOTAL_DONORS_PER_MONTH
ALBEDO_DATA_DIR
SNOWFREE_ALBEDO_DIR
SNOWCOVERED_ALBEDO_DIR
LANDCOVER_FRACTION_DIR
SCF_DIR
ALBEDO_OUTPUT_DIR
RF_DATA_DIR
RADIATIVE_KERNEL_DIR
ALBEDO_ROOT
RF_OUTPUT_DIR
```

The two `HISTSNOW_MAX_*` donor caps are disabled by default so that the
manuscript calculation uses all available candidates. They may be set to
positive integers only for memory-constrained exploratory runs.

`ALBEDO_OUTPUT_DIR` and `ALBEDO_ROOT` must refer to the same albedo-synthesis
directory when non-default locations are used.

The default parallel settings can be changed through environment variables:

```bat
set HISTSNOW_N_JOBS=12
set HISTSNOW_FULLFIELD_N_JOBS=12
set ALBEDO_WORKERS=8
set RF_MAX_WORKERS=8
```

Use conservative worker counts initially. The historical pixel-level snow
reconstruction and the RF kernel cache can require substantial memory.

## Software requirements

- Python 3.10 or later
- NumPy
- pandas
- xarray
- h5py
- scikit-learn
- joblib
- matplotlib
- rasterio
- GDAL Python bindings
- a NetCDF backend such as netCDF4 or h5netcdf

Example:

```bash
conda install -c conda-forge python=3.11 numpy pandas xarray h5py scikit-learn joblib matplotlib rasterio gdal netcdf4
```

## Run

```bash
python reconstruct_historical_snow_cover.py
python synthesize_surface_albedo_5LC.py
python calculate_albedo_radiative_forcing.py
```
