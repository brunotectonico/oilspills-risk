# oilspills-risk

This repository provides a modular coastal oil-spill risk screening workflow for tanker traffic hotspots (GMTDS) and simplified OSCAR-driven trajectories.

## Current status

### ✅ Implemented
1. **Hotspot extraction from GMTDS tanker rasters**
   - Reads monthly GeoTIFFs inside GMTDS ZIP archives.
   - Filters to region limits (defaults around Djibouti / Bab el Mandeb).
   - Detects high-density pixels with percentile thresholding.
   - Clusters hotspot candidates with DBSCAN.
   - Exports detailed and monthly summary CSV outputs.

2. **Optional mean density raster outputs (all pixels)**
   - Can save GeoTIFFs with mean density computed from **all valid raster pixels**, not only hotspot pixels.
   - Supports output frequency:
     - `monthly`: 12 mean rasters (`M01`..`M12`) aggregated across all analyzed years
     - `seasonal`: one mean raster per seasonal window (e.g., 3-month period)

3. **Optional seasonal window (e.g., 3 months)**
   - You can limit hotspot analysis to a period with `--season-start-month` and `--season-length-months`.
   - Summary output includes a `period_id` to keep analyses separated.

4. **Reusable trajectory/risk module (simple version)**
   - Particle advection + random walk diffusion.
   - Simplified weathering as daily bulk mass loss (default 30%/day).
   - Composite score: `density_factor × coastal_hit_fraction × survival_fraction`.

5. **Modular OSCAR download and grid-processing workflow**
   - Shared dataclasses and study-area validation live in `oilspill_risk/models.py`.
   - Date windows live in `oilspill_risk/periods.py`.
   - PO.DAAC command/auth helpers live in `oilspill_risk/podaac.py`.
   - Longitude/latitude reconstruction, clipping, and GeoTIFF export live in `oilspill_risk/gridding.py`.
   - Period download orchestration lives in `oilspill_risk/oscar_workflow.py`.
   - `oilspill_risk/oscar.py` remains as a backward-compatible facade.

## Repository structure

- `density_hotspots.py`
  CLI wrapper for hotspot extraction.
- `oilspill_risk/hotspots.py`
  Reusable hotspot detection and outputs.
- `oilspill_risk/density_rasters.py`
  Separate mean-density raster aggregation module.
- `oilspill_risk/trajectory.py`
  Reusable trajectory simulation and coastal risk scoring primitives.
- `oilspill_risk/coastlines.py`
  Coastline shapefile/GeoPackage reading and point sampling utilities.
- `oilspill_risk/exposure.py`
  Grid-based coastward current exposure probability and GeoTIFF outputs.
- `oilspill_risk/mapping.py`
  Cartography helpers for currents, coastlines, and traffic-density hotspots.
- `oilspill_risk/models.py`
  OSCAR `StudyArea`, `OscarDownloadConfig`, and study-area helpers.
- `oilspill_risk/periods.py`
  Month-aligned period generation for downloads/exports.
- `oilspill_risk/podaac.py`
  PO.DAAC downloader command construction and Earthdata `.netrc` setup.
- `oilspill_risk/gridding.py`
  OSCAR coordinate standardization, clipping, and north-up EPSG:4326 GeoTIFF export.
- `oilspill_risk/oscar_workflow.py`
  High-level OSCAR period download and optional standardization workflow.
- `oilspill_risk/oscar.py`
  Compatibility imports for existing notebooks/scripts.
- `hotspots4density.ipynb`
  Notebook exploration.

## Study area used for OSCAR Red Sea / Djibouti tests

```python
from oilspill_risk.models import StudyArea

area = StudyArea(lon_min=41.5, lon_max=45.75, lat_min=9.75, lat_max=14.75)
```

## Run hotspot extraction

```bash
python density_hotspots.py /path/to/gmtds_data \
  --pattern "*Tankers.zip" \
  --mean-raster-dir mean_density_rasters \
  --mean-raster-frequency seasonal \ # or monthly
  --season-start-month 1 \
  --season-length-months 3
```

### Expected GMTDS input filename patterns

The hotspot CLI expects one ZIP archive per analyzed year and one or more monthly GeoTIFFs inside each ZIP. Year and month are parsed from filenames before the raster is read:

- ZIP archive names must contain an underscore followed by a 4-digit year, for example `GMTDS_Tankers_2020.zip` or `GMTDS_2020_Tankers.zip`.
- GeoTIFF member names inside each ZIP must contain an underscore, a 2-digit month, and another underscore, for example `GMTDS_Tankers_01_density.tif` for January.
- The default CLI pattern is `*Tankers.zip`; use `--pattern` if your archive names differ while still following the year parsing rule.
- Files that do not match these patterns raise a parsing error. During batch ZIP processing, errors are logged with the offending TIFF/ZIP name and processing continues with the next monthly raster.

## Secure Earthdata auth with PO.DAAC downloader

For the `podaac-data-downloader` mode (from `podaac/data-subscriber`), avoid hardcoding credentials in scripts.

Recommended pattern:
- Set environment variables `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` in your shell/session.
- Call `run_podaac_downloader(...)`; the helper writes a local netrc with restricted permissions and runs the CLI.

```python
from pathlib import Path
from oilspill_risk.gridding import export_oscar_uv_geotiff, standardize_oscar_uv_netcdf
from oilspill_risk.models import StudyArea

bbox = StudyArea(lon_min=41.5, lon_max=45.75, lat_min=9.75, lat_max=14.75)
result = run_podaac_downloader(
    collection="OSCAR_L4_OC_FINAL_V2.0",
    output_dir=Path("oscar_downloads"),
    start_date="2020-01-01T00:00:00Z",
    end_date="2020-01-03T23:59:59Z",
    bbox=bbox,
    dry_run=True,
)
print(result.stdout)
```

## Standardize the already downloaded OSCAR NetCDF files

The downloaded OSCAR final files should be standardized from the global grid, not clipped by the downloader first.  This keeps the original global coordinate metadata available for reconstructing true degree coordinates before the local clip and GeoTIFF export.
To test coordinate rearrangement without clipping, pass `area=None` (or omit it). This writes the full global OSCAR U/V grids after longitude normalization and north-up raster orientation.

```python
from pathlib import Path
from oilspill_risk.gridding import export_oscar_uv_geotiff, standardize_oscar_uv_netcdf
from oilspill_risk.models import StudyArea

area = StudyArea(lon_min=41.5, lon_max=45.75, lat_min=9.75, lat_max=14.75)
for raw_nc in sorted(Path("files").glob("oscar_currents_final_*.nc")):
    date_id = raw_nc.stem.rsplit("_", 1)[-1]
    clipped_nc = raw_nc.with_name(f"oscar_uv_std_{date_id}.nc")
    standardize_oscar_uv_netcdf(raw_nc, clipped_nc, area) # area = None if no clipping to be done
    export_oscar_uv_geotiff(clipped_nc, raw_nc.parent, area)
```

### NetCDF / GeoTIFF coordinate-placement notes

- `standardize_oscar_uv_netcdf(...)` reconstructs index-like OSCAR coordinates from `geospatial_*` metadata before clipping.
- Longitudes are normalized to `[-180, 180]`, and NetCDF U/V variables are written in conventional non-spatial, `lat`, `lon` dimension order for QGIS.
- Standardized NetCDF latitude is written north-to-south so the NetCDF grid opens north-up in GIS; GeoTIFF export still re-sorts internally as needed.
- GeoTIFFs are written north-up using a transform derived from coordinate **centers** expanded by half a grid cell. This avoids the half-cell placement error caused by using center min/max values as raster bounds.
- The GeoTIFF write flips rows vertically only when latitude is stored south-to-north; it does **not** flip columns, which would mirror the Red Sea/Djibouti subset east-west.

### Which current format should trajectories use?

For the current `trajectory.py` workflow, either NetCDF or GeoTIFF can be loaded into the same `CurrentField` structure. Use NetCDF when you want to preserve time slices or average across time; use paired U/V GeoTIFFs when you want the currents to follow the same QGIS/GDAL raster conventions as marine-traffic density rasters.

```python
from pathlib import Path
from oilspill_risk.trajectory import current_field_from_geotiff, current_field_from_netcdf

# QGIS/GDAL-style raster workflow, aligned with traffic-density rasters
currents_from_raster = current_field_from_geotiff(
    Path("files/oscar_uv_clip_20200101_u.tif"),
    Path("files/oscar_uv_clip_20200101_v.tif"),
)

# NetCDF workflow, useful when retaining the time dimension
currents_from_netcdf = current_field_from_netcdf(Path("files/oscar_uv_clip_20200101.nc"))
```

## Mapping currents and traffic-density hotspots

`oilspill_risk/mapping.py` hosts reusable cartography helpers. It can read U/V currents from paired GeoTIFFs or NetCDF through the same loaders used by `trajectory.py`, plot current intensity as a color field, draw current orientation vectors, and overlay hotspots with symbol size scaled by density.

```python
from pathlib import Path
from oilspill_risk.mapping import plot_current_orientation_intensity, plot_hotspots

ax, mesh, quiver = plot_current_orientation_intensity(
    u_tif=Path("files/oscar_uv_clip_20200101_u.tif"),
    v_tif=Path("files/oscar_uv_clip_20200101_v.tif"),
    stride=3,
)
plot_hotspots(Path("gmtds_tanker_hotspots_multi.csv"), ax=ax, density_col="mean_density")
```

To plot directly from a standardized NetCDF current file instead of paired GeoTIFFs, pass `nc_path` and any NetCDF loader options through the same helper:

```python
ax, mesh, quiver = plot_current_orientation_intensity(
    nc_path=Path("files/oscar_uv_clip_20200101.nc"),
    time_index=0,  # omit to average over time by default
    stride=3,
)
plot_hotspots(Path("gmtds_tanker_hotspots_multi.csv"), ax=ax, density_col="mean_density")
```

Cartopy coastlines can be added with `add_cartopy_coastlines(ax)`. Cartopy is optional and only required when that coastline helper is called.

## Coastward current exposure maps

The grid-based exposure workflow estimates where daily OSCAR current vectors point toward coastlines or user-selected coastal target segments. This is a current-direction screening layer, separate from the particle trajectory functions that remain available for later source-based simulations.

The workflow is:

1. load standardized daily currents from NetCDF or paired U/V GeoTIFFs into a `CurrentField`;
2. read a coastline Shapefile/GeoPackage and sample line, polygon-boundary, or point geometries into coastline points;
3. optionally keep only selected coastline segments via an attribute column;
4. for every current-grid pixel, find the nearest sampled coast/target point;
5. compare the local current vector with the pixel-to-coast vector;
6. compute a daily probability layer from `max(0, cos(theta))`, optionally weighted by current speed and distance decay;
7. average daily probability layers into monthly or seasonal products according to the user-selected period.

```python
from pathlib import Path
from oilspill_risk.coastlines import coast_points_from_shapefile
from oilspill_risk.exposure import (
    aggregate_exposure_probabilities_for_periods,
    exposure_from_netcdf,
    write_exposure_geotiff,
)
from oilspill_risk.periods import seasonal_periods

periods = seasonal_periods(
    "2020-01-01T00:00:00Z",
    "2020-03-31T23:59:59Z",
    season_length_months=1,
)

coast_points = coast_points_from_shapefile(
    Path("coast/coastline.shp"),
    spacing=0.01,  # decimal degrees when working with EPSG:4326 OSCAR grids
    target_crs="EPSG:4326",
    segment_id_col="segment_id",
    selected_segments={"djibouti", "yemen"},
)

daily_results = [
    exposure_from_netcdf(
        nc_path,
        coast_points,
        alpha=2.0,
        include_speed=True,
        current_kwargs={"input_units": "m/s"},
    )
    for _, _, period_id in periods
    for nc_path in sorted(Path("files").glob(f"oscar_uv_std_clip_{period_id}_*.nc"))
]
period_results = aggregate_exposure_probabilities_for_periods(daily_results, periods)
for period_id, result in period_results.items():
    write_exposure_geotiff(result, Path(f"outputs/coastward_probability_{period_id}.tif"))
```

Use the same coordinate reference system for coast points and current-grid coordinates. With the current OSCAR lon/lat workflow, `target_crs="EPSG:4326"` means `spacing` is interpreted in decimal degrees. Standardized OSCAR outputs from `download_oscar_for_periods(..., standardize=True)` include the `period_id` returned by `seasonal_periods()` in their filenames, so exposure results can infer the period suffix and aggregate with `aggregate_exposure_probabilities_for_periods(...)`. Combining these exposure maps with GMTDS traffic density is intentionally left for a later step.

## Trajectory and spill-risk analysis logic

`HotspotSource` is the bridge between the traffic-density workflow and the spill-trajectory workflow: it stores a candidate spill origin (`lon`, `lat`), its traffic-derived `density_weight`, and a stable `hotspot_id`. Use `hotspot_sources_from_records(...)` to convert a hotspot CSV or table into these source objects. The optional `mean_density_min` filter keeps only hotspots whose selected density column is high enough to be treated as potential spill sources.

```python
import numpy as np
from pathlib import Path
from oilspill_risk.trajectory import (
    SimulationConfig,
    current_field_from_netcdf,
    estimate_coastal_risk_for_sources,
    hotspot_sources_from_records,
)

sources = hotspot_sources_from_records(
    Path("gmtds_tanker_hotspots_multi.csv"),
    density_col="mean_density",
    mean_density_min=100.0,  # choose after inspecting the density distribution
)
currents = current_field_from_netcdf(Path("files/oscar_uv_clip_20200101.nc"))
coast_points = np.array([
    [43.0, 11.5],
    [43.1, 11.6],
])
risk_rows = estimate_coastal_risk_for_sources(
    sources=sources,
    currents=currents,
    coast_points=coast_points,
    cfg=SimulationConfig(),
)
```

The current probabilistic workflow works in these stages:

1. extract GMTDS hotspot candidates and keep the `period_id`, `mean_density`, and hotspot coordinates explicit;
2. filter hotspot candidates into `HotspotSource` objects with a documented density threshold such as `mean_density_min`;
3. aggregate or select OSCAR currents for the matching period, loading them from NetCDF or paired GeoTIFFs into `CurrentField`;
4. simulate particles from each source with advection, random-walk diffusion, and simplified weathering;
5. score each source as `density_weight × coastal_hit_fraction × survival_fraction`;
6. combine event-level scores by period, and only average across periods after keeping period IDs explicit.
