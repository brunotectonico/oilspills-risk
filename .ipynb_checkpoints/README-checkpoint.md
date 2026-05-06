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
from oilspill_risk.oscar import StudyArea

area = StudyArea(
    lon_min=41.50803970311112,
    lon_max=45.69349690701655,
    lat_min=9.72568026223306,
    lat_max=14.679851304878618,
)
```

## Run hotspot extraction

```bash
python density_hotspots.py /path/to/gmtds_data \
  --pattern "*Tankers.zip" \
  --mean-raster-dir mean_density_rasters \
  --mean-raster-frequency seasonal \
  --season-start-month 1 \
  --season-length-months 3
```

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
from oilspill_risk.oscar import StudyArea, export_oscar_uv_geotiff, standardize_oscar_uv_netcdf

area = StudyArea(lon_min=41.5, lon_max=45.75, lat_min=9.75, lat_max=14.75)
for raw_nc in sorted(Path("files").glob("oscar_currents_final_*.nc")):
    date_id = raw_nc.stem.rsplit("_", 1)[-1]
    clipped_nc = raw_nc.with_name(f"oscar_uv_clip_{date_id}.nc")
    standardize_oscar_uv_netcdf(raw_nc, clipped_nc, area)
    export_oscar_uv_geotiff(clipped_nc, raw_nc.parent, area)
```

### NetCDF / GeoTIFF coordinate-placement notes

- `standardize_oscar_uv_netcdf(...)` reconstructs index-like OSCAR coordinates from `geospatial_*` metadata before clipping.
- Longitudes are normalized to `[-180, 180]`, and latitude/longitude axes are sorted before writing.
- GeoTIFFs are written north-up using a transform derived from coordinate **centers** expanded by half a grid cell. This avoids the half-cell placement error caused by using center min/max values as raster bounds.
- The GeoTIFF write flips rows vertically only when latitude is stored south-to-north; it does **not** flip columns, which would mirror the Red Sea/Djibouti subset east-west.

## Still to complete

- Robust OSCAR provider defaults (dataset IDs/URLs can vary by server).
- Coastline preparation utilities (extract target coast points from shapefiles).
- Domain aggregation/mapping for gridded coastline risk products.
- Calibration of incident-rate scaling for absolute probability interpretation.
