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

5. **OSCAR download module**
   - Infers study-area bounds from hotspot CSV.
   - Downloads NetCDF files for a bbox + time range following PO.DAAC logic.
   - Supports per-period downloads (e.g., one OSCAR file per 3-month seasonal window).

## Repository structure

- `density_hotspots.py`  
  CLI wrapper for hotspot extraction.
- `oilspill_risk/hotspots.py`  
  Reusable hotspot detection and outputs.
- `oilspill_risk/density_rasters.py`  
  Separate mean-density raster aggregation module.
- `oilspill_risk/trajectory.py`  
  Reusable trajectory simulation and coastal risk scoring primitives.
- `oilspill_risk/oscar.py`  
  OSCAR study-area inference, ERDDAP URL generation, and period-based NetCDF downloads.
- `hotspots4density.ipynb`  
  Notebook exploration.

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
from datetime import date
from pathlib import Path

from oilspill_risk.oscar import StudyArea, run_podaac_downloader

bbox = StudyArea(lon_min=40.0, lon_max=45.0, lat_min=10.0, lat_max=14.0)
result = run_podaac_downloader(
    collection="OSCAR_L4_OC_FINAL_V2.0",
    output_dir=Path("oscar_downloads"),
    start_date=datetime(2020, 1, 1, 0, 0, 0).strftime('%Y-%m-%dT%H:%M:%SZ'), #dates should be strings in YYYYMMDDhhmmssZ format
    end_date=datetime(2020, 3, 31, 23, 59, 59).strftime('%Y-%m-%dT%H:%M:%SZ'),
    bbox=bbox,
    dry_run=True,
)
print(result.stdout)
```

## Example OSCAR period downloads

```python
from datetime import date
from pathlib import Path

from oilspill_risk.oscar import (
    OscarDownloadConfig,
    download_oscar_for_periods,
    infer_study_area_from_hotspots,
    seasonal_periods,
)

area = infer_study_area_from_hotspots(Path("gmtds_tanker_hotspots_multi.csv"), pad_deg=0.7)
periods = seasonal_periods(
    start_date=datetime(2019, 1, 1, 0, 0, 0).strftime('%Y-%m-%dT%H:%M:%SZ'), #dates should be strings in YYYYMMDDhhmmssZ format
    end_date=datetime(2021, 12, 31, 23, 59, 59).strftime('%Y-%m-%dT%H:%M:%SZ'),
    season_start_month=1,
    season_length_months=3,
)

cfg = OscarDownloadConfig(
    output_dir=Path("oscar_downloads"),
    podaac_collection="OSCAR_L4_OC_FINAL_V2.0",
)

downloaded = download_oscar_for_periods(cfg, area, periods, standardize=True)
```


### NetCDF standardization for GIS and trajectory

If OSCAR files are in `0..360` longitude and appear twisted in GIS:
- use `standardize_oscar_uv_netcdf(...)` to normalize longitude to `-180..180`,
- clip to your hotspot study area,
- and keep only `u` and `v` variables for trajectory runs.

You can then load currents directly into the trajectory module with `current_field_from_netcdf(...)`.

## Still to complete

- Robust OSCAR provider defaults (dataset IDs/URLs can vary by server).
- Coastline preparation utilities (extract target coast points from shapefiles).
- Domain aggregation/mapping for gridded coastline risk products.
- Calibration of incident-rate scaling for absolute probability interpretation.
