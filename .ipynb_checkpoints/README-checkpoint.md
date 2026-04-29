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
     - `monthly`: one mean raster per `YYYY-MM`
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
   - Builds ERDDAP subset URLs and downloads NetCDF files for a bbox + time range.
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
    start_date=date(2019, 1, 1),
    end_date=date(2021, 12, 31),
    season_start_month=1,
    season_length_months=3,
)

cfg = OscarDownloadConfig(
    output_dir=Path("oscar_subsets"),
    base_griddap_url="https://your-erddap-server/erddap/griddap",
    dataset_id="oscar_dataset_id",
)

downloaded = download_oscar_for_periods(cfg, area, periods)
```

## Still to complete

- Robust OSCAR provider defaults (dataset IDs/URLs can vary by server).
- Coastline preparation utilities (extract target coast points from shapefiles).
- Domain aggregation/mapping for gridded coastline risk products.
- Calibration of incident-rate scaling for absolute probability interpretation.
