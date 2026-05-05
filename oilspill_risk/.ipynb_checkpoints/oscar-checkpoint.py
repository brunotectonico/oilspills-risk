"""OSCAR download and NetCDF utilities using PO.DAAC downloader workflow."""

from __future__ import annotations

import os
import stat
import subprocess
import warnings

from dataclasses import dataclass
from datetime import date, datetime
from calendar import monthrange
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr



@dataclass(frozen=True)
class StudyArea:
    """Bounding box used to subset OSCAR fields."""

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


@dataclass(frozen=True)
class OscarDownloadConfig:
    """PO.DAAC downloader options for OSCAR data pulls."""

    output_dir: Path
    podaac_collection: str
    u_var: str = "u"
    v_var: str = "v"


def infer_study_area_from_hotspots(
    hotspot_csv: Path,
    lon_col: str = "lon",
    lat_col: str = "lat",
    pad_deg: float = 0.5,
) -> StudyArea:
    """Infer a current-download bounding box from hotspot coordinates."""
    df = pd.read_csv(hotspot_csv)
    return StudyArea(
        lon_min=float(df[lon_col].min() - pad_deg),
        lon_max=float(df[lon_col].max() + pad_deg),
        lat_min=float(df[lat_col].min() - pad_deg),
        lat_max=float(df[lat_col].max() + pad_deg),
    )


def write_earthdata_netrc(netrc_path: Path, username: str, password: str) -> Path:
    """Write Earthdata credentials to a netrc file with restricted permissions."""
    netrc_path.parent.mkdir(parents=True, exist_ok=True)
    netrc_path.write_text(
        "machine urs.earthdata.nasa.gov"
        f"  login {username}"
        f"  password {password}",
        encoding="utf-8",
    )
    netrc_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return netrc_path


def build_podaac_downloader_cmd(
    collection: str,
    output_dir: Path,
    *,
    start_date: str | None = None,#Assumes date already in string format (based in init.py)
    end_date: str | None = None,
    bbox: StudyArea | None = None,
    provider: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Build a podaac-data-downloader command list."""
    cmd = [
        "podaac-data-downloader",
        "-c",
        collection,
        "-d",
        str(output_dir),
    ]

    if start_date is not None:
        cmd += ["-sd", start_date] 
    if end_date is not None:
        cmd += ["-ed", end_date]
    if bbox is not None:
        cmd += ["-b", f"{bbox.lon_min},{bbox.lat_min},{bbox.lon_max},{bbox.lat_max}"]
    if provider is not None:
        cmd += ["-p", provider]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if dry_run:
        cmd += ["--dry-run"]

    return cmd


def run_podaac_downloader(
    collection: str,
    output_dir: Path,
    *,
    earthdata_username: str | None = None,
    earthdata_password: str | None = None,
    netrc_path: Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    bbox: StudyArea | None = None,
    provider: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run the PO.DAAC downloader with Earthdata auth from netrc or env vars.

    Credentials can be passed directly, or via env vars
    EARTHDATA_USERNAME / EARTHDATA_PASSWORD.
    """
    username = earthdata_username or os.getenv("EARTHDATA_USERNAME")
    password = earthdata_password or os.getenv("EARTHDATA_PASSWORD")

    env = os.environ.copy()
    if username and password:
        resolved_netrc = netrc_path or (Path.home() / (".netrc" if os.name == "nt" else ".netrc")) #Some windows versions uses _netrc
        write_earthdata_netrc(resolved_netrc, username, password)
        env["NETRC"] = str(resolved_netrc)
        
    cmd = build_podaac_downloader_cmd(
        collection=collection,
        output_dir=output_dir,
        start_date=start_date,
        end_date=end_date,
        bbox=bbox,
        provider=provider,
        limit=limit,
        dry_run=dry_run,
    )

    try: # to handle errors in download
        return subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stdout}")
        print(f"Details: {e.stderr}") 
        raise    


def _subset_lon_lat_robust(ds: xr.Dataset, area: StudyArea, lon_name: str, lat_name: str) -> xr.Dataset:
    """Subset lon/lat robustly for descending coords and dateline-crossing ranges."""
    lon = ds[lon_name].values
    lat = ds[lat_name].values

    # Convert area to OSCAR's 0-360 convention
    lon_min_360 = (area.lon_min + 360) % 360
    lon_max_360 = (area.lon_max + 360) % 360

    print(f"Raw lon range: {lon.min():.1f}→{lon.max():.1f}, target: {lon_min_360:.1f}→{lon_max_360:.1f}")
    print(f"Raw lat range: {lat.min():.1f}→{lat.max():.1f}, target: {min(area.lat_min, area.lat_max):.1f}→{max(area.lat_max, area.lat_min):.1f}")

    if lon_min_360 <= lon_max_360:
        lon_mask = (lon >= lon_min_360) & (lon <= lon_max_360)
    else:  # Dateline cross
        lon_mask = (lon >= lon_min_360) | (lon <= lon_max_360)
    
    lat_mask = (lat >= min(area.lat_min, area.lat_max)) & (lat <= max(area.lat_max, area.lat_min))
    
    lon_idx = np.where(lon_mask)[0]
    lat_idx = np.where(lat_mask)[0]
    print(f"Selected: lon={len(lon_idx)} pts, lat={len(lat_idx)} pts")
    if len(lon_idx) == 0 or len(lat_idx) == 0:
        raise ValueError(f"No overlap! lon_idx={lon_idx[:5]}, lat_idx={lat_idx[:5]}")
    
    return ds.isel({lon_name: lon_idx, lat_name: lat_idx})


def standardize_oscar_uv_netcdf(
    input_nc: Path,
    output_nc: Path,
    area: StudyArea,
    *,
    u_var: str = "u",
    v_var: str = "v",
    lon_name: str = "lon",
    lat_name: str = "lat",
) -> Path:
    """Normalize longitude to [-180,180], sort longitude, then clip and keep u/v only."""
    ds = None
    try:
        ds = xr.open_dataset(input_nc)

        # Automatic detection of coordinates
        actual_lon = next((n for n in [lon_name, 'longitude', 'x', 'long'] if n in ds.dims), None)
        actual_lat = next((n for n in [lat_name, 'latitude', 'y'] if n in ds.dims), None)
        if not actual_lon or not actual_lat:
            raise KeyError(f"No coordinates found. Variables present: {list(ds.dims)}")

        # Automatic detection of variables (U/V)        
        actual_u = next((n for n in [u_var, 'u_current', 'ugos', 'u_curr'] if n in ds.data_vars), None)
        actual_v = next((n for n in [v_var, 'v_current', 'vgos', 'v_curr'] if n in ds.data_vars), None)
        if not actual_u or not actual_v:
            raise KeyError(f"No U/V variables found. Variables: {list(ds.data_vars)}")

        # CRITICAL: 1. SUBSET RAW → 2. NORMALIZE → 3. u/v only
        raw_subset = _subset_lon_lat_robust(ds, area, actual_lon, actual_lat)
        lon_norm = ((raw_subset[actual_lon] + 180) % 360) - 180
        norm_subset = raw_subset.assign_coords({actual_lon: lon_norm}).sortby(actual_lon)
        only_uv = norm_subset[[actual_u, actual_v]]
        
        output_nc.parent.mkdir(parents=True, exist_ok=True)
        only_uv.to_netcdf(output_nc)
        print(f"Output {output_nc}: u.shape={only_uv[actual_u].shape}")
        
        only_uv.close()
        return output_nc
        
    except Exception as e:
        print(f"Failed {input_nc}: {e}")
        raise
    finally:
        if ds: ds.close()

def seasonal_periods(
    start_date: str,
    end_date: str,
    season_length_months: int = 3,
) -> list[tuple[date, date, str]]:
    """Create non-overlapping seasonal windows starting from start_date.
    
    Each window is exactly season_length_months long. Next window starts
    exactly after the previous one ends. Warns if any window is incomplete."""
    # From string (input) to datetime
    # Should be the following format: 2020-01-01T00:00:00Z
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    s_date = datetime.strptime(start_date, fmt).date()
    e_date = datetime.strptime(end_date, fmt).date()

    span = max(1, min(12, season_length_months))
    periods: list[tuple[date, date, str]] = []

    current_year, current_month = s_date.year, s_date.month
    window_count = 0

    def _add_months(y: int, m: int, delta: int) -> tuple[int, int]: #nested function to calculate the time window
        total = (y * 12 + (m - 1)) + delta
        return total // 12, (total % 12) + 1
    
    while True:
        # Window start = current month
        period_start = date(current_year, current_month, 1)
        
        # Window end = span-1 months ahead
        end_year, end_month = _add_months(current_year, current_month, span - 1)
        period_end = date(end_year, end_month, monthrange(end_year, end_month)[1])

        # Stop if this window starts after end_date
        if period_start > e_date:
            break

        clipped_start = max(period_start, s_date)
        clipped_end = min(period_end, e_date)
        
        # Check if complete (matches full span exactly)
        is_complete = (clipped_start == period_start) and (clipped_end == period_end)
        
        if not is_complete:
            warnings.warn(
                f"Skipping incomplete season {window_count+1}: "
                f"[{period_start:%Y-%m} to {period_end:%Y-%m}] "
                f"only available [{clipped_start:%Y-%m-%d} to {clipped_end:%Y-%m-%d}] "
                f"({(clipped_end-clipped_start).days+1}/"
                f"{(period_end-period_start).days+1} days)"
            )
            # Skip incomplete periods
            pass
        else:
            pid = f"S{window_count+1}_{period_start:%Y%m}_{period_end:%Y%m}"
            periods.append((period_start, period_end, pid))
            window_count += 1

        # Next window starts exactly after this one ends
        next_year, next_month = _add_months(end_year, end_month, 1)
        current_year, current_month = next_year, next_month

    return periods

def download_oscar_for_periods(
    cfg: OscarDownloadConfig,
    area: StudyArea,
    periods: list[tuple[date, date, str]],
    *,
    standardize: bool = False,
) -> list[Path]:
    """Download one OSCAR file per period using only PO.DAAC downloader logic."""
    outputs: list[Path] = []

    for pstart, pend, pid in periods:
        existing = {p.resolve() for p in cfg.output_dir.glob("*.nc")}

        # to string..
        start_str = pstart.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = pend.strftime('%Y-%m-%dT%H:%M:%SZ')

        downloader_bbox = None if standardize else area
        run_podaac_downloader(
            collection=cfg.podaac_collection,
            output_dir=cfg.output_dir,
            start_date=start_str,
            end_date=end_str,
            bbox=downloader_bbox,
            dry_run=False,
        )

        current = sorted(cfg.output_dir.glob("*.nc"), key=lambda p: p.stat().st_mtime)
        new_files = [p for p in current if p.resolve() not in existing]
        if not new_files:
            raise FileNotFoundError(f"No new NetCDF files found in {cfg.output_dir} for period {pid}")

        raw_out = new_files[-1]
        if standardize:
            raw_day = raw_out.stat().st_mtime
            day_str = datetime.fromtimestamp(raw_day).strftime('%Y%m%d')
            std_out = cfg.output_dir / f"oscar_uv_clip_{day_str}.nc"
            outputs.append(standardize_oscar_uv_netcdf(raw_out, std_out, area, u_var=cfg.u_var, v_var=cfg.v_var))
        else:
            outputs.append(raw_out)

    return outputs
