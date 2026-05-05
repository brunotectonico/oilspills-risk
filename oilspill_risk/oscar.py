"""OSCAR download and NetCDF utilities using PO.DAAC downloader workflow."""

from __future__ import annotations

import os
import stat
import subprocess
from dataclasses import dataclass
from datetime import date
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
        "machine urs.earthdata.nasa.gov\n"
        f"  login {username}\n"
        f"  password {password}\n",
        encoding="utf-8",
    )
    netrc_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return netrc_path


def build_podaac_downloader_cmd(
    collection: str,
    output_dir: Path,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
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
        cmd += ["-sd", start_date.isoformat()]
    if end_date is not None:
        cmd += ["-ed", end_date.isoformat()]
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
    start_date: date | None = None,
    end_date: date | None = None,
    bbox: StudyArea | None = None,
    provider: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run PO.DAAC downloader with Earthdata auth from netrc or env vars."""
    username = earthdata_username or os.getenv("EARTHDATA_USERNAME")
    password = earthdata_password or os.getenv("EARTHDATA_PASSWORD")

    env = os.environ.copy()
    if username and password:
        resolved_netrc = netrc_path or (Path.home() / ("_netrc" if os.name == "nt" else ".netrc"))
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

    return subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)


def _subset_lon_lat_robust(ds: xr.Dataset, area: StudyArea, lon_name: str, lat_name: str) -> xr.Dataset:
    """Subset lon/lat robustly for descending coords and dateline-crossing ranges."""
    lon = ds[lon_name]
    lat = ds[lat_name]

    lon_vals = lon.values
    lat_vals = lat.values

    lon_min = ((area.lon_min + 180) % 360) - 180
    lon_max = ((area.lon_max + 180) % 360) - 180

    if lon_min <= lon_max:
        lon_mask = (lon_vals >= lon_min) & (lon_vals <= lon_max)
    else:
        lon_mask = (lon_vals >= lon_min) | (lon_vals <= lon_max)

    lat_lo = min(area.lat_min, area.lat_max)
    lat_hi = max(area.lat_min, area.lat_max)
    lat_mask = (lat_vals >= lat_lo) & (lat_vals <= lat_hi)

    lon_idx = np.where(lon_mask)[0]
    lat_idx = np.where(lat_mask)[0]
    if lon_idx.size == 0 or lat_idx.size == 0:
        raise ValueError("No grid cells found within requested lon/lat bounds")

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
    ds = xr.open_dataset(input_nc)

    lon_norm = ((ds[lon_name] + 180) % 360) - 180
    ds = ds.assign_coords({lon_name: lon_norm}).sortby(lon_name)

    only_uv = ds[[u_var, v_var]]
    clipped = _subset_lon_lat_robust(only_uv, area, lon_name=lon_name, lat_name=lat_name)

    output_nc.parent.mkdir(parents=True, exist_ok=True)
    clipped.to_netcdf(output_nc)
    ds.close()
    clipped.close()
    return output_nc


def seasonal_periods(
    start_date: date,
    end_date: date,
    season_start_month: int,
    season_length_months: int = 3,
) -> list[tuple[date, date, str]]:
    """Create year-based seasonal windows (e.g., 3-month periods)."""
    periods: list[tuple[date, date, str]] = []
    span = max(1, min(12, season_length_months))

    for year in range(start_date.year, end_date.year + 1):
        period_start = date(year, season_start_month, 1)
        end_month = ((season_start_month - 1 + span - 1) % 12) + 1
        end_year = year + (1 if season_start_month + span - 1 > 12 else 0)
        period_end = date(end_year, end_month, 1) + pd.offsets.MonthEnd(1)
        period_end = period_end.date()

        if period_end < start_date or period_start > end_date:
            continue

        clipped_start = max(period_start, start_date)
        clipped_end = min(period_end, end_date)
        pid = f"{year}-M{season_start_month:02d}_M{end_month:02d}"
        periods.append((clipped_start, clipped_end, pid))

    return periods


def download_oscar_for_periods(
    cfg: OscarDownloadConfig,
    area: StudyArea,
    periods: list[tuple[date, date, str]],
    *,
    standardize: bool = False,
) -> list[Path]:
    """Download one OSCAR file per period using PO.DAAC downloader logic."""
    outputs: list[Path] = []

    for pstart, pend, pid in periods:
        existing = {p.resolve() for p in cfg.output_dir.glob("*.nc")}

        downloader_bbox = None if standardize else area
        run_podaac_downloader(
            collection=cfg.podaac_collection,
            output_dir=cfg.output_dir,
            start_date=pstart,
            end_date=pend,
            bbox=downloader_bbox,
            dry_run=False,
        )

        current = sorted(cfg.output_dir.glob("*.nc"), key=lambda p: p.stat().st_mtime)
        new_files = [p for p in current if p.resolve() not in existing]
        if not new_files:
            raise FileNotFoundError(f"No new NetCDF files found in {cfg.output_dir} for period {pid}")

        raw_out = new_files[-1]
        if standardize:
            std_out = cfg.output_dir / f"oscar_uv_clip_{pid}_{pstart:%Y%m%d}_{pend:%Y%m%d}.nc"
            outputs.append(standardize_oscar_uv_netcdf(raw_out, std_out, area, u_var=cfg.u_var, v_var=cfg.v_var))
        else:
            outputs.append(raw_out)

    return outputs
