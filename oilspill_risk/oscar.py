"""OSCAR data download and period utilities."""

from __future__ import annotations

import os
import stat
import subprocess
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlretrieve

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
    """Download options for OSCAR NetCDF subsets."""

    output_dir: Path
    base_griddap_url: str
    dataset_id: str
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
        cmd += ["-sd", start_date] #Assumes date already in string format (based in init.py)
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
    start_date: date | None = None,
    end_date: date | None = None,
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


def build_erddap_subset_url(
    cfg: OscarDownloadConfig,
    area: StudyArea,
    start_dt: datetime,
    end_dt: datetime,
) -> str:
    """Build an ERDDAP griddap subset URL for OSCAR u/v variables."""

    def _time_str(ts: datetime) -> str:
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    query = (
        f"{cfg.u_var}[({_time_str(start_dt)}):1:({_time_str(end_dt)})]"
        f"[({area.lat_min}):1:({area.lat_max})][({area.lon_min}):1:({area.lon_max})],"
        f"{cfg.v_var}[({_time_str(start_dt)}):1:({_time_str(end_dt)})]"
        f"[({area.lat_min}):1:({area.lat_max})][({area.lon_min}):1:({area.lon_max})]"
    )
    encoded = quote(query, safe="[]():,._-+")
    return f"{cfg.base_griddap_url.rstrip('/')}/{cfg.dataset_id}.nc?{encoded}"


def download_oscar_subset(
    cfg: OscarDownloadConfig,
    area: StudyArea,
    start_dt: datetime,
    end_dt: datetime,
    output_name: str | None = None,
) -> Path:
    """Download a subsetted OSCAR NetCDF file for a study area and period."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if output_name is None:
        output_name = f"oscar_{start_dt:%Y%m%d}_{end_dt:%Y%m%d}.nc"

    out_path = cfg.output_dir / output_name
    url = build_erddap_subset_url(cfg=cfg, area=area, start_dt=start_dt, end_dt=end_dt)
    urlretrieve(url, out_path)
    return out_path


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
    """Normalize OSCAR longitude to [-180,180], clip to bbox, and keep only u/v."""
    ds = xr.open_dataset(input_nc)

    lon = ds[lon_name]
    lon_norm = ((lon + 180) % 360) - 180
    ds = ds.assign_coords({lon_name: lon_norm}).sortby(lon_name)

    clipped = ds[[u_var, v_var]].sel(
        {
            lon_name: slice(area.lon_min, area.lon_max),
            lat_name: slice(area.lat_min, area.lat_max),
        }
    )

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
    """Download one OSCAR file per period."""
    outputs: list[Path] = []
    for pstart, pend, pid in periods:
        raw_out = download_oscar_subset(
            cfg=cfg,
            area=area,
            start_dt=datetime(pstart.year, pstart.month, pstart.day),
            end_dt=datetime(pend.year, pend.month, pend.day, 23),
            output_name=f"oscar_{pid}_{pstart:%Y%m%d}_{pend:%Y%m%d}.nc",
        )
        if standardize:
            std_out = cfg.output_dir / f"oscar_uv_clip_{pid}_{pstart:%Y%m%d}_{pend:%Y%m%d}.nc"
            outputs.append(standardize_oscar_uv_netcdf(raw_out, std_out, area, u_var=cfg.u_var, v_var=cfg.v_var))
        else:
            outputs.append(raw_out)
    return outputs
