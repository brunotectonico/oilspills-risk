"""OSCAR download and NetCDF utilities using PO.DAAC downloader workflow."""

from __future__ import annotations

import os
import re
import stat
import subprocess
import warnings

import rasterio
from rasterio.transform import from_bounds

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


def validate_study_area(area: StudyArea) -> StudyArea:
    """Validate and normalize StudyArea bounds."""
    lon_min, lon_max = sorted([float(area.lon_min), float(area.lon_max)])
    lat_min, lat_max = sorted([float(area.lat_min), float(area.lat_max)])

    if not (-180.0 <= lon_min <= 360.0 and -180.0 <= lon_max <= 360.0):
        raise ValueError(f"Invalid longitude bounds: {lon_min}, {lon_max}")
    if not (-90.0 <= lat_min <= 90.0 and -90.0 <= lat_max <= 90.0):
        raise ValueError(f"Invalid latitude bounds: {lat_min}, {lat_max}")
    if lon_min == lon_max or lat_min == lat_max:
        raise ValueError("StudyArea bounds collapse to zero width/height")

    return StudyArea(lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max)

    
def infer_study_area_from_hotspots(
    hotspot_csv: Path,
    lon_col: str = "lon",
    lat_col: str = "lat",
    pad_deg: float = 0.5,
) -> StudyArea:
    """Infer a current-download bounding box from hotspot coordinates."""
    df = pd.read_csv(hotspot_csv)
    area = StudyArea(
        lon_min=float(df[lon_col].min() - pad_deg),
        lon_max=float(df[lon_col].max() + pad_deg),
        lat_min=float(df[lat_col].min() - pad_deg),
        lat_max=float(df[lat_col].max() + pad_deg),
    )
    area = validate_study_area(area)
    print(f"[DEBUG] inferred StudyArea: {area}")
    return area


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

        
def _to_360(lon: float) -> float:
    return lon % 360


def _infer_lon_lat_names(ds: xr.Dataset, lon_name: str, lat_name: str) -> tuple[str, str]:
    """Resolve lon/lat coordinate names from dataset contents and metadata."""

    def _is_lon(var_key: str, var_obj: xr.DataArray | xr.Variable) -> bool:
        attrs = getattr(var_obj, "attrs", {})
        std = str(attrs.get("standard_name", "")).lower()
        units = str(attrs.get("units", "")).lower()
        name = str(var_key).lower()
        return (
            "longitude" in std
            or "degrees_east" in units
            or name in {"lon", "longitude", "x"}
        )

    def _is_lat(var_key: str, var_obj: xr.DataArray | xr.Variable) -> bool:
        attrs = getattr(var_obj, "attrs", {})
        std = str(attrs.get("standard_name", "")).lower()
        units = str(attrs.get("units", "")).lower()
        name = str(var_key).lower()
        return (
            "latitude" in std
            or "degrees_north" in units
            or name in {"lat", "latitude", "y"}
        )

    merged = {**dict(ds.coords.items()), **dict(ds.variables.items())}

    lon_found = [k for k, v in merged.items() if _is_lon(k, v)]
    lat_found = [k for k, v in merged.items() if _is_lat(k, v)]

    resolved_lon = lon_found[0] if lon_found else lon_name
    resolved_lat = lat_found[0] if lat_found else lat_name

    # fallback to explicit names if both got same var by loose matching
    # if resolved_lon == resolved_lat:
    #     if lon_name in ds.dims:
    #         resolved_lon = lon_name
    #     if lat_name in ds.dims:
    #         resolved_lat = lat_name
    if resolved_lon == resolved_lat:
        if lon_name in ds.variables or lon_name in ds.coords:
            resolved_lon = lon_name
        if lat_name in ds.variables or lat_name in ds.coords:
            resolved_lat = lat_name
            
    if resolved_lon == resolved_lat:
        raise ValueError(f"Could not reliably infer lon/lat coordinate names (got {resolved_lon})")

    return resolved_lon, resolved_lat


def _reconstruct_degree_coords(ds: xr.Dataset, lon_name: str, lat_name: str) -> xr.Dataset:
    """Rebuild lon/lat degrees if dataset stores index-like coordinates."""
    lon = np.asarray(ds[lon_name].values, dtype=float)
    lat = np.asarray(ds[lat_name].values, dtype=float)

    lon_index_like = np.allclose(lon, np.arange(lon.size))
    lat_index_like = np.allclose(lat, np.arange(lat.size))

    if lon_index_like:
        glon_min = ds.attrs.get("geospatial_lon_min")
        glon_max = ds.attrs.get("geospatial_lon_max")
        glon_res = ds.attrs.get("geospatial_lon_resolution")
        if glon_res is not None and isinstance(glon_res, str):
            glon_res = float(glon_res.split()[0])
        if glon_min is not None and glon_max is not None:
            if glon_res is None:
                glon_res = (float(glon_max) - float(glon_min)) / max(1, lon.size - 1)
            lon = float(glon_min) + np.arange(lon.size) * float(glon_res)
            ds = ds.assign_coords({lon_name: lon})

    if lat_index_like:
        glat_min = ds.attrs.get("geospatial_lat_min")
        glat_max = ds.attrs.get("geospatial_lat_max")
        glat_res = ds.attrs.get("geospatial_lat_resolution")
        if glat_res is not None and isinstance(glat_res, str):
            glat_res = float(glat_res.split()[0])
        if glat_min is not None and glat_max is not None:
            if glat_res is None:
                glat_res = (float(glat_max) - float(glat_min)) / max(1, lat.size - 1)
            lat = float(glat_min) + np.arange(lat.size) * float(glat_res)
            ds = ds.assign_coords({lat_name: lat})

    return ds
  
  
def _subset_lon_lat_robust(ds: xr.Dataset, area: StudyArea, lon_name: str, lat_name: str) -> xr.Dataset:
    """Subset lon/lat robustly for descending coords and dateline-crossing ranges."""
    if lon_name not in ds.variables and lon_name not in ds.coords and lon_name not in ds.dims:
        raise KeyError(f"Longitude coordinate '{lon_name}' not found. Available: {list(ds.variables)}")
    if lat_name not in ds.variables and lat_name not in ds.coords and lat_name not in ds.coords:
        raise KeyError(f"Latitude coordinate '{lat_name}' not found. Available: {list(ds.variables)}")
    
    lon_vals = np.asarray(ds[lon_name].values)
    lat_vals = np.asarray(ds[lat_name].values)
    lon_dim = ds[lon_name].dims[0]
    lat_dim = ds[lat_name].dims[0]

    lon_uses_360 = float(np.nanmax(lon_vals)) > 180.0
    if lon_uses_360:
        lon_min = _to_360(area.lon_min)
        lon_max = _to_360(area.lon_max)
    else:
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

    return ds.isel({lon_dim: lon_idx, lat_dim: lat_idx})


def _extract_data_date(raw_file: Path) -> str:
    """Extract YYYYMMDD from a raw OSCAR filename."""
    m = re.search(r"(19|20)\d{6}", raw_file.name)
    if not m:
        raise ValueError(f"Could not parse data date from filename: {raw_file.name}")
    return m.group(0)


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
    """Subset raw data, normalize lon to [-180,180], sort, then keep u/v only."""
    ds = None
    try:
        ds = xr.open_dataset(input_nc)
        lon_name, lat_name = _infer_lon_lat_names(ds, lon_name=lon_name, lat_name=lat_name)
        ds = _reconstruct_degree_coords(ds, lon_name=lon_name, lat_name=lat_name)

        print(f"[DEBUG] raw dims={dict(ds.dims)} lon_name={lon_name} lat_name={lat_name}")
        print(f"[DEBUG] raw lon range=({float(ds[lon_name].min())},{float(ds[lon_name].max())})")
        print(f"[DEBUG] raw lat range=({float(ds[lat_name].min())},{float(ds[lat_name].max())})")
        
        # Automatic detection of variables (U/V)        
        actual_u = next((n for n in [u_var, 'u_current', 'ugos', 'u_curr'] if n in ds.data_vars), None)
        actual_v = next((n for n in [v_var, 'v_current', 'vgos', 'v_curr'] if n in ds.data_vars), None)
        if not actual_u or not actual_v:
            raise KeyError(f"No U/V variables found. Variables: {list(ds.data_vars)}")

        # CRITICAL: 1. SUBSET RAW
        subset_raw = _subset_lon_lat_robust(ds, area, lon_name, lat_name)
        # subset_raw = subset_raw.set_coords([lon_name, lat_name])
        
        # 2. normalize lon to [-180, 180] and sort
        norm_subset = subset_raw.assign_coords({lon_name: ((subset_raw[lon_name] + 180) % 360) - 180})
        norm_subset = norm_subset.sortby(lon_name)
        norm_subset = norm_subset.sortby(lat_name)

        # enforce canonical GIS-friendly coord names and dimensions 
        lon_dim = norm_subset[lon_name].dims[0]
        lat_dim = norm_subset[lat_name].dims[0]
        rename_map = {}
        if lon_dim != "lon":
            rename_map[lon_dim] = "lon"
        if lat_dim != "lat":
            rename_map[lat_dim] = "lat"
        if lon_name != "lon":
            rename_map[lon_name] = "lon"
        if lat_name != "lat":
            rename_map[lat_name] = "lat"
        if rename_map:
            norm_subset = norm_subset.rename(rename_map)
        
        # 3) select u and v only, replacing fill values with NaN for GIS stats
        only_uv = norm_subset[[actual_u, actual_v]]
        for var_name in [actual_u, actual_v]:
            fill_value = only_uv[var_name].attrs.get("_FillValue")
            if fill_value is not None:
                only_uv[var_name] = only_uv[var_name].where(only_uv[var_name] != fill_value)
            # only_uv[var_name].attrs["coordinates"] = "lon lat"
            
        # CRS hints for GIS readers
        only_uv["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east", "axis": "X"})
        only_uv["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north", "axis": "Y"})
        
        output_nc.parent.mkdir(parents=True, exist_ok=True)
        only_uv.to_netcdf(output_nc)
        finite_u = int(np.isfinite(only_uv[actual_u].values).sum())
        finite_v = int(np.isfinite(only_uv[actual_v].values).sum())
        print(f"[DEBUG] standardized={output_nc.name} finite_u={finite_u} finite_v={finite_v}")
                
        ds.close()
        subset_raw.close()
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
  
  
def _raw_oscar_files(output_dir: Path) -> set[Path]:
    """Track raw OSCAR .nc files only (exclude clipped outputs)."""
    return {
        p.resolve()
        for p in output_dir.glob("*.nc")
        if "clip" not in p.name.lower() and p.name.lower().startswith("oscar")
    }

  
def export_oscar_uv_geotiff(
    input_nc: Path,
    output_dir: Path,
    area: StudyArea,
    *,
    u_var: str = "u",
    v_var: str = "v",
) -> tuple[Path, Path]:
    """Alternative output for GIS: export clipped U/V as GeoTIFF rasters."""
    area = validate_study_area(area)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(input_nc)
    lon_name, lat_name = _infer_lon_lat_names(ds, lon_name="lon", lat_name="lat")
    ds = _reconstruct_degree_coords(ds, lon_name=lon_name, lat_name=lat_name)

    subset = _subset_lon_lat_robust(ds, area, lon_name=lon_name, lat_name=lat_name)
    subset = subset.assign_coords({lon_name: ((subset[lon_name] + 180) % 360) - 180}).sortby(lon_name).sortby(lat_name)

    if lon_name != "lon" or lat_name != "lat":
        subset = subset.rename({lon_name: "lon", lat_name: "lat"})

    if "time" in subset[u_var].dims:
        subset_u = subset[u_var].isel(time=0)
        subset_v = subset[v_var].isel(time=0)
    else:
        subset_u = subset[u_var]
        subset_v = subset[v_var]

    fill_u = subset_u.attrs.get("_FillValue")
    fill_v = subset_v.attrs.get("_FillValue")
    if fill_u is not None:
        subset_u = subset_u.where(subset_u != fill_u)
    if fill_v is not None:
        subset_v = subset_v.where(subset_v != fill_v)
        
    lon = subset_u["lon"].values
    lat = subset_u["lat"].values
    width = lon.size
    height = lat.size
    transform = from_bounds(float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()), width, height)

    u_path = output_dir / f"{input_nc.stem}_{u_var}.tif"
    v_path = output_dir / f"{input_nc.stem}_{v_var}.tif"

    with rasterio.open(
        u_path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan
    ) as dst:
        dst.write(np.flipud(subset_u.values.astype("float32")), 1)

    with rasterio.open(
        v_path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan
    ) as dst:
        dst.write(np.flipud(subset_v.values.astype("float32")), 1)

    ds.close()
    print(f"[DEBUG] wrote GeoTIFFs: {u_path.name}, {v_path.name}")
    return u_path, v_path      

def download_oscar_for_periods(
    cfg: OscarDownloadConfig,
    area: StudyArea,
    periods: list[tuple[date, date, str]],
    *,
    standardize: bool = False,
) -> list[Path]:
    """Download OSCAR files for period using PO.DAAC downloader logic."""
    outputs: list[Path] = []
    area = validate_study_area(area)
    print(f"[DEBUG] download StudyArea: {area}")

    for pstart, pend, pid in periods:
        existing = _raw_oscar_files(cfg.output_dir)
        print(f"[DEBUG] period={pid} existing_raw={len(existing)}")
        
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

        current_raw = sorted(_raw_oscar_files(cfg.output_dir), key=lambda p: p.stat().st_mtime)
        new_files = [p for p in current_raw if p.resolve() not in existing]
        print(f"[DEBUG] period={pid} new_raw={len(new_files)}")
        if not new_files:
            raise FileNotFoundError(f"No new raw NetCDF files found in {cfg.output_dir} for period {pid}")

        for raw_out in new_files:
            if standardize:
                data_date = _extract_data_date(raw_out)
                std_out = cfg.output_dir / f"oscar_uv_clip_{data_date}.nc"
                print(f"[DEBUG] standardize raw={raw_out.name} -> clip={std_out.name}")
                outputs.append(
                    standardize_oscar_uv_netcdf(raw_out, std_out, area, u_var=cfg.u_var, v_var=cfg.v_var)
                )
            else:
                outputs.append(raw_out)

    return outputs
