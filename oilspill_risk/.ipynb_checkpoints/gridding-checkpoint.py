"""OSCAR grid processing logic: focus on latitude/lonigtude handling, and subsetting u/v variables, with GIS compatibility module (TIFF exporting function)"""

from __future__ import annotations

import os
import re
import warnings

import rasterio
from rasterio.transform import from_bounds

from pathlib import Path
import numpy as np
import xarray as xr


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
    if resolved_lon == resolved_lat:
        if lon_name in ds.variables or lon_name in ds.coords:
            resolved_lon = lon_name
        if lat_name in ds.variables or lat_name in ds.coords:
            resolved_lat = lat_name
            
    if resolved_lon == resolved_lat:
        raise ValueError(f"Could not reliably infer lon/lat coordinate names (got {resolved_lon})")

    return resolved_lon, resolved_lat


def _reconstruct_degree_coords(ds: xr.Dataset, lon_name: str, lat_name: str) -> xr.Dataset:
    """
    Convert index-like lon/lat coords to real degree coords using global metadata.
    OSCAR v2.0 is 0..359.75 lon and -89.75..89.75 lat at 0.25° resolution.
    """
    lon = np.asarray(ds[lon_name].values, dtype=float)
    lat = np.asarray(ds[lat_name].values, dtype=float)

    lon_is_index = np.array_equal(lon, np.arange(lon.size))
    lat_is_index = np.array_equal(lat, np.arange(lat.size))

    if lon_is_index:
        lon_min = ds.attrs.get("geospatial_lon_min")
        lon_res = ds.attrs.get("geospatial_lon_resolution", 0.25)
        if isinstance(lon_res, str):
            lon_res = float(lon_res.split()[0])
        if lon_min is None:
            raise ValueError("Missing geospatial_lon_min for reconstructing longitude")
        ds = ds.assign_coords({lon_name: float(lon_min) + np.arange(lon.size) * float(lon_res)})

    if lat_is_index:
        lat_min = ds.attrs.get("geospatial_lat_min")
        lat_res = ds.attrs.get("geospatial_lat_resolution", 0.25)
        if isinstance(lat_res, str):
            lat_res = float(lat_res.split()[0])
        if lat_min is None:
            raise ValueError("Missing geospatial_lat_min for reconstructing latitude")
        ds = ds.assign_coords({lat_name: float(lat_min) + np.arange(lat.size) * float(lat_res)})

    return ds

    
def ensure_lon_lat(ds: xr.Dataset) -> xr.Dataset:
    """
    Return dataset with:
    - dims named 'lon' and 'lat'
    - coordinates 'lon' in [-180, 180], 'lat' ascending
    - lon/lat set as coords
    """
    lon_name, lat_name = _infer_lon_lat_names(ds, lon_name="lon", lat_name="lat")
    ds = _reconstruct_degree_coords(ds, lon_name=lon_name, lat_name=lat_name)

    # rename dims and coords to lon/lat
    rename = {}
    if lon_name != "lon":
        rename[lon_name] = "lon"
    if lat_name != "lat":
        rename[lat_name] = "lat"
    if rename:
        ds = ds.rename(rename)

    # now guarantee coords
    ds = ds.set_coords(["lon", "lat"])

    # normalize lon to [-180,180]
    lon = ds["lon"]
    if float(lon.max()) > 180:
        ds = ds.assign_coords(lon=((lon + 180) % 360) - 180)

    # ensure lat ascending
    if ds["lat"][0] > ds["lat"][-1]:
        ds = ds.sortby("lat")
    ds = ds.sortby("lon")

    return ds


def subset_lon_lat(ds: xr.Dataset, area: StudyArea, lon_name: str, lat_name: str) -> xr.Dataset:
    lon_vals = np.asarray(ds[lon_name].values)
    lat_vals = np.asarray(ds[lat_name].values)

    lon_dim = ds[lon_name].dims[0]
    lat_dim = ds[lat_name].dims[0]

    lon_min = area.lon_min
    lon_max = area.lon_max
    lat_lo = min(area.lat_min, area.lat_max)
    lat_hi = max(area.lat_min, area.lat_max)

    if lon_min <= lon_max:
        lon_mask = (lon_vals >= lon_min) & (lon_vals <= lon_max)
    else:
        lon_mask = (lon_vals >= lon_min) | (lon_vals <= lon_max)

    lat_mask = (lat_vals >= lat_lo) & (lat_vals <= lat_hi)

    lon_idx = np.where(lon_mask)[0]
    lat_idx = np.where(lat_mask)[0]

    if lon_idx.size == 0 or lat_idx.size == 0:
        raise ValueError("No grid cells found within requested lon/lat bounds")

    return ds.isel({lon_dim: lon_idx, lat_dim: lat_idx})


def standardize_oscar_uv_netcdf(
    input_nc: Path,
    output_nc: Path,
    area: StudyArea,
    *,
    u_var: str = "u",
    v_var: str = "v",
) -> Path:
    ds = xr.open_dataset(input_nc)
    try:
        ds = ensure_lon_lat(ds)
        print(f"[DEBUG] lon range=({float(ds.lon.min())},{float(ds.lon.max())})")
        print(f"[DEBUG] lat range=({float(ds.lat.min())},{float(ds.lat.max())})")

        # detect u/v once
        actual_u = next((n for n in [u_var, "u_current", "ugos", "u_curr"] if n in ds.data_vars), None)
        actual_v = next((n for n in [v_var, "v_current", "vgos", "v_curr"] if n in ds.data_vars), None)
        if not actual_u or not actual_v:
            raise KeyError(f"No U/V variables found. Variables: {list(ds.data_vars)}")

        # sub = subset_lon_lat(ds[[actual_u, actual_v]], "lon", "lat"]], area)
        sub = subset_lon_lat(ds[[actual_u, actual_v, "lon", "lat"]], area, "lon", "lat")

        # replace fill values
        for vname in [actual_u, actual_v]:
            fv = sub[vname].attrs.get("_FillValue")
            if fv is not None:
                sub[vname] = sub[vname].where(sub[vname] != fv)

        # CRS hints
        sub["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east", "axis": "X"})
        sub["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north", "axis": "Y"})

        output_nc.parent.mkdir(parents=True, exist_ok=True)
        sub[[actual_u, actual_v]].to_netcdf(output_nc)

        finite_u = int(np.isfinite(sub[actual_u].values).sum())
        finite_v = int(np.isfinite(sub[actual_v].values).sum())
        print(f"[DEBUG] standardized={output_nc.name} finite_u={finite_u} finite_v={finite_v}")
        return output_nc
    finally:
        ds.close()


def export_oscar_uv_geotiff(
    input_nc: Path,
    output_dir: Path,
    area: StudyArea,
    *,
    u_var: str = "u",
    v_var: str = "v",
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(input_nc)
    ds = ensure_lon_lat(ds)
    # subset = subset_lon_lat(ds, area)
    subset = subset_lon_lat(ds, area, "lon", "lat")

    u = subset[u_var].isel(time=0) if "time" in subset[u_var].dims else subset[u_var]
    v = subset[v_var].isel(time=0) if "time" in subset[v_var].dims else subset[v_var]

    # Ensure latitude is ascending south -> north before writing
    subset = subset.sortby("lat")
    subset = subset.sortby("lon")
    
    lon = subset["lon"].values
    lat = subset["lat"].values
    width, height = lon.size, lat.size
    transform = from_bounds(float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()), width, height)

    u_path = output_dir / f"{input_nc.stem}_{u_var}.tif"
    v_path = output_dir / f"{input_nc.stem}_{v_var}.tif"

    with rasterio.open(
        u_path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan
    ) as dst:
        # dst.write(np.flipud(u.values.astype("float32")), 1)
        dst.write(np.fliplr(u.values.astype("float32")), 1)

    with rasterio.open(
        v_path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan
    ) as dst:
        # dst.write(np.flipud(v.values.astype("float32")), 1)
        dst.write(np.fliplr(v.values.astype("float32")), 1)

    ds.close()
    print(f"[DEBUG] wrote GeoTIFFs: {u_path.name}, {v_path.name}")
    return u_path, v_path