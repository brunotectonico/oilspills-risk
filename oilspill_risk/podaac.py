"""Functions to write cmd to call for PO.DAAC routine and download OSCAR files"""

from __future__ import annotations

import os
import stat
import subprocess
import warnings

from pathlib import Path


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