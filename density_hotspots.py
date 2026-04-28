"""CLI wrapper for GMTDS tanker hotspot extraction."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from oilspill_risk.hotspots import HotspotConfig, RunOptions, run_hotspot_extraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract GMTDS tanker hotspots")
    parser.add_argument("data_dir", type=Path, help="Directory containing GMTDS ZIP files")
    parser.add_argument("--pattern", default="*Tankers.zip", help="Glob pattern for ZIP files")
    parser.add_argument("--start", type=int, default=0, help="Start index in the sorted ZIP list")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of ZIP files to process")
    parser.add_argument("--output", default="gmtds_tanker_hotspots_multi.csv", help="Detailed output CSV name")
    parser.add_argument(
        "--summary-output",
        default="gmtds_tanker_hotspots_monthly_summary.csv",
        help="Monthly summary output CSV name",
    )
    parser.add_argument(
        "--hotspot-raster-output",
        default=None,
        help="Optional GeoTIFF output with mean density on hotspot pixels",
    )
    parser.add_argument(
        "--season-start-month",
        type=int,
        default=None,
        help="Optional first month for a seasonal window (1-12), e.g. 1 for Jan-Mar",
    )
    parser.add_argument(
        "--season-length-months",
        type=int,
        default=3,
        help="Seasonal window size in months when --season-start-month is provided",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    options = RunOptions(
        data_dir=args.data_dir,
        zip_pattern=args.pattern,
        output_csv=args.output,
        output_summary_csv=args.summary_output,
        output_hotspot_raster=args.hotspot_raster_output,
        start=args.start,
        limit=args.limit,
        season_start_month=args.season_start_month,
        season_length_months=args.season_length_months,
    )
    cfg = HotspotConfig()
    run_hotspot_extraction(options, cfg)


if __name__ == "__main__":
    main()
