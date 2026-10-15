#!/usr/bin/env python3
"""Generate full-day northern-hemisphere DFF vectors for selected events.

The scientific and cleaning settings come directly from rbf_fit's packaged
default_config.json.  This script only adapts this project's download layout,
builds rbf_fit's per-radar input cache, and streams fitted vectors to one
Parquet file per unique event date.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import functools
import hashlib
import json
import os
from pathlib import Path
from multiprocessing.pool import ThreadPool
import sys
import tempfile
import traceback
from typing import Iterable, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parent
MPL_CONFIG_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
RBF_FIT_SOURCE = PROJECT_ROOT.parent / "rbf_fit" / "src"
if RBF_FIT_SOURCE.is_dir() and str(RBF_FIT_SOURCE) not in sys.path:
    sys.path.insert(0, str(RBF_FIT_SOURCE))

try:
    import rbf_fit
    from rbf_fit import DFFConfig, MakeDFF
    from rbf_fit.io import fitacf as fitacf_io
    from rbf_fit.io import map2 as map2_io
except ImportError as error:
    raise SystemExit(
        "rbf_fit or one of its dependencies is unavailable. Install the sibling "
        "package with: python -m pip install ../rbf_fit"
    ) from error

from event_list import selected_events


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_CACHE_ROOT = DEFAULT_DATA_ROOT / "fitacf_cache"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "dff"

# This matches the hemisphere classification used by rbf_fit's ReadData.
SOUTHERN_RADARS = frozenset(
    {
        "bpk",
        "dce",
        "dcn",
        "fir",
        "hal",
        "ker",
        "mcm",
        "san",
        "sps",
        "sye",
        "sys",
        "tig",
        "unw",
        "zho",
    }
)


def install_scipy_lpmn_compatibility() -> None:
    """Provide SciPy >=1.17's supported replacement for removed ``lpmn``."""
    special = map2_io.special
    if hasattr(special, "lpmn"):
        return

    def lpmn(order: int, degree: int, value):
        result = special.assoc_legendre_p_all(
            degree,
            order,
            value,
            branch_cut=2,
            norm=False,
            diff_n=1,
        )
        # assoc_legendre_p_all is indexed [derivative, degree, order].
        # lpmn returned [order, degree] arrays for values and derivatives.
        values = result[0, :, : order + 1].T
        derivatives = result[1, :, : order + 1].T
        return values, derivatives

    special.lpmn = lpmn


install_scipy_lpmn_compatibility()


def parse_args() -> argparse.Namespace:
    events = selected_events()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event",
        dest="event_names",
        action="append",
        choices=sorted(events),
        help="process only this event (repeatable); default: every event",
    )
    parser.add_argument(
        "--cadence",
        default="1min",
        help="DFF fit cadence understood by pandas (default: 1min)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"downloaded event data root (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help=f"rbf_fit input-cache root (default: {DEFAULT_CACHE_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"DFF Parquet destination (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--cache-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="threads used while converting raw fitacf3 records (default: up to 8)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="optional rbf_fit JSON override; omitted means packaged defaults",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing daily DFF files instead of skipping them",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop on the first failed fit instead of recording and continuing",
    )
    parser.add_argument(
        "--verbose-errors",
        action="store_true",
        help="print a traceback for every failed timestamp",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and show the work without reading fitacf files",
    )
    return parser.parse_args()


def events_by_date(event_names: Iterable[str] | None) -> dict[dt.date, list[str]]:
    events = selected_events()
    names = list(event_names) if event_names else list(events)
    grouped: dict[dt.date, list[str]] = {}
    for name in names:
        grouped.setdefault(events[name].event_date.date(), []).append(name)
    return dict(sorted(grouped.items()))


def full_day_times(event_date: dt.date, cadence: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(event_date)
    end = start + pd.Timedelta(days=1)
    step = pd.Timedelta(cadence)
    if step <= pd.Timedelta(0):
        raise ValueError("cadence must be positive")
    return pd.date_range(start, end, freq=step, inclusive="left")


def northern_radars(day_dir: Path) -> list[str]:
    radars = sorted(
        entry.name.lower()
        for entry in day_dir.iterdir()
        if entry.is_dir()
        and entry.name.lower() not in SOUTHERN_RADARS
        and any(entry.glob("*.fitacf3.bz2"))
    )
    if not radars:
        raise FileNotFoundError(f"No northern fitacf3 radar folders found in {day_dir}")
    return radars


def validate_event_inputs(data_root: Path, event_date: dt.date) -> tuple[list[str], Path]:
    day = event_date.strftime("%Y%m%d")
    day_dir = data_root / day
    if not day_dir.is_dir():
        raise FileNotFoundError(f"Missing event data directory: {day_dir}")
    radars = northern_radars(day_dir)
    map2_file = day_dir / f"{day}.north.map2.bz2"
    if not map2_file.is_file():
        raise FileNotFoundError(f"Missing northern MAP2 file: {map2_file}")
    return radars, map2_file


@contextmanager
def archive_view(
    data_root: Path,
    event_dates: Iterable[dt.date],
) -> Iterator[Path]:
    """Expose the downloaded files using rbf_fit's archive-style hierarchy."""
    with tempfile.TemporaryDirectory(prefix="steve-stats-rbf-archive-") as temp_dir:
        archive_root = Path(temp_dir)
        for event_date in event_dates:
            day = event_date.strftime("%Y%m%d")
            year = event_date.strftime("%Y")
            day_dir = data_root / day
            radars, map2_file = validate_event_inputs(data_root, event_date)

            for radar in radars:
                target_dir = archive_root / year / "fitacf3" / radar
                target_dir.mkdir(parents=True, exist_ok=True)
                for source in sorted((day_dir / radar).glob(f"{day}.*.fitacf3.bz2")):
                    target = target_dir / source.name
                    if not target.exists():
                        target.symlink_to(source.resolve())

            map_target = (
                archive_root
                / year
                / "map2"
                / "north"
                / f"{day}.north.map2.bz2"
            )
            map_target.parent.mkdir(parents=True, exist_ok=True)
            map_target.symlink_to(map2_file.resolve())

        yield archive_root


@contextmanager
def use_local_map2(archive_root: Path):
    """Point rbf_fit's MAP2 calls at the temporary local archive view."""
    original_get_vectors = map2_io.get_fit_vectors_map_file
    original_read_map = map2_io._read_map_file
    cached_read_map = functools.lru_cache(maxsize=None)(original_read_map)

    def configured_get_vectors(*args, **kwargs):
        kwargs["parent_dir"] = str(archive_root)
        kwargs["hemi"] = "north"
        return original_get_vectors(*args, **kwargs)

    map2_io._read_map_file = cached_read_map
    map2_io.get_fit_vectors_map_file = configured_get_vectors
    try:
        yield
    finally:
        map2_io.get_fit_vectors_map_file = original_get_vectors
        map2_io._read_map_file = original_read_map
        cached_read_map.cache_clear()


def build_fitacf_cache(
    event_date: dt.date,
    radars: list[str],
    archive_root: Path,
    cache_root: Path,
    workers: int,
) -> None:
    """Build missing daily input Parquets without changing fitting defaults."""
    day = event_date.strftime("%Y%m%d")
    cache_dir = cache_root / day
    suffix = event_date.strftime("_%Y-%m-%d.parquet")
    missing = [radar for radar in radars if not (cache_dir / f"{radar}{suffix}").is_file()]
    if not missing:
        print(f"{day}: all {len(radars)} radar input caches already exist.")
        return
    if workers < 1:
        raise ValueError("cache workers must be at least 1")

    cache_root.mkdir(parents=True, exist_ok=True)
    print(f"{day}: building input caches for {len(missing)} radar(s): {', '.join(missing)}")
    # Raw ingestion needs skip_if_file_not_found=False. MakeDFF itself is still
    # called with the packaged default (True) after these caches exist.
    original_pool = fitacf_io.multiprocessing.Pool
    fitacf_io.multiprocessing.Pool = lambda: ThreadPool(processes=workers)
    try:
        fitacf_io.ReadData(
            dt.datetime.combine(event_date, dt.time.min),
            radars=missing,
            hemisphere="North",
            outdata_dir=str(cache_root),
            sd_data_base_dir=str(archive_root),
            skip_if_file_not_found=False,
        )
    finally:
        fitacf_io.multiprocessing.Pool = original_pool


def config_identity(config: DFFConfig) -> tuple[str, str]:
    source = Path(config.source).resolve()
    effective_config = json.dumps(
        config.values, sort_keys=True, separators=(",", ":")
    ).encode()
    return str(source), hashlib.sha256(effective_config).hexdigest()


def fit_one_time(
    fit_time: pd.Timestamp,
    radars: list[str],
    archive_root: Path,
    cache_root: Path,
    output_dir: Path,
    config: DFFConfig,
    event_names: list[str],
) -> pd.DataFrame:
    job = MakeDFF(
        fit_time.to_pydatetime(),
        radars,
        hemisphere="North",
        data_base_dir=str(cache_root),
        sd_data_base_dir=str(archive_root),
        outdata_dir=str(output_dir),
        dff_config=config,
    )
    _, fitted, _ = job.make_hi_res_fit()
    if "fit_source" in fitted.columns:
        fitted = fitted.loc[fitted["fit_source"] != "map_prior"].copy()
    if fitted.empty:
        raise ValueError("fit produced no radar-location DFF vectors")

    config_source, config_hash = config_identity(config)
    fitted["fit_time"] = fit_time.to_pydatetime()
    fitted["event_names"] = ",".join(event_names)
    fitted["hemisphere"] = "north"
    fitted["input_radars"] = ",".join(radars)
    fitted["rbf_fit_version"] = rbf_fit.__version__
    fitted["dff_config_source"] = config_source
    fitted["dff_config_sha256"] = config_hash
    return fitted.reset_index(drop=True)


def write_full_day(
    event_date: dt.date,
    event_names: list[str],
    radars: list[str],
    archive_root: Path,
    cache_root: Path,
    output_dir: Path,
    cadence: str,
    config: DFFConfig,
    overwrite: bool,
    fail_fast: bool,
    verbose_errors: bool,
) -> tuple[Path, int, list[dict[str, str]]]:
    day = event_date.strftime("%Y%m%d")
    output = (output_dir / f"dff_{day}.parquet").resolve()
    if output.exists() and not overwrite:
        print(f"{day}: output already exists; skipping {output}")
        return output, -1, []

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{output.stem}.", suffix=".parquet", dir=output.parent, delete=False
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    temporary_path.unlink()

    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    columns: list[str] | None = None
    total_rows = 0
    failures: list[dict[str, str]] = []

    try:
        times = full_day_times(event_date, cadence)
        for index, fit_time in enumerate(times):
            if index % max(1, int(pd.Timedelta("1h") / pd.Timedelta(cadence))) == 0:
                print(f"{day}: fitting {fit_time:%H:%M} ({index + 1}/{len(times)})", flush=True)
            try:
                frame = fit_one_time(
                    fit_time,
                    radars,
                    archive_root,
                    cache_root,
                    output_dir,
                    config,
                    event_names,
                )
                if writer is None:
                    columns = frame.columns.tolist()
                    table = pa.Table.from_pandas(frame, preserve_index=False)
                    schema = table.schema
                    writer = pq.ParquetWriter(temporary_path, schema, compression="zstd")
                else:
                    frame = frame.reindex(columns=columns)
                    table = pa.Table.from_pandas(
                        frame, schema=schema, preserve_index=False, safe=False
                    )
                writer.write_table(table)
                total_rows += len(frame)
            except Exception as error:
                failure = {"fit_time": fit_time.isoformat(), "error": str(error)}
                failures.append(failure)
                print(f"{day} {fit_time:%H:%M}: skipped: {error}", file=sys.stderr)
                if verbose_errors:
                    traceback.print_exc()
                if fail_fast:
                    raise

        if writer is None:
            raise RuntimeError(f"No DFF vectors were generated for {day}")
        writer.close()
        writer = None
        temporary_path.replace(output)
    finally:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)

    return output, total_rows, failures


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    grouped_events = events_by_date(args.event_names)
    config = DFFConfig.load(args.config)

    print(f"rbf_fit config: {Path(config.source).resolve()}")
    print(f"rbf_fit backend: {config.fitting_config.backend}")
    runnable_events: dict[dt.date, list[str]] = {}
    input_failures: dict[str, list[dict[str, str]]] = {}
    for event_date, names in grouped_events.items():
        try:
            radars, map2_file = validate_event_inputs(data_root, event_date)
        except FileNotFoundError as error:
            print(f"{event_date:%Y%m%d}: unavailable: {error}", file=sys.stderr)
            input_failures[event_date.strftime("%Y%m%d")] = [
                {"fit_time": "full_day", "error": str(error)}
            ]
            if args.fail_fast:
                raise
            continue
        runnable_events[event_date] = names
        print(
            f"{event_date:%Y%m%d} ({', '.join(names)}): "
            f"{len(radars)} northern radar(s), MAP2={map2_file.name}"
        )

    if args.dry_run:
        return 1 if input_failures else 0
    if not runnable_events:
        print("No event dates have complete fitacf3 and MAP2 inputs.", file=sys.stderr)
        return 2

    all_failures = dict(input_failures)
    with archive_view(data_root, runnable_events) as archive_root:
        with use_local_map2(archive_root):
            for event_date, names in runnable_events.items():
                radars, _ = validate_event_inputs(data_root, event_date)
                build_fitacf_cache(
                    event_date,
                    radars,
                    archive_root,
                    cache_root,
                    args.cache_workers,
                )
                output, rows, failures = write_full_day(
                    event_date=event_date,
                    event_names=names,
                    radars=radars,
                    archive_root=archive_root,
                    cache_root=cache_root,
                    output_dir=output_dir,
                    cadence=args.cadence,
                    config=config,
                    overwrite=args.overwrite,
                    fail_fast=args.fail_fast,
                    verbose_errors=args.verbose_errors,
                )
                if rows >= 0:
                    print(
                        f"{event_date:%Y%m%d}: wrote {rows:,} vectors to {output} "
                        f"with {len(failures)} skipped fit time(s)."
                    )
                if failures:
                    all_failures[event_date.strftime("%Y%m%d")] = failures

    if all_failures:
        report = output_dir / "dff_failures.json"
        report.write_text(json.dumps(all_failures, indent=2) + "\n", encoding="utf-8")
        print(f"Failure details: {report}", file=sys.stderr)
    return 1 if input_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
