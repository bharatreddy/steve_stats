#!/usr/bin/env python3
"""Create LOS, DFF, and side-by-side comparison movies matching radar_los_maps.ipynb.

Default: 2011-04-02 04:30--06:30 UT, 10-minute windows, AACGM-v2 maps.
LOS frames contain all returns in [start, end), without averaging; DFF frames
are fits at each window midpoint using config/dff_config.json. DFF requires
local MAP2 files and the sibling rbf_fit package. MP4 requires ffmpeg; GIF
uses Pillow (which retains frames in memory). No notebook execution is needed.

Examples:
    python radar_vector_movies.py
    python radar_vector_movies.py --kind los --format gif --step 2min
    python radar_vector_movies.py --kind dff --fps 4 --coordinates geographic
    python radar_vector_movies.py --kind combined --fps 4
"""
import bz2
import datetime as dt
from functools import lru_cache
from pathlib import Path
import os
import warnings

try:
    import cmcrameri as cmc
except ImportError:
    cmc = None

PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "event_list.py").exists():
    raise RuntimeError("Start the notebook with the steve_stats project as the working directory.")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache/matplotlib"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
from pyproj import Geod
import pydarn
import pydarnio
import aacgmv2
from sd_carto import SDCarto  # copied from meso_tracers/plotting_utils/sd_carto.py
from pydarn.utils.coordinates import gate2geographic_location
from matplotlib.animation import FFMpegWriter, PillowWriter, writers
import argparse

def display(value):
    """Print notebook inventory tables when running from the command line."""
    print(value.to_string(index=False))


EVENT_DATE = "2016-07-25"#"2015-09-11"#"2011-04-02"
START_TIME = "03:30"#"04:00"#"04:30"
END_TIME = "07:30"#"07:00"#"06:30"
TIME_DELTA = "10min"  # pandas duration string or datetime.timedelta
RADARS = ["cve", "cvw", "bks"]
SCATTER = "both"  # "ground", "ionospheric", or "both"

cmap = (cmc.cm.berlin_r if cmc else plt.get_cmap("RdBu_r"))#plt.get_cmap("RdBu_r")

DATA_ROOT = "data"
SAVE_MAPS = False
OUTPUT_ROOT = "outputs/los_maps"
MAP_EXTENT = [-130, -45, 30, 85]  # west, east, south, north (geographic degrees)
VELOCITY_LIMIT = 400  # m/s; colors saturate beyond +/- this value
VECTOR_SECONDS = 120  # display scale: 1000 m/s arrow spans 120 km
COASTLINES = True  # first use may download Cartopy Natural Earth coastlines

GRID_COLUMNS = 3  # number of maps per row

COORDINATES = "aacgmv2"  # "aacgmv2", "aacgmv2_mlt", or "geographic"
AACGM_HEIGHT_KM = 300  # same conversion altitude for data, radar markers and coastlines
AACGM_EXTENT = [-100, 50, 45, 85]  # magnetic longitude, magnetic latitude
MLT_EXTENT = [-180, 180, 40, 90]  # longitude = MLT hours * 15, wrapped to +/-180

GROUND_SCATTER_CONSTANT_COLOR = True  # False colors ground scatter by velocity
GROUND_SCATTER_COLOR = "grey"  # Matplotlib color, applied to ground dots and arrows
GROUND_DOT_SIZE = 6  # marker area in points squared
IONOSPHERIC_DOT_SIZE = 10  # marker area in points squared


DFF_SAVE_MAPS = False  # save DFF grid independently of LOS SAVE_MAPS
DFF_VELOCITY_LIMIT = 800  # speed colorbar: 0 to this value (m/s)

DFF_CMAP = (cmc.cm.batlow if cmc else plt.get_cmap("viridis"))
DFF_ARROW_WIDTH = 0.003#0.006  # same width as tracers_sd/notebooks/dff_polar_panel.ipynb
DFF_ARROW_HEADWIDTH = 4
DFF_ARROW_HEADLENGTH = 5
DFF_ARROW_HEADAXISLENGTH = 3
DFF_VECTOR_SECONDS = 600  # display scale: 1000 m/s spans 600 km; increase for longer arrows


def configure():
    global data_root, end, radars, start, step, windows
    if any(not np.isfinite(size) or size < 0 for size in (GROUND_DOT_SIZE, IONOSPHERIC_DOT_SIZE)):
        raise ValueError("Dot sizes must be finite, nonnegative areas in points squared.")

    start = pd.Timestamp(f"{EVENT_DATE} {START_TIME}")
    end = pd.Timestamp(f"{EVENT_DATE} {END_TIME}")
    if start == end:
        raise ValueError("Start and end must differ.")
    if end < start:
        end += pd.Timedelta(days=1)
    step = pd.Timedelta(TIME_DELTA)
    if pd.isna(step) or step <= pd.Timedelta(0):
        raise ValueError("TIME_DELTA must be positive.")
    if SCATTER not in {"both", "ground", "ionospheric"}:
        raise ValueError("SCATTER must be both, ground, or ionospheric.")
    radars = list(dict.fromkeys(str(r).strip().lower() for r in RADARS))
    if not radars or VELOCITY_LIMIT <= 0 or VECTOR_SECONDS <= 0:
        raise ValueError("Select radars and positive velocity/display scales.")
    data_root = (PROJECT_ROOT / DATA_ROOT).resolve()
    if COORDINATES not in {"geographic", "aacgmv2", "aacgmv2_mlt"}:
        raise ValueError("COORDINATES must be geographic, aacgmv2, or aacgmv2_mlt.")
    output_dir = PROJECT_ROOT / OUTPUT_ROOT / start.strftime("%Y%m%d") / SCATTER / COORDINATES
    windows = [(t, min(t + step, end)) for t in pd.date_range(start, end, freq=step, inclusive="left")]
    print(f"{len(windows)} windows: {start} to {end} UT; scatter={SCATTER}")


def record_time(record):
    return dt.datetime(*(int(record[f"time.{key}"]) for key in
                         ("yr", "mo", "dy", "hr", "mt", "sc")),
                       microsecond=int(record.get("time.us", 0)))


def read_records(path):
    with bz2.open(path, "rb") as stream:
        raw = stream.read()
    if hasattr(pydarn, "SuperDARNRead"):
        return pydarn.SuperDARNRead(raw, True).read_fitacf()
    records = pydarnio.read_fitacf(raw)
    return records[0] if isinstance(records, tuple) else records


@lru_cache(maxsize=None)
def gate_position(station, radar, day, beam, gate, frang, rsep):
    # pyDARN's geometry reads hardware from its registry. Restore after use.
    radar_info = pydarn.SuperDARNRadars.radars[pydarn.RadarID(station)]
    original = radar_info.hardware_info
    hardware = pydarn.read_hdw_file(radar, date=dt.datetime.fromisoformat(day))
    try:
        pydarn.SuperDARNRadars.radars[pydarn.RadarID(station)] = radar_info._replace(hardware_info=hardware)
        lat, lon = gate2geographic_location(
            stid=pydarn.RadarID(station), beam=beam, range_gate=gate,
            frang=frang, rsep=rsep, center=True, height=300)
    finally:
        pydarn.SuperDARNRadars.radars[pydarn.RadarID(station)] = radar_info
    return float(lat), float(lon), hardware.geographic.lat, hardware.geographic.lon


def load_observations():
    global rows, observations
    rows, inventory = [], []
    for day in pd.date_range(start.normalize(), (end - pd.Timedelta(nanoseconds=1)).normalize(), freq="D"):
        for radar in radars:
            folder = data_root / day.strftime("%Y%m%d") / radar
            files = sorted(folder.glob("*.fitacf3.bz2")) or sorted(folder.glob("*.fitacf.bz2"))
            inventory.append({"date": str(day.date()), "radar": radar, "files": len(files)})
            if not files:
                warnings.warn(f"No files for {radar} on {day.date()} in {folder}")
            for path in files:
                # Read every daily file: avoids assuming archive file duration.
                for record in read_records(path):
                    time = record_time(record)
                    if not start <= time < end or "slist" not in record:
                        continue
                    gates = record["slist"]
                    velocities, flags = record.get("v", []), record.get("gflg", [])
                    if not (len(gates) == len(velocities) == len(flags)):
                        warnings.warn(f"Skipping inconsistent arrays in {path.name} at {time}")
                        continue
                    for gate, velocity, flag in zip(gates, velocities, flags):
                        if flag not in (0, 1) or not np.isfinite(velocity):
                            continue
                        lat, lon, radar_lat, radar_lon = gate_position(
                            int(record["stid"]), radar, time.date().isoformat(),
                            int(record["bmnum"]), int(gate), float(record["frang"]), float(record["rsep"]))
                        if np.isfinite(lat) and np.isfinite(lon):
                            rows.append((time, radar, lat, lon, radar_lat, radar_lon, float(velocity), int(flag)))
    observations = pd.DataFrame(rows, columns=["time", "radar", "lat", "lon", "radar_lat", "radar_lon", "velocity", "gflg"])
    display(pd.DataFrame(inventory))
    if observations.empty:
        raise RuntimeError("No finite LOS observations in this interval. Check the inventory, date and times.")
    if SCATTER != "both":
        observations = observations.loc[observations.gflg.eq(1 if SCATTER == "ground" else 0)].copy()
    print(f"Selected {len(observations):,} observations.")


def map_coordinates(lon, lat, timestamp):
    """Convert geographic points; the same timestamp is used throughout a panel."""
    lon, lat = np.atleast_1d(lon).astype(float), np.atleast_1d(lat).astype(float)
    if COORDINATES == "geographic":
        return lon, lat
    mlat, mlon, _ = aacgmv2.convert_latlon_arr(
        lat, lon, AACGM_HEIGHT_KM, timestamp.to_pydatetime(), method_code="G2A")
    if COORDINATES == "aacgmv2_mlt":
        mlon = np.asarray(aacgmv2.convert_mlt(mlon, timestamp.to_pydatetime())) * 15
    return (np.asarray(mlon) + 180) % 360 - 180, np.asarray(mlat)

def setup_projection():
    global extent, geo, geod, norm, projection
    geod = Geod(ellps="WGS84")
    if COORDINATES == "geographic":
        projection = ccrs.LambertConformal(central_longitude=-95, central_latitude=55)
        extent = MAP_EXTENT
    elif COORDINATES == "aacgmv2":
        projection = ccrs.NorthPolarStereo(central_longitude=-25)
        extent = AACGM_EXTENT
    else:
        projection = ccrs.NorthPolarStereo(central_longitude=0)
        extent = MLT_EXTENT



    geo = ccrs.PlateCarree()
    norm = Normalize(-VELOCITY_LIMIT, VELOCITY_LIMIT)



def render_los(left, right, fig=None):
    windows = [(left, right)]
    summary = []
    columns = 1
    if columns < 1:
        raise ValueError("GRID_COLUMNS must be a positive integer.")
    row_count = (len(windows) + columns - 1) // columns
    # Match subplot slots to the projected map bounds, which can be much wider
    # than tall in AACGM mode. Square slots otherwise leave large gaps between rows.
    probe_fig = plt.figure()
    probe_ax = probe_fig.add_subplot(projection=projection)
    probe_ax.set_extent(extent, crs=geo)
    xmin, xmax = probe_ax.get_xlim()
    ymin, ymax = probe_ax.get_ylim()
    panel_aspect = (ymax - ymin) / (xmax - xmin)
    plt.close(probe_fig)
    panel_width = 6
    panel_height = panel_width * panel_aspect + 0.65  # room for titles and tick labels
    standalone = fig is None
    if standalone:
        fig = plt.figure(figsize=(panel_width * columns + 0.9,
                                  panel_height * row_count + 0.8), layout="constrained")
        fig.get_layout_engine().set(h_pad=0.04, w_pad=0.04, hspace=0.02, wspace=0.02)
    axes = []
    for index, (left, right) in enumerate(windows):
        panel_time = left + (right - left) / 2
        ax = fig.add_subplot(row_count, columns, index + 1, projection="sdcarto",
                             map_projection=projection, plot_date=panel_time.to_pydatetime(),
                             coords="geo" if COORDINATES == "geographic" else COORDINATES)
        ax.aacgm_height = AACGM_HEIGHT_KM
        axes.append(ax)
        frame = observations.loc[(observations.time >= left) & (observations.time < right)]
        ax.set_extent(extent, crs=geo)
        if COASTLINES:
            ax.coastlines(resolution="110m", linewidth=0.6)
        grid = ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, linewidth=0.4, alpha=0.4)
        if COORDINATES == "aacgmv2_mlt":
            from matplotlib.ticker import FuncFormatter
            grid.xformatter = FuncFormatter(lambda value, pos: f"{(value / 15) % 24:g}h")
        for flag, marker, size in [(0, ".", IONOSPHERIC_DOT_SIZE), (1, "o", GROUND_DOT_SIZE)]:
            subset = frame.loc[frame.gflg.eq(flag)]
            if subset.empty:
                continue
            lon, lat, velocity = (subset[c].to_numpy() for c in ("lon", "lat", "velocity"))
            toward, _, _ = geod.inv(lon, lat, subset.radar_lon.to_numpy(), subset.radar_lat.to_numpy())
            bearing = toward + np.where(velocity >= 0, 0, 180)
            tip_lon, tip_lat, _ = geod.fwd(lon, lat, bearing, np.abs(velocity) * VECTOR_SECONDS)
            lon, lat = map_coordinates(lon, lat, panel_time)
            tip_lon, tip_lat = map_coordinates(tip_lon, tip_lat, panel_time)
            base_xy = projection.transform_points(geo, lon, lat)
            tip_xy = projection.transform_points(geo, tip_lon, tip_lat)
            scatter_zorder = 4 if flag == 0 else 2  # ionospheric above ground
            constant_ground = flag == 1 and GROUND_SCATTER_CONSTANT_COLOR
            if constant_ground:
                quiver_values = ()
                quiver_colors = {"color": GROUND_SCATTER_COLOR}
                dot_colors = {"color": GROUND_SCATTER_COLOR}
            else:
                quiver_values = (velocity,)
                quiver_colors = {"cmap": cmap, "norm": norm}
                dot_colors = {"c": velocity, "cmap": cmap, "norm": norm}
            ax.quiver(base_xy[:, 0], base_xy[:, 1], tip_xy[:, 0] - base_xy[:, 0],
                      tip_xy[:, 1] - base_xy[:, 1], *quiver_values, **quiver_colors,
                      angles="xy", scale_units="xy", scale=1, width=0.0018,
                      zorder=scatter_zorder)
            ax.scatter(lon, lat, **dot_colors, marker=marker,
                       s=size, transform=geo, linewidths=0.4, zorder=scatter_zorder + 0.1)
        # Include available selected radar sites even in windows without returns.
        sites = pd.DataFrame(rows, columns=["time", "radar", "lat", "lon", "radar_lat", "radar_lon", "velocity", "gflg"])
        for radar, site in sites.groupby("radar", sort=False):
            row = site.iloc[0]
            site_lon, site_lat = map_coordinates(row.radar_lon, row.radar_lat, panel_time)
            ax.plot(site_lon, site_lat, "k^", ms=5, transform=geo)
            ax.text(site_lon[0] + 1, site_lat[0], radar.upper(), transform=geo, fontsize=9)
        if frame.empty:
            ax.text(0.5, 0.5, "No selected scatter in this window", transform=ax.transAxes, ha="center")
        handles = [Line2D([], [], color=GROUND_SCATTER_COLOR if flag == 1 and GROUND_SCATTER_CONSTANT_COLOR else "k",
                          marker=m, linestyle="None", label=label)
                   for flag, m, label in [(0, ".", "Ionospheric"), (1, "o", "Ground")]
                   if SCATTER == "both" or (flag == 1) == (SCATTER == "ground")]
        ax.legend(handles=handles, loc="lower left")
        ax.set_title(f"{left:%H:%M} – {right:%H:%M} UT | {len(frame):,} returns", fontsize=11)
        summary.append({"start_UT": left, "end_UT": right, "returns": len(frame),
                        "radars_with_returns": ", ".join(sorted(frame.radar.unique()))})
    if standalone:
        fig.suptitle(f"{start:%Y-%m-%d %H:%M} – {end:%Y-%m-%d %H:%M} UT | {SCATTER} scatter | {COORDINATES}\n"
                 f"{', '.join(r.upper() for r in radars)} | 1000 m/s arrow = {VECTOR_SECONDS:g} km; no time averaging",
                     fontsize=15)
    else:
        fig.suptitle(f"Line-of-sight | {SCATTER} scatter\n"
                     f"1000 m/s arrow = {VECTOR_SECONDS:g} km; no time averaging", fontsize=11)
    if SCATTER != "ground" or not GROUND_SCATTER_CONSTANT_COLOR:
        colorbar_label = "LOS velocity (m/s); positive toward radar"
        if GROUND_SCATTER_CONSTANT_COLOR and SCATTER == "both":
            colorbar_label = "Ionospheric " + colorbar_label
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                     shrink=0.7, label=colorbar_label, extend="both")
    return fig

def fit_dff():
    global dff_vectors
    # Fit once per window at its midpoint using only available selected radars.
    # This can take a few minutes. Run the setup/data cells first.
    from generate_event_dff import (
        archive_view, use_local_map2, build_fitacf_cache, validate_event_inputs,
        DFFConfig, MakeDFF,
    )


    if DFF_VELOCITY_LIMIT <= 0:
        raise ValueError("DFF_VELOCITY_LIMIT must be positive.")
    dff_frames = []
    DFF_CONFIG_PATH = PROJECT_ROOT / "config/dff_config.json"
    dff_config = DFFConfig.load(DFF_CONFIG_PATH)
    print(f"DFF configuration: {DFF_CONFIG_PATH}")
    dff_days = sorted({(left + (right - left) / 2).date() for left, right in windows})
    with archive_view(data_root, dff_days) as archive_root:
        with use_local_map2(archive_root):
            for day in dff_days:
                available, _, filetype = validate_event_inputs(data_root, day)
                selected = [radar for radar in radars if radar in available]
                if not selected:
                    raise ValueError(f"No selected radar data available on {day}")
                missing = sorted(set(radars) - set(selected))
                if missing:
                    warnings.warn(f"DFF {day}: unavailable radars: {', '.join(missing)}")
                build_fitacf_cache(day, selected, filetype, archive_root,
                                   data_root / "fitacf_cache", workers=2)
                for left, right in windows:
                    midpoint = left + (right - left) / 2
                    if midpoint.date() != day:
                        continue
                    print(f"Fitting DFF {midpoint} UT: {', '.join(selected)}", flush=True)
                    job = MakeDFF(midpoint.to_pydatetime(), selected, hemisphere="North",
                                  data_base_dir=str(data_root / "fitacf_cache"),
                                  sd_data_base_dir=str(archive_root),
                                  outdata_dir=str(PROJECT_ROOT / "outputs/dff_notebook"),
                                  dff_config=dff_config)
                    _, fitted, _ = job.make_hi_res_fit()
                    if "fit_source" in fitted:
                        fitted = fitted.loc[fitted.fit_source.ne("map_prior")].copy()
                    fitted = fitted.copy()
                    fitted["panel_time"] = midpoint
                    fitted["selected_radars"] = ",".join(selected)
                    dff_frames.append(fitted)
    dff_vectors = pd.concat(dff_frames, ignore_index=True)


def render_dff(left, right, fig=None):
    windows = [(left, right)]
    norm = Normalize(0, DFF_VELOCITY_LIMIT)
    dff_output_dir = PROJECT_ROOT / OUTPUT_ROOT / start.strftime("%Y%m%d") / "dff" / COORDINATES

    summary = []
    columns = 1
    if columns < 1:
        raise ValueError("GRID_COLUMNS must be a positive integer.")
    row_count = (len(windows) + columns - 1) // columns
    # Match subplot slots to the projected map bounds, which can be much wider
    # than tall in AACGM mode. Square slots otherwise leave large gaps between rows.
    probe_fig = plt.figure()
    probe_ax = probe_fig.add_subplot(projection=projection)
    probe_ax.set_extent(extent, crs=geo)
    xmin, xmax = probe_ax.get_xlim()
    ymin, ymax = probe_ax.get_ylim()
    panel_aspect = (ymax - ymin) / (xmax - xmin)
    plt.close(probe_fig)
    panel_width = 6
    panel_height = panel_width * panel_aspect + 0.65  # room for titles and tick labels
    standalone = fig is None
    if standalone:
        fig = plt.figure(figsize=(panel_width * columns + 0.9,
                                  panel_height * row_count + 0.8), layout="constrained")
        fig.get_layout_engine().set(h_pad=0.04, w_pad=0.04, hspace=0.02, wspace=0.02)
    axes = []
    for index, (left, right) in enumerate(windows):
        panel_time = left + (right - left) / 2
        ax = fig.add_subplot(row_count, columns, index + 1, projection="sdcarto",
                             map_projection=projection, plot_date=panel_time.to_pydatetime(),
                             coords="geo" if COORDINATES == "geographic" else COORDINATES)
        ax.aacgm_height = AACGM_HEIGHT_KM
        axes.append(ax)
        frame = dff_vectors.loc[dff_vectors.panel_time.eq(panel_time)].copy()
        frame = frame.dropna(subset=["mlat", "mlt", "east_vel", "north_vel"])
        ax.set_extent(extent, crs=geo)
        if COASTLINES:
            ax.coastlines(resolution="110m", linewidth=0.6)
        grid = ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, linewidth=0.4, alpha=0.4)
        if COORDINATES == "aacgmv2_mlt":
            from matplotlib.ticker import FuncFormatter
            grid.xformatter = FuncFormatter(lambda value, pos: f"{(value / 15) % 24:g}h")
        if not frame.empty:
            mlat = frame.mlat.to_numpy()
            mlon = np.asarray(aacgmv2.convert_mlt(frame.mlt.to_numpy(), panel_time.to_pydatetime(), m2a=True))
            east, north = frame.east_vel.to_numpy(), frame.north_vel.to_numpy()
            speed = np.hypot(east, north)
            bearing = np.degrees(np.arctan2(east, north))
            # Step in the magnetic tangent plane, then transform both endpoints.
            magnetic_geod = Geod(a=6371000 + AACGM_HEIGHT_KM * 1000, f=0)
            tip_mlon, tip_mlat, _ = magnetic_geod.fwd(mlon, mlat, bearing, speed * DFF_VECTOR_SECONDS)
            if COORDINATES == "geographic":
                lat, lon, _ = aacgmv2.convert_latlon_arr(mlat, mlon, AACGM_HEIGHT_KM,
                                                        panel_time.to_pydatetime(), method_code="A2G")
                tip_lat, tip_lon, _ = aacgmv2.convert_latlon_arr(tip_mlat, tip_mlon, AACGM_HEIGHT_KM,
                                                                panel_time.to_pydatetime(), method_code="A2G")
            elif COORDINATES == "aacgmv2_mlt":
                lat, tip_lat = mlat, tip_mlat
                lon = np.asarray(aacgmv2.convert_mlt(mlon, panel_time.to_pydatetime())) * 15
                tip_lon = np.asarray(aacgmv2.convert_mlt(tip_mlon, panel_time.to_pydatetime())) * 15
            else:
                lon, lat, tip_lon, tip_lat = mlon, mlat, tip_mlon, tip_mlat
            base_xy = projection.transform_points(geo, np.asarray(lon), np.asarray(lat))
            tip_xy = projection.transform_points(geo, np.asarray(tip_lon), np.asarray(tip_lat))
            ax.quiver(base_xy[:, 0], base_xy[:, 1], tip_xy[:, 0] - base_xy[:, 0],
                      tip_xy[:, 1] - base_xy[:, 1], speed, cmap=DFF_CMAP, norm=norm,
                      angles="xy", scale_units="xy", scale=1,
                      width=DFF_ARROW_WIDTH, headwidth=DFF_ARROW_HEADWIDTH,
                      headlength=DFF_ARROW_HEADLENGTH, headaxislength=DFF_ARROW_HEADAXISLENGTH,
                      minlength=0, pivot="tail", zorder=30)
        # Include available selected radar sites even in windows without returns.
        sites = pd.DataFrame(rows, columns=["time", "radar", "lat", "lon", "radar_lat", "radar_lon", "velocity", "gflg"])
        for radar, site in sites.groupby("radar", sort=False):
            row = site.iloc[0]
            site_lon, site_lat = map_coordinates(row.radar_lon, row.radar_lat, panel_time)
            ax.plot(site_lon, site_lat, "k^", ms=5, transform=geo)
            ax.text(site_lon[0] + 1, site_lat[0], radar.upper(), transform=geo, fontsize=9)
        if frame.empty:
            ax.text(0.5, 0.5, "No DFF vectors at window midpoint", transform=ax.transAxes, ha="center")
        ax.set_title(f"{left:%H:%M} – {right:%H:%M} UT | {len(frame):,} DFF vectors", fontsize=11)
        summary.append({"start_UT": left, "end_UT": right, "returns": len(frame),
                        "radars_with_returns": ", ".join(sorted(frame.selected_radars.unique()))})
    if standalone:
        fig.suptitle(f"{start:%Y-%m-%d %H:%M} – {end:%Y-%m-%d %H:%M} UT | DFF | {COORDINATES}\n"
                 f"{', '.join(r.upper() for r in radars)} | 1000 m/s arrow = {DFF_VECTOR_SECONDS:g} km; snapshot at each window midpoint",
                     fontsize=15)
    else:
        fig.suptitle(f"DFF vectors | midpoint snapshot\n"
                     f"1000 m/s arrow = {DFF_VECTOR_SECONDS:g} km", fontsize=11)
    fig.colorbar(ScalarMappable(norm=norm, cmap=DFF_CMAP), ax=axes,
                 shrink=0.7, label="DFF speed (m/s)", extend="max")
    return fig


def render_combined(left, right):
    """Render synchronized LOS and DFF maps with independent velocity scales."""
    probe_fig = plt.figure()
    try:
        probe_ax = probe_fig.add_subplot(projection=projection)
        probe_ax.set_extent(extent, crs=geo)
        xmin, xmax = probe_ax.get_xlim()
        ymin, ymax = probe_ax.get_ylim()
        panel_aspect = (ymax - ymin) / (xmax - xmin)
    finally:
        plt.close(probe_fig)
    fig = plt.figure(figsize=(13.8, 6 * panel_aspect + 2), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.04, w_pad=0.04, hspace=0.02, wspace=0.02)
    try:
        los_panel, dff_panel = fig.subfigures(1, 2)
        render_los(left, right, fig=los_panel)
        render_dff(left, right, fig=dff_panel)
        fig.suptitle(f"{left:%Y-%m-%d %H:%M} – {right:%Y-%m-%d %H:%M} UT | {COORDINATES}\n"
                     f"{', '.join(r.upper() for r in radars)}", fontsize=15)
    except Exception:
        plt.close(fig)
        raise
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", default=EVENT_DATE)
    parser.add_argument("--start", default=START_TIME, help="Start time in UT (HH:MM)")
    parser.add_argument("--end", default=END_TIME, help="End time in UT (HH:MM); earlier means next day")
    parser.add_argument("--step", default=TIME_DELTA, help="Frame window, e.g. 2min or 10min")
    parser.add_argument("--radars", nargs="+", default=RADARS)
    parser.add_argument("--kind", choices=["los", "dff", "combined", "both"], default="both",
                        help="Movie to generate; both generates LOS, DFF, and combined movies")
    parser.add_argument("--coordinates", choices=["geographic", "aacgmv2", "aacgmv2_mlt"], default=COORDINATES)
    parser.add_argument("--scatter", choices=["ground", "ionospheric", "both"], default=SCATTER)
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    parser.add_argument("--fps", type=float, default=2)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/radar_movies")
    parser.add_argument("--no-coastlines", action="store_true")
    return parser.parse_args()


def main():
    global EVENT_DATE, START_TIME, END_TIME, TIME_DELTA, RADARS, COORDINATES, SCATTER, DATA_ROOT, COASTLINES
    args = parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0 or args.dpi <= 0:
        raise ValueError("FPS and DPI must be positive and finite.")
    if args.format == "mp4" and not writers.is_available("ffmpeg"):
        raise RuntimeError("MP4 output needs ffmpeg on PATH. Install ffmpeg or use --format gif.")
    if cmc is None:
        warnings.warn("cmcrameri unavailable: using RdBu_r/viridis. Install cmcrameri for notebook colors.")
    EVENT_DATE, START_TIME, END_TIME = args.date, args.start, args.end
    TIME_DELTA, RADARS = args.step, args.radars
    COORDINATES, SCATTER, DATA_ROOT = args.coordinates, args.scatter, args.data_root
    COASTLINES = not args.no_coastlines
    configure()
    load_observations()
    # Projection and coordinate conversion helpers use the configured coordinate mode.
    setup_projection()
    if args.kind in ("dff", "combined", "both"):
        fit_dff()
    output = args.output_root / start.strftime("%Y%m%d") / COORDINATES
    output.mkdir(parents=True, exist_ok=True)
    for kind in (["los", "dff", "combined"] if args.kind == "both" else [args.kind]):
        suffix = f"_{SCATTER}" if kind in ("los", "combined") else ""
        path = output / f"{kind}{suffix}_{start:%Y%m%dT%H%M%S}_{end:%Y%m%dT%H%M%S}.{args.format}"
        writer = FFMpegWriter(fps=args.fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]) if args.format == "mp4" else PillowWriter(fps=args.fps)
        renderer = {"los": render_los, "dff": render_dff, "combined": render_combined}[kind]
        first = None
        try:
            for index, (left, right) in enumerate(windows):
                fig = renderer(left, right)
                try:
                    if first is None:
                        first = fig
                        # Every frame uses the same map extent and fixed canvas size.
                        writer.setup(fig, str(path), dpi=args.dpi)
                    fig.set_size_inches(first.get_size_inches(), forward=True)
                    writer.fig = fig
                    writer.grab_frame()
                finally:
                    plt.close(fig)
                print(f"{kind.upper()} frame {index + 1}/{len(windows)}: {left} UT", flush=True)
        finally:
            if first is not None:
                writer.finish()
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
