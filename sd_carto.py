#!/usr/bin/env python

"""Copied from meso_tracers/plotting_utils/sd_carto.py.

Local adaptation: configurable AACGM feature conversion altitude.
"""

import datetime as dt

import aacgmv2
import cartopy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.projections import register_projection
from shapely.geometry import LineString, MultiLineString

class SDCarto(GeoAxes):
    name = "sdcarto"

    def __init__(self, *args, **kwargs):
        self.supported_coords = ["geo", "aacgmv2", "aacgmv2_mlt"]
        if "coords" in kwargs:
            self.coords = kwargs.pop("coords")
            if self.coords not in self.supported_coords:
                err_str = "Coordinates not supported, choose from : "
                for _n, _sc in enumerate(self.supported_coords):
                    if _n + 1 != len(self.supported_coords):
                        err_str += _sc + ", "
                    else:
                        err_str += _sc
                raise TypeError(err_str)
        else:
            self.coords = "geo"
        if "map_projection" in kwargs:
            self.map_projection = kwargs.pop("map_projection")
        else:
            self.map_projection = cartopy.crs.PlateCarree()
        self.rad = kwargs.pop("rad", "bks")
        if "plot_date" in kwargs:
            self.plot_date = kwargs.pop("plot_date")
        else:
            self.plot_date = None
            if self.coords in {"aacgmv2", "aacgmv2_mlt"}:
                raise TypeError("Need to provide a date using 'plot_date' keyword for aacgmv2 plotting")
        super().__init__(*args, projection=self.map_projection, **kwargs)
        return

    def overaly_coast_lakes(self, resolution="50m", color="black", **kwargs):
        """Overlay AACGM coastlines and lakes."""
        kwargs["edgecolor"] = color
        kwargs["facecolor"] = "none"
        self.add_feature(
            cartopy.feature.NaturalEarthFeature("physical", "coastline", resolution),
            **kwargs,
        )
        self.add_feature(
            cartopy.feature.NaturalEarthFeature("physical", "lakes", resolution),
            **kwargs,
        )
        return

    def coastlines(self, resolution="50m", color="black", **kwargs):
        kwargs["edgecolor"] = color
        kwargs["facecolor"] = "none"
        feature = cartopy.feature.NaturalEarthFeature("physical", "coastline", resolution)
        return self.add_feature(feature, **kwargs)

    def add_feature(self, feature, **kwargs):
        if "edgecolor" not in kwargs:
            kwargs["edgecolor"] = "black"
        if self.coords == "geo":
            return super().add_feature(feature, **kwargs)
        else:
            aacgm_geom = self.get_aacgm_geom(feature, out_height=getattr(self, "aacgm_height", 300.0))
            aacgm_feature = cartopy.feature.ShapelyFeature(aacgm_geom, cartopy.crs.PlateCarree())
            kwargs["facecolor"] = "none"
            return super().add_feature(aacgm_feature, **kwargs)

    def grid_on(
        self,
        tx=cartopy.crs.PlateCarree(),
        draw_labels=(True, True, True, True),
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    ):
        """Add lat/lon grid lines to the map."""
        import matplotlib.ticker as mticker
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

        gl = self.gridlines(
            crs=tx,
            draw_labels=True,
            linewidth=linewidth,
            color=color,
            alpha=alpha,
            linestyle=linestyle,
        )
        gl.top_labels = draw_labels[0]
        gl.bottom_labels = draw_labels[2]
        gl.right_labels = draw_labels[1]
        gl.left_labels = draw_labels[3]
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 15))
        gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, 15))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        return
    
    def add_dn_terminator(self, **kwargs):
        """Add the day/night terminator."""
        from cartopy.feature.nightshade import Nightshade

        if self.plot_date:
            ns_feature = Nightshade(self.plot_date, alpha=0.2)
            return super().add_feature(ns_feature, **kwargs)
        return

    def _iter_coordinate_sequences(self, geom):
        geom_type = geom.geom_type
        if geom_type == "LineString":
            yield list(geom.coords)
        elif geom_type == "LinearRing":
            yield list(geom.coords)
        elif geom_type == "MultiLineString":
            for part in geom.geoms:
                yield list(part.coords)
        elif geom_type == "Polygon":
            yield list(geom.exterior.coords)
            for interior in geom.interiors:
                yield list(interior.coords)
        elif geom_type == "MultiPolygon":
            for polygon in geom.geoms:
                yield list(polygon.exterior.coords)
                for interior in polygon.interiors:
                    yield list(interior.coords)
        elif hasattr(geom, "geoms"):
            for part in geom.geoms:
                yield from self._iter_coordinate_sequences(part)

    def get_aacgm_geom(self, feature, out_height=300.0):
        """Convert feature lines, splitting at invalid points and longitude seams."""
        import logging

        segments = []
        logger = aacgmv2.logger
        previous_level = logger.level
        # Global coastlines cross the equator, where AACGM is undefined.
        # Those vertices are expected gaps rather than plotting failures.
        logger.setLevel(logging.CRITICAL)
        try:
            for geometry in feature.geometries():
                for coordinates in self._iter_coordinate_sequences(geometry):
                    segment = []
                    for lon, lat, *_ in coordinates:
                        mlat, mlon, mlt = aacgmv2.get_aacgm_coord(
                            lat, lon, out_height, self.plot_date)
                        x = mlon if self.coords == "aacgmv2" else mlt * 15
                        x = (x + 180) % 360 - 180
                        valid = np.isfinite(mlat) and np.isfinite(x)
                        if not valid or (segment and abs(x - segment[-1][0]) > 10):
                            if len(segment) > 1:
                                segments.append(segment)
                            segment = []
                        if valid:
                            segment.append((x, mlat))
                    if len(segment) > 1:
                        segments.append(segment)
        finally:
            logger.setLevel(previous_level)
        return MultiLineString(segments)

    def mark_latitudes(self, lat_arr, lon_location=45, **kwargs):
        """
        Label latitude circles on non-rectangular map projections.
        """
        if isinstance(lat_arr, list):
            lat_arr = np.array(lat_arr)
        else:
            if not isinstance(lat_arr, np.ndarray):
                raise TypeError("lat_arr must either be a list or numpy array")
        lon_location_arr = np.full(lat_arr.shape, lon_location)
        proj_xyz = self.projection.transform_points(
            cartopy.crs.Geodetic(), lon_location_arr, lat_arr
        )
        out_extent_lats = False
        for _np, _pro in enumerate(proj_xyz[..., :2].tolist()):
            lat_lim = self.get_extent(crs=cartopy.crs.Geodetic())[2::]
            if (lat_arr[_np] >= min(lat_lim)) and (lat_arr[_np] <= max(lat_lim)):
                self.text(_pro[0], _pro[1], str(lat_arr[_np]), **kwargs)
            else:
                out_extent_lats = True
        if out_extent_lats:
            print("some lats were out of extent ignored them")

    def mark_longitudes(self, lon_arr=np.arange(-180, 180, 60), **kwargs):
        """
        Label longitudes on non-rectangular map projections.
        """
        if isinstance(lon_arr, list):
            lon_arr = np.array(lon_arr)
        else:
            if not isinstance(lon_arr, np.ndarray):
                raise TypeError("lon_arr must either be a list or numpy array")
        [x1, y1], [x2, y2] = self.viewLim.get_points()
        right_bound = LineString(([-x1, y1], [x2, y2]))
        top_bound = LineString(([x1, -y1], [x2, y2]))
        bottom_bound = LineString(([x1, y1], [x2, -y2]))
        left_bound = LineString(([x1, y1], [-x2, y2]))
        plot_outline = MultiLineString([right_bound, top_bound, bottom_bound, left_bound])
        plot_extent = self.get_extent(cartopy.crs.Geodetic())
        line_constructor = lambda t, n, b: np.vstack(
            (np.zeros(n) + t, np.linspace(b[2], b[3], n))
        ).T
        for t in lon_arr:
            xy = line_constructor(t, 30, plot_extent)
            proj_xyz = self.projection.transform_points(cartopy.crs.PlateCarree(), xy[:, 0], xy[:, 1])
            xyt = proj_xyz[..., :2]
            ls = LineString(xyt.tolist())
            locs = plot_outline.intersection(ls)
            if locs.is_empty:
                continue
            closest_bound = min(
                [
                    right_bound.distance(locs),
                    top_bound.distance(locs),
                    bottom_bound.distance(locs),
                    left_bound.distance(locs),
                ]
            )
            if closest_bound == right_bound.distance(locs):
                ha = "left"
                va = "top"
            elif closest_bound == top_bound.distance(locs):
                ha = "left"
                va = "bottom"
            elif closest_bound == bottom_bound.distance(locs):
                ha = "left"
                va = "top"
            else:
                ha = "right"
                va = "top"
            if self.coords == "aacgmv2_mlt":
                marker_text = str(int((t / 15.0) % 24))
            else:
                marker_text = str(t)
            self.text(locs.bounds[0], locs.bounds[1], marker_text, ha=ha, va=va, **kwargs)

register_projection(SDCarto)


class RangeTimeIntervalPlot(object):
    """
    Create plots for velocity, width, power, elevation angle, etc.
    """
    
    def __init__(self, nrang, unique_times, fig_title="", num_subplots=3):
        self.nrang = nrang
        self.unique_gates = np.linspace(1, nrang, nrang)
        self.unique_times = unique_times
        self.num_subplots = num_subplots
        self._num_subplots_created = 0
        self.fig = plt.figure(figsize=(8, 3*num_subplots), dpi=100) # Size for website
        plt.suptitle(fig_title, x=0.075, y=0.99, ha='left', fontweight='bold', fontsize=15)
        mpl.rcParams.update({'font.size': 10})
        return
    
    def addPlot(self, data_dict, beam, title, vel_max=200, vel_step=25, xlabel='Time UT'):
        # add new axis
        self.vel_ax = self._add_axis()
        # set up variables for plotter
        time = np.hstack(data_dict['time'])
        gate = np.hstack(data_dict['gate'])
        allbeam = np.hstack(data_dict['beam'])
        flags = np.hstack(data_dict['vel'])
        bounds = list(range(-vel_max, vel_max+1, vel_step))
        cmap = plt.cm.jet
        mask = allbeam == beam
        self._create_colormesh(self.vel_ax, time, gate, flags, mask, bounds, cmap, xlabel)
        self._tight_layout()    # need to do this before adding the colorbar, because it depends on the axis position
        self._add_colorbar(self.fig, self.vel_ax, bounds, cmap, label='Velocity [m/s]')
        self.vel_ax.set_title(title, loc='left', fontdict={'fontweight': 'bold'})
        return
    
    def _add_axis(self):
        self._num_subplots_created += 1
        ax = self.fig.add_subplot(self.num_subplots, 1, self._num_subplots_created)
        return ax
    
    def _add_colorbar(self, fig, ax, bounds, colormap, label=''):
        """
        Add a colorbar to the right of an axis.
        """
        import matplotlib as mpl
        pos = ax.get_position()
        cpos = [pos.x1 + 0.025, pos.y0 + 0.0125,
                0.015, pos.height * 0.9]                # this list defines (left, bottom, width, height
        cax = fig.add_axes(cpos)
        norm = mpl.colors.BoundaryNorm(bounds, colormap.N)
        cb2 = mpl.colorbar.ColorbarBase(cax, cmap=colormap,
                                        norm=norm,
                                        ticks=bounds,
                                        spacing='uniform',
                                        orientation='vertical')
        cb2.set_label(label)
