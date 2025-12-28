# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 16:02:28 2025
@author: santaro



"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
import matplotlib.transforms as mtransforms
import matplotlib.font_manager as fm
from matplotlib.textpath import TextPath
from matplotlib.animation import ArtistAnimation
from matplotlib.animation import FuncAnimation

from enum import Enum, auto
from pathlib import Path
import json

import config

# plt.rcParams["font.family"] = "DejaVu Sans"
# plt.rcParams["font.family"] = "monospace"
# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.family"] = "serif"

class PlotSizeCode(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()
    SQUARE_ILLUST = auto()
    SQUARE_FIG = auto()
    RECTANGLE_FIG = auto()
    LANDSCAPE_FIG_21 = auto()
    LANDSCAPE_FIG_31 = auto()
    TRAJECTORY = auto()
    TRAJECTORY_2 = auto()
    TRAJECTORY_WITH_TIMESERIES = auto()
    TRAJECTORY_WITH_TIMESERIES2 = auto()

class MyPlotter:
    @staticmethod
    def cnvt_val2list(N, *args):
        vals = []
        for val in args:
            if not isinstance(val, list):
                val = [val] * N
            # elif len(val) == 1:
                # val = [val[0]] * N
            elif len(val) < N:
                for i in range(N-len(val)):
                    val.append(None)
            vals.append(val)
        return vals

    @staticmethod
    def make_formatter(decimal_places, hide0=False):
        def formatter(x, pos=None):
            if x == 0:
                if hide0:
                    return ""
                else:
                    return '0'
            else:
                return f"{x:.{decimal_places}f}"
        return ticker.FuncFormatter(formatter)

    @staticmethod
    def get_axsfromfig(fig):
        axs_load = []
        for _ax in fig.axes:
            l_list = []
            for _l in _ax.get_lines():
                _l_dict = {
                    'xdata': _l.get_xdata(),
                    'ydata': _l.get_ydata(),
                    'color': _l.get_color(),
                    'linestyle': _l.get_linestyle(),
                    'marker': _l.get_marker(),
                    'label': _l.get_label()
                    }
                l_list.append(_l_dict)
            coll_list = []
            for _coll in _ax.collections:
                xydata = _coll.get_offsets().T
                _coll_dict ={
                    'xdata': xydata[0],
                    'ydata': xydata[1],
                    'size': _coll.get_sizes(),
                    'facecolors': _coll.get_facecolors(),
                    'edgecolors': _coll.get_edgecolors()
                    }
                coll_list.append(_coll_dict)
            _ax_dict = {
                'ax': _ax,
                'xlim': _ax.get_xlim(),
                'ylim': _ax.get_ylim(),
                'xticks': _ax.get_xticks(),
                'yticks': _ax.get_yticks(),
                'lines': l_list,
                'collections': coll_list
                }
            axs_load.append(_ax_dict)
        return axs_load

    @staticmethod
    def measure_text_size_px(fig, text, fontsize, fontfamily):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        t = fig.text(0, 0, text, fontsize=fontsize, fontfamily=fontfamily, transform=fig.transFigure)
        bbox = t.get_window_extent(renderer=renderer)
        width_px, height_px = bbox.width, bbox.height
        t.remove()
        return width_px, height_px

    @staticmethod
    def calc_text_offset_prcisely(fig, ax, text, fontsize, fontfamily, xem=1, yem=0):
        w_px, h_px = MyPlotter.measure_text_size_px(fig, text, fontsize, fontfamily)
        dx_inch = w_px / fig.dpi * xem
        dy_inch = h_px / fig.dpi * yem
        return mtransforms.ScaledTranslation(dx_inch, dy_inch, fig.dpi_scale_trans)

    @staticmethod
    def measure_text_size_pt(text, fontsize, fontfamily):
        tp = TextPath((0, 0), text, size=fontsize, prop=fm.FontProperties(family=fontfamily))
        bbox = tp.get_extents()
        width_pt, height_pt = bbox.width, bbox.height
        return width_pt, height_pt

    @staticmethod
    def calc_text_offset(fig, ax, text, fontsize, fontfamily, xem=1, yem=0):
        string_width_pt, char_height_pt = MyPlotter.measure_text_size_pt(text, fontsize, fontfamily)
        char_width_pt = string_width_pt / len(text)
        dx_inch = char_width_pt * xem / 72
        dy_inch = char_height_pt * yem / 72
        return mtransforms.ScaledTranslation(dx_inch, dy_inch, fig.dpi_scale_trans)

    @staticmethod
    def offsetpx2axAxes(fig, ax, text, fontsize, fontfamily, xem=1, yem=1):
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        ax_w_inch, ax_h_inch = bbox.width, bbox.height
        ax_w_px, ax_h_px = ax_w_inch * fig.dpi, ax_h_inch * fig.dpi
        offset_px = MyPlotter.calc_text_offset(fig=fig, ax=ax, text=text, fontsize=fontsize, fontfamily=fontfamily, xem=xem, yem=yem).transform((0, 0))
        offset_Axes = offset_px / np.array([ax_w_px, ax_h_px])
        return offset_Axes

    @staticmethod
    def offset_em(fig, fontsize, xem=1, yem=0):
        xpt = fontsize * xem
        ypt = fontsize * yem
        dx_inch = xpt / 72
        dy_inch = ypt / 72
        return mtransforms.ScaledTranslation(dx_inch, dy_inch, fig.dpi_scale_trans)

    def __init__(self, sizecode, json_path=config.ROOT/"config"/"plot_settings.json"):
        self.json_path = json_path
        with open(json_path, "r") as f:
            self.fig_settings = json.load(f)
        self.sizecode = sizecode
        self.load_settings()

    def _apply_settings(self, settings, suffix):
        keys_std = ["num_axes", "labelsize", "ticklength", "tickwidth", "major_grid_lw", "minor_grid_lw"]
        keys_tuple = ["figsize", "marge_size", "notell_pos", "notelr_pos",
                        "gs_shape", "gs_width_ratios", "gs_height_ratios", "gs_whspace",
                        "gsub_shape", "gsub_height_ratios", "gsub_width_ratios", "gsub_whspace"
                        ]
        for k in keys_std:
            if not k in settings:
                settings[k] = None
                # raise KeyError(k)
            setattr(self, f"{k}{suffix}", settings[k])
        for k in keys_tuple:
            if not k in settings:
                # raise KeyError(k)
                setattr(self, f"{k}{suffix}", None)
            else:
                setattr(self, f"{k}{suffix}", tuple(settings[k]))

    def _get_settings(self, *args, **kwargs):
        isslide = kwargs["slide"]
        attrs = []
        for attr_name in args:
            suffix = "_slide" if isslide else ""
            attr = getattr(self, f"{attr_name}{suffix}")
            attrs.append(attr)
        return attrs

    def load_settings(self):
        try:
            settings = self.fig_settings[self.sizecode.value]["analysis"]
            settings_slide = self.fig_settings[self.sizecode.value]["slide"]
        except KeyError as e:
            raise ValueError(f"{self.sizecode.value} cannot be found in self.fig_settings.") from e
        self._apply_settings(settings, suffix="")
        self._apply_settings(settings_slide, suffix="_slide")

    def myfig(self, sharex=False, sharey=False, title=None, notell=None, notelr=None, xlabel=None, ylabel=None, xrange=None, yrange=None, xtick=None, ytick=None, xtick_0center=True, ytick_0center=True, xsigf=2, ysigf=2, grid=False, slide=False):
        figsize, num_axes, marge_size = self._get_settings("figsize", "num_axes", "marge_size", slide=slide)
        gs_shape, gs_height_ratios, gs_width_ratios, gs_whspace = self._get_settings("gs_shape", "gs_height_ratios", "gs_width_ratios", "gs_whspace", slide=slide)
        gsub_shape, gsub_height_ratios, gsub_width_ratios, gsub_whspace, labelsize, ticklength, tickwidth, major_grid_lw, minor_grid_lw, marge_size, notell_pos, notelr_pos = self._get_settings("gsub_shape", "gsub_height_ratios", "gsub_width_ratios", "gsub_whspace", "labelsize", "ticklength", "tickwidth", "major_grid_lw", "minor_grid_lw", "marge_size", "notell_pos", "notelr_pos", slide=slide)
        labelsize, ticklength, tickwidth, major_grid_lw, minor_grid_lw = self._get_settings("labelsize", "ticklength", "tickwidth", "major_grid_lw", "minor_grid_lw", slide=slide)
        notell_pos, notelr_pos = self._get_settings("notell_pos", "notelr_pos", slide=slide)

        sharex, sharey, xlabel, ylabel, xrange, yrange, xtick, ytick, grid, xsigf, ysigf , xtick_0center, ytick_0center = MyPlotter.cnvt_val2list(num_axes, sharex, sharey, xlabel, ylabel, xrange, yrange, xtick, ytick, grid, xsigf, ysigf, xtick_0center, ytick_0center)

        fig = plt.figure(figsize=figsize)
        axs = np.empty(num_axes, dtype=object)

        self.gs_master = GridSpec(gs_shape[0], gs_shape[1], figure=fig,
                    left=marge_size[0], right=1-marge_size[1], bottom=marge_size[2], top=1-marge_size[3],
                    wspace=gs_whspace[0], hspace=gs_whspace[1], width_ratios=gs_width_ratios, height_ratios=gs_height_ratios)

        self.gs = np.array([self.gs_master[r, c] for r in range(self.gs_master.nrows) for c in range(self.gs_master.ncols)]).flatten()

        i = 0
        self.gsub_masters = []
        if gsub_shape is None: gsub_shape = [[1, 1] for i in range(len(self.gs))]
        if gsub_height_ratios is None: gsub_height_ratios = [[1] for i in range(len(self.gs))]
        if gsub_width_ratios is None: gsub_width_ratios = [[1] for i in range(len(self.gs))]
        if gsub_whspace is None: gsub_whspace = [[1, 1] for i in range(len(self.gs))]
        for j in range(len(self.gs)):
            gsub_master = self.gs[j].subgridspec(gsub_shape[j][0], gsub_shape[j][1], wspace=gsub_whspace[j][0], hspace=gsub_whspace[j][1], width_ratios=gsub_width_ratios[j], height_ratios=gsub_height_ratios[j])
            self.gsub_masters.append(gsub_master)
            gsub = np.array([gsub_master[r, c] for r in range(gsub_master.nrows) for c in range(gsub_master.ncols)]).flatten()
            for k in range(len(gsub)):
                ss = gsub[k]
                sx, sy = sharex[i], sharey[i]
                if (sx is False or sx == i) and (sy is False or sy == i):
                    axs[i] = fig.add_subplot(ss)
                elif (sx is not False and sx != i) and sy is False:
                    axs[i] = fig.add_subplot(ss, sharex=axs[sx])
                elif sx is False and (sy is not False and sy != i):
                    axs[i] = fig.add_subplot(ss, sharey=axs[sy])
                else:
                    axs[i] = fig.add_subplot(ss, sharex=axs[sx], sharey=axs[sy])
                i += 1

        fig.suptitle(title, fontsize=10)
        alpha = 0.05 if slide else 1
        color = 'r' if slide else 'k'
        fig.text(notell_pos[0], notell_pos[1], notell, ha='left', va='center', fontsize=8, alpha=alpha, color=color)
        fig.text(notelr_pos[0], notelr_pos[1], notelr, ha='left', va='center', fontsize=8, alpha=alpha, color=color)
        for i in range(num_axes):
            axs[i].set_xlabel(xlabel[i], fontsize=labelsize)
            axs[i].set_ylabel(ylabel[i], fontsize=labelsize)
            axs[i].set_xlim(xrange[i])
            axs[i].set_ylim(yrange[i])
            if xtick[i] is not None and xrange[i] is not None:
                if xtick_0center[i]:
                    axs[i].set_xticks(np.hstack([np.arange(0, xrange[i][0]-xtick[i]/10, -xtick[i])[::-1], np.arange(0, xrange[i][1]+xtick[i]/10, xtick[i])]))
                else:
                    axs[i].set_xticks(np.arange(xrange[i][0], xrange[i][1]+xtick[i]/10, xtick[i]))
            elif xrange[i] is None:
                axs[i].autoscale(enable=True, axis='x')
            if ytick[i] is not None and yrange[i] is not None:
                if ytick_0center[i]:
                    axs[i].set_yticks(np.hstack([np.arange(0, yrange[i][0]-ytick[i]/10, -ytick[i])[::-1], np.arange(0, yrange[i][1]+ytick[i]/10, ytick[i])]))
                else:
                    axs[i].set_yticks(np.arange(yrange[i][0], yrange[i][1]+ytick[i]/10, ytick[i]))
            elif yrange[i] == None:
                axs[i].autoscale(enable=True, axis='y')
            hide0 = True if axs[i].get_xlim()[0] == 0 and axs[i].get_ylim()[0] == 0 else False
            axs[i].xaxis.set_major_formatter(MyPlotter.make_formatter(xsigf[i], hide0=hide0))
            axs[i].yaxis.set_major_formatter(MyPlotter.make_formatter(ysigf[i], hide0=hide0))
            if hide0:
                offset = MyPlotter.calc_text_offset(fig, axs[i], text='0', fontsize=labelsize, fontfamily="DejaVu Sans" , xem=1.4, yem=1.4)
                axs[i].text(0, 0, "0", transform=axs[i].transAxes-offset, fontsize=labelsize, clip_on=False, zorder=10)
            axs[i].tick_params(axis='both', which="both", direction="in", length=ticklength, width=tickwidth, labelsize=labelsize)
            if grid[i]:
                axs[i].minorticks_on()
                axs[i].grid(which='major', lw=major_grid_lw)
                axs[i].xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
                axs[i].yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
                axs[i].grid(which='minor', lw=minor_grid_lw)
        return fig, axs

    def slidefig(self, fig, axs, lw=2, ms=20):
        # self.gs.update(left=self.marge_size_slide[0], right=1-self.marge_size_slide[1], top=1-self.marge_size_slide[2], bottom=self.marge_size_slide[3],
        #                wspace=self.marge_size_slide[4], hspace=self.marge_size_slide[5])
        self.gs_master.update(left=self.marge_size_slide[0], right=1-self.marge_size_slide[1], bottom=self.marge_size[2], top=1-self.marge_size_slide[3],
                                wspace=self.gs_whspace_slide[0], hspace=self.gs_whspace_slide[1])

        # for i, gsub_master in enumerate(self.gsub_masters):
        #     gsub_master.set_spacing(wspace=self.gsub_whspace_slide[i][0], hspace=self.gsub_hspace_slide[i][1])

        self.gsub_masters = []
        i = 0
        for j in range(len(self.gs_shape)):
            gsub_master = self.gs[j].subgridspec(self.gsub_shape[j][0], self.gsub_shape[j][1], wspace=self.gsub_whspace[j][0], hspace=self.gsub_whspace[j][1], width_ratios=self.gsub_width_ratios[j], height_ratios=self.gsub_height_ratios[j])
            self.gsub_masters.append(gsub_master)
            gsub = np.array([gsub_master[r, c] for r in range(gsub_master.nrows) for c in range(gsub_master.ncols)]).flatten()
            for k in range(len(gsub)):
                ss = gsub[k]
                axs[i] = fig.add_subplot(ss)
                i += 1

        if hasattr(fig, "_suptitle") and fig._suptitle:
            fig._suptitle.remove()
            fig._suptitle = None
        for _text in fig.texts[:]: # [:] means shallow copy
            _text.remove()
        for i in range(len(axs)):
            hide0 = True if axs[i].get_xlim()[0] == 0 and axs[i].get_ylim()[0] == 0 else False
            if hide0:
                for _t in axs[i].texts:
                    _t.remove()
                offset = MyPlotter.calc_text_offset(fig, axs[i], text='0', fontsize=self.labelsize_slide, fontfamily="DejaVu Sans" , xem=1.4, yem=1)
                axs[i].text(0, 0, "0", transform=axs[i].transAxes-offset, fontsize=self.labelsize_slide, clip_on=False, zorder=10)
            axs[i].set_xlabel(axs[i].get_xlabel(), fontsize=self.labelsize_slide)
            axs[i].set_ylabel(axs[i].get_ylabel(), fontsize=self.labelsize_slide)
            axs[i].tick_params(axis='both', length=self.ticklength_slide, width=self.tickwidth_slide, labelsize=self.labelsize_slide)
            lines = axs[i].get_lines()
            colls = axs[i].collections
            for _l in lines:
                _l.set_linewidth(lw)
            for _c in colls:
                _c.set_sizes([ms])
        return fig, axs

    @staticmethod
    def draw_center_line(fig, ax, length=0.05, lw=0.4, color='k'):
        ax.axvline(x=0, ymin=0.5-length, ymax=0.5+length, lw=lw, color='k')
        ax.axhline(y=0, xmin=0.5-length, xmax=0.5+length, lw=lw, color='k')
        return fig, ax

    @staticmethod
    def fill_ring(fig, ax, r_inner, r_outer, num_points=99, facecolor='r', edgecolor='none', alpha=0.2):
        theta = np.linspace(0, 2*np.pi, num_points)
        c_inner = np.array([r_inner * np.cos(theta), r_inner * np.sin(theta)]).T
        c_outer = np.array([r_outer * np.cos(theta), r_outer * np.sin(theta)]).T
        vertices = np.concatenate([c_outer, c_inner[::-1]])
        codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(vertices) - 1)
        ring_path = mpath.Path(vertices, codes)
        patch = patches.PathPatch(ring_path, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
        ax.add_patch(patch)
        return fig, ax

    @staticmethod
    def add_auxiliary_cicles(fig, ax, radii, colors='k', lws=1, alphas=1):
        colors, lws, alphas = MyPlotter.cnvt_val2list(len(radii), colors, lws, alphas)
        for _r, _c, _lw, _alpha in zip(radii, colors, lws, alphas):
            _circle = patches.Circle((0, 0), _r, edgecolor=_c, facecolor='none', zorder=5, ls='-', lw=_lw, alpha=_alpha)
            ax.add_patch(_circle)
        return fig, ax


class PlotterForCage:
    def __init__(self, name=""):
        self.name = name

    # def plot_trajectory(self, y, z, trjcolor='k', trjlw=0.4, auxiliary_circles_radii=None, auxiliary_circles_colors=None, auxiliary_circles_lws=2, auxiliary_circles_alphas=0.4, title='trajectory of cage center', xlabel='y [mm]', ylabel='z [mm]', xrange=(-0.5, 0.5), yrange=(-0.5, 0.5), xtick=0.1, ytick=0.1, xtick_0center=True, ytick_0center=True, xsigf=2, ysigf=2, notell='', notelr='', grid=False):
    #     plotter = MyPlotter(sizecode=PlotSizeCode.SQUARE_FIG)
    #     fig, axs = plotter.myfig(title=title, xlabel=xlabel, ylabel=ylabel, xrange=xrange, yrange=yrange, xtick=xtick, ytick=ytick, xtick_0center=xtick_0center, ytick_0center=ytick_0center, xsigf=xsigf, ysigf=ysigf, notell=notell, notelr=notelr, grid=grid)
    #     axs[0].plot(y, z, c=trjcolor, lw=trjlw)
    #     axs[0].set_aspect(1)
    #     fig, axs[0] = MyPlotter.draw_center_line(fig, axs[0])
    #     if auxiliary_circles_radii is not None:
    #         fig, axs[0] = MyPlotter.add_auxiliary_cicles(fig, axs[0], radii=auxiliary_circles_radii, colors=auxiliary_circles_colors, lws=auxiliary_circles_lws, alphas=auxiliary_circles_alphas)
    #     return fig, axs

    def plot_trajectory(self, ys, zs, trjcolors=['r', 'b', 'g', 'm', 'c', 'y']*100, trjlws=[0.4]*100, auxiliary_circles_radii=None, auxiliary_circles_colors=None, auxiliary_circles_lws=2, auxiliary_circles_alphas=0.4, title='', xlabel='y [mm]', ylabel='z [mm]', xrange=(-0.5, 0.5), yrange=(-0.5, 0.5), xtick=0.1, ytick=0.1, xtick_0center=True, ytick_0center=True, xsigf=2, ysigf=2, notell='', notelr='', grid=False):
        plotter = MyPlotter(sizecode=PlotSizeCode.SQUARE_FIG)
        fig, axs = plotter.myfig(title=title, xlabel=xlabel, ylabel=ylabel, xrange=xrange, yrange=yrange, xtick=xtick, ytick=ytick, xtick_0center=xtick_0center, ytick_0center=ytick_0center, xsigf=xsigf, ysigf=ysigf, notell=notell, notelr=notelr, grid=grid)
        axs[0].set_aspect(1)
        for i in range(len(ys)):
            axs[0].plot(ys[i], zs[i], c=trjcolors[i], lw=trjlws[i])
        fig, axs[0] = MyPlotter.draw_center_line(fig, axs[0])
        if auxiliary_circles_radii is not None:
            fig, axs[0] = MyPlotter.add_auxiliary_cicles(fig, axs[0], radii=auxiliary_circles_radii, colors=auxiliary_circles_colors, lws=auxiliary_circles_lws, alphas=auxiliary_circles_alphas)
        return fig, axs

    def plot_vstime2(self, ts, fts, colors=['k']*2, lws=[0.4]*2, alphas=[1]*2, xlabel='time [sec]', ylabel=None, xrange=None, yrange=None, xsigf=2, ysigf=2, xtick=None, ytick=None, xtick_0center=True, ytick_0center=True, title='', notell='', notelr='', plottype=['plot']*2):
        plotter = MyPlotter(sizecode=PlotSizeCode.LANDSCAPE_FIG_21)
        fig, axs = plotter.myfig(sharex=[0, 0], sharey=False, title=title, xlabel=xlabel, ylabel=ylabel, xrange=xrange, yrange=yrange, xtick=xtick, ytick=ytick, xtick_0center=xtick_0center, ytick_0center=ytick_0center, grid=0, xsigf=xsigf, ysigf=ysigf, notell=notell, notelr=notelr)
        for i in range(len(axs)):
            if plottype[i] == 'plot':
                axs[i].plot(ts[i], fts[i], lw=lws[i], c=colors[i], alpha=alphas[i])
            elif plottype[i] == 'scatter':
                axs[i].scatter(ts[i], fts[i], s=lws[i], c=colors[i], alpha=alphas[i])
        return fig, axs

    def plot_vstime3(self, ts, fts, colors=['k']*3, lws=[0.4]*3, alphas=[1]*3, xlabel=['', '', 'time [sec]'], ylabel=None, xrange=None, yrange=None, xsigf=2, ysigf=2, xtick=None, ytick=None, xtick_0center=True, ytick_0center=True, title='', notell='', notelr='', plottype=['plot']*3):
        plotter = MyPlotter(sizecode=PlotSizeCode.LANDSCAPE_FIG_31)
        fig, axs = plotter.myfig(sharex=[0, 0, 0], sharey=False, title=title, xlabel=xlabel, ylabel=ylabel, xrange=xrange, yrange=yrange, xtick=xtick, ytick=ytick, xtick_0center=xtick_0center, ytick_0center=ytick_0center, grid=0, xsigf=xsigf, ysigf=ysigf, notell=notell, notelr=notelr)
        for i in range(len(axs)):
            if plottype[i] == 'plot':
                axs[i].plot(ts[i], fts[i], lw=lws[i], c=colors[i], alpha=alphas[i])
            elif plottype[i] == 'scatter':
                axs[i].scatter(ts[i], fts[i], s=lws[i], c=colors[i], alpha=alphas[i])
        return fig, axs

    def plot_probability(self, probability_map, bins=100, cmap='viridis', vmin=None, vmax=None, auxiliary_circles_radii=None, auxiliary_circles_colors=None, auxiliary_circles_lws=2, auxiliary_circles_alphas=1, title='', xlabel='y [mm]', ylabel='z [mm]', xrange=(-0.5, 0.5), yrange=(-0.5, 0.5), xtick=0.1, ytick=0.1, xtick_0center=True, ytick_0center=True, xsigf=2, ysigf=2, notell='', notelr=''):
        plotter = MyPlotter(sizecode=PlotSizeCode.SQUARE_FIG)
        fig, axs = plotter.myfig(title=title, xlabel=xlabel, ylabel=ylabel, xrange=xrange, yrange=yrange, xtick=xtick, ytick=ytick, xtick_0center=xtick_0center, ytick_0center=ytick_0center, grid=0, xsigf=xsigf, ysigf=ysigf, notell=notell, notelr=notelr)
        axs[0].set_aspect(1)
        axs[0].imshow(probability_map.T, origin='lower', cmap=cmap, extent=([xrange[0], xrange[1], yrange[0], yrange[1]]), vmin=vmin, vmax=vmax)
        fig, axs[0] = MyPlotter.draw_center_line(fig, axs[0])
        if auxiliary_circles_radii is not None:
            fig, axs[0] = MyPlotter.add_auxiliary_cicles(fig, axs[0], radii=auxiliary_circles_radii, colors=auxiliary_circles_colors, lws=auxiliary_circles_lws, alphas=auxiliary_circles_alphas)
        return fig, axs[0]

    def animate_trajectory(self, ys, zs, trjcolors=['k', 'r', 'b'] + ['k']*97, trjdispmax=[100, 100] + [1]*98, trjlws=[0.4]*100, trjlalphas=[0.4]*100, trjmarkersizes=[8, 20, 20] + [8]*97, trjmarkeralphas=[1]*100, auxiliary_circles_radii=None, auxiliary_circles_colors=None, auxiliary_circles_lws=2, auxiliary_circles_alphas=0.4, title='', xlabel='y [mm]', ylabel='z [mm]', xrange=(-0.5, 0.5), yrange=(-0.5, 0.5), xtick=0.1, ytick=0.1, xtick_0center=True, ytick_0center=True, xsigf=2, ysigf=2, notell='', notelr='', grid=False):
        plotter = MyPlotter(sizecode=PlotSizeCode.TRAJECTORY)
        fig, axs = plotter.myfig(title=title, xlabel=xlabel, ylabel=ylabel, xrange=xrange, yrange=yrange, xtick=xtick, ytick=ytick, xtick_0center=xtick_0center, ytick_0center=ytick_0center, xsigf=xsigf, ysigf=ysigf, notell=notell, notelr=notelr, grid=grid)
        axs[0].set_aspect(1)
        fig, axs[0] = MyPlotter.draw_center_line(fig, axs[0])
        axs[1].axis("off")
        if auxiliary_circles_radii is not None:
            fig, axs[0] = MyPlotter.add_auxiliary_cicles(fig, axs[0], radii=auxiliary_circles_radii, colors=auxiliary_circles_colors, lws=auxiliary_circles_lws, alphas=auxiliary_circles_alphas)
        data_list = []
        for i in range(len(ys)):
            _data = {"id": 0, "data": [ys[i], zs[i]], "color": trjcolors[i], "markersize": trjmarkersizes[i], "malpha": trjmarkeralphas[i], "lw": trjlws[i], "lalpha": trjlalphas[i], "disp_max": trjdispmax[i]}
            data_list.append(_data)
        vct_list = None
        vline_list = None
        hline_list = None
        offset = MyPlotter.offsetpx2axAxes(fig, axs[1], text=f"frame: 10000", fontsize=10, fontfamily="monospace", xem=1.2, yem=1.2)
        note_list = [
            {"id": 1, "prefix": "frame: ", "data": np.arange(len(t)), "sigf": 0, "disp_width": 5, "suffix": "", "position": (0, 1-offset[1]), "fontsize": 10, "fontfamily": "monospace"},
            {"id": 1, "prefix": "time:  ", "data": np.round(t, 3), "sigf": 3, "disp_width": 5, "suffix": "", "position": (0, 1-offset[1]*2), "fontsize": 10, "fontfamily": "monospace"},
            ]
        animator = MyAnimator(fig, axs, data_list=data_list, vct_list=vct_list, vline_list=vline_list, hline_list=hline_list, note_list=note_list)
        ani = animator.make_func_ani(skip=2, interval=10)
        return fig, axs, ani

    def animate_trajectory2(self, ys, zs, ys1, zs1, gravity_angle, time, trjcolors=['k', 'r', 'b'] + ['k']*97, trjdispmax=[100, 100] + [1]*98, trjlws=[0.4]*100, trjlalphas=[0.4]*100, trjmarkersizes=[8, 20, 20] + [8]*97, trjmarkeralphas=[1]*100, auxiliary_circles_radii=None, auxiliary_circles_colors=None, auxiliary_circles_lws=2, auxiliary_circles_alphas=0.4, title='', xlabel='y [mm]', ylabel='z [mm]', xrange=(-0.5, 0.5), yrange=(-0.5, 0.5), xtick=0.1, ytick=0.1, xtick_0center=True, ytick_0center=True, xsigf=2, ysigf=2, notell='', notelr='', grid=False):
        plotter = MyPlotter(sizecode=PlotSizeCode.TRAJECTORY_2)
        fig, axs = plotter.myfig(title=title, xlabel=xlabel, ylabel=ylabel, xrange=xrange, yrange=yrange, xtick=xtick, ytick=ytick, xtick_0center=xtick_0center, ytick_0center=ytick_0center, xsigf=xsigf, ysigf=ysigf, notell=notell, notelr=notelr, grid=grid)
        for i in range(2):
            fig, axs[i] = MyPlotter.draw_center_line(fig, axs[i])
            axs[i].set_aspect(1)
            if auxiliary_circles_radii is not None:
                fig, axs[i] = MyPlotter.add_auxiliary_cicles(fig, axs[i], radii=auxiliary_circles_radii, colors=auxiliary_circles_colors, lws=auxiliary_circles_lws, alphas=auxiliary_circles_alphas)
        axs[2].axis("off")
        data_list = []
        for i in range(len(ys)):
            _data0 = {"id": 0, "data": [ys[i], zs[i]], "color": trjcolors[i], "markersize": trjmarkersizes[i], "malpha": trjmarkeralphas[i], "lw": trjlws[i], "lalpha": trjlalphas[i], "disp_max": trjdispmax[i]}
            _data1 = {"id": 1, "data": [ys1[i], zs1[i]], "color": trjcolors[i], "markersize": trjmarkersizes[i], "malpha": trjmarkeralphas[i], "lw": trjlws[i], "lalpha": trjlalphas[i], "disp_max": trjdispmax[i]}
            data_list.append(_data0)
            data_list.append(_data1)
        vct_list = [
            {"mode": "force", "id": 0, "data": [np.zeros_like(ys[0]), np.zeros_like(zs[0]), np.zeros_like(ys[0]), -np.ones_like(zs[0])], "width": 0.004, "scale": 10, "color": 'k', "alpha": 0.8},
            {"mode": "force", "id": 1, "data": [np.zeros_like(ys[0]), np.zeros_like(zs[0]), np.cos(gravity_angle), np.sin(gravity_angle)], "width": 0.004, "scale": 10, "color": 'k', "alpha": 0.8},
        ]
        vline_list = None
        hline_list = None
        offset = MyPlotter.offsetpx2axAxes(fig, axs[2], text=f"frame: 10000", fontsize=10, fontfamily="monospace", xem=1.2, yem=1.2)
        note_list = [
            {"id": 2, "prefix": "frame: ", "data": np.arange(len(time)), "sigf": 0, "disp_width": 5, "suffix": "", "position": (0, 1-offset[1]), "fontsize": 10, "fontfamily": "monospace"},
            {"id": 2, "prefix": "time:  ", "data": np.round(time, 3), "sigf": 3, "disp_width": 5, "suffix": "", "position": (0, 1-offset[1]*2), "fontsize": 10, "fontfamily": "monospace"},
            ]
        animator = MyAnimator(fig, axs, data_list=data_list, vct_list=vct_list, vline_list=vline_list, hline_list=hline_list, note_list=note_list)
        ani = animator.make_func_ani(skip=2, interval=10)
        return fig, axs, ani

    def animate_trajectory3(self, ys, zs, ys1, zs1, time, ts, fts, gravity_angle, trjcolors=['k', 'r', 'b'] + ['k']*97, trjdispmax=[100, 100] + [1]*98, trjlws=[0.4]*100, trjlalphas=[0.4]*100, trjmarkersizes=[8, 20, 20] + [8]*97, trjmarkeralphas=[1]*100, auxiliary_circles_radii=None, auxiliary_circles_colors=None, auxiliary_circles_lws=2, auxiliary_circles_alphas=0.4, ftcolors=['k']*100, ftlws=[0.4]*100, ftalphas=[1]*100, title='', xlabel=["y [mm]", "y [mm]", "time [sec]", None], ylabel=["y [mm]", "y [mm]", "time [sec]", None], xrange=[(-0.5, 0.5), (-0.5, 0.5), (0, 1), None], yrange=[(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), None], xtick=[0.1, 0.1, 0.2, None], ytick=[0.1, 0.1, 0.1, None], xtick_0center=True, ytick_0center=True, xsigf=[2, 2, 1, None], ysigf=[2, 2, 2, None], notell='', notelr='', grid=False):
        plotter = MyPlotter(sizecode=PlotSizeCode.TRAJECTORY_WITH_TIMESERIES)
        fig, axs = plotter.myfig(title=title, xlabel=xlabel, ylabel=ylabel, xrange=xrange, yrange=yrange, xtick=xtick, ytick=ytick, xtick_0center=xtick_0center, ytick_0center=ytick_0center, xsigf=xsigf, ysigf=ysigf, notell=notell, notelr=notelr, grid=grid)
        for i in range(2):
            fig, axs[i] = MyPlotter.draw_center_line(fig, axs[i])
            axs[i].set_aspect(1)
            if auxiliary_circles_radii is not None:
                fig, axs[i] = MyPlotter.add_auxiliary_cicles(fig, axs[i], radii=auxiliary_circles_radii, colors=auxiliary_circles_colors, lws=auxiliary_circles_lws, alphas=auxiliary_circles_alphas)
        for i in range(len(ts)):
            axs[2].plot(ts[i], fts[i], color=ftcolors[i], lw=ftlws[i], alpha=ftalphas[i])
        axs[2].axhline(y=0, xmin=-10000, xmax=10000, lw=0.8, alpha=1, color='k')
        axs[3].axis("off")
        data_list = []
        for i in range(len(ys)):
            _data0 = {"id": 0, "data": [ys[i], zs[i]], "color": trjcolors[i], "markersize": trjmarkersizes[i], "malpha": trjmarkeralphas[i], "lw": trjlws[i], "lalpha": trjlalphas[i], "disp_max": trjdispmax[i]}
            _data1 = {"id": 1, "data": [ys1[i], zs1[i]], "color": trjcolors[i], "markersize": trjmarkersizes[i], "malpha": trjmarkeralphas[i], "lw": trjlws[i], "lalpha": trjlalphas[i], "disp_max": trjdispmax[i]}
            data_list.append(_data0)
            data_list.append(_data1)
        vct_list = [
            {"mode": "force", "id": 0, "data": [np.zeros_like(ys[0]), np.zeros_like(zs[0]), np.zeros_like(ys[0]), -np.ones_like(zs[0])], "width": 0.004, "scale": 10, "color": 'k', "alpha": 0.8},
            {"mode": "force", "id": 1, "data": [np.zeros_like(ys[0]), np.zeros_like(zs[0]), np.cos(gravity_angle), np.sin(gravity_angle)], "width": 0.004, "scale": 10, "color": 'k', "alpha": 0.8},
        ]
        vline_list = [
            {"id": 2, "data": ts[0], "color": 'k', "lw": 0.4, "alpha": 1, "ymin": -10000, "ymax": 10000},
        ]
        hline_list = [
            {"id": 2, "data": fts[0], "color": 'k', "lw": 0.4, "alpha": 1, "xmin": -10000, "xmax": 10000},
        ]
        offset = MyPlotter.offsetpx2axAxes(fig, axs[3], text=f"frame: 10000", fontsize=10, fontfamily="monospace", xem=1.2, yem=1.2)
        note_list = [
            {"id": 3, "prefix": "frame: ", "data": np.arange(len(time)), "sigf": 0, "disp_width": 5, "suffix": "", "position": (0, 1-offset[1]), "fontsize": 10, "fontfamily": "monospace"},
            {"id": 3, "prefix": "time:  ", "data": np.round(time, 3), "sigf": 3, "disp_width": 5, "suffix": " [sec]", "position": (0, 1-offset[1]*2), "fontsize": 10, "fontfamily": "monospace"},
            ]
        animator = MyAnimator(fig, axs, data_list=data_list, vct_list=vct_list, vline_list=vline_list, hline_list=hline_list, note_list=note_list)
        ani = animator.make_func_ani(skip=2, interval=10)
        return fig, axs, ani

class MyAnimator:
    @staticmethod
    def make_point_and_line(fig, ax, x, y, frame, disp_max, color, markersize, malpha, lw, lalpha):
        if disp_max == 1:
            l, = ax.plot([], [])
            p = ax.scatter(x[frame], y[frame], c=color, s=markersize)
        elif disp_max > 1:
            startf = frame - disp_max if frame > disp_max else 0
            l, = ax.plot(x[startf: frame+1], y[startf: frame+1], c=color, lw=lw, alpha=lalpha)
            p = ax.scatter(x[frame], y[frame], c=color, s=markersize, alpha=malpha)
        return fig, ax, p, l

    @staticmethod
    def make_auxiliary_line_endpoint(x, y, scale1=100, scale2=100):
        angle = np.arctan2(y, x)
        x1 = -scale1 * x
        x2 = scale2 * x
        y1 = np.tan(angle) * x1
        y2 = np.tan(angle) * x2
        xs = np.vstack([x1, x2]).T
        ys = np.vstack([y1, y2]).T
        endpoints = [xs, ys]
        return endpoints

    def __init__(self, fig, axs, data_list, vct_list=None, vline_list=None, hline_list=None, fline_list=None, note_list=None):
        self.fig = fig
        self.axs = axs
        self.data_list_original = data_list # point and line for timeseries data like trajectory
        self.data_list = None
        self.vct_list_original = vct_list # vector
        self.vct_list = None
        self.vline_list_original = vline_list # vertical line with ax.axvline
        self.vline_list = None
        self.hline_list_original = hline_list # horizontal line with ax.axhline
        self.hline_list = None
        self.fline_list_original = fline_list # free line with ax.plot
        self.fline_list = None
        self.note_list_original = note_list # note with ax.text
        self.note_list = None
        self.num_frames_original = len(self.data_list_original[0]["data"][0])

    def skip_frames(self, frange, skip):
        if frange:
            s, e = frange
        else:
            s, e = 0, self.num_frames_original
        self.data_list = []
        for d in self.data_list_original:
            x, y = d["data"]
            new_entry = {**d,"data": [x[s:e:skip], y[s:e:skip]]}
            self.data_list.append(new_entry)
        if self.vct_list_original:
            self.vct_list = []
            for d in self.vct_list_original:
                if d["mode"] == "force":
                    x, y, u, v = d["data"]
                    new_entry = {**d,"data": [x[s:e:skip], y[s:e:skip], u[s:e:skip], v[s:e:skip]]}
                elif d["mode"] == "field":
                    x, y, u, v, c = d["data"]
                    new_entry = {**d,"data": [x, y, u[s:e:skip], v[s:e:skip], c[s:e:skip]]}
                self.vct_list.append(new_entry)
        if self.vline_list_original:
            self.vline_list = []
            for d in self.vline_list_original:
                data = d["data"]
                new_entry = {**d,"data": data[s:e:skip]}
                self.vline_list.append(new_entry)
        if self.hline_list_original:
            self.hline_list = []
            for d in self.hline_list_original:
                data = d["data"]
                new_entry = {**d,"data": data[s:e:skip]}
                self.hline_list.append(new_entry)
        if self.fline_list_original:
            self.fline_list = []
            for d in self.fline_list_original:
                data = d["data"]
                new_entry = {**d,"data": [data[0][s:e:skip], data[1][s:e:skip]]}
                self.fline_list.append(new_entry)
        if self.note_list_original:
            self.note_list = []
            for d in self.note_list_original:
                data = d["data"]
                new_entry = {**d,"data": data[s:e:skip]}
                self.note_list.append(new_entry)

    def make_artist_ani(self, disp_max=100, frange=None, skip=1, interval=100):
        self.skip_frames(frange, skip)
        num_frames = len(self.data_list[0]["data"][0])
        artists = []
        for f in range(num_frames):
            container = []
            for entry in self.data_list:
                i_ax = entry["id"]
                data = entry["data"]
                color = entry["color"]
                markersize = entry["markersize"]
                malpha = entry["malpha"]
                lw = entry["lw"]
                lalpha = entry["lalpha"]
                disp_max = entry["disp_max"]
                self.fig, self.axs[i_ax], p, l = MyAnimator.make_point_and_line(fig=self.fig, ax=self.axs[i_ax], x=data[0], y=data[1], frame=f, disp_max=disp_max, color=color, markersize=markersize, malpha=malpha, lw=lw, lalpha=lalpha)
                container.append(p)
                container.append(l)
            if self.vct_list:
                for entry in self.vct_list:
                    i_ax = entry["id"]
                    width = entry["width"]
                    scale = entry["scale"]
                    if entry["mode"] == "force":
                        color = entry["color"]
                        alpha = entry["alpha"]
                        x, y, u, v = entry["data"]
                        v = self.axs[i_ax].quiver(x[f], y[f], u[f], v[f], angles="xy", scale_units="xy", scale=scale, width=width, color=color, alpha=alpha)
                    elif entry["mode"] == "field":
                        cmap = entry["cmap"]
                        clim = entry["clim"]
                        x, y, u, v, c = entry["data"]
                        v = self.axs[i_ax].quiver(x, y, u[f], v[f], c[f], angles="xy", scale_units="xy", scale=scale, width=width, cmap=cmap, clim=clim)
                    container.append(v)
            if self.vline_list:
                for entry in self.vline_list:
                    i_ax = entry["id"]
                    x = entry["data"][f]
                    color = entry["color"]
                    lw = entry["lw"]
                    alpha = entry["alpha"]
                    ymin, ymax = entry["ymin"], entry["ymax"]
                    vl = self.axs[i_ax].axvline(x=x, ymin=ymin, ymax=ymax, c=color, lw=lw, alpha=alpha)
                    container.append(vl)
            if self.hline_list:
                for entry in self.hline_list:
                    i_ax = entry["id"]
                    y = entry["data"][f]
                    color = entry["color"]
                    lw = entry["lw"]
                    alpha = entry["alpha"]
                    xmin, xmax = entry["xmin"], entry["xmax"]
                    hl = self.axs[i_ax].axhline(y=y, xmin=xmin, xmax=xmax, c=color, lw=lw, alpha=alpha)
                    container.append(hl)
            if self.fline_list:
                for entry in self.fline_list:
                    i_ax = entry["id"]
                    x, y = entry["data"][f]
                    color = entry["color"]
                    lw = entry["lw"]
                    alpha = entry["alpha"]
                    fl, = self.axs[i_ax].plot(x, y, c=color, lw=lw, alpha=alpha)
                    container.append(fl)
            if self.note_list:
                for entry in self.note_list:
                    i_ax = entry["id"]
                    prefix = entry["prefix"]
                    data = entry["data"][f]
                    suffix = entry["suffix"]
                    note = f"{prefix}{data:.3f}{suffix}"
                    pos = entry["position"]
                    fontsize = entry["fontsize"]
                    fontfamily = entry["fontfamily"]
                    t = self.axs[i_ax].text(pos[0], pos[1], note, transform=self.axs[i_ax].transAxes, fontsize=fontsize, fontfamily=fontfamily)
                    container.append(t)
            artists.append(container)
        ani = ArtistAnimation(self.fig, artists, interval=interval)
        return ani

    def init_func_ani(self):
        self.scats = []
        self.lines = []
        self.vectors = []
        self.vlines = []
        self.hlines = []
        self.flines = []
        self.notes = []
        for entry in self.data_list:
            i_ax = entry["id"]
            color = entry["color"]
            markersize = entry["markersize"]
            malpha = entry["malpha"]
            lw = entry["lw"]
            lalpha = entry["lalpha"]
            p = self.axs[i_ax].scatter([], [], c=color, s=markersize, alpha=malpha)
            l, = self.axs[i_ax].plot([], [], c=color, lw=lw, alpha=lalpha)
            self.scats.append(p)
            self.lines.append(l)
        if self.vct_list:
            for entry in self.vct_list:
                i_ax = entry["id"]
                width = entry["width"]
                scale = entry["scale"]
                if entry["mode"] == "force":
                    color = entry["color"]
                    alpha = entry["alpha"]
                    x, y, u, v = entry["data"]
                    v = self.axs[i_ax].quiver(x[0], y[0], np.zeros_like(u[0]), np.zeros_like(v[0]), angles="xy", scale_units="xy", scale=scale, width=width, color=color, alpha=alpha)
                elif entry["mode"] == "field":
                    cmap = entry["cmap"]
                    clim = entry["clim"]
                    x, y, u, v, c = entry["data"]
                    v = self.axs[i_ax].quiver(x, y, np.zeros_like(u[0]), np.zeros_like(v[0]), c[0], angles="xy", scale_units="xy", scale=scale, width=width, cmap=cmap, clim=clim)
                self.vectors.append(v)
        if self.vline_list:
            for entry in self.vline_list:
                i_ax = entry["id"]
                color = entry["color"]
                lw = entry["lw"]
                alpha = entry["alpha"]
                ymin, ymax = entry["ymin"], entry["ymax"]
                vl = self.axs[i_ax].axvline(x=0, ymin=ymin, ymax=ymax, c=color, lw=lw, alpha=alpha)
                self.vlines.append(vl)
        if self.hline_list:
            for entry in self.hline_list:
                i_ax = entry["id"]
                color = entry["color"]
                lw = entry["lw"]
                alpha = entry["alpha"]
                xmin, xmax = entry["xmin"], entry["xmax"]
                hl = self.axs[i_ax].axhline(y=0, xmin=xmin, xmax=xmax, c=color, lw=lw, alpha=alpha)
                self.hlines.append(hl)
        if self.fline_list:
            for entry in self.fline_list:
                i_ax = entry["id"]
                color = entry["color"]
                lw = entry["lw"]
                alpha = entry["alpha"]
                fl, = self.axs[i_ax].plot([], [], c=color, lw=lw, alpha=alpha)
                self.flines.append(fl)
        if self.note_list:
            for entry in self.note_list:
                i_ax = entry["id"]
                pos = entry["position"]
                fontsize = entry["fontsize"]
                fontfamily = entry["fontfamily"]
                t = self.axs[i_ax].text(pos[0], pos[1], "", transform=self.axs[i_ax].transAxes, fontsize=fontsize, fontfamily=fontfamily)
                self.notes.append(t)
        return self.scats + self.lines + self.vectors + self.vlines + self.hlines + self.flines + self.notes

    def update(self, frame):
        for i, entry in enumerate(self.data_list):
            disp_max = entry["disp_max"]
            startf = frame - disp_max if frame > disp_max else 0
            data = entry["data"]
            self.scats[i].set_offsets([data[0][frame], data[1][frame]])
            if disp_max > 1:
                self.lines[i].set_data(data[0][startf:frame+1], data[1][startf:frame+1])
        if self.vct_list:
            for i, entry in enumerate(self.vct_list):
                if entry["mode"] == "force":
                    x, y, u, v = entry["data"]
                    self.vectors[i].set_UVC(u[frame], v[frame])
                    self.vectors[i].set_offsets([x[frame], y[frame]])
                elif entry["mode"] == "field":
                    x, y, u, v, c = entry["data"]
                    self.vectors[i].set_UVC(u[frame], v[frame], c[frame])
        if self.vline_list:
            for i, entry in enumerate(self.vline_list):
                x = entry["data"][frame]
                self.vlines[i].set_xdata([x])
        if self.hline_list:
            for i, entry in enumerate(self.hline_list):
                y = entry["data"][frame]
                self.hlines[i].set_ydata([y])
        if self.fline_list:
            for i, entry in enumerate(self.fline_list):
                x, y = entry["data"]
                self.flines[i].set_data(x[frame], y[frame])
        if self.note_list:
            for i, entry in enumerate(self.note_list):
                prefix = entry["prefix"]
                data = entry["data"][frame]
                sigf = entry["sigf"]
                width = entry["disp_width"]
                suffix = entry["suffix"]
                note = f"{prefix}{data:{width}.{sigf}f}{suffix}"
                self.notes[i].set_text(note)
        return self.scats + self.lines + self.vectors + self.vlines + self.hlines + self.flines + self.notes

    def make_func_ani(self, frange=None, skip=1, interval=100, blit=True):
        self.skip_frames(frange, skip)
        num_frames = len(self.data_list[0]["data"][0])
        ani = FuncAnimation(self.fig, self.update, fargs=(), frames=num_frames, init_func=self.init_func_ani, interval=interval, blit=blit)
        return ani

def check_fonts():
    for f in fm.findSystemFonts(fontpaths=None, fontext="ttf"):
        # if "Times" in f or "Roman" in f:
        print(f)
    # print(fm.findfont("Times New Roman"))


if __name__ == '__main__':
    print('---- test ----')
    scrdir = Path(__file__).resolve().parent

    #### sample data
    t = np.linspace(0, 10, 10000)
    x = 0.2 * np.cos(2*np.pi*t)
    y = 0.2 * np.sin(2*np.pi*t)
    z = np.sin(2*np.pi*t*4)**2
    ft = 4*np.sin(2*np.pi*t) + 0.2*np.sin(2*np.pi*20*t)


    # plotter = MyPlotter(sizecode=PlotSizeCode.SQUARE_ILLUST)
    # plotter = MyPlotter(sizecode=PlotSizeCode.SQUARE_FIG)
    # plotter = MyPlotter(sizecode=PlotSizeCode.RECTANGLE_FIG)
    # plotter = MyPlotter(sizecode=PlotSizeCode.LANDSCAPE_FIG_31)
    # plotter = MyPlotter(sizecode=PlotSizeCode.TRAJECTORY_2)
    plotter = MyPlotter(sizecode=PlotSizeCode.TRAJECTORY_WITH_TIMESERIES)


    fig, axs = plotter.myfig(xlabel="x [mm]", ylabel="y [mm]", notell="tc02_sc02_2000rpm", notelr="0.222sec\ntest")

    axs[0].set_aspect(1)
    axs[1].set_aspect(1)

    # axs[3].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    # axs[3].set_xlabel("")
    # axs[3].set_ylabel("")

    axs[0].plot(x, y)
    # fig, axs = plotter.slidefig(fig, axs)

    # import mycage
    # cage = mycage.SimpleCage()
    # cage.time_series_data()

    # import mycoord
    # transformer = mycoord.CoordTransformer2D(theta=2*np.pi*t)
    # res = transformer.transform_coord(np.vstack([x, y]).T, towhich="tolocal")
    # x_lcs, y_lcs = res.T

    # for field
    # xx = np.linspace(-10, 10, 21)
    # yy = np.linspace(-10, 10, 21)
    # X, Y = np.meshgrid(xx, yy)
    # U = np.cos((X[np.newaxis, :, :])*0.4 + t[:, np.newaxis, np.newaxis] * 6)
    # V = np.sin((Y[np.newaxis, :, :])*0.4 + t[:, np.newaxis, np.newaxis] * 6)
    # C = np.sqrt(U**2+V**2)

    # plotter = MyPlotter(sizecode=PlotSizeCode.TRAJECTORY_WITH_TIMESERIES)
    # fig, axs = plotter.myfig(notell='testll', notelr='testlr', sharex=[0, False, False, False], sharey=False, xrange=[(-4, 4), (-10, 10), None, None], yrange=[(-4, 4), (-10, 10), None, None], xtick=1, ytick=1, xsigf=1, ysigf=1)
    # axs[0].set_aspect(1)
    # axs[1].set_aspect(1)
    # axs[3].axis("off")
    # fig, axs = plotter.slidefig(fig, axs)

    # plotter = PlotterForCage()
    # fig, axs = plotter.plot_trajectory(x, y)
    # fig, axs = plotter.plot_vstime2([t, t, t], [x, y, z])
    # fig, axs, ani = plotter.animate_trajectory([x], [y])
    # fig, axs, ani = plotter.animate_trajectory2([x], [y], [x_lcs], [y_lcs], gravity_angle=-2*np.pi*t-np.pi/2, time=t)
    # fig, axs, ani = plotter.animate_trajectory3([cage.p_cage[:, 1]], [cage.p_cage[:, 2]], [cage.p_cage_lcs[:, 1]], [cage.p_cage_lcs[:, 2]], cage.t, [cage.t], [cage.p_cage[:, 2]], gravity_angle=-cage.omega_rot_avg*cage.t-np.pi/2)

    plt.show()

    for i in vars(plotter):
        print(i)


