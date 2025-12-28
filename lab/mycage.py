# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 10:22:23 2025
@author: santaro

useful functions for cage

Modification
- delete the legacy function to make cage sample data, because the more utilized class is implemented.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import polars as pl

from matplotlib.animation import ArtistAnimation
# import matplotlib
# matplotlib.use('Qt5Agg')
# matplotlib.use('TkAgg')

import config

import sys
from pathlib import Path
import time

if __name__ == '__main__':
    import mytools, myplotter, mycoord, config
else:
    import mytools, myplotter, mycoord

class SimpleCage:
    def __init__(self, name='', PCD=50, ID=48, OD=52, width=10, num_pockets=8, num_markers=8, num_mesh=100, Dp=6.25, Dw=5.953):
        self.name = name
        self.PCD = PCD
        self.ID = ID
        self.OD = OD
        self.width = width
        self.num_pockets = num_pockets
        self.num_markers = num_markers
        self.num_mesh = num_mesh
        self.num_nodes = num_mesh + 1
        self.Dp = Dp
        self.Dw = Dw
        self.dp = Dp - Dw
        self.pos_pockets = np.linspace(0, 2*np.pi, self.num_pockets, endpoint=False) + np.pi/2
        self.pos_markers = np.linspace(0, 2*np.pi, self.num_markers, endpoint=False) + np.pi/2
        self.pos_node = np.linspace(0, 2*np.pi, self.num_nodes, endpoint=True) + np.pi/2

    @staticmethod
    def omega2p(omega, r, dt, num_frames=None):
        omega = np.full(num_frames, omega) if np.ndim(omega) == 0 else omega
        angle_rev = np.cumsum(np.hstack([0, omega[1:]])) * dt
        y = r * np.cos(angle_rev + np.pi/2)
        z = r * np.sin(angle_rev + np.pi/2)
        x = np.zeros_like(y)
        return np.vstack([x, y, z]).T

    @staticmethod
    def p2omega(p, dt):
        angle = np.unwrap(np.arctan2(p[:, 2], p[:, 1]))
        omega = np.gradient(angle, dt)
        return omega

    @staticmethod
    def make_cage_points(num_frames, num_points, x_value, a, b, deform_angle, transformer):
        p_lcs = np.full((num_frames, num_points, 3), np.nan) # points of pockets on local coordinate system
        p_global = np.full((num_frames, num_points, 3), np.nan)
        pos_points = np.linspace(0, 2*np.pi, num_points, endpoint=False) + np.pi/2
        if deform_angle != 0: rot_euler_for_ellipse = np.vstack([deform_angle, np.zeros(num_frames), np.zeros(num_frames)]).T
        for i in range(num_points):
            _x = np.full(num_frames, x_value)
            _theta = np.full(num_frames, pos_points[i])
            _y = a * np.cos(_theta)
            _z = b * np.sin(_theta)
            p_lcs[:, i, :] = np.vstack([_x, _y, _z]).T
            if deform_angle != 0: p_lcs[:, i, :] = mycoord.CoordTransformer3D.rotate_euler(p_lcs[:, i, :], euler_angles=rot_euler_for_ellipse, rot_order="zyx")
            p_global[:, i, :] = transformer.transform_coord(p_lcs[:, i, :], towhich='toglobal')
        return p_global, p_lcs

    def time_series_data(self, fps=10000, p_cage=np.vstack([np.zeros(10001), 0.4*np.cos(np.linspace(0, 2*np.pi*40, 10001)), 0.4*np.sin(np.linspace(0, 2*np.pi*40, 10001))]).T, R_cage=np.vstack([np.linspace(0, 2*np.pi*40, 10001), np.zeros(10001), np.zeros(10001)]).T,  a=1, b=1, noise_type="normal", noise_max=1):
        self.fps = fps
        self.num_frames = p_cage.shape[0]
        self.duration = (self.num_frames - 1) / self.fps
        self.t = np.linspace(0, self.duration, self.num_frames, endpoint=True)
        self.dt = 1 / self.fps
        self.p_cage = p_cage
        self.R_cage = R_cage
        self.omega_rot = np.gradient(R_cage[:, 0], self.dt)
        self.angle_rev = np.unwrap(np.arctan2(self.p_cage[:, 2], self.p_cage[:, 1]))
        self.omega_rev = np.gradient(self.angle_rev, self.dt)
        self.r_rev = np.linalg.norm(self.p_cage[:, 1:], axis=1)

        self.omega_rot_avg = np.nanmean(self.omega_rot)
        self.rpm_avg = np.nanmean(self.omega_rot) / (2*np.pi) * 60

        self.euler_angles1 = np.vstack([self.omega_rot_avg * self.t, np.zeros(self.num_frames), np.zeros(self.num_frames)]).T # euler angle for rotation frame with constant velocity
        self.euler_angles2 = np.vstack([self.R_cage[:, 0], np.zeros(self.num_frames), np.zeros(self.num_frames)]).T # euler angle for rotation frame with instant velocity

        self.transformer_SI = mycoord.CoordTransformer3D(coordsys_name="cage_coordsys", local_origin=np.zeros((self.num_frames, 3)), euler_angles=self.R_cage, rot_order='zyx')
        self.p_cage_lcs = self.transformer_SI.transform_coord(self.p_cage, towhich="tolocal")

        self.transformer_CI= mycoord.CoordTransformer3D(coordsys_name="cage_coordsys", local_origin=self.p_cage, euler_angles=self.R_cage, rot_order='zyx')

        #### generate points on cage
        self.p_pockets, self.p_pockets_lcs = SimpleCage.make_cage_points(num_frames=self.num_frames, num_points=self.num_pockets, x_value=0, a=a, b=b, deform_angle=0, transformer=self.transformer_CI)
        self.p_markers, self.p_markers_lcs = SimpleCage.make_cage_points(num_frames=self.num_frames, num_points=self.num_markers, x_value=self.width/2, a=a, b=b, deform_angle=0, transformer=self.transformer_CI)
        self.p_nodes, self.p_nodes_lcs = SimpleCage.make_cage_points(num_frames=self.num_frames, num_points=self.num_nodes, x_value=0, a=a, b=b, deform_angle=0, transformer=self.transformer_CI)

        #### add noise to the marker
        rng = np.random.default_rng(seed=0)
        if noise_type == 'uniform':
            noise = rng.uniform(-1, 1, (self.num_frames, self.num_markers, 3)) * noise_max
        elif noise_type == 'normal':
            noise = rng.normal(0, 1/3, (self.num_frames, self.num_markers, 3)) * noise_max
        self.p_markers_noise_lcs = self.p_markers_lcs + noise
        self.p_markers_noise = np.full((self.num_frames, self.num_markers, 3), np.nan)
        for i in range(self.num_markers):
            self.p_markers_noise[:, i, :] = self.transformer_CI.transform_coord(self.p_markers_noise_lcs[:, i, :], towhich='toglobal')

    def time_series_data2(self, fps=10000, duration=1, omega_rot=40*np.pi, omega_rev=40*np.pi, r_rev=0.4, a=1, b=1, omega_deform=0, noise_type="normal", noise_max=1):
        num_frames = fps * duration + 1
        t = np.linspace(0, duration , num_frames)
        dt = 1 / fps
        x = np.zeros(num_frames)
        y = r_rev * np.cos(omega_rev*t)
        z = r_rev * np.sin(omega_rev*t)
        p_cage = np.vstack([x, y, z]).T
        Rx = np.cumsum(np.hstack([0, np.full(num_frames-1, omega_rot)])) * dt
        Ry = np.zeros(num_frames)
        Rz = np.zeros(num_frames)
        R_cage = np.vstack([Rx, Ry, Rz]).T
        self.time_series_data(fps=fps, p_cage=p_cage, R_cage=R_cage, a=a, b=b, noise_type=noise_type, noise_max=noise_max)


    def export_with_TEMAformat(self, outdir=None):
        #### export with visualization test format
        if outdir is None: outdir = Path(__file__).resolve().parent / 'data'
        if outfname == None: outfname = f'{self.name}_sc{self.shooting_code}_{self.rpm_avg}rpm_{self.fps}fps'
        markers = [self.t]
        for i in range(self.num_pockets):
            markers.append(self.p_markers[:, i, 1])
            markers.append(self.p_markers[:, i, 2])
        markers = np.array(markers).T
        markers_noise = [self.t]
        for i in range(self.num_pockets):
            markers_noise.append(self.p_markers_noise[:, i, 1])
            markers_noise.append(self.p_markers_noise[:, i, 2])
        markers_noise = np.array(markers_noise).T
        zero_markers = []
        _markers_name = []
        for i in range(self.num_markers+1):
            _markers_name.append(f'point#{i}')
        zero_markers.append(_markers_name)
        zero_markers.append(np.hstack([self.p_zero_markers[:, 1], 0]))
        zero_markers.append(np.hstack([self.p_zero_markers[:, 2], 0]))
        _bring = []
        for i in range(self.num_markers):
            _bring.append('-')
        _bring.append(1)
        zero_markers.append(_bring)
        zero_markers = np.array(zero_markers).T
        header_markers = ['time [sec]']
        for i in range(self.num_markers):
            for j in ['y [pixel]', 'z [pixel]']:
                header_markers.append(f'p{i}{j}')
        mytools.save_csv(markers, header=header_markers, outfname=f'{outfname}_ideal.csv')
        mytools.save_csv(markers_noise, header=header_markers, outfname=f'{outfname}_noise.csv')
        header_zero_markers = ['name', 'y [pixel]', 'z [pixel]', 'area [pixel^2]']
        mytools.save_csv(zero_markers, header=header_zero_markers, outfname=f'{self.name}_zero.csv')

    def export_time_series_data(self, outdir=None):
        #### export primary data
        if outdir is None: outdir = Path(__file__).resolve().parent / 'data'
        if outfname == None: outfname = f'{self.name}_sc{self.shooting_code}_{self.rpm_avg}rpm_{self.fps}fps'
        cage = np.vstack([np.arange(self.num_frames), # 0
                          self.t, # 1
                          self.x, # 2
                          self.y, # 3
                          self.z, # 4
                          self.Rx, # 5
                          self.Ry, # 6
                          self.Rz, # 7
                          ])
        header_cage = ['frame', 'time [sec]', 'x', 'y', 'z', 'Rx [rad]', 'Ry [rad]', 'Rz [rad]']
        mytools.save_csv(cage, header=header_cage, outfname=f'{outfname}_cage.csv')
        header_markers = ['frame', 'time [sec]']
        for i in range(self.num_markers):
            for j in list(['x', 'y']):
                header_markers.append(f'p{i}{j}')
        markers = np.vstack([np.arange(self.num_frames), self.t, self.p_markers]).T
        mytools.save_csv(markers, header=header_markers, outfname=f'{outfname}_markers.csv')
        markers_noise = np.vstack([np.arange(self.num_frames), self.t, self.p_markers_noise]).T
        mytools.save_csv(markers_noise, header=header_markers, outfname=f'{outfname}_markers_noise.csv')
        header_pockets = ['frame', 'time [sec]']
        for i in range(self.num_markers):
            for j in list(['x', 'y']):
                header_pockets.append(f'p{i}_{j}')
        pockets = np.vstack([np.arange(self.num_frames), self.t, self.p_pockets]).T
        mytools.save_csv(pockets, header=header_pockets, outfname=f'{outfname}_pockets.csv')

    def plot_trajectory(self, pltrange=[0, 10000]):
        y, z = self.p_cage[:, 1], self.p_cage[:, 2]
        fig, ax = self.plotter.plot_trajectory(y[pltrange[0]:pltrange[1]], z[pltrange[0]:pltrange[1]], xrange=(-1, 1), yrange=(-1, 1))
        return fig, ax

    def plot_markers(self, pltrange=[0, 10000], num='all'):
        ys, zs = [], []
        if num == 'all':
            num = self.num_markers
        for i in range(num):
            p_markers_rotframe2 = self.transformer.transform_coord(self.p_markers[:, i, :], euler_angles=self.euler_angles, towhich='tolocal')
            ys.append(p_markers_rotframe2[:, 1][pltrange[0]:pltrange[1]])
            zs.append(p_markers_rotframe2[:, 2][pltrange[0]:pltrange[1]])
        fig, ax = self.plotter.plot_trajectories(ys, zs, xrange=(-40, 40), yrange=(-40, 40), xtick=10, ytick=10)
        return fig, ax

    def append_all_object(self, container, *args):
        for a in args:
            if a is not None:
                container.append(a)
        return container

    def make_line_and_point_for_ani(self, fig, ax, x, y, f, disp_max, color='k', markersize=8):
        if disp_max == 1:
            l, = ax.plot([0, 0], [0, 0], alpha=0)
            p = ax.scatter(x[f], y[f], c=color, s=markersize)
        elif disp_max > 1:
            if f <= disp_max:
                l, = ax.plot(x[:f+1], y[:f+1], c=color, lw=1, alpha=0.2)
                p = ax.scatter(x[f], y[f], c=color, s=markersize)
            elif f > disp_max:
                start = f - disp_max
                end = f + 1
                l, = ax.plot(x[start:end], y[start:end], c=color, lw=1, alpha=0.2)
                p = ax.scatter(x[f], y[f], c=color, s=markersize)
        return fig, ax, p, l

    def make_cage_shape_for_ani(self, fig, ax, x, y, f, color='k', lw=4, alpha=0.4, markersize=False):
        l, = ax.plot(x[f], y[f], c=color, lw=lw, alpha=alpha)
        if markersize:
            p = ax.scatter(x[f], y[f], c=color, s=markersize) # display nodes
        else:
            p = None
        return fig, ax, p, l

    def transform_multiple_pointsets(self, pointsets, local_origin, euler_angles, towhich):
        for i in range(len(pointsets[0])):
            pointsets[:, i, :] = self.transformer.transform_coord(pointsets[:, i, :], local_origin=local_origin, euler_angles=euler_angles, towhich=towhich)
        return pointsets

    def make_current_frame_and_time(self, fig, ax, f, num_frames, fontsize=8, position=(0.9, 0.025, 0.014)):
        curf = f'{f} / {num_frames} [frame]'
        curt = f'{round(self.t[f], 4):#.04f} [sec]'
        txt1 = fig.text(position[0], position[1], curf, ha='left', va='center', fontsize=fontsize)
        txt2 = fig.text(position[0], position[1]+position[2], curt, ha='left', va='center', fontsize=fontsize)
        return fig, ax, txt1, txt2

    def make_ani_trajectory(self, num_frames=1000, interval=1000/30, skip=1, disp_max=None, xyrange=(-1, 1), xytick=0.2, xysigf=2, markersize=8, color='k', coordsys='global'):
        plotter = myplotter.CustomPlotter()
        fig, ax = plotter.myfig(sizecode='sizecode01', title='trajectory of cage center', xrange=xyrange, yrange=xyrange, xtick=xytick, ytick=xytick)
        ax.axvline(x=0, ymin=0.44, ymax=0.56, c='k', lw=0.4)
        ax.axhline(y=0, xmin=0.44, xmax=0.56, c='k', lw=0.4)
        num_frames = self.num_frames if num_frames is None else num_frames
        if disp_max is None: disp_max = num_frames
        if coordsys == 'global':
            cx, cy, cz = self.p_cage[::skip, :].T
        # rotating frame: S mean system, C mean cage, A mean average, I mean instantaneous
        elif coordsys == 'rotframeSA':
            cx, cy, cz = self.transformer.transform_coord(self.p_cage, local_origin=self.system_center, euler_angles=self.euler_angles1, towhich='tolocal')[::skip, :].T
        elif coordsys == 'rotframeSI':
            cx, cy, cz = self.transformer.transform_coord(self.p_cage, local_origin=self.system_center, euler_angles=self.euler_angles2, towhich='tolocal')[::skip, :].T
        elif coordsys == 'rotframeCA':
            cx, cy, cz = self.transformer.transform_coord(self.p_cage, euler_angles=self.euler_angles1, towhich='tolocal')[::skip, :].T
        elif coordsys == 'rotframeCI':
            cx, cy, cz = self.transformer.transform_coord(self.p_cage, euler_angles=self.euler_angles2, towhich='tolocal')[::skip, :].T
        artists = []
        for f in range(num_frames):
            fig, ax, p, l = self.make_line_and_point_for_ani(fig=fig, ax=ax, x=cy, y=cz, f=f, disp_max=disp_max, color=color, markersize=8)
            fig, ax, txt1, txt2 = self.make_current_frame_and_time(fig=fig, ax=ax, f=f, num_frames=num_frames)
            artists.append([p, l, txt1, txt2])
        ani = ArtistAnimation(fig, artists, interval=interval)
        return ani

    def make_ani_trajectories(self, num_frames=1000, interval=1000/30, skip=1, disp_max=None, center_scale=1, xyrange=(-30, 30), xytick=5, xysigf=0, markersize=8, color='r', coordsys='global', markers_type='ideal'):
        plotter = myplotter.CustomPlotter(myplotter.PlotSizeCode.SQUARE_FIG)
        fig, ax = plotter.myfig(title='trajectory of markers', xrange=xyrange, yrange=xyrange, xtick=xytick, ytick=xytick, xsigf=xysigf, ysigf=xysigf)
        ax = ax[0]
        ax.axvline(x=0, ymin=0.44, ymax=0.56, c='k', lw=0.4)
        ax.axhline(y=0, xmin=0.44, xmax=0.56, c='k', lw=0.4)
        num_frames = self.num_frames if num_frames is None else num_frames
        if disp_max is None: disp_max = num_frames
        colors = ['r', 'b'] + ['k']*100
        if markers_type == 'ideal':
            if coordsys == 'global':
                x, y, z = self.p_markers[::skip, :, :].transpose(2, 0, 1)
                cx, cy, cz = self.p_cage[::skip, :].T * center_scale
                x_node, y_node, z_node = self.p_nodes[::skip, :, :].transpose(2, 0, 1)
            # rotating frame: S mean system, C mean cage, A mean average, I mean instantaneous
            elif coordsys == 'rotframeSA':
                x, y, z = self.transform_multiple_pointsets(self.p_markers, local_origin=self.system_center, euler_angles=self.euler_angles1, towhich='tolocal')[::skip, :, :].transpose(2, 0, 1)
                cx, cy, cz = self.transformer.transform_coord(self.p_cage, local_origin=self.system_center, euler_angles=self.euler_angles1, towhich='tolocal')[::skip, :].T * center_scale
                x_node, y_node, z_node = self.transform_multiple_pointsets(self.p_nodes, local_origin=self.system_center, euler_angles=self.euler_angles1, towhich='tolocal')[::skip, :, :].transpose(2, 0, 1)
            elif coordsys == 'rotframeSI':
                x, y, z = self.transform_multiple_pointsets(self.p_markers, local_origin=self.system_center, euler_angles=self.euler_angles2, towhich='tolocal')[::skip, :, :].transpose(2, 0, 1)
                cx, cy, cz = self.transformer.transform_coord(self.p_cage, local_origin=self.system_center, euler_angles=self.euler_angles2, towhich='tolocal')[::skip, :].T * center_scale
                x_node, y_node, z_node = self.transform_multiple_pointsets(self.p_nodes, local_origin=self.system_center, euler_angles=self.euler_angles2, towhich='tolocal')[::skip, :, :].transpose(2, 0, 1)
            elif coordsys == 'rotframeCA':
                x, y, z = self.transform_multiple_pointsets(self.p_markers, local_origin=self.p_cage, euler_angles=self.euler_angles1, towhich='tolocal')[::skip, :, :].transpose(2, 0, 1)
                cx, cy, cz = self.transformer.transform_coord(self.p_cage, local_origin=self.p_cage, euler_angles=self.euler_angles1, towhich='tolocal')[::skip, :].T * center_scale
                x_node, y_node, z_node = self.transform_multiple_pointsets(self.p_nodes, local_origin=self.p_cage, euler_angles=self.euler_angles1, towhich='tolocal')[::skip, :, :].transpose(2, 0, 1)
            elif coordsys == 'rotframeCI':
                x, y, z = self.transform_multiple_pointsets(self.p_markers, local_origin=self.p_cage, euler_angles=self.euler_angles2, towhich='tolocal')[::skip, :, :].transpose(2, 0, 1)
                cx, cy, cz = self.transformer.transform_coord(self.p_cage, local_origin=self.p_cage, euler_angles=self.euler_angles2, towhich='tolocal')[::skip, :].T * center_scale
                x_node, y_node, z_node = self.transform_multiple_pointsets(self.p_nodes, local_origin=self.p_cage, euler_angles=self.euler_angles2, towhich='tolocal')[::skip, :, :].transpose(2, 0, 1)
        elif markers_type == 'noise':
            if coordsys == 'global':
                x, y, z = self.p_markers_noise[::skip, :, :].transpose(2, 0, 1)
                cx, cy, cz = self.p_cage[::skip, :].T * center_scale
                x_node, y_node, z_node = self.p_nodes[::skip, :, :].transpose(2, 0, 1)
            elif coordsys == 'rotframeSA':
                x, y, z = self.transform_multiple_pointsets(self.p_markers_noise, local_origin=self.system_center, euler_angles=self.euler_angles1, towhich='tolocal')[::skip, :, :].transpose(2, 0, 1)
                cx, cy, cz = self.transformer.transform_coord(self.p_cage, local_origin=self.system_center, euler_angles=self.euler_angles1, towhich='tolocal')[::skip, :].T * center_scale
                x_node, y_node, z_node = self.transform_multiple_pointsets(self.p_nodes, local_origin=self.system_center, euler_angles=self.euler_angles1, towhich='tolocal')[::skip, :, :].transpose(2, 0, 1)
            elif coordsys == 'rotframeSI':
                x, y, z = self.transform_multiple_pointsets(self.p_markers_noise, local_origin=self.system_center, euler_angles=self.euler_angles2, towhich='tolocal')[::skip, :, :].transpose(2, 0, 1)
                cx, cy, cz = self.transformer.transform_coord(self.p_cage, local_origin=self.system_center, euler_angles=self.euler_angles2, towhich='tolocal')[::skip, :].T * center_scale
                x_node, y_node, z_node = self.transform_multiple_pointsets(self.p_nodes, local_origin=self.system_center, euler_angles=self.euler_angles2, towhich='tolocal')[::skip, :, :].transpose(2, 0, 1)
            elif coordsys == 'rotframeCA':
                x, y, z = self.transform_multiple_pointsets(self.p_markers_noise, local_origin=self.p_cage, euler_angles=self.euler_angles1, towhich='tolocal')[::skip, :, :].transpose(2, 0, 1)
                cx, cy, cz = self.transformer.transform_coord(self.p_cage, local_origin=self.p_cage, euler_angles=self.euler_angles1, towhich='tolocal')[::skip, :].T * center_scale
                x_node, y_node, z_node = self.transform_multiple_pointsets(self.p_nodes, local_origin=self.p_cage, euler_angles=self.euler_angles1, towhich='tolocal')[::skip, :, :].transpose(2, 0, 1)
            elif coordsys == 'rotframeCI':
                x, y, z = self.transform_multiple_pointsets(self.p_markers_noise, local_origin=self.p_cage, euler_angles=self.euler_angles2, towhich='tolocal')[::skip, :, :].transpose(2, 0, 1)
                cx, cy, cz = self.transformer.transform_coord(self.p_cage, local_origin=self.p_cage, euler_angles=self.euler_angles2, towhich='tolocal')[::skip, :].T * center_scale
                x_node, y_node, z_node = self.transform_multiple_pointsets(self.p_nodes, local_origin=self.p_cage, euler_angles=self.euler_angles2, towhich='tolocal')[::skip, :, :].transpose(2, 0, 1)
        else:
            print('**** markers_type is invalid.')
        artists = []
        for f in range(num_frames):
            container = []
            for i in range(self.num_markers):
                fig, ax, p, l = self.make_line_and_point_for_ani(fig=fig, ax=ax, x=y[:, i], y=z[:, i], f=f, disp_max=disp_max, color=colors[i], markersize=8)
                container = self.append_all_object(container, p, l)

            fig, ax, p_center, l_center = self.make_line_and_point_for_ani(fig=fig, ax=ax, x=cy, y=cz, f=f, disp_max=disp_max, color='k', markersize=8)
            fig, ax, p_node, l_mesh = self.make_cage_shape_for_ani(fig=fig, ax=ax, x=y_node, y=z_node, f=f, color='g')
            fig, ax, txt1, txt2 = self.make_current_frame_and_time(fig=fig, ax=ax, f=f, num_frames=num_frames)
            container = self.append_all_object(container, p_center, l_center, p_node, l_mesh, txt1, txt2)
            artists.append(container)
        ani = ArtistAnimation(fig, artists, interval=interval)
        return ani

if __name__ == '__main__':
    print('---- test ----')

    # plotter = myplotter.PlotterForCage(config.ROOT/'config'/'plot_settings.json')
    # transformer = mycoord.CoordTransformer3D()

    st = time.perf_counter()

    import specification_loader
    spec_loader = specification_loader.SpecificationLoader()
    sample_code = specification_loader.SampleCode
    spec = spec_loader.specification_factory(sample_code.SIMPLE50)

    cage = SimpleCage(name=spec.name, PCD=spec.PCD, ID=spec.ID, OD=spec.OD, width=spec.width, num_pockets=spec.num_pockets, num_markers=spec.num_markers, num_mesh=100, Dp=spec.Dp, Dw=spec.Dw)

    import time_param_loader
    param_loader = time_param_loader.TimeParamLoader()
    motion_code = time_param_loader.MotionCode
    param = param_loader.param_factory(motion_code.ROT_REV)

    cage.time_series_data2(fps=param.fps, duration=param.duration, omega_rot=param.omega_rot, omega_rev=param.omega_rev, r_rev=param.r_rev, a=param.a, b=param.b, omega_deform=param.omega_deform, noise_type=param.noise_type, noise_max=param.noise_max)

    ani = cage.make_ani_trajectories(num_frames=1000, interval=1000/30, skip=1, disp_max=None, center_scale=1, xyrange=(-30, 30), xytick=5, xysigf=0, markersize=8, color='r', coordsys='global', markers_type='ideal')
    # sample_cage.plot_trajectory()
    # sample_cage.plot_markers(pltrange=(0, 10))


    import myplotter
    plotter = myplotter.PlotterForCage()
    # fig, axs, ani = plotter.animate_trajectory2([x], [y], [x_lcs], [y_lcs], gravity_angle=-2*np.pi*t-np.pi/2, time=t)
    # fig, axs, ani = plotter.animate_trajectory3([cage.p_cage[:, 1]], [cage.p_cage[:, 2]], [cage.p_cage_lcs[:, 1]], [cage.p_cage_lcs[:, 2]], cage.t, [cage.t], [cage.p_cage[:, 2]], gravity_angle=-cage.omega_rot_avg*cage.t-np.pi/2)

    et = time.perf_counter()
    # print(f'elapsed time: {round((et-st)/1000, 4)} [sec]')
    plt.show()




