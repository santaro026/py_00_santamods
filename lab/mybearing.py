"""
Created on Wed Oct 29 16:54:32 2025
@author: santaro




"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import polars as pl

from matplotlib.animation import ArtistAnimation
import matplotlib
matplotlib.use('Qt5Agg')
# matplotlib.use('TkAgg')

import sys
from pathlib import Path
import time

if __name__ == '__main__':
    import mytools, myplotter, mycoord, config
else:
    import mytools, myplotter, mycoord

class Ring:
    def __init__(self, name=None, PCD=50, R=3.225, D_groove=50+3.225):
        self.name = name
        self.PCD = PCD
        self.R = R
        self.D_groove = D_groove

    @staticmethod
    def omega2p(num_frames, dt, omega_rot, omega_rev, r_rev):
        omega_rot = np.full(num_frames, omega_rot) if np.ndim(omega_rot) == 0 else omega_rot
        omega_rev = np.full(num_frames, omega_rev) if np.ndim(omega_rev) == 0 else omega_rev
        angle_rev = np.cumsum(np.hstack([0, omega_rev[1:]])) * dt
        y = r_rev * np.cos(angle_rev + np.pi/2)
        z = r_rev * np.sin(angle_rev + np.pi/2)
        x = np.zeros_like(y)
        Rx = np.cumsum(np.hstack([0, omega_rot[1:]])) * dt
        Ry = np.full(num_frames, 0)
        Rz = np.full(num_frames, 0)
        return np.vstack([x, y, z]).T, np.vstack([Rx, Ry, Rz]).T

    @staticmethod
    def p2omega(p, euler_angle):
        omega_rot = np.gradient(euler_angle[:, 0])
        angle_rev = np.cumsum(np.hstack([np.unwrap(np.arctan2(p[:, 2], p[:, 1]))]))
        omega_rev = np.gradient(angle_rev)
        return omega_rot, omega_rev

    def time_series_data(self, shooting_code='', fps=10000, duration=1, x=0, y=None, z=None, omega_rot=100*2*np.pi, omega_rev=100*2*np.pi, r_rev=0, Rx=None, Ry=0, Rz=0):
        self.shooting_code = shooting_code
        self.fps = fps
        self.duration = duration
        self.num_frames = self.fps * self.duration + 1
        self.t = np.linspace(0, duration, self.num_frames, endpoint=True)
        self.dt = self.duration / self.fps
        self.omega_rot = np.full(self.num_frames, omega_rot) if np.ndim(omega_rot) == 0 else omega_rot
        self.omega_rev = np.full(self.num_frames, omega_rev) if np.ndim(omega_rev) == 0 else omega_rev
        self.r_rev = np.full(self.num_frames, r_rev) if np.ndim(r_rev) == 0 else r_rev
        self.rpm_avg = np.nanmean(self.omega_rot) / (2*np.pi) * 60
        self.omega_rot_avg = np.nanmean(self.omega_rot)
        self.Rx = np.cumsum(np.hstack([0, self.omega_rot[1:]])) * self.dt if Rx is None else Rx
        self.Ry = np.full(self.num_frames, Ry) if np.ndim(Ry) == 0 else Ry
        self.Rz = np.full(self.num_frames, Rz) if np.ndim(Rz) == 0 else Rz
        self.euler_angles = np.vstack([self.Rx, self.Ry, self.Rz]).T
        self.euler_angles1 = np.vstack([self.omega_rot_avg * self.t, np.zeros(self.num_frames), np.zeros(self.num_frames)]).T
        self.euler_angles2 = np.vstack([self.Rx, np.zeros(self.num_frames), np.zeros(self.num_frames)]).T
        self.angle_rev = np.cumsum(np.hstack([0, self.omega_rev[1:]])) * self.dt
        self.x = np.full(self.t.shape, x) if np.ndim(x) == 0 else x
        self.y = self.r_rev * np.cos(self.angle_rev + np.pi/2) if y is None else y
        self.z = self.r_rev * np.sin(self.angle_rev + np.pi/2) if z is None else z
        self.p_ring = np.vstack([self.x, self.y, self.z]).T

class Ball:
    def __init__(self, name=None, PCD=50, Dw=5.953):
        self.name = name
        self.PCD = PCD
        self.Dw = Dw

    @staticmethod
    def omega2p(num_frames, dt, omega_rot, omega_rev, r_rev):
        omega_rot = np.full(num_frames, omega_rot) if np.ndim(omega_rot) == 0 else omega_rot
        omega_rev = np.full(num_frames, omega_rev) if np.ndim(omega_rev) == 0 else omega_rev
        angle_rev = np.cumsum(np.hstack([0, omega_rev[1:]])) * dt
        y = r_rev * np.cos(angle_rev + np.pi/2)
        z = r_rev * np.sin(angle_rev + np.pi/2)
        x = np.zeros_like(y)
        Rx = np.cumsum(np.hstack([0, omega_rot[1:]])) * dt
        Ry = np.full(num_frames, 0)
        Rz = np.full(num_frames, 0)
        return np.vstack([x, y, z]).T, np.vstack([Rx, Ry, Rz]).T

    @staticmethod
    def p2omega(p, euler_angle):
        omega_rot = np.gradient(euler_angle[:, 0])
        angle_rev = np.cumsum(np.hstack([np.unwrap(np.arctan2(p[:, 2], p[:, 1]))]))
        omega_rev = np.gradient(angle_rev)
        return omega_rot, omega_rev

    def time_series_data(self, shooting_code='', fps=10000, duration=1, x=0, y=None, z=None, omega_rot=100*2*np.pi, omega_rev=100*2*np.pi, Rx=None, Ry=0, Rz=0):
        self.shooting_code = shooting_code
        self.fps = fps
        self.duration = duration
        self.num_frames = self.fps * self.duration + 1
        self.t = np.linspace(0, duration, self.num_frames, endpoint=True)
        self.dt = self.duration / self.fps
        self.omega_rot = np.full(self.num_frames, omega_rot) if np.ndim(omega_rot) == 0 else omega_rot
        self.omega_rev = np.full(self.num_frames, omega_rev) if np.ndim(omega_rev) == 0 else omega_rev
        self.rpm_avg = np.nanmean(self.omega_rot) / (2*np.pi) * 60
        self.omega_rot_avg = np.nanmean(self.omega_rot)

        self.Rx = np.cumsum(np.hstack([0, self.omega_rot[1:]])) * self.dt if Rx is None else Rx
        self.Ry = np.full(self.num_frames, Ry) if np.ndim(Ry) == 0 else Ry
        self.Rz = np.full(self.num_frames, Rz) if np.ndim(Rz) == 0 else Rz

        self.euler_angles = np.vstack([self.Rx, self.Ry, self.Rz]).T
        self.angle_rev = np.cumsum(np.hstack([0, self.omega_rev[1:]])) * self.dt

        self.x = np.full(self.t.shape, x) if np.ndim(x) == 0 else x
        self.y = self.r_rev * np.cos(self.angle_rev + np.pi/2) if y is None else y
        self.z = self.r_rev * np.sin(self.angle_rev + np.pi/2) if z is None else z
        self.p_ball = np.vstack([self.x, self.y, self.z]).T

if __name__ == '__main__':
    print('---- test ----')








