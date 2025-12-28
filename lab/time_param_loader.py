"""
Created on Wed Oct 22 22:31:35 2025
@author: santaro



"""
import numpy as np

from dataclasses import dataclass
import enum
from typing import Union

time_params = {
    "ROT": {
        "duration": 1,
        "fps": 10000,
        "system_center": np.zeros(3),
        "omega_rot": 100,
        "omega_rev": 0,
        "r_rev": 0,
        "x": 0,
        "Ry": 0,
        "Rz": 0,
        "a": 1,
        "b": 1,
        "omega_deform": 0,
        "deformation_mode": "circle",
        "noise_type": "normal",
        "noise_max": 1
    },
    "ROT_REV": {
        "duration": 1,
        "fps": 10000,
        "system_center": np.zeros(3),
        "omega_rot": 100,
        "omega_rev": 100,
        "r_rev": 0.2,
        "x": 0,
        "Ry": 0,
        "Rz": 0,
        "a": 1,
        "b": 1,
        "omega_deform": 0,
        "deformation_mode": "circle",
        "noise_type": "normal",
        "noise_max": 1
    },
    "ROT_REV5": {
        "duration": 1,
        "fps": 10000,
        "system_center": np.zeros(3),
        "omega_rot": 100,
        "omega_rev": 500,
        "r_rev": 0.2,
        "x": 0,
        "Ry": 0,
        "rz": 0,
        "a": 1,
        "b": 1,
        "omega_deform": 0,
        "deformation_mode": "circle",
        "noise_type": "normal",
        "noise_max": 1
    },
    "ROT_REV10": {
        "duration": 1,
        "fps": 10000,
        "system_center": np.zeros(3),
        "omega_rot": 100,
        "omega_rev": 1000,
        "r_rev": 0.2,
        "x": 0,
        "Ry": 0,
        "Rz": 0,
        "a": 1,
        "b": 1,
        "omega_deform": 0,
        "deformation_mode": "circle",
        "noise_type": "normal",
        "noise_max": 1
    },
    "ROT_REV_ELLIPSE": {
        "duration": 1,
        "fps": 10000,
        "system_center": np.zeros(3),
        "omega_rot": 100,
        "omega_rev": 100,
        "r_rev": 0.2,
        "x": 0,
        "Ry": 0,
        "Rz": 0,
        "a": 1,
        "b": 1,
        "omega_deform": 100,
        "deformation_mode": "ellipse",
        "noise_type": "normal",
        "noise_max": 1
    },
    "ROT_REV5_ELLIPSE5": {
        "duration": 1,
        "fps": 10000,
        "system_center": np.zeros(3),
        "omega_rot": 100,
        "omega_rev": 500,
        "r_rev": 0.2,
        "x": 0,
        "Ry": 0,
        "Rz": 0,
        "a": 1,
        "b": 1,
        "omega_deform": 500,
        "deformation_mode": "ellipse",
        "noise_type": "normal",
        "noise_max": 1
    },
    "ROT_REV10_ELLIPSE10": {
        "duration": 1,
        "fps": 10000,
        "system_center": np.zeros(3),
        "omega_rot": 100,
        "omega_rev": 1000,
        "r_rev": 0.2,
        "x": 0,
        "Ry": 0,
        "Rz": 0,
        "a": 1,
        "b": 1,
        "omega_deform": 1000,
        "deformation_mode": "ellipse",
        "noise_type": "normal",
        "noise_max": 1
    }
}

@dataclass
class TimeParam:
    duration: float
    fps: float
    system_center: np.ndarray
    omega_rot: Union[float, np.ndarray]
    omega_rev: Union[float, np.ndarray]
    r_rev: float
    x: Union[float, np.ndarray]
    Ry: Union[float, np.ndarray]
    Rz: Union[float, np.ndarray]
    a: Union[float, np.ndarray]
    b: Union[float, np.ndarray]
    omega_deform: Union[float, np.ndarray]
    deformation_mode: str
    noise_type: str
    noise_max: float

class MotionCode(enum.Enum):
    ROT = 'ROT'
    ROT_REV = 'ROT_REV'
    ROT_REV5 = 'ROT_REV5'
    ROT_REV10 = 'ROT_REV10'
    ROT_REV_ELLIPSE = 'ROT_REV_ELLIPSE'
    ROT_REV5_ELLIPSE5 = 'ROT_REV5_ELLIPSE5'
    ROT_REV10_ELLIPSE10 = 'ROT_REV10_ELLIPSE10'

class TimeParamLoader:
    def __init__(self, time_params=time_params):
        self.time_params = time_params
    def param_factory(self, motion_code):
        param = self.time_params[motion_code.name]
        time_param = TimeParam(
            duration = param['duration'],
            fps = param['fps'],
            system_center = param['system_center'],
            omega_rot = param['omega_rot'],
            omega_rev = param['omega_rev'],
            r_rev = param['r_rev'],
            x = param['x'],
            Ry = param['Ry'],
            Rz = param['Rz'],
            a = param['a'],
            b = param['b'],
            deformation_mode = param['deformation_mode'],
            omega_deform = param['omega_deform'],
            noise_type = param['noise_type'],
            noise_max = param['noise_max']
        )
        return time_param

if __name__ == '__main__':
    print('---- test ----')

    param_loader = TimeParamLoader()
    param = param_loader.param_factory(MotionCode.ROT)
    print(param)
