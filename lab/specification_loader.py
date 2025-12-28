"""
Created on Wed Oct 22 22:31:35 2025
@author: santaro


"""
import numpy as np

from pathlib import Path

from dataclasses import dataclass
import enum
from typing import Union
import json
import config

@dataclass
class Specification:
    name: str
    PCD: float
    ID: float
    OD: float
    width: float
    num_pockets: int
    num_markers: int
    # num_mesh: int
    # num_node: int
    Dp: float
    Dw: float

class SampleCode(enum.Enum):
    SIMPLE50 = "SIMPLE50"

class SpecificationLoader():
    def __init__(self, infpath=config.ROOT/'assets'/'cage_specifications.json'):
        with open(infpath, 'r') as f:
            self.specifications = json.load(f)
    def specification_factory(self, sample_code):
        specification = self.specifications[sample_code.name]
        spec = Specification(
            name = specification['name'],
            PCD = specification['PCD'],
            ID = specification['ID'],
            OD = specification['OD'],
            width = specification['width'],
            num_pockets = specification['num_pockets'],
            num_markers = specification['num_markers'],
            # num_mesh = specification['num_mesh'],
            # num_node = specification['num_node'],
            Dp = specification['Dp'],
            Dw = specification['Dw']
        )
        return spec

if __name__ == '__main__':
    print('---- test ----')

    spec_loader = SpecificationLoader()
    spec = spec_loader.specification_factory(SampleCode.SIMPLE50)

    print(spec)

