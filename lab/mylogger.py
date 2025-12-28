"""
Created on Wed Oct 15 14:43:29 2025
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from pathlib import Path
import time

import logging
import config

class MyLogger(logging.Logger):
    def __init__(self, name, outdir=Path.cwd(), mode="w"):
        super().__init__(name)
        outdir.mkdir(parents=True, exist_ok=True)
        self.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.DEBUG)
        self.console_handler.setFormatter(self.formatter)
        self.file_handler = logging.FileHandler(outdir / f'{name}.log', mode=mode)
        self.file_handler.setLevel(logging.INFO)
        self.file_handler.setFormatter(self.formatter)
        if not self.handlers:
            self.addHandler(self.console_handler)
            self.addHandler(self.file_handler)
        self.cache = {}

    def measure_time(self, name, mode):
        if mode == 's':
            st = time.perf_counter()
            self.cache[f"{name}_st"] = st
        elif mode == 'e':
            et = time.perf_counter()
            self.cache[f"{name}_et"] = et
            elapsed_time = et - self.cache[f"{name}_st"]
            self.cache[f"{name}_elapsed"] = elapsed_time
            self.info(f"{name} elapsed time: {elapsed_time}")


if __name__ == '__main__':
    print('---- test ----')

    # logger.debug('debug message')
    # logger.info('information')
    # logger.warning('warning')

    # logger = MyLogger(name=__name__)
    # logger.info('test')
    # logger.debug('debug test')
    # logger.warning('****')

    # logger2 = MyLogger(name="logger2")
    # logger2.info("test2")
    # logger2.debug("debug test2")
    # logger2.warning("**** 2")

    # logger = MyLogger(name=__name__)
    # logger.warning('**** second time')
