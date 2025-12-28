# -*- coding: utf-8 -*-
"""
Created on Wed Feb 5 09:02:00 2025
@author: santaro

"""

import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import scipy

import time
import sys
from pathlib import Path
import re
from datetime import datetime
import dill
import os

def find_projectroot(curdir=None, markfile=".myprojectroot"):
    # if curdir == None: curdir = get_main_script_path()
    if curdir == None: curdir = Path(os.getcwd())
    while True:
        if (curdir / markfile).exists(): break
        if curdir == curdir.parent : raise FileNotFoundError(f'{markfile} was not found.')
        curdir = curdir.parent
    return curdir

def get_versioninfo(scrdir):
    releasenote = str(list(scrdir.glob('released_at_*'))[0])
    releasedate = re.search(r'released_at_(\d{6})', releasenote).group(1)
    version = re.search(r'(v_\d+_\d+_\d+)', releasenote).group(1)
    return releasedate, version

def get_main_script_path():
    main_script = sys.modules['__main__'].__file__
    res = Path(main_script).resolve().parent
    return res

def file_exist_checker(func, *args, **kwargs):
    """
    It verifies the existence of the target file and executes a function like saving the file if it doesn't exist.

    """
    file = Path(args[0])
    if file.exists():
        print(f'**** There is already a file with the same name in the destination:\n{file}')
        res = 1
    else:
        func(*args, **kwargs)
        res = 0
    return res

def save_fig(fig, outdir=None, outfname='ouput.png', pkl=False, mkchildir=False, offbackend=False):
    if offbackend:
        import matplotlib
        matplotlib.use('Agg')
    if outdir is None:
        outdir = get_main_script_path() / 'data'
    outfname = Path(outfname)
    if mkchildir:
        outdir_png = outdir / outfname.suffix[1:]
        outdir_png.mkdir(parents=True, exist_ok=True)
    else:
        outdir_png = outdir
    outfname_buf = outfname
    outfpath = outdir_png / outfname_buf
    file_exist = outfpath.exists()
    count = 1
    while file_exist:
        print(f'**** There is already a file with the same name in the destination:\n{outfpath}')
        outfpath = outdir_png / (outfname_buf.stem + f'_o{count}' + outfpath.suffix)
        file_exist = outfpath.exists()
        count += 1
    fig.savefig(outfpath, dpi=300)
    if pkl:
        if mkchildir:
            outdir_pkl = outdir / 'pkl'
            outdir_pkl.mkdir(parents=True, exist_ok=True)
        else:
            outdir_pkl = outdir
        outfpath_pkl = outdir_pkl / (outfpath.stem + '.pkl')
        with open(outfpath_pkl, 'wb') as f:
            dill.dump(fig, f)
    plt.close()
    if offbackend:
        matplotlib.use('qt5agg')
    return 0

def save_csv(data, trans=True, header=None, outdir=None, outfname='output.csv', pkl=False, mkchildir=False):
    if outdir is None:
        outdir = get_main_script_path() / 'data'
    outfname = Path(outfname)
    if mkchildir:
        outdir_csv = outdir / outfname.suffix[1:]
        outdir_csv.mkdir(parents=True, exist_ok=True)
    else:
        outdir_csv = outdir
    outfname_buf = outfname
    outfpath = outdir_csv / outfname_buf
    file_exist = outfpath.exists()
    count = 1
    while file_exist:
        print(f'**** There is already a file with the same name in the destination:\n{outfpath}')
        outfpath = outdir_csv / (outfname_buf.stem + f'_o{count}' + outfpath.suffix)
        file_exist = outfpath.exists()
        count += 1
    if trans:
        df = pl.from_numpy(data.T, schema=header)
    else:
        df = pl.from_numpy(data, schema=header)
    df.write_csv(outfpath)
    if pkl:
        if mkchildir:
            outdir_pkl = outdir / 'pkl'
            outdir_pkl.mkdir(parents=True, exist_ok=True)
        else:
            outdir_pkl = outdir
        outfpath_pkl = outdir_pkl / (outfpath.stem + '.pkl')
        with open(outfpath_pkl, 'wb') as f:
            dill.dump(data, f)
    return 0

def summarize_res_csv(data, schema, wkdir=None, fname='output.csv'):
    if wkdir is None:
        wkdir = get_main_script_path() / 'data'
    outfile = wkdir / fname
    if outfile.exists():
        df_read = pl.scan_csv(outfile)
        df_add = pl.DataFrame(data, schema=schema).cast(df_read.collect_schema()).lazy()
        df_res = pl.concat([df_read, df_add], how='vertical')
    else:
        df_res = pl.DataFrame(data, schema=schema).lazy()
    return df_res

def save_txt(data, outdir=None, outfname='text.txt'):
    if isinstance(data, str):
        note = data
    elif isinstance(data, dict):
        note = ''
        for key, value in data.items():
            note += f'{key}: {value}\n'
    elif isinstance(data, list):
        note = ','.join(str(e) for e in data)

    if outdir is None:
        outdir = get_main_script_path() / 'data'
    outfname = Path(outfname)
    outdir.mkdir(parents=True, exist_ok=True)
    outfile_buf = outdir / outfname
    outfile = outfile_buf
    count = 1
    while True:
        try:
            with open(outfile, mode='x') as f:
                f.write(note)
                break
        except FileExistsError:
            outfile = outfile_buf.parent / (outfile_buf.stem + f'_o{count}' + outfile_buf.suffix)
            count += 1

def natural_keys(text):
    parts = re.split(r'(\d+)', text)
    res = [int(part) if part.isdigit() else part for part in parts]
    return res

def sort_bynumber(texts):
    res = sorted(texts, key=natural_keys)
    return res

def read_matdata(input_file):
    """
    extract sound data from .mat file for atremis

    """
    mat_data = scipy.io.loadmat(input_file)
    struct_array = mat_data['shdf']
    void_entry = struct_array[0][0]
    fields = void_entry.dtype.names
    data = void_entry['Data'][0]
    return data

class ReiterWrapper(object):
    def __init__(self, f):
        self._f = f
    def __iter__(self):
        return self._f()

class Logger:
    def __init__(self):
        self.log_entries = []
        self.mgs_entries = []
        self.time_entries = {}

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if type(message) != str:
            message = str(message)
        self.log_entries.append(f'[{timestamp}] {message}')

    def msg(self, message):
        if type(message) != str:
            message = str(message)
        self.mgs_entries.append(f'{message}')

    def rectime(self, mode, name='main'):
        if mode == 0:
            self.log(message=f'{name} start')
            self.time_entries[f'st_{name}'] = time.perf_counter()
        elif mode == 1:
            et = time.perf_counter()
            excution_time = et - self.time_entries[f'st_{name}']
            self.log(message=f'{name} complete')
            self.log(message=f'excution time of {name}: {excution_time} [sec]')
            return excution_time

    def printlog(self):
        print('\n**** printlog\n')
        print('\n'.join(self.log_entries))

    def printmsg(self):
        print('\n**** printmsg\n')
        print('\n'.join(self.mgs_entries))

    def export(self, outdir=None, outfname=""):
        if outdir is None:
            outdir = get_main_script_path()
        count = 1
        outfile_buf = outdir / (outfname + "_log.txt")
        outfile = outfile_buf
        while True:
            try:
                with open(outfile, "x") as f:
                    f.write("\n".join(self.log_entries))
                    break
            except FileExistsError:
                outfile = outfile_buf.parent / (str(outfile_buf.stem) + f'_o{count}' + str(outfile_buf.suffix))
                count += 1
        count = 1
        outfile_buf = outdir / (outfname + "_msg.txt")
        outfile = outfile_buf
        while True:
            try:
                with open(outfile, "x") as f:
                    f.write("\n".join(self.mgs_entries))
                    break
            except FileExistsError:
                outfile = outfile_buf.parent / (str(outfile_buf.stem) + f'_o{count}' + str(outfile_buf.suffix))
                count += 1

def print_progressbar(current, total, bar_length=40, row=1):
    percent = round(current / total * 100)
    filled_length = int(current / total * bar_length)
    bar = "#" * filled_length + " " * (bar_length - filled_length)
    print(f'{percent:3.0f} % |{bar}| {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
    if row > 1:
        for i in range(row-1):
            print(f'      |{bar}|', flush=True)

def print_pathlist(path_list, digit_width=2):
    # _p = '\n'.join(map(str, path_list))
    _p = '\n'.join(f'{i:0{digit_width}}: {p}' for i, p in enumerate(path_list))
    print(_p)
    return 0

def extract_runs(mask):
    runs = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    if len(runs) == 0:
        runs = None
    return runs

def merge_ranges(time_ranges1, time_ranges2, tmax=None):
    if isinstance(time_ranges1, np.ndarray): time_ranges1 = time_ranges1.tolist()
    if isinstance(time_ranges2, np.ndarray): time_ranges2 = time_ranges2.tolist()
    if time_ranges1 is None and time_ranges2 is None:
        merged = None
        merged_filtered = None
    elif time_ranges1 is None and time_ranges2 is not None:
        merged = time_ranges2
    elif time_ranges1 is not None and time_ranges2 is None:
        merged = time_ranges1

    else:
        if not any(isinstance(r, list) for r in time_ranges1):
            time_ranges1 = [time_ranges1]
        if not any(isinstance(r, list) for r in time_ranges2):
            time_ranges2 = [time_ranges2]
        all_ranges = sorted(time_ranges1 + time_ranges2, key=lambda x: x[0])
        merged = []
        for current in all_ranges:
            if not merged or merged[-1][1] < current[0]:
                merged.append(current)
            else:
                merged[-1][1] = max(merged[-1][1], current[1])
    if tmax is not None and merged is not None:
        merged_filtered = []
        if not any(isinstance(_m, list) for _m in merged):
            merged = [merged]
        for _m in merged:
            st, et = _m
            if st < tmax and et < tmax:
                merged_filtered.append([st, et])
            elif st < tmax and tmax < et:
                merged_filtered.append([st, tmax])
            elif tmax < st:
                pass
        if len(merged_filtered) == 0:
            merged_filtered = None
    return merged_filtered

def cnvt_trange2frange(time_ranges, fps):
    if time_ranges is None:
        frame_ranges = None
        frame_ids = False
    else:
        if isinstance(time_ranges, np.ndarray): time_ranges = time_ranges.tolist()
        if not any(isinstance(r, list) for r in time_ranges):
            time_ranges = [time_ranges]
        frame_ranges, frame_ids = [], []
        for time_range in time_ranges:
            st = time_range[0]
            et = time_range[1]
            sf = st * fps
            if sf - int(sf) != 0:
                sf += 1
            ef = int(et * fps)
            frame_range = [sf, ef]
            frame_id = np.arange(sf, ef).astype(np.int64)
            frame_ranges.append(frame_range)
            frame_ids.append(frame_id)
    return frame_ranges, frame_ids

def detect_noise(sound, sampling_rate, window_time=0.1, threshold_factor=10, which='rms'):
    from scipy import signal
    num_perwin = int(window_time*sampling_rate)
    _kernel = np.ones(num_perwin) / num_perwin
    sound_pw = signal.fftconvolve(sound**2, _kernel, mode='same')
    sound_rms = np.sqrt(sound_pw)
    sound_range = [np.min(sound_rms), np.max(sound_rms)]
    threshold = threshold_factor * sound_range[0]
    mask = sound_rms > threshold
    okruns = extract_runs(mask) # get runs [start, end), with end not included.
    okruns_id = np.where(mask)[0]
    ngruns = extract_runs(~mask)
    ngruns_id = np.where(~mask)[0]
    return okruns, okruns_id, ngruns, ngruns_id, threshold, sound_pw, sound_rms, sound_range

if __name__ == '__main__':
    print(f'\n---- test ----\n')

    wkdir = Path(__file__).resolve().parent

    t = np.linspace(0, 1, 101)
    x = 10 * np.sin(10*t)

    res = get_main_script_path()
    print(res)

