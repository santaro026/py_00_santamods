"""
Created on Thu Dec 04 19:35:12 2025
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

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import threading

import config

def get_main_script_path():
    main_script = sys.modules['__main__'].__file__
    res = Path(main_script).resolve().parent
    return res

def check_file_exist(func):
    def myinner(*args, **kwargs):
        file = Path(args[0])
        if file.exists():
            print(f'**** There is already a file with the same name in the destination:\n{file}')
            newfile = file
            return Path(newfile)
        else:
            func(*args, **kwargs)
            return 0
    return myinner

def get_unique_outfile(output_file: Path):
    output_file_new = output_file
    count = 1
    while output_file_new.exists():
        print(f'**** There is already a file with the same name in the destination:\n{output_file}')
        output_file_new = output_file.parent / (output_file.stem + f'_{count}' + output_file.suffix)
        count += 1
    return output_file_new

def savefig(fig, output_file, pkl=False, mkchildir=False, backendoff=True, dpi=300):
    outdir = output_file.parent
    outdir.mkdir(parents=True, exist_ok=True)
    if backendoff:
        import matplotlib
        matplotlib.use('Agg')
    if mkchildir:
        outdir_png = outdir / output_file.suffix.lstrip('.')
        outdir_png.mkdir(parents=True, exist_ok=True)
    else:
        outdir_png = outdir
    output_file_png = outdir_png / output_file.name
    output_file_png = get_unique_outfile(output_file_png)
    fig.savefig(output_file_png, dpi=dpi)
    if pkl:
        if mkchildir:
            outdir_pkl = outdir / 'pkl'
            outdir_pkl.mkdir(parents=True, exist_ok=True)
        else:
            outdir_pkl = outdir
        output_file_pkl = outdir_pkl / (output_file_png.stem + '.pkl')
        with open(output_file_pkl, 'wb') as f:
            dill.dump(fig, f)
    return output_file_png

def savecsv(data, output_file, header=None, pkl=False, mkchildir=False):
    outdir = output_file.parent
    outdir.mkdir(parents=True, exist_ok=True)
    if mkchildir:
        outdir_csv = outdir / output_file.suffix.lstrip('.')
        outdir_csv.mkdir(parents=True, exist_ok=True)
    else:
        outdir_csv = outdir
    output_file_csv = outdir_csv / output_file.name
    output_file_csv = get_unique_outfile(output_file_csv)
    df = pl.from_numpy(data, schema=header)
    df.write_csv(output_file_csv)
    if pkl:
        if mkchildir:
            outdir_pkl = outdir / 'pkl'
            outdir_pkl.mkdir(parents=True, exist_ok=True)
        else:
            outdir_pkl = outdir
        output_file_pkl = outdir_pkl / (output_file_csv.stem + '.pkl')
        with open(output_file_pkl, 'wb') as f:
            dill.dump(data, f)
    return output_file_csv

def savetxt(data, output_file, pkl=False, mkchildir=False):
    if isinstance(data, str):
        note = data
    elif isinstance(data, dict):
        note = ''
        for key, value in data.items():
            note += f'{key}: {value}\n'
    elif isinstance(data, list):
        note = ','.join(str(e) for e in data)
    else:
        note = str(data)
    outdir = output_file.parent
    outdir.mkdir(parents=True, exist_ok=True)
    if mkchildir:
        outdir_txt = outdir / output_file.suffix.lstrip('.')
        outdir_txt.mkdir(parents=True, exist_ok=True)
    else:
        outdir_txt = outdir
    output_file_txt = outdir_txt / output_file.name
    output_file_txt = get_unique_outfile(output_file_txt)
    with open(output_file_txt, mode='x') as f:
        f.write(note)
    if pkl:
        if mkchildir:
            outdir_pkl = outdir / 'pkl'
            outdir_pkl.mkdir(parents=True, exist_ok=True)
        else:
            outdir_pkl = outdir
        output_file_pkl = outdir_pkl / (output_file_txt.stem + '.pkl')
        with open(output_file_pkl, 'wb') as f:
            dill.dump(data, f)
    return output_file_txt

class MyTkProgress():
    def __init__(self, title="", maximum=100, size="400x200"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(size)
        self.width, self.height = map(int, size.split('x'))

        self.maximum = maximum
        self.progress = tk.IntVar()
        self.pb = ttk.Progressbar(self.root, maximum=self.maximum, mode="determinate", variable=self.progress, length=self.width*0.8)
        self.pb.place(x=self.width*0.1, y=self.height*0.1)

        # self.empty_label = tk.Label(self.root, text="", relief="solid", width=int(40), height=int(4))
        # self.empty_label.place(x=self.width*0.5, y=self.height*0.5, anchor="center")
        self.note_label = tk.Label(self.root, text="")
        self.note_label.place(x=self.width*0.5, y=self.height*0.3, anchor="n")

        self.time_label= tk.Label(self.root, text="")
        self.time_label.place(x=self.width*0.5, y=self.height*0.85, anchor="center")
        self.start_time = time.perf_counter()
        self.update_timer()

        # self.count_btn = tk.Button(self.root, text="c++", command=self.count_up)
        # self.count_btn.pack()
        # self.count_btn = tk.Button(self.root, text="c--", command=self.count_down)
        # self.count_btn.pack()
        # self.stop_btn = tk.Button(self.root, text="stop", command=self.stop_progress)
        # self.stop_btn.pack()

        self.root.bind("<q>", lambda event: self.root.destroy())

    def mainloop(self):
        self.root.mainloop()

    def stop_progress(self):
        self.pb.stop()
    def count_up(self):
        if self.progress.get() == 100:
            messagebox.showinfo("info", "complete")
        if self.progress.get() < 100:
            self.progress.set(self.progress.get() + 10)
    def count_down(self):
        if self.progress.get() != 0:
            self.progress.set(self.progress.get() - 10)

    def update_timer(self):
        elapsed = int(time.perf_counter() - self.start_time)
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        self.time_label.config(text=f"{h:02}:{m:02}:{s:02}")
        self.root.after(1000, self.update_timer)

    def update_progress(self, progress):
        self.root.after(0, lambda: self.progress.set(progress))

    def update_note(self, note):
        self.note_label.config(text=note)

    def update_status(self, progress, note):
        if progress >= self.maximum:
            self.update_progress(progress)
            self.update_note(f"complete\n"+note)
        else:
            self.update_progress(progress)
            self.update_note(note)

    def start_task(self, main, *args, **kwargs):
        t = threading.Thread(target=main, args=args, kwargs=kwargs, daemon=False)
        t.start()



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

def print_progressbar(current, total, bar_length=40, row=1):
    percent = round(current / total * 100)
    filled_length = int(current / total * bar_length)
    bar = "#" * filled_length + " " * (bar_length - filled_length)
    print(f'{percent:3.0f} % |{bar}| {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
    if row > 1:
        for i in range(row-1):
            print(f'      |{bar}|', flush=True)


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

def test_main(step=100):
    for i in range(step):
        print(i)
        time.sleep(0.1)
        # mybar.update_progress(i+1)
        # mybar.update_note(f"now: {i+1}")
        mybar.update_status(i+1, f"now: {i+1}")



if __name__ == '__main__':
    print(f'\n---- test ----\n')
    datadir = config.ROOT / "data"

    # data_path = datadir / "sample_data1.csv"
    # data_path = config.ROOT / "results" / "sample_data.png"

    # fig, ax = plt.subplots()
    # x = np.arange(10)
    # y = x**2
    # d = np.vstack([x, y]).T

    # savecsv(d, config.ROOT/"results"/"sampel.csv", pkl=True, mkchildir=True)


    mybar = MyTkProgress()
    mybar.start_task(test_main, 100)
    mybar.mainloop()

