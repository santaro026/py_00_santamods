# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:51:39 2024
@author: santaro

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fft as scfft
from numpy import fft as npfft

from pathlib import Path

if __name__ == '__main__':
    import myutils, myplotter, config
else:
    from . import myutils, myplotter
    import config

class Myfft:
    def __init__(self, t, ft, sample_rate):
        self._t = t
        self._ft = ft
        self._sample_rate = sample_rate
        self._N = int(len(self._t))
        self._duration = float(self._t[-1] - self._t[0])
        self._dt = 1 / self._sample_rate
        self._check_dt()
        self.cache = []
    @property
    def t(self):
        return self._t
    @property
    def ft(self):
        return self._ft
    @property
    def sample_rate(self):
        return self._sample_rate
    @property
    def N(self):
        return self._N
    @property
    def duration(self):
        return self._duration
    @property
    def dt(self):
        return self._dt

    def _check_dt(self):
        diffs = np.diff(self.t)
        dt_est = float(np.mean(diffs))
        rel_err = abs(dt_est - self._dt) / self._dt
        if rel_err > 0.01:
            raise ValueError(f"dt value error is too large: dt={self._dt}, acutial_dt_mean={dt_est}, relative_error={rel_err}")

    def _get_window(self, window_name, n, fft_backend) -> np.ndarray:
        window_name = window_name.lower()
        if fft_backend == "numpy":
            if window_name in ("hann", "hanning"):
                w = np.hanning(n)
            elif window_name == "hamming":
                w = np.hamming(n)
            elif window_name == "blackman":
                w = np.blackman(n)
            elif window_name in ("rect", "rectangular", "boxcar"):
                w = np.ones(n).astype(float)
            else:
                raise ValueError(f"Unsupported window: {window_name}")
        elif fft_backend == "scipy":
            if window_name in ("hann", "hanning"):
                w = signal.windows.get_window("hann", n)
            elif window_name == "hamming":
                w = signal.windows.get_window("hamming", n)
            elif window_name == "blackman":
                w = signal.windows.get_window("blackman", n)
            elif window_name in ("rect", "rectangular", "boxcar"):
                w = np.ones(n).astype(float)
            else:
                raise ValueError(f"Unsupported window: {window_name}")
        return w

    def _normalize_amplitude_scale(self, sp, n) -> np.ndarray:
        y = sp.copy()
        y *= (2.0 / n)
        y[0] *= 0.5
        is_even = (n%2==0)
        if is_even and len(y) > 1:
            y[-1] *= 0.5
        return y

    def _trange2frange(self, trange):
        f_start = np.where(self.t>=trange[0])[0][0]
        f_end = np.where(self.t<=trange[1])[0][-1]
        return (f_start, f_end)

    def split_data(self, fft_size, overlap=0.5, lastseg="pad", tranges=None):
        franges = []
        if tranges:
            for trange in tranges:
                frange = self._trange2frange(trange)
                franges.append(frange)
        else:
            tranges = [(self.t[0], self.t[-1])]
            franges = [(0, self.N)]
        t_target = np.zeros(0)
        ft_target = np.zeros(0)
        for frange in franges:
            t_target = np.hstack([t_target, self.t[frange[0]:frange[1]]])
            ft_target = np.hstack([ft_target, self.ft[frange[0]:frange[1]]])
        _N = len(t_target)
        self.t_splitted = []
        self.ft_splitted = []
        num_overlap  = int(fft_size * overlap)
        start_frame = 0
        end_frame = start_frame + fft_size
        pad_len = None
        while start_frame < _N:
            if end_frame > _N:
                if lastseg == "cut":
                    break
                else:
                    end_frame = _N
            _t = t_target[start_frame:end_frame]
            _ft =ft_target[start_frame:end_frame]
            if lastseg == "pad":
                if len(_t) < fft_size:
                    pad_len = fft_size - len(_t)
                    _t = np.hstack([_t, np.arange(1, pad_len+1)/self.sample_rate])
                    _ft = np.pad(_ft, (0, pad_len), mode='constant', constant_values=0)
            self.t_splitted.append(_t)
            self.ft_splitted.append(_ft)
            start_frame += (fft_size - num_overlap)
            end_frame = start_frame + fft_size
        _cache = {
            "tranges": tranges,
            "franges": franges,
            "t_splitted": self.t_splitted,
            "ft_splitted": self.ft_splitted,
            "t_target": t_target,
            "ft_target": ft_target,
            "fft_size": fft_size,
            "overlap": overlap,
            "lastseg": lastseg,
            "zero_padding": pad_len
        }
        return _cache

    def compute_fft(self, ft, window_func, mode="psd", is_real=True, use_fftshift=True, fft_backend="scipy"):
        n = len(ft)
        if fft_backend == "numpy":
            FFT = npfft
        elif fft_backend == "scipy":
            FFT = scfft
        window = self._get_window(window_func, n, fft_backend)
        pw_window = np.mean(window**2)
        rms_window = np.sqrt(pw_window)
        # avg_window = sum(window)/n
        ft_windowed = ft * window
        if is_real:
            sp = FFT.rfft(ft_windowed)
            freq = FFT.rfftfreq(n, self.dt)
        elif not is_real:
            sp = FFT.fft(ft_windowed)
            freq = FFT.fftfreq(n, self.dt)
            if use_fftshift:
                sp = FFT.fftshift(sp)
                freq = FFT.fftshift(freq)
            else:
                sp = sp[freq>=0]
                freq = freq[freq>=0]
        if mode == "psd":
            sp = abs(sp)**2
            sp /= (n * pw_window * self.sample_rate)
            sp[1:-1] *= 2
        elif mode == "spectrum":
            sp = abs(sp)**2
            sp /= (n * n * pw_window)
            if is_real:
                sp[1:-1] *= 2
            else:
                sp[1:] *= 2
            # if window_func in ("hann", "hanning"):
                # sp[1:-1] *= 1.5
        elif mode == "magnitude":
            sp = self._normalize_amplitude_scale(abs(sp), n)
            sp /= rms_window
            # sp /= avg_window
        else:
            raise ValueError(f"invalid argument: mode {mode}")
        return freq, sp

    def compute_segmented_fft(self, mode="psd", tranges=None, fft_size=2**12, overlap=0.5, lastseg="pad", window_func='hann', is_real=True, use_fftshift=True, fft_backend="scipy"):
        _cache = self.split_data(fft_size=fft_size, overlap=overlap, lastseg=lastseg, tranges=tranges)
        res = []
        for _t, _ft in zip(self.t_splitted, self.ft_splitted):
            if len(_t) < fft_size:
                pad_len = fft_size - len(_t)
                _ft = np.pad(_ft, (0, pad_len), mode='constant', constant_values=0)
                _cache["zero_padding"] = pad_len
            _res = self.compute_fft(_ft, window_func, mode=mode, is_real=is_real, use_fftshift=use_fftshift ,fft_backend=fft_backend)
            res.append(_res)
        freq = res[0][0]
        sp = np.zeros(len(freq))
        for i in range(len(res)):
            sp += res[i][1]
        sp = sp/(len(res))
        self.sp = sp
        self.freq = freq
        _cache2 = {
            "mode": mode,
            "window_func": window_func,
            "sp": self.sp,
            "freq": freq
        }
        _cache.update(_cache2)
        self.cache.append(_cache)
        return freq, sp

    def plot_result(self, plot_mode="plot", is_log=True, show_peaks=True, findpeak_height=10**-3, findpeak_distance=1, xrange=None, yrange=None, xtick=[1, 2000], ytick=None, xsigf=0, ysigf=0, ylabel=["time [sec]", "amplitude"], notell=""):
        t_per_window = self.cache[-1]["fft_size"] / self.sample_rate
        _x = self.t[-1]*0.02
        plotter = myplotter.MyPlotter(sizecode=myplotter.PlotSizeCode.LANDSCAPE_FIG_21)
        notelr = f'sample rate: {self.sample_rate:.0f} [/sec], window func: {self.cache[-1]["window_func"]}\nfft_size: {self.cache[-1]["fft_size"]} ({t_per_window*1000:.1f} [ms]), overlap: {self.cache[-1]["overlap"]}, zero padding: {self.cache[-1]["zero_padding"]}'
        if xrange is None: xrange = [(self.t[0], self.t[-1]), (0, self.sample_rate//2)]
        fig, axs = plotter.myfig(xrange=xrange, yrange=yrange, xtick=xtick, ytick=ytick, xsigf=xsigf, ysigf=ysigf, xlabel=["time [sec]", "frequency [Hz]"], ylabel=ylabel, notell=notell, notelr=notelr)
        axs[0].plot(self.t, self.ft, c='k', lw=0.2, alpha=1)
        for trange in self.cache[-1]["tranges"]:
            st, et = trange[0], trange[-1]
            axs[0].axvline(st, c='b', lw=0.4, alpha=0.2)
            axs[0].axvline(et, c='b', lw=0.4, alpha=0.2)
            axs[0].fill_between(x=np.linspace(st, et, 100), y1=-100, y2=100, color='b', alpha=0.1)
        axs[0].text(0, 0.96, f"window size: {t_per_window*1000:.1f} [ms]", fontsize=8, transform=axs[0].transAxes)
        axs[0].axvline(x=_x, c='g', lw=0.4)
        axs[0].axvline(x=_x+t_per_window, c='g', lw=0.4)
        mode = self.cache[-1]["mode"]
        _sp = self.sp
        if is_log:
            eps = np.finfo(float).tiny
            if mode in ("psd", "spectrum"):
                _sp = 10 * np.log10(np.maximum(_sp, eps))
            elif mode == "magnitude":
                _sp = 20 * np.log10(np.maximum(_sp, eps))
            elif mode == "complex":
                amp = np.abs(_sp)
                _sp = 20 * np.log10(np.maximum(amp, eps))
            else:
                pass
        if plot_mode == "plot":
            axs[1].plot(self.freq, _sp, c='b', lw=1, alpha=1)
        elif plot_mode == "stem":
            mline, sline, bline = axs[1].stem(self.freq, _sp, linefmt='-b', markerfmt='None', basefmt='b')
        if show_peaks:
            height_max = np.max(_sp[1:])
            if findpeak_height == 'auto':
                findpeak_height = height_max / 10
            peaks, props = signal.find_peaks(_sp, height=findpeak_height, distance=findpeak_distance)
            heights = props['peak_heights']
            arrowprops = dict(arrowstyle="-")
            for _p, _h in zip(peaks, heights):
                axs[1].annotate(f'{round(self.freq[_p])}\n{round(_h, 2)}', (self.freq[_p], _sp[_p]), textcoords='offset points', xytext=(0, 40), ha='center', arrowprops=arrowprops)
                # axs[1].annotate(f'f: {round(freq[_p])}\na: {round(_h, 2)}', (freq[_p], f_abs_amp[_p]), textcoords='data', xytext=(_p, height_max), ha='center')
        return fig, axs

    def make_spectrogram(self, window='hann', nperseg=None, noverlap=None, scaling="density", mode="psd", is_log=False, cmap="viridis", shading="auto", vrange=None, title='', yrange=[None, (0, 24000)], ylabel=[None, "frequency [Hz]"]):
        # cmap: viridis, jet, gray, bone, ocean
        # shading: auto, flat, nearest, gouraud
        freq, t_segment, sxx = signal.spectrogram(x=self.ft, fs=self.sample_rate, window=window, nperseg=nperseg, noverlap=noverlap, scaling=scaling, mode=mode)
        if is_log:
            eps = np.finfo(float).tiny
            if mode in ("psd", "spectrum"):
                sxx = 10 * np.log10(np.maximum(sxx, eps))
            elif mode == "magnitude":
                sxx = 20 * np.log10(np.maximum(sxx, eps))
            elif mode == "complex":
                amp = np.abs(sxx)
                sxx = 20 * np.log10(np.maximum(amp, eps))
            else:
                pass
        tperseg = nperseg / self.sample_rate
        _x = self.t[-1]*0.2
        axis_mode = "logarithmic" if is_log else "linear"
        notelr = f'sample rate: {self.sample_rate:.0f} [/sec]\nwindow func: {window}, nperseg: {nperseg} ({tperseg*1000:.1f} [ms]), noverlap: {noverlap}\namp axis: {axis_mode}'
        notell = ""
        plotter = myplotter.MyPlotter(sizecode=myplotter.PlotSizeCode.LANDSCAPE_FIG_21)
        fig, axs = plotter.myfig(title=title, xrange=(self.t[0], self.t[-1]), yrange=yrange, xtick=[1, None], ytick=None, xsigf=[1, 1], ysigf=[1, 0], xlabel="time [sec]", ylabel=ylabel, notell=notell, notelr=notelr)
        axs[0].plot(self.t, self.ft, c='k', lw=0.2, alpha=1)
        axs[0].text(0.2, 0.96, f"window size: {tperseg*1000:.1f} [ms]", fontsize=8, transform=axs[0].transAxes)
        axs[0].axvline(x=_x, c='g', lw=0.4)
        axs[0].axvline(x=_x+tperseg, c='g', lw=0.4)
        pcm = axs[1].pcolormesh(t_segment, freq, sxx, cmap=cmap, shading=shading)
        if vrange:
            pcm.set_clim(vmin=vrange[0], vmax=vrange[1])
        return fig, axs

def generate_periodic_data(duration, sample_rate, num_elements=1, freqs=None, amps=None, has_leakage=False, noise_level=0, noise_type="normal"):
    dt = 1 / sample_rate
    fundamental_freq = 1 / duration
    N = duration * sample_rate
    if N.is_integer():
        N = int(N)
    else:
        raise ValueError(f"number of sample N must be integer: {N}")
    t = np.linspace(0, 1 , N, endpoint=False) * duration
    if not freqs or not amps: rng = np.random.default_rng(seed=0)
    if not freqs:
        if has_leakage:
            freqs = rng.uniform(0, sample_rate//2, num_elements)
        else:
            freqs = rng.integers(0, N//2, num_elements) * fundamental_freq
    if not amps:
        amps = rng.uniform(0, 100, num_elements)
    ft = 0
    for f, a in zip(freqs, amps):
        ft += a * np.sin(2*np.pi*f*t)
    if noise_type == "normal":
        ft += rng.normal(0, 0.1, ft.size) * noise_level
    freqs_preview = np.array2string(freqs, precision=2)
    amps_preview = np.array2string(amps, precision=2)
    print(f"duration: {duration}, sample_rate: {sample_rate}, dt: {dt}, num_elements: {num_elements}")
    print(f"freqs: {freqs_preview}\namps: {amps_preview}")
    return t, ft, freqs, amps

def pw2db(ft):
    eps = np.finfo(float).tiny
    db = 10 * np.log10(np.maximum(ft, eps))
    return db
def mag2db(ft):
    eps = np.finfo(float).tiny
    db = 20 * np.log10(np.maximum(ft, eps))
    return db
def complex2db(ft):
    eps = np.finfo(float).tiny
    ft = np.abs(ft)
    db = 20 * np.log10(np.maximum(ft, eps))
    return db

def calc_analytic_psd(freqs, amps, bin_size=None, freq_axis=None):
    if bin_size:
        psd = np.zeros_like(freqs)
        for c, a in enumerate(amps):
            psd[c] = a**2 / 2 / bin_size
    elif freq_axis:
        psd = np.zeros_like(freq_axis)
        df = freq_axis[1] - freq_axis[0]
        for f, a in zip(freqs, amps):
            idx = np.argmin(np.abs(freq_axis - f))
            psd[idx] = a**2 / 2 / df
    return psd

def calc_analytic_spectrum(freqs, amps, freq_axis=None):
    if not freq_axis:
        psd = np.zeros_like(freqs)
        for c, a in enumerate(amps):
            psd[c] = a**2 / 2
    elif freq_axis:
        psd = np.zeros_like(freq_axis)
        for f, a in zip(freqs, amps):
            idx = np.argmin(np.abs(freq_axis - f))
            psd[idx] = a**2 / 2
    return psd

if __name__ == '__main__':
    print('---- test ----')
    #### generate sample data
    # rng = np.random.default_rng(seed=0)
    sample_rate = 48000
    duration = (2**15) / sample_rate
    dt = 1 / sample_rate
    num_elements = 4
    has_leakage = False
    noise_level = 0
    t, ft, freqs, amps = generate_periodic_data(duration, sample_rate, num_elements=num_elements, has_leakage=has_leakage, noise_level=noise_level)

    #### params
    fft_size = 2**15
    overlap = 0
    window_func = "rectangular"
    window_func = "hann"
    lastseg = "cut"
    is_log = False
    mode = "spectrum"
    # mode = "psd"

    if mode == "psd":
        scaling = "density"
    elif mode == "spectrum":
        scaling = "spectrum"

    analyzer = Myfft(t, ft, sample_rate)
    analyzer.compute_segmented_fft(fft_size=fft_size, window_func=window_func, overlap=overlap, mode=mode, lastseg=lastseg, is_real=True, use_fftshift=True, fft_backend="scipy")
    fig, axs = analyzer.plot_result(show_peaks=False, is_log=is_log)
    # axs[1].plot(analyzer.freq, analyzer.sp, c='b', lw=2, alpha=0.4)
    f_welch, Pxx_welch = signal.welch(x=ft, fs=sample_rate, window=window_func, nperseg=fft_size, noverlap=int(overlap*fft_size), scaling=scaling)
    axs[1].plot(f_welch, Pxx_welch, c='r', alpha=0.4, lw=2)

    # amps = calc_analytic_psd(freqs, amps, bin_size=sample_rate/fft_size)
    amps = calc_analytic_spectrum(freqs, amps)
    if is_log:
        _amps = mag2db(amps)
        bottom = -100
    else:
        _amps = amps
        bottom = 0
    axs[1].bar(freqs, _amps, width=200, bottom=bottom, color='g', alpha=0.4)
    # mline, sline, bline = axs[1].stem(freqs, _amps, bottom=bottom, linefmt='-r', markerfmt='None', basefmt='r')

    # fig, axs = analyzer.make_spectrogram(nperseg=2**8, noverlap=0.5, is_log=True, yrange=[None, (0, 10000)], shading="nearest", vrange=(-100, -40))

    # print(analyzer.cache)
    plt.show()

    # fig, ax = plt.subplots(figsize=(15, 8))

    # res = analyzer.split_data(fft_size=fft_size, lastseg="pad", tranges=[(0, 0.2), (0.5, 0.8)], overlap=0)
    # x = res["t_target"]
    # y = res["ft_target"]
    # xs = res["t_splitted"]
    # ys = res["ft_splitted"]

    # x, y = np.zeros(0), np.zeros(0)
    # for _x, _y in zip(xs, ys):
    #     x = np.hstack([x, _x])
    #     y = np.hstack([y, _y])

    # ax.plot(x, y, lw=1, c='b')
    # ax.plot(np.arange(len(x)), x, lw=1, c='b')
    # ax.plot(np.arange(len(x)), y, lw=1, c='b')


    # plt.show()


    """
    datafile_sc02 = config.ROOT / "data" / "REC2469.mat"
    datafile_sc03 = config.ROOT / "data" / "REC2475.mat"
    datafile_sc05 = config.ROOT / "data" / "REC2491.mat"
    datafile_sc28 = config.ROOT / "data" / "REC2673.mat"
    datafile_sc29 = config.ROOT / "data" / "REC2674.mat"
    datafile_sc32 = config.ROOT / "data" / "REC2683.mat"
    datafile_sc33 = config.ROOT / "data" / "REC2684.mat"

    res_list = []

    audio_dataloader_sc02 = myav.AudioDataLoader(datafile_sc02)
    analyzer_sc02 = Myfft(audio_dataloader_sc02.t, audio_dataloader_sc02.sound, 48000)
    res_sc02 = analyzer_sc02.compute_segmented_fft(trange=(0, 0.6), is_real=True, fft_backend="scipy", fft_size=fft_size)
    # fig, axs = analyzer_sc02.plot_result(show_peaks=False, yrange=[(-20, 20), None])
    # analyzer_sc02.make_spectrogram(nperseg=2**8, noverlap=0.5)

    audio_dataloader_sc03 = myav.AudioDataLoader(datafile_sc03)
    analyzer_sc03 = Myfft(audio_dataloader_sc03.t, audio_dataloader_sc03.sound, 48000)
    res_sc03 = analyzer_sc03.compute_segmented_fft(trange=(5, 5.6), is_real=True, fft_backend="scipy", fft_size=fft_size)
    # fig, axs = analyzer_sc03.plot_result(show_peaks=False, yrange=[(-20, 20), None])
    # analyzer_sc03.make_spectrogram(nperseg=2**8, noverlap=0.5)

    audio_dataloader_sc05 = myav.AudioDataLoader(datafile_sc05)
    analyzer_sc05 = Myfft(audio_dataloader_sc05.t, audio_dataloader_sc05.sound, 48000)
    res_sc05 = analyzer_sc05.compute_segmented_fft(trange=(5, 5.6), is_real=True, fft_backend="scipy", fft_size=fft_size)
    # fig, axs = analyzer_sc05.plot_result(show_peaks=False, yrange=[(-20, 20), None])

    audio_dataloader_sc28 = myav.AudioDataLoader(datafile_sc28)
    # print(audio_dataloader_sc28)
    analyzer_sc28 = Myfft(audio_dataloader_sc28.t, audio_dataloader_sc28.sound, 48000)
    res_sc28 = analyzer_sc28.compute_segmented_fft(trange=(2.6, 3.2), is_real=True, fft_backend="scipy", fft_size=fft_size)
    # fig, axs = analyzer_sc28.plot_result(show_peaks=False)

    audio_dataloader_sc29 = myav.AudioDataLoader(datafile_sc29)
    analyzer_sc29 = Myfft(audio_dataloader_sc29.t, audio_dataloader_sc29.sound, 48000)
    res_sc29 = analyzer_sc29.compute_segmented_fft(trange=(1, 1.6), is_real=True, fft_backend="scipy", fft_size=fft_size)
    # fig, axs = analyzer_sc29.plot_result(show_peaks=False)

    audio_dataloader_sc32 = myav.AudioDataLoader(datafile_sc32)
    analyzer_sc32 = Myfft(audio_dataloader_sc32.t, audio_dataloader_sc32.sound, 48000)
    res_sc32 = analyzer_sc32.compute_segmented_fft(trange=(3.2, 3.8), is_real=True, fft_backend="scipy", fft_size=fft_size)
    # fig, axs = analyzer_sc32.plot_result(show_peaks=False)
    analyzer_sc32.make_spectrogram(nperseg=2**10, noverlap=0.5)

    audio_dataloader_sc33 = myav.AudioDataLoader(datafile_sc33)
    analyzer_sc33 = Myfft(audio_dataloader_sc33.t, audio_dataloader_sc33.sound, 48000)
    res_sc33 = analyzer_sc33.compute_segmented_fft(trange=(3.2, 3.8), is_real=True, fft_backend="scipy", fft_size=fft_size)
    # fig, axs = analyzer_sc33.plot_result(show_peaks=False)

    res_list = [res_sc02, res_sc28, res_sc29, res_sc33]
    color_list = ['k', 'r', 'orange', 'b']

    # plotter = myplotter.MyPlotter(sizecode=myplotter.PlotSizeCode.RECTANGLE_FIG)
    # fig, axs = plotter.myfig(ysigf=3)
    # for i, res in enumerate(res_list):
    #     freq = res[0]
    #     sp = res[1]
    #     axs[0].plot(freq, sp, c=color_list[i], lw=1, alpha=1)

    # axs[0].set(xlim=(0, 20000))

    plt.show()


    """





