"""
Created on Sat Nov 29 13:14:50 2025
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import scipy

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable, Tuple, Union

import config

class DataMapLoader:
    @staticmethod
    def load_datamap(datamapfile):
        df = pl.read_excel(datamapfile, sheet_name="all", has_header=True, drop_empty_cols=True, drop_empty_rows=True, infer_schema_length=1000)
        datamap = df.select(
            pl.col("test_code").cast(pl.Int32, strict=False),
            pl.col("shooting_code").cast(pl.Int32, strict=False),
            pl.col("date").cast(pl.String, strict=False),
            pl.col("cage").cast(pl.String, strict=False),
            pl.col("material").cast(pl.String, strict=False),
            pl.col("commanded_rot_speed").cast(pl.Int32, strict=False),
            pl.col("fps").cast(pl.Int32, strict=False),
            pl.col("recording_number").cast(pl.Int32, strict=False),
            pl.col("sample_rate").cast(pl.Int32, strict=False)
        ).drop_nulls(subset=["test_code"])
        return datamap

    def __init__(self, datamap_path):
        self._datamap_path = datamap_path
        self._datamap = DataMapLoader.load_datamap(self._datamap_path)

    @property
    def datamap_path(self):
        return self._datamap_path
    @property
    def datamap(self):
        return self._datamap

    def extract_info_from_tcsc(self, tc, sc):
        info = self.datamap.filter((pl.col("test_code") == tc) & (pl.col("shooting_code") == sc)).to_dicts()
        if len(info) == 0:
            raise ValueError(f"no data was found in datamap by tc{tc}-sc{sc}.")
        elif len(info) > 1:
            raise RuntimeError(f"multiple data ({len(info)} was found by tc{tc}-sc{sc}.")
        return info[0]

    def extract_info_from_rec(self, rec):
        info = self.datamap.filter(pl.col("recording_number") == rec).to_dicts()
        if len(info) == 0:
            raise ValueError(f"no data was found in datamap by rec{rec}.")
        elif len(info) > 1:
            raise RuntimeError(f"multiple data ({len(info)} was found by rec{rec}.")
        return info[0]

    def __repr__(self):
        return (
            f"DataMapLoader:\n"
            f"datamap_path: {self.datamap_path}\n"
            f"num_data: {self.datamap.shape[0]}\n"
        )

class CoordDataLoader:
    @staticmethod
    def get_label_from_filename(filename):
        tc_match = re.search(r"tc(\d+)", filename)
        sc_match = re.search(r"sc(\d+)", filename)
        tc = int(tc_match.group(1)) if tc_match else None
        sc = int(sc_match.group(1)) if sc_match else None
        return tc, sc

    @staticmethod
    def get_info_from_filename(filename):
        fps_match = re.search(r"(\d+)fps", filename)
        rpm_match = re.search(r"(\d+)rpm", filename)
        rec_match = re.search(r"rec(\d+)", filename)
        fps = int(fps_match.group(1)) if fps_match else None
        rpm = int(rpm_match.group(1)) if rpm_match else None
        rec = int(rec_match.group(1)) if rec_match else None
        return fps, rpm, rec

    @staticmethod
    def load_markers(data_path=None, data_format="tema", num_cage_markers=8):
        if data_format == "tema":
            skip_rows = 3
            skip_data = 0
            separator = '\t'
        data = pl.read_csv(data_path, has_header=False, skip_rows=skip_rows, separator=separator, infer_schema_length=50000).cast(pl.Float64, strict=False).to_numpy()[:, skip_data:]
        t = data[:, 0]
        if t[0] != 0:
            raise ValueError(f"loaded data does not start from 0 [sec], {data_path}, start form {t[0]}")
        points = data[:, 1:]
        if len(points[0]) % 2 == 0: # check data shape
            num_points = len(points[0]) // 2
            num_ring_markers = num_points - num_cage_markers
            if not (num_points == num_cage_markers or num_points == num_cage_markers + 1):
                raise RuntimeError(f"loading data shape does not match: {data_path}, num_points is {num_points}")
        else:
            raise RuntimeError(f"the number of coordinate data points is odd, confirm the input data: {data_path}")
        num_frames = len(t)
        points = points.reshape((num_frames, num_points, 2))
        cage_markers = points[:, :num_cage_markers]
        if num_ring_markers > 0:
            ring_markers = data[:, num_cage_markers:num_cage_markers+num_ring_markers]
        else:
            ring_markers = None
        return t, cage_markers, ring_markers

    @staticmethod
    def load_markers_zero(data_path, data_format="tema", num_cage_markers=8):
        if data_format == "tema":
            data = pl.read_csv(data_path, has_header=False, skip_rows=3, separator='\t', infer_schema_length=10).cast(pl.Float64, strict=False).to_numpy()[:, 1:].astype(float)
            cage_markers = data[:-1, :2].astype(float)
            ring_center = data[-1, :2].astype(float)
            ring_area = data[-1, 2].astype(float)
            if len(cage_markers) % 2 == 0:
                if len(cage_markers) != num_cage_markers:
                    raise RuntimeError(f"loading data shape does not match: {data_path}")
            else:
                raise RuntimeError(f"loading data shape does not match, number of points data is odd: {data_path}")
        return cage_markers, ring_center, ring_area

    @staticmethod
    def calc_scaling_factor_pixel2mm(measured_value=1, reference_value=1, reference_mode="area"):
        if reference_mode == "area":
            scaling_factor_pixel2mm = np.sqrt(reference_value/measured_value)
        return scaling_factor_pixel2mm

    def __init__(self, data_path, zero_data_path, data_format="tema", num_cage_markers=8, zero_data_format="tema", pixel2mm_reference_mode="area", reference_value=np.pi*(49.1/2)**2):
        self._data_path = data_path
        self._zero_data_path = zero_data_path
        self._tc, self._sc = CoordDataLoader.get_label_from_filename(self._data_path.name)
        self._fps, self._rpm, self._rec = CoordDataLoader.get_info_from_filename(self._data_path.name)
        self.t_data, self.cage_markers_pixel, self.ring_markers_pixel = CoordDataLoader.load_markers(data_path=data_path, data_format=data_format, num_cage_markers=num_cage_markers)
        self._num_frames = len(self.t_data)
        self.cage_markers_zero_pixel, self.ring_center_zero_pixel, self.ring_area_zero_pixel = CoordDataLoader.load_markers_zero(data_path=self._zero_data_path, data_format=zero_data_format, num_cage_markers=num_cage_markers)
        self._pixel2mm_reference_mode = pixel2mm_reference_mode
        self._reference_value = reference_value
        self.pixel2mm = CoordDataLoader.calc_scaling_factor_pixel2mm(measured_value=self.ring_area_zero_pixel, reference_value=self._reference_value, reference_mode=self._pixel2mm_reference_mode)
        self.t = np.arange(self._num_frames) / self._fps
        self.ring_center_zero = (self.pixel2mm * self.ring_center_zero_pixel)[np.newaxis, np.newaxis, :]
        self.cage_markers = self.pixel2mm * self.cage_markers_pixel - self.ring_center_zero
        self.cage_markers_zero = (self.pixel2mm * self.cage_markers_zero_pixel - self.ring_center_zero)[np.newaxis, :, :]
        if self.ring_markers_pixel is not None: self.ring_markers = self.pixel2mm * self.ring_markers_pixel - self.ring_center_zero
        self._duration = float(self.t[-1] - self.t[0])

    @property
    def data_path(self):
        return self._data_path
    @property
    def zero_data_path(self):
        return self._zero_data_path
    @property
    def tc(self):
        return self._tc
    @property
    def sc(self):
        return self._sc
    @property
    def rpm(self):
        return self._rpm
    @property
    def fps(self):
        return self._fps
    @property
    def num_frames(self):
        return self._num_frames
    @property
    def duration(self):
        return self._duration
    @property
    def num_cage_markers(self):
        return self._num_cage_markers
    @property
    def rec(self):
        return self._rec
    @property
    def pixel2mm_reference_mode(self):
        return self._pixel2mm_reference_mode
    @property
    def reference_value(self):
        return self._reference_value

    def __repr__(self):
        ring_center_zero_preview = np.array2string(self.ring_center_zero.squeeze(), precision=9, separator=", ")
        ring_center_zero_pixel_preview = np.array2string(self.ring_center_zero_pixel, precision=9, separator=", ")
        return (
            f"data_path: {self.data_path}\n"
            f"zero_data_path: {self.zero_data_path}\n"
            f"tc: {self.tc}, sc: {self.sc}\n"
            f"rpm: {self.rpm}, rec: {self.rec}\n"
            f"fps: {self.fps} [frame/sec], duration: {self.duration} [sec], num_frames: {self.num_frames},\n"
            f"biring area: {self.ring_area_zero_pixel} [pixel] ({self.reference_value} [mm**2]), pexel2mm: {self.pixel2mm}\n"
            f"ring_center: {ring_center_zero_preview} [mm] ({ring_center_zero_pixel_preview} [pixel])\n"
            f"cage_markers: {self.cage_markers.shape}, ring_markers: {self.ring_markers.shape}\n"
        )

@dataclass
class CoordSeries:
    loader: CoordDataLoader
    datamap: DataMapLoader | None = None
    def __post_init__(self):
        if self.datamap is not None:
            info = self.datamap.extract_info_from_tcsc(self.loader.tc, self.loader.sc)
            if self.loader.rpm != info["commanded_rot_speed"]:
                raise ValueError(f"data condition of rpm does not match.\nfilename info: {self.loader.rpm}, datamap info: {info["commanded_rot_speed"]}")
            if self.loader.fps !=  info["fps"]:
                raise ValueError(f"data condition of fps does not match.\nfilename info: {self.loader.fps}, datamap info: {info["fps"]}")
            if self.loader.rec != info["recording_number"]:
                raise ValueError(f"data condition of rec does not match.\nfilename info: {self.loader.rec}, datamap info: {info["recording_number"]}")

    @property
    def meta(self) -> dict[str, int | float]:
        return {
            "tc": self.loader.tc,
            "sc": self.loader.sc,
            "fps": self.loader.fps,
            "duration": self.loader.duration,
            "num_frames": self.loader.num_frames
        }
    @property
    def t(self) -> np.ndarray:
        return self.loader.t
    @property
    def cage_markers(self) -> np.ndarray:
        return self.loader.cage_markers
    @property
    def ring_markers(self) -> np.ndarray:
        return self.loader.ring_markers

    def __repr__(self):
        return f"CoordSeries:\n{repr(self.loader)}"

class AudioDataLoader:
    @staticmethod
    def get_label_from_filename(filename):
        tc_match = re.search(r"tc(\d+)", filename)
        sc_match = re.search(r"sc(\d+)", filename)
        tc = int(tc_match.group(1)) if tc_match else None
        sc = int(sc_match.group(1)) if sc_match else None
        return tc, sc

    @staticmethod
    def get_info_from_filename(filename):
        rec_match = re.search(r"REC(\d+)", filename)
        rec = int(rec_match.group(1)) if rec_match else None
        return rec

    @staticmethod
    def load_sound(data_path=None):
        data_format = data_path.suffix
        if data_format == ".mat":
            matdata = scipy.io.loadmat(data_path)
            struct_array = matdata["shdf"]
            void_entry = struct_array[0][0]
            fields = void_entry.dtype.names
            # print(fields)
            t = void_entry["Absc1Data"][0]
            sound = void_entry["Data"][0]
            # data_volt = void_entry["Data"][1]
            if t[0] != 0:
                raise ValueError(f"input audio data has problem, time data does not start form 0, start from {t[0]}.")
        elif data_format == ".csv":
            df = pl.read_csv(data_path, has_header=False, skip_rows=19, separator=',', infer_schema_length=10).cast(pl.Float64, strict=False) #.to_numpy()[:, 1].astype(float)
            data = df.to_numpy()[:, 0:2].astype(float)
            t = data[:, 0]
            sound = data[:, 1]
            if t[0] != 0:
                raise ValueError(f"input audio data has problem, time data does not start form 0, start from {t[0]}.")
        return t, sound

    def __init__(self, data_path):
        self._data_path = data_path
        self._tc, self._sc = AudioDataLoader.get_label_from_filename(self._data_path.name)
        self._rec = AudioDataLoader.get_info_from_filename(self._data_path.name)
        self._t, self._sound = AudioDataLoader.load_sound(self._data_path)
        self._num_samples = len(self._t)
        self._duration = float(self._t[-1] - self._t[0])
        self._sample_rate = float(1 / (self._t[1] - self._t[0]))

    @property
    def data_path(self):
        return self._data_path
    @property
    def tc(self):
        return self._tc
    @tc.setter
    def tc(self, value):
        if self._tc is None:
            self._tc = value
        else: raise AttributeError("tc has already certain value, you cannot rewrite tc.")
    @property
    def sc(self):
        return self._sc
    @sc.setter
    def sc(self, value):
        if self._sc is None:
            self._sc = value
        else: raise AttributeError("sc has already certain value, you cannot rewrite sc.")
    @property
    def rec(self):
        return self._rec
    @property
    def sample_rate(self):
        return self._sample_rate
    @sample_rate.setter
    def sample_rate(self, value):
        self._sample_rate = value
    @property
    def num_samples(self):
        return self._num_samples
    @property
    def duration(self):
        return self._duration
    @property
    def t(self):
        return self._t
    @property
    def sound(self):
        return self._sound

    def __repr__(self):
        t_preview = np.array2string(self.t[:5], precision=10, separator=", ")
        sound_preview = np.array2string(self.sound[:5], precision=10, separator=", ")
        return (
            f"data_path: {self.data_path}\n"
            f"tc: {self.tc}, sc: {self.sc}, rec: {self.rec}\n"
            f"sample_rate: {self.sample_rate} [Hz], duration: {self.duration} [sec], num_samples: {self.num_samples}\n"
            f"t: {self.t.shape}, {t_preview}\n"
            f"sound: {self.sound.shape}, {sound_preview}\n"
        )

@dataclass
class AudioSeries:
    loader: AudioDataLoader
    datamap: DataMapLoader | None = None
    def __post_init__(self):
        if self.datamap is not None:
            info = self.datamap.extract_info_from_rec(self.loader.rec)
            if self.loader.tc is None:
                self.loader.tc = info["test_code"]
            if self.loader.sc is None:
                self.loader.sc = info["shooting_code"]
            sample_rate_accuracy = abs(1 - self.loader.sample_rate/info["sample_rate"])
            if sample_rate_accuracy >= 0.01:
                raise ValueError(f"data condition of sample_rate does not match.\ndatamap info: {info["sample_rate"]}, actual data: {self.loader.sample_rate}")

    @property
    def meta(self) -> dict[str, int | float]:
        return {
            "tc": self.loader.tc,
            "sc": self.loader.sc,
            "rec": self.loader.rec,
            "sample_rate": self.loader.sample_rate,
            "duration": self.loader.duration,
            "num_samples": self.loader.num_samples
        }
    @property
    def t(self) -> np.ndarray:
        return self.loader.t
    @property
    def sound(self) -> np.ndarray:
        return self.loader.sound

    def __repr__(self):
        return f"AudioSeries:\n{repr(self.loader)}"

@dataclass
class Series:
    _coord: CoordSeries | None = None
    _audio: AudioSeries | None = None
    datamap: DataMapLoader | None = None
    aligned_reference: str | None = None
    cache: dict[str, object] = field(default_factory=dict)
    _tc: int | None = None
    _sc: int | None = None
    def __post_init__(self):
        if self._coord: self.tc = self.coord.meta["tc"]
        if self._audio: self.sc = self.coord.meta["sc"]

    def __repr__(self):
        coord_meta = None
        audio_meta = None
        if self.coord:
            coord_meta = self.coord.meta
        if self.audio:
            audio_meta = self.audio.meta
        return (
            f"tc: {self.tc}, sc: {self.sc}, coord: {coord_meta}, audio: {audio_meta}"
        )

    @property
    def tc(self):
        return self._tc
    @tc.setter
    def tc(self, value):
        self._tc = value
    @property
    def sc(self):
        return self._sc
    @sc.setter
    def sc(self, value):
        self._sc = value
    @property
    def coord(self):
        return self._coord
    @coord.setter
    def coord(self, value):
        if not isinstance(value, CoordSeries):
            raise TypeError(f"coord must be CoordSeries object, {type(value)} was passed.")
        if self._coord:
            raise TypeError(f"coord already has a CoordSeries object {self.coord}, you cannot overwrite.")
        elif self._coord is None:
            self._coord = value
            self.tc = self.coord.meta["tc"]
            self.sc = self.coord.meta["sc"]
    @property
    def audio(self):
        return self._audio
    @audio.setter
    def audio(self, value):
        if not isinstance(value, AudioSeries):
            raise TypeError(f"audio must be audioSeries object, {type(value)} was passed.")
        if self._audio:
            raise TypeError(f"audio already has a CoordSeries object {self.audio}, you cannot overwrite.")
        elif self._audio is None:
            self._audio = value
    @property
    def meta(self) -> dict[str, int | float]:
        universal_code = {"tc": self.tc, "sc": self.sc}
        if self.coord:
            universal_code.update(self.coord.meta)
        if self.audio:
            universal_code.update({k: v for k, v in self.audio.meta.items() if k not in universal_code})
        return universal_code
    @property
    def map(self):
        return self.datamap.datamap

    def has_both(self) -> bool:
        return (self.coord is not None) and (self.audio is not None)

@dataclass
class HandlerConfig:
    resample_mode: str = "linear"
    interpolation_fill: str = "edge"
    logging_level: str = "INFO"

class DataSeriesHandler:
    def __init__(self, config: HandlerConfig | None = None):
        self.config = config
        self.seriesmap: dict[tuple[int, int], Series] = {}
        self.datamap: DataMapLoader | None = None
        self.unloaded_coord: list[Path] = []
        self.unloaded_audio: list[Path] = []
        self._logs: list[tuple[str, str]] = []

    @staticmethod
    def search_zero_coord_file(tgtdir, tgtfile):
        tc_match = re.search(r"tc(\d+)", tgtfile.name)
        tc = tc_match.group(1) if tc_match else None
        suffix = tgtfile.suffix
        if tc:
            p = tgtdir.glob(f"tc{tc}_sc00*{suffix}")
        if p is None:
            p = tgtdir.glob(f"zero*{suffix}")
        if p is None:
            raise FileNotFoundError(f"zero coord file was not found for {tgtfile}.")
        p = list(p)
        if len(p) != 1:
            raise FileNotFoundError(f"zero data file must be a single, but {len(p)} file was found.")
        return p[0]

    def add_coord_file(self, data_path, zero_data_path):
        try:
            loader = CoordDataLoader(data_path=data_path, zero_data_path=zero_data_path)
            tc = loader.tc
            sc = loader.sc
        except Exception as e:
            self._log("ERROR", f"Coord add failed: {data_path} ({e})")
            self.unloaded_coord.append(data_path)
            return None
        if (tc is None) or (sc is None):
            series = None
        else:
            series = self.seriesmap.get((tc, sc))
        if series is None:
            series = Series()
            self.seriesmap[(tc, sc)] = series
        series.coord = CoordSeries(loader)
        self._log("INFO", f"Coord added: tc={tc}, sc={sc}, file={data_path}")
        return series

    def add_audio_file(self, data_path):
        try:
            loader = AudioDataLoader(data_path)
            rec = loader.rec
            info = self.datamap.extract_info_from_rec(rec)
            tc = info["test_code"]
            sc = info["shooting_code"]
        except Exception as e:
            self._log("ERROR", f"Audio add failed: {data_path} ({e})")
            self.unloaded_audio.append(data_path)
            return None
        if tc is None or sc is None:
            series = None
        else:
            series = self.seriesmap.get((tc, sc))
        if series is None:
            series = Series()
            self.seriesmap[(tc, sc)] = series
        series.audio = AudioSeries(loader)
        self._log("INFO", f"Audio added: tc={tc}, sc={sc}, file={data_path}")
        return series

    def scan_directory(self, coord_dir, audio_dir, datamap_dir, coord_glob, audio_glob, datamap_glob):
        datamap_list = list(datamap_dir.glob(datamap_glob))
        if len(datamap_list) != 1:
            raise FileNotFoundError(f"multiple datamap file was found, it msut be a single file")
        self.datamap = DataMapLoader(datamap_list[0])
        for p in coord_dir.glob(coord_glob):
            if p.match(r"*sc00*"):
                continue
            zero_data_path = DataSeriesHandler.search_zero_coord_file(coord_dir, p)
            self.add_coord_file(p, zero_data_path)
        if audio_dir and audio_glob:
            for p in audio_dir.glob(audio_glob):
                self.add_audio_file(p)
        self.seriesmap = dict(sorted(self.seriesmap.items(), key=lambda x: (x[0][0], x[0][1])))

    def report_pairing(self):
        paired = sum(1 for s in self.seriesmap.values() if s.has_both())
        nocoord = [f"{s.tc}-{s.sc}" for s in self.seriesmap.values() if s.coord is None]
        noaudio = [f"{s.tc}-{s.sc}" for s in self.seriesmap.values() if s.audio is None]
        return {
            "paired_count": paired,
            "num_series": len(self.seriesmap),
            "missing_coord": nocoord,
            "missing_audio": noaudio,
            "unloaded_coord_files": [p.name for p in self.unloaded_coord],
            "unloaded_audio_files": [p.name for p in self.unloaded_audio],
        }

    def filter(self, tc, sc):
        for s in self.seriesmap.values():
            if s.coord is None:
                continue
            m = s.coord.meta
            if tc is not None and m.get("tc") != tc:
                continue
            if sc is not None and m.get("sc") != sc:
                continue
            yield s

    #### editing
    def align_series(self, tc, sc, reference="coord", t0_offset=0.0):
        s = self.seriesmap.get((tc, sc))
        if s is None:
            self._log("WARN", f"align skipped because data was not found: (tc, sc) = ({tc}, {sc})")
            return None
        if not s.has_both():
            self._log("WARN", f"align skipped because coord or audio data was missing: (tc, sc) = ({tc}, {sc})")
            return None
        t_coord = s.coord.t
        t_audio = s.audio.t + t0_offset
        sound = s.audio.sound
        aligned_sound = np.interp(t_coord, t_audio, sound)
        self.aligned_reference = reference
        s.cache["aligned"] = {
            "t": t_coord,
            "coord_cage_markers": s.coord.cage_markers,
            "coord_ring_markers": s.coord.ring_markers,
            "audio":aligned_sound,
        }
        return s.cache["aligned"]

    #### editing
    def slice(self, tc, sc, t_start, t_end):
        s = self.seriesmap.get(tc, sc)
        if not s or "aligned" not in s.cache:
            self._log("WARN", f"slice skipped: tc, sc = ({tc}, {sc})")
            return None
        data = s.cache["aligned"]
        t = data["t"]
        idx = (t >= t_start) & (t <= t_end)
        out = {
            "t": t[idx],
            "coord": data["coord"][idx],
            "audio": data["audio"][idx],
        }
        return out

    def _log(self, level, message):
        if self.config:
            if self.config.logging_level in ("DEBUG", "INFO", "WARN", "ERROR"):
                self._logs.append((level, message))

if __name__ == "__main__":
    print("---- test ----")


    datadir = config.ROOT / "data" / "tc99_290101_testCage_PA99GF30"

    datamapfile = datadir / "list_visualization_test.xlsx"
    datamap_loader = DataMapLoader(datamapfile)
    datamap = datamap_loader.datamap
    # print(datamap.head())
    # info = datamap_loader.extract_info_from_tcsc(1, 2)
    # print(info)
    # datamap.write_csv(datadir/"datamap.csv")

    datafile_coord = datadir / "tc23_sc01_4000rpm_8000fps_rec2153.txt"
    datafile_coord_zero = datadir / "tc23_sc00_0rpm_8000fps_rec2113.txt"
    coord_loader = CoordDataLoader(data_path=datafile_coord, num_cage_markers=8, zero_data_path=datafile_coord_zero)
    # print(repr(coord_loader))
    # coord_series = CoordSeries(coord_loader, datamap_loader)
    # print(coord_series.meta)
    # print(repr(coord_series))


    # datafile_audio = datadir / "REC2153.mat"
    # audio_loader = AudioDataLoader(data_path=datafile_audio)
    # print(audio_loader)
    # audio_series = AudioSeries(audio_loader, datamap_loader)
    # print(audio_series.meta)
    # print(audio_series)


    # series = Series(coord_series, audio_series, datamap_loader)
    # print(series.meta)

    dataseries = DataSeriesHandler()
    dataseries.scan_directory(datadir, datadir, datadir, "tc*", "REC*", "*.xlsx")
    print(dataseries.report_pairing())
    # print(dataseries)
    # print(dataseries.datamap)
    for k, v in dataseries.seriesmap.items():
        print(f"{k}: {v}")

    # print(dataseries._logs)


    # all_list = dataseries.list_all()
    # for l in all_list:
        # print(l)

