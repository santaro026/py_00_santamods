"""
Created on Wed Dec 03 19:18:42 2025
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

import tempfile
import subprocess
from pydub import AudioSegment
from pydub.playback import play

import cv2 as cv

import sys
import os
from pathlib import Path

import config

class MyAudioEditor:
    def __init__(self, data_path, tempdir=None):
        self._data_path = data_path
        self._audio = AudioSegment.from_file(self._data_path)
        self._channels = self._audio.channels
        self._sample_rate = self._audio.frame_rate
        self._duration = self._audio.duration_seconds
        self._sample_width = self._audio.sample_width
        self._bit_depth = self._sample_width * 8

        self.t = np.arange(0, self._duration, 1/self._sample_rate)
        self.soundl = np.array(self._audio.get_array_of_samples())[0::self._channels]
        self.soundr = np.array(self._audio.get_array_of_samples())[1::self._channels]

        self.tempdir = config.ROOT / ".tmp" if tempdir is None else tempdir

    @property
    def data_path(self):
        return self._data_path
    @property
    def audio(self):
        return self._audio
    @property
    def channels(self):
        return self._channels
    @property
    def sample_rate(self):
        return self._sample_rate
    @property
    def duration(self):
        return self._duration
    @property
    def sample_width(self):
        return self._sample_width
    @property
    def bit_width(self):
        return self._bit_depth

    def __repr__(self):
        return (
                f"data_path: {self.data_path}\n"
                f"channels: {self.channels}, sample_rate: {self.sample_rate}, duration: {self.duration}, bit_widht: {self.bit_width}"
        )

    def play(self, trange=None, volume_db=0):
        audio_to_play = self._audio
        if trange is not None:
            start_ms, end_ms = trange
            audio_to_play = audio_to_play[max(0, start_ms):max(0, end_ms)]
        if volume_db != 0:
            audio_to_play = audio_to_play.apply_gain(volume_db)
        if self._data_path.suffix == ".wav":
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()
            audio_to_play.export(tmp_path, format="wav")
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", tmp_path],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            try:
                os.unlink(tmp_path)
            except OSError:
                print(f"**** OSError in os.unlink(tmp_path), temp file still remained: {tmp_path}")
        else:
            play(self._audio)

    def adjust_audio_offset(self, offset_ms):
        audio_to_mix = self._audio
        if offset_ms < 0:
            silence = AudioSegment.silent(duration=(offset_ms))
            audio_to_mix = silence + audio_to_mix
        elif offset_ms >0:
            audio_to_mix = audio_to_mix[offset_ms:]
        temp_audio_path = self.tempdir / "temp.mp4"
        audio_to_mix.export(temp_audio_path, codec="aac", bitrate="192k")
        return temp_audio_path, audio_to_mix

    def mix_video(self, video_file, offset_ms, output_file=None):
        temp_audio_path, temp_audio = self.adjust_audio_offset(offset_ms)
        if output_file is None: output_file = config.ROOT / "results" / "mixed_video.mp4"
        subprocess.run(["ffmpeg", "-y",
                        "-i", str(video_file.resolve()),
                        "-i", str(temp_audio_path.resolve()),
                        "-map", "0:v:0",
                        "-map", "1:a:0",
                        "-c:v", "copy",
                        "-c:a", "copy",
                        # "-shortest",
                        str(output_file.resolve())], check=True)
        os.remove(temp_audio_path)

class MyVideoEditor:
    def __init__(self, data_path):
        self._data_path = data_path
        self._video = cv.VideoCapture(self._data_path)
        self._width = int(self._video.get(cv.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._video.get(cv.CAP_PROP_FRAME_HEIGHT))
        self._fps = int(self._video.get(cv.CAP_PROP_FPS))
        self._num_frames = int(self._video.get(cv.CAP_PROP_FRAME_COUNT))
        self._duration = self._num_frames / self._fps

    @property
    def data_path(self):
        return self._data_path
    @property
    def video(self):
        return self._video
    @property
    def width(self):
        return self._width
    @property
    def height(self):
        return self._height
    @property
    def fps(self):
        return self._fps
    @property
    def num_frames(self):
        return self._num_frames
    @property
    def duration(self):
        return self._duration

    def __repr__(self):
        return (
            f"data path: {self._data_path}\n"
            f"(width, height): ({self._width}, {self._height})\n"
            f"fps: {self._fps}, num_frames: {self._num_frames}, duration: {self._duration}"
        )


def check_ffmpeg():
    print(sys.executable)
    print(os.environ.get("PATH", "")[:200], "...")
    print(os.environ.get("FFMPEG_BINARY"))
    print(os.environ.get("FFPROBE_BINARY"))

def check_temp():
    import tempfile, os
    print("TEMP:", os.environ.get("TEMP"))
    print("TMP:", os.environ.get("TMP"))

if __name__ == "__main__":
    print("---- test ----")
    print(f"ROOT: {config.ROOT}")

    audio_editor = MyAudioEditor(config.ROOT/"data"/"sampledata"/"wav"/"desk_hand.wav")
    # audio_editor = MyAudioEditor(config.ROOT/"data"/"sampledata"/"wav"/"desk_iphone.wav")
    print(repr(audio_editor))
    # audio_editor.play(trange=[0, 10000], volume_db=0)

    # import myfft

    fig, ax = plt.subplots(figsize=(20, 12))
    ax.plot(audio_editor.t, audio_editor.soundl)
    plt.show()


    # video_editor = MyVideoEditor(config.ROOT/"data"/"video_sample.mp4")
    # print(repr(video_editor))



