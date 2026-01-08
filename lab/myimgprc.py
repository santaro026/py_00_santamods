"""
Created on Wed Dec 17 22:04:25 2025
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from pathlib import Path

import cv2 as cv

import config

def crop_rectangle(img, target_size, center):
    top = center[0] - target_size[0]//2
    bottom = center[0] + target_size[0]//2
    left = center[1] - target_size[1]//2
    right = center[1] + target_size[1]//2
    return img[top:bottom, left:right]

def mask_circle(img, center, radius):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask = cv.circle(mask, center, radius, (255), -1)
    return cv.bitwise_and(img, img, mask=mask)

def rorate_img(img, center, angle, direction=1):
    direction = -1 if direction < 0 else 1
    angle = direction * angle
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    return cv.warpAffine(img, M, img.shape[:2])

def adjust_gamma(img, gamma):
    table = ((np.arange(256) / 255) ** (1/gamma) * 255).astype(np.uint8)
    return cv.LUT(img, table)

def equalize_histogram(img):
    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    ycrcb[:, 0] = cv.equalizeHist(ycrcb[:, 0])
    return cv.cvtColor(ycrcb, cv.COLOR_YCrCb2BGR)

def roi_rectangle(img, yx, size, color=(255, 0, 0), lw=2):
    y1 = yx[0]-size[0]
    y2 = yx[0]+size[0]
    x1 = yx[1]-size[1]
    x2 = yx[1]+size[1]
    top_left = (x1, y1)
    bottom_right = (x2, y2)
    img = cv.rectangle(img, top_left, bottom_right, color, lw)
    roiyx = ((y1, y2), (x1, x2))
    roi = img[y1:y2, x1:x2]
    return roi, roiyx, img

def roi_circle(mg, center, radius, color=(0, 0, 255), lw=2):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask = cv.circle(mask, center, radius, 255, -1)
    roi = cv.bitwise_and(img, img, mask=mask)
    img = cv.circle(img, center, radius, color, lw)
    roi_id = (mask == 255)
    return roi, roi_id, img


class VideoLoader:
    def __init__(self, data_path):
        self._data_path = data_path
        self._cap = cv.VideoCapture(self._data_path)
        self._width = int(self._cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self._fps = int(self._cap.get(cv.CAP_PROP_FPS))
        self._num_frames = int(self._cap.get(cv.CAP_PROP_FRAME_COUNT))
        ret, frame = self._cap.read()
        self._channels = frame.shape[2]

    @property
    def data_path(self):
        return self._data_path
    @property
    def cap(self):
        return self._cap
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
    def channels(self):
        return self._channels

    def __repr__(self):
        return (
            f"data_path: {self.data_path}\n"
            f"fps: {self.fps}, num_frames: {self.num_frames}\n"
            f"(width, height, channels): ({self.width}, {self.height}, {self.channels})\n"
        )


class VideoProcessor:
    def __init__(self, video_loader):
        self._video_loader = video_loader
        self._original = self._video_loader.cap
        self.window_name = self._video_loader.data_path.name
        self.window = cv.namedWindow(self.window_name)
        self.cache = {}

    @property
    def video_loader(self):
        return self._video_loader
    @property
    def original(self):
        return self._original

    def process_video(self, *funcs_with_args):
        # for i in range(self.video_loader.num_frames):
        while True:
            ret, frame = self.video_loader.cap.read()
            # if not ret:
                # break
            for _i, (func, args, kwargs) in enumerate(funcs_with_args):
                frame = func(frame, *args, **kwargs)
            cv.imshow(self.window_name, frame)
            k = cv.waitKey(1)
            if k == ord('q'):
                break
        cv.destroyAllWindows()

    def rotate_video(self, center, angles, target_size, pos):
        # fourcc = cv.VideoWriter_fourcc(*"mp4v")
        # video_writer = cv.VideoWriter(config.ROOT/"data"/self.window_name, fourcc, 30, target_size)
        for i in range(self.video_loader.num_frames):
        # while True:
            ret, frame = self.video_loader.cap.read()
            if not ret:
                break

            if i > 1000:
                break

            frame = crop_rectangle(frame, target_size, center)
            center2 = (target_size[0]//2, target_size[1]//2)
            frame = mask_circle(frame, center2, target_size[0]//2)
            frame = rorate_img(frame, center2, angles[i], direction=1)
            frame = cv.circle(frame, center2, 400, (0, 0, 255), 2)

            frame = adjust_gamma(frame, gamma=3)

            frame = cv.convertScaleAbs(frame, alpha=1, beta=0)

            r = 441
            size = (20, 20)
            # r = 200
            # size = (80, 80)
            for _i in range(8):
                x = int(r * np.cos(pos[_i])) + center2[0]
                y = int(r * np.sin(pos[_i])) + center2[1]
                roi, roiyx, frame = roi_rectangle(frame, (y, x), size)

                roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
                _, roi = cv.threshold(roi, 200, 255, cv.THRESH_BINARY)
                # _, roi = cv.threshold(roi, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                fill_id = (roi == 255)
                roi = cv.cvtColor(roi, cv.COLOR_GRAY2BGR)
                if i == 0:
                    roi[fill_id, 0] = 0
                    roi[fill_id, 1] = 0
                    roi[fill_id, 2] = 255
                frame[roiyx[0][0]:roiyx[0][1], roiyx[1][0]:roiyx[1][1]] = roi

            #### test
            if i == 0:
                img0 = frame
            elif i > 0:
                frame = cv.addWeighted(frame, 0.5, img0, 0.5, 0)

            cv.imshow(self.window_name, frame)
            k = cv.waitKey(1)
            if k == ord('q'):
                break




            # video_writer.write(frame)
        # video_writer.release()
        cv.destroyAllWindows()



class VideoViewer:
    def __init__(self, video_loader, name=None):
        self.video_loader = video_loader
        self.name = self.video_loader.data_path.name if name is None else name
        self.window = cv.namedWindow(self.name)

    def view(self):
        while True:
            ret, frame = self.video_loader.cap.read()
            cv.imshow(self.name, frame)
            k = cv.waitKey(1)
            if k == ord('q'):
                break
        cv.destroyAllWindows()

    def view_frame(self, frame):
        self.video_loader.cap.set(cv.CAP_PROP_POS_FRAMES, frame)
        ret, img = self.video_loader.cap.read()
        while True:
            cv.imshow(self.name, img)
            k = cv.waitKey(1)
            if k == ord('q'):
                break
        cv.destroyAllWindows()


if __name__ == "__main__":
    print("---- test ----")

    datadir = config.ROOT / "data" / "251210_tc26_v652" / "avi"
    # dataname = "tc26_sc28_5000rpm_8000fps_rec2673_0to4000.avi"
    dataname = "tc26_sc28_5000rpm_8000fps_rec2673_34000to38000.avi"
    data_path = datadir / dataname

    video_loader = VideoLoader(data_path)
    print(repr(video_loader))

    # video_viewer = VideoViewer(video_loader)
    # video_viewer.view()
    # video_viewer.view_frame(100)

    video_processor = VideoProcessor(video_loader)
    center = (511, 507)
    center = (513, 507)
    target_size = (1000, 1000)
    num_frames = video_loader.num_frames
    fps = 8000
    deg_per_frame = np.degrees(2138.46/60*2*np.pi) / fps
    angles = np.arange(num_frames) * deg_per_frame

    # pos = np.linspace(0, 2*np.pi, 8, endpoint=False) - np.radians(-4)
    pos = np.linspace(0, 2*np.pi, 8, endpoint=False) - np.radians(4)


    # video_processor.process_video(
    #     (crop_rectangle, ((1000, 1000), center), {}),
    #     (mask_circle, (center, 500), {}),
    #     (rorate_img, (center, 20), {"direction": 1})
    # )

    video_processor.rotate_video(center=center, angles=angles, target_size=target_size, pos=pos)

