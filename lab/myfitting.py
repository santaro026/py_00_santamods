"""
Created on Wed Dec 17 22:03:47 2025
@author: santaro



"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

#### least squares fitting of line
def lsm_for_line(points):
    if points.shape[1] != 2:
        raise ValueError("argument of points must be shape of (N, 2).")
    valid_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
    valid_id = np.where(valid_mask)[0]
    x = points[valid_id, 0]
    y = points[valid_id, 1]
    num_points = len(x)
    M1 = np.vstack([x, np.ones(num_points)]).T
    coef, residuals, rank, s = np.linalg.lstsq(M1, y, rcond=None)
    slope, intercept = coef
    info = {
        "rss": float(residuals[0]) if residuals.size > 0 else None,
        "rank": int(rank),
        "singular_values": s,
        "num_points": len(x),
        "valid_indices": valid_id
    }
    # M2 = np.array(y)
    # res = np.linalg.inv(M1.T @ M1) @ M1.T @ M2
    return slope, intercept, info

#### least squares fitting of circles
def lsm_for_circle(points):
    if points.shape[1] != 2:
        raise ValueError("argument of points must be shape of (N, 2).")
    valid_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
    valid_id = np.where(valid_mask)[0]
    if len(valid_id) < 3:
        raise ValueError(f"need at least 3 valid points")
    x = points[valid_id, 0]
    y = points[valid_id, 1]
    num_points = len(x)
    M1 = np.vstack([x, y, np.ones(num_points)]).T
    M2 = -(x**2 + y**2)
    coef, residuals, rank, s = np.linalg.lstsq(M1, M2, rcond=None)
    A, B, C = coef
    # A, B, C = np.dot(np.linalg.inv(np.dot(M1.T, M1)), np.dot(M1.T, M2))
    cx, cy = -A/2, -B/2
    r = (cx**2 + cy**2 - C)**0.5
    xyr = np.array([cx, cy, r])
    d_center = np.sqrt((x - cx)**2 +(y - cy)**2)
    geom_err = np.abs(d_center - r)
    info = {
        "rss": float(residuals[0]) if residuals.size > 0 else None,
        "rank": int(rank),
        "singular_values": s,
        "num_points": len(x),
        "valid_indices": valid_id,
        "radii": d_center,
        "geom_error_mean": float(np.mean(geom_err)),
        "geom_error_std": float(np.std(geom_err)),
        "geom_error_max": float(np.max(geom_err)),
    }
    return xyr, info

def lsm_for_circles(points):
    """
    Least squares fitting of circles for sequential frames.
    it doesn't use linalg.lstsq of numpy to avoid for loops for frames, is useful when the number of frames is greater than that of points.

    """
    if points.shape[2] != 2:
        raise ValueError("argument of points must be shape of (M, N, 2).")
    valid_mask = (~np.isnan(points).any(axis=2) & ~np.isinf(points).any(axis=2)).all(axis=1)
    valid_id = np.where(valid_mask)[0]
    if len(valid_id) < 3:
        raise ValueError(f"need at least 3 valid points")
    num_frames = points.shape[0]
    num_points = points.shape[1]
    xyrs = np.full((num_frames, 3), np.nan)
    d_center = np.full((num_frames, num_points), np.nan)
    xs = points[valid_id, :, 0]
    ys = points[valid_id, :, 1]
    num_frames_valid = xs.shape[0]
    M1 = np.stack([xs, ys, np.ones((num_frames_valid, num_points))], axis=2)
    M1T = M1.transpose(0, 2, 1)
    M2 = -(xs**2 + ys**2)[:, :, np.newaxis]
    inv_M1T_M1 = np.linalg.inv(M1T @ M1)
    # inv_M1T_M1 = np.linalg.pinv(M1T @ M1)
    M1T_M2 = M1T @ M2
    ABC = (inv_M1T_M1 @ M1T_M2).T.squeeze()
    # A, B, C = ABC[0, 0], ABC[0, 1], ABC[0, 2]
    # print(ABC.shape)
    A, B, C = ABC[0], ABC[1], ABC[2]
    # Compute the circle centers (cx, cy) and radius (r)
    cx = -A / 2
    cy = -B / 2
    r = np.sqrt(cx**2 + cy**2 - C)  # Shape (num_frames,)
    xyrs[valid_id, :] = np.column_stack([cx, cy, r])
    d_center[valid_id, :] = np.sqrt((xs - cx[:, np.newaxis])**2 + (ys - cy[:, np.newaxis])**2)
    info = {
        "num_points": num_points,
        "num_frames": num_frames,
        "nun_valid_frames": num_frames_valid,
        "radii": d_center,
    }
    return xyrs, info

def lsm_for_ellipse(p):
    """
    Least squares fitting of ellipse

    """
    x, y = p[0], p[1]
    num_points = len(x)
    M1 = np.vstack([x*y, y**2, x, y, np.ones(num_points)]).T
    M2 = -x**2
    A, B, C, D, E = np.linalg.lstsq(M1, M2, rcond=None)[0]
    cx = (A*D - 2*B*C) / (4*B - A**2)
    cy = (A*C - 2*D) / (4*B - A**2)

    if abs(A/(1-B)) > 10**6: #45degのエラー回避のための応急処置
        a , b = 0, 0
        theta = 45
    else:
        theta = np.degrees(np.arctan(A/(1-B)) / 2)
        sin = np.sin(np.radians(theta))
        cos = np.cos(np.radians(theta))
        a = np.sqrt(
            (cx*cos + cy*sin)**2 - E*cos**2
                - ((cx*sin - cy*cos)**2 - E*sin**2)
                    *(sin**2 - B*cos**2) / (cos**2 - B*sin**2)
            )
        b = np.sqrt(
            (cx*sin - cy*cos)**2 - E*sin**2
                - ((cx*cos + cy*sin)**2 - E*cos**2)
                    *(cos**2 - B*sin**2) / (sin**2 - B*cos**2)
            )
    res = [cx, cy, a, b, theta]
    return res

#### calculate elliptical deformation
def calc_elliptical_deformation(points, points_zero):
    """
    evaluate elliptical deformation; roundness, major and minor axis, ect
    parameter:
    points: in the shape of (num_points*2, num_frames)
    points_zero: in the shape of (num_points*2) like [x0, y0, x1, y1, x2, y2, ...]

    """
    num_frames, num_points = int(len(points[0, :])), int(len(points[:, 0])/2)
    num_axes = int(num_points / 2)
    #### calclate diameters
    dias = np.linalg.norm(
        points.reshape(num_points, 2, num_frames)[:num_axes]
        - points.reshape(num_points, 2, num_frames)[num_axes:], axis=1
        )
    #### calclate zero diameters
    dias_zero = np.linalg.norm(
        points_zero.reshape(num_points, 2)[:num_axes]
        - points_zero.reshape(num_points, 2)[num_axes:], axis=1
        )
    delta_dias = dias - dias_zero.reshape(-1, 1)
    rnd = (np.amax(delta_dias, axis=0) - np.amin(delta_dias, axis=0)) / 2
    elp_id = np.array([np.argmin(delta_dias, axis=0), np.argmax(delta_dias, axis=0)])
    #calclate vector of diameters
    vcts = points[:num_points] - points[num_points:] # vectors for diameters
    thetas = np.arctan2(vcts.reshape(num_axes, 2, -1).swapaxes(0, 1).reshape(2, -1)[1, :],
                        vcts.reshape(num_axes, 2, -1).swapaxes(0, 1).reshape(2, -1)[0, :]).reshape(num_axes, num_frames) # angles of diameter vector
    thetas_elp_minor = thetas[elp_id[0], np.arange(num_frames)]
    thetas_elp_major = thetas[elp_id[1], np.arange(num_frames)]
    dthetas_elp_axes = abs(thetas_elp_major - thetas_elp_minor)
    dthetas_elp_axes = np.array([2*np.pi - x if x > np.pi else x for x in dthetas_elp_axes]) # adjust the values of angle
    elp_angle = np.degrees(np.vstack([thetas_elp_minor, thetas_elp_major, dthetas_elp_axes]))
    res = [dias, delta_dias, rnd, elp_id, elp_angle]
    return res

def calc_cumulative_angles(angles, threshold=300, unit='deg'):
    full_angle = 360 if unit=='deg' else 2*np.pi
    if np.ndim(angles) == 1:
        d_angles = angles[1:] - angles[:-1]
        flag_forward = np.hstack([0, np.where(d_angles<-threshold, 1, 0)])
        flag_backward = np.hstack([0, np.where(d_angles>threshold, -1, 0)])
        flag = np.cumsum(flag_forward) + np.cumsum(flag_backward)
        angles_corrected = angles + flag * full_angle
    elif np.ndim(angles) > 1:
        d_angles = angles[:, 1:] - angles[:, :-1]
        flag_forward = np.hstack([np.zeros((len(angles), 1)), np.where(d_angles<np.full((len(angles), 1), -threshold), 1, 0)])
        flag_backward = np.hstack([np.zeros((len(angles), 1)), np.where(d_angles>np.full((len(angles), 1), threshold), -1, 0)])
        flag = np.cumsum(flag_forward, axis=1) + np.cumsum(flag_backward, axis=1)
        angles_corrected = angles + flag * full_angle
    return angles_corrected, flag

if __name__ == '__main__':
    print('---- test ----')
    duration = 1
    num_frames = 1000
    rng = np.random.default_rng(seed=0)

    t = np.linspace(0, 1, num_frames, endpoint=False)
    r = 0.4
    x = np.zeros(num_frames)
    y = r * np.cos(6*t)
    z = r * np.sin(6*t)

    pos = np.linspace(0, 2*np.pi, 8, endpoint=False)

    noise_level = 0.05
    noise = rng.uniform(-1, 1, (num_frames, 2)) * r * noise_level
    p = np.column_stack([y, z]) + noise

    import mylogger
    logger = mylogger.MyLogger("myfitting_test")

    ps = np.zeros((num_frames, 8, 2))
    for i, theta in enumerate(pos):
        t = np.linspace(0, 1, num_frames, endpoint=False)
        r = 0.4
        y = r * np.cos(6*t+theta)
        z = r * np.sin(6*t+theta)
        ps[:, i, 0] = y
        ps[:, i, 1] = z
    # ps[20:80, 0, 0] = np.nan
    # print(ps[0, :, :])
    noise_level = 0.05
    noise = rng.uniform(-1, 1, (num_frames, 8, 2)) * r * noise_level
    ps += noise
    print(ps.shape)
    # res = lsm_for_circles(ps)
    # print(res)

    logger.measure_time("lsm_for_circle", 's')
    res1 = []
    for i in range(num_frames):
        _r = lsm_for_circle(ps[i, :, :])[0]
        res1.append(_r)
    logger.measure_time("lsm_for_circle", 'e')
    res1 = np.array(res1)

    logger.measure_time("lsm_for_circles", 's')
    res2 = lsm_for_circles(ps)[0]
    logger.measure_time("lsm_for_circles", 'e')

    err = res2 - res1
    logger.info(f"lsm2 - lsm1:\n{err}")



    if 1:
        fig, ax = plt.subplots(figsize=(20, 15))
        # ax.set_aspect(1)
        ax.grid()

        # ax.plot(p[:, 0], p[:, 1])
        # ax.plot(xfit, yfig, c='r', lw=1)

        ax.plot(t, res1[:, 2], c='b', lw=0.2)
        ax.plot(t, res2[:, 2], c='r', lw=0.2)
        ax.plot(t, err[:, 2], c='g', lw=1)
        ax.set(ylim=(-0.2, 0.5))

        plt.show()

