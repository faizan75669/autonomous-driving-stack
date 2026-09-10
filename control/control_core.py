"""ROS-independent lateral-control equations used by the ROS 2 nodes."""
from __future__ import annotations

import math
import numpy as np


def pure_pursuit_steering(target_x: float, target_y: float, wheelbase: float,
                          lookahead: float, max_steer: float = 0.5) -> float:
    if lookahead <= 1e-9:
        return 0.0
    alpha = math.atan2(target_y, target_x)
    delta = math.atan2(2.0 * wheelbase * math.sin(alpha), lookahead)
    return float(np.clip(delta, -max_steer, max_steer))


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def project_to_segment(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    d2 = vx * vx + vy * vy
    if d2 < 1e-12:
        return ax, ay
    t = float(np.clip(((px - ax) * vx + (py - ay) * vy) / d2, 0.0, 1.0))
    return ax + t * vx, ay + t * vy


def stanley_steering(path: np.ndarray, speed: float, wheelbase: float,
                     gain: float, softening: float, max_steer: float) -> float:
    if len(path) < 2:
        return 0.0
    fx, fy = wheelbase, 0.0
    idx = int(np.argmin(np.linalg.norm(path - np.array([fx, fy]), axis=1)))
    j = min(idx + 1, len(path) - 1)
    if j == idx:
        j = max(0, idx - 1)
    ax, ay = path[idx]
    bx, by = path[j]
    cx, cy = project_to_segment(fx, fy, ax, ay, bx, by)
    tx, ty = bx - ax, by - ay
    length = math.hypot(tx, ty)
    if length < 1e-9:
        return 0.0
    heading_error = normalize_angle(math.atan2(ty, tx))
    cross_track = ((fx - cx) * (-ty) + (fy - cy) * tx) / length
    correction = math.atan2(gain * cross_track, abs(speed) + softening)
    return float(np.clip(normalize_angle(heading_error + correction), -max_steer, max_steer))
