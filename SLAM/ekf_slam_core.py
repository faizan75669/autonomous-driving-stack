"""Pure NumPy EKF-SLAM core for 2-D cone landmarks.

State: [x, y, yaw, lx1, ly1, ...]. Observations are range/bearing pairs
expressed in the vehicle/body frame. This module is ROS-independent so the
math can be unit-tested separately from message plumbing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _motion_jacobian(state: np.ndarray, v: float, yaw_rate: float, dt: float) -> np.ndarray:
    n = state.size
    g = np.eye(n)
    yaw = state[2]
    g[0, 2] = -v * math.sin(yaw) * dt
    g[1, 2] = v * math.cos(yaw) * dt
    return g


@dataclass
class EKFConfig:
    process_std_xy: float = 0.05
    process_std_yaw: float = math.radians(2.0)
    range_std: float = 0.15
    bearing_std: float = math.radians(3.0)
    association_gate: float = 9.21  # chi-square gate, 2 DoF, ~99%


class EKFSLAMCore:
    def __init__(self, config: Optional[EKFConfig] = None) -> None:
        self.cfg = config or EKFConfig()
        self.state = np.zeros(3, dtype=float)
        self.covariance = np.diag([0.1, 0.1, math.radians(5.0) ** 2]).astype(float)

    @property
    def landmark_count(self) -> int:
        return max(0, (self.state.size - 3) // 2)

    def predict(self, v: float, yaw_rate: float, dt: float) -> None:
        if dt <= 0.0:
            return
        yaw = self.state[2]
        self.state[0] += v * math.cos(yaw) * dt
        self.state[1] += v * math.sin(yaw) * dt
        self.state[2] = wrap_angle(self.state[2] + yaw_rate * dt)

        G = _motion_jacobian(self.state, v, yaw_rate, dt)
        q = np.diag([
            self.cfg.process_std_xy ** 2 * dt,
            self.cfg.process_std_xy ** 2 * dt,
            self.cfg.process_std_yaw ** 2 * dt,
        ])
        Q = np.zeros_like(self.covariance)
        Q[:3, :3] = q
        self.covariance = G @ self.covariance @ G.T + Q
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def _measurement(self, landmark_index: int) -> tuple[np.ndarray, np.ndarray]:
        i = 3 + 2 * landmark_index
        dx = self.state[i] - self.state[0]
        dy = self.state[i + 1] - self.state[1]
        r2 = max(dx * dx + dy * dy, 1e-12)
        r = math.sqrt(r2)
        zhat = np.array([r, wrap_angle(math.atan2(dy, dx) - self.state[2])])

        n = self.state.size
        H = np.zeros((2, n))
        H[0, 0] = -dx / r
        H[0, 1] = -dy / r
        H[0, i] = dx / r
        H[0, i + 1] = dy / r
        H[1, 0] = dy / r2
        H[1, 1] = -dx / r2
        H[1, 2] = -1.0
        H[1, i] = -dy / r2
        H[1, i + 1] = dx / r2
        return zhat, H

    def add_landmark(self, measurement: np.ndarray) -> int:
        r, b = float(measurement[0]), float(measurement[1])
        yaw = self.state[2]
        lx = self.state[0] + r * math.cos(yaw + b)
        ly = self.state[1] + r * math.sin(yaw + b)
        self.state = np.concatenate((self.state, [lx, ly]))
        old_n = self.covariance.shape[0]
        new_cov = np.zeros((old_n + 2, old_n + 2))
        new_cov[:old_n, :old_n] = self.covariance
        new_cov[-2:, -2:] = np.eye(2) * max(self.cfg.range_std ** 2, 0.01)
        self.covariance = new_cov
        return self.landmark_count - 1

    def update(self, measurement: np.ndarray, landmark_index: int) -> float:
        zhat, H = self._measurement(landmark_index)
        innovation = np.array([
            float(measurement[0]) - zhat[0],
            wrap_angle(float(measurement[1]) - zhat[1]),
        ])
        R = np.diag([self.cfg.range_std ** 2, self.cfg.bearing_std ** 2])
        S = H @ self.covariance @ H.T + R
        nis = float(innovation.T @ np.linalg.solve(S, innovation))
        if nis > self.cfg.association_gate:
            return nis
        K = self.covariance @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ innovation
        self.state[2] = wrap_angle(self.state[2])
        I = np.eye(self.state.size)
        self.covariance = (I - K @ H) @ self.covariance
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        return nis

    def associate(self, measurement: np.ndarray) -> Optional[int]:
        if self.landmark_count == 0:
            return None
        best_idx, best_nis = None, float("inf")
        for i in range(self.landmark_count):
            zhat, H = self._measurement(i)
            innovation = np.array([
                float(measurement[0]) - zhat[0],
                wrap_angle(float(measurement[1]) - zhat[1]),
            ])
            R = np.diag([self.cfg.range_std ** 2, self.cfg.bearing_std ** 2])
            S = H @ self.covariance @ H.T + R
            nis = float(innovation.T @ np.linalg.solve(S, innovation))
            if nis < best_nis:
                best_nis, best_idx = nis, i
        return best_idx if best_nis <= self.cfg.association_gate else None
