import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "SLAM"))
sys.path.insert(0, str(ROOT / "planning"))
sys.path.insert(0, str(ROOT / "control"))

from ekf_slam_core import EKFSLAMCore
from rrt_star import RRTStar
from control_core import pure_pursuit_steering, stanley_steering


def test_ekf_add_landmark_and_update():
    ekf = EKFSLAMCore()
    idx = ekf.add_landmark(np.array([5.0, 0.0]), "blue")
    assert idx == 0
    assert ekf.landmark_count == 1
    assert ekf.state.shape == (5,)
    nis = ekf.update(np.array([5.0, 0.0]), idx)
    assert nis <= ekf.cfg.association_gate
    assert np.all(np.isfinite(ekf.state))
    assert np.all(np.linalg.eigvalsh(ekf.covariance) >= -1e-8)


def test_ekf_prediction_moves_forward():
    ekf = EKFSLAMCore()
    ekf.predict(v=2.0, yaw_rate=0.0, dt=0.5)
    assert math.isclose(ekf.state[0], 1.0, rel_tol=1e-6)
    assert math.isclose(ekf.state[1], 0.0, abs_tol=1e-6)


def test_rrt_collision_check():
    planner = RRTStar([0.0, 0.0], [4.0, 0.0], obstacles=[(np.array([2.0, 0.0]), 0.5)])
    assert not planner.collision_free([0.0, 0.0], [4.0, 0.0])
    assert planner.collision_free([0.0, 1.0], [4.0, 1.0])


def test_rrt_direct_path_when_clear():
    planner = RRTStar([0.0, 0.0], [2.0, 0.0], obstacles=[], max_iterations=100)
    path = planner.plan()
    assert len(path) >= 2
    assert np.allclose(path[0], [0.0, 0.0])
    assert np.allclose(path[-1], [2.0, 0.0])


def test_pure_pursuit_straight_path_is_zero():
    assert abs(pure_pursuit_steering(4.0, 0.0, 1.53, 4.0)) < 1e-9


def test_stanley_straight_path_is_zero():
    path = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    assert abs(stanley_steering(path, 2.0, 1.53, 2.5, 1.0, 0.5)) < 1e-9
