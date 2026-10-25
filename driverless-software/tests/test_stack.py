"""
Tests for the driverless stack. They run without ROS:

    cd driverless-software
    python3 -m pytest tests -q

The real node classes are used, with tools/offline_sim/fake_ros.py standing
in for rclpy and the message types.
"""

import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools", "offline_sim"))
import sim  # noqa: E402  (installs fake ROS + package paths)
import fake_ros  # noqa: E402


# ------------------------------------------------------------------ helpers
def straight_track(length=30.0, spacing=3.0, width=3.5):
    x = np.arange(2.0, length, spacing)
    blue = np.column_stack([x, np.full_like(x, width / 2)])
    yellow = np.column_stack([x, np.full_like(x, -width / 2)])
    return blue, yellow


def plan_once(planner, blue, yellow):
    fake_ros.reset()
    fake_ros.PARAMS["seed"] = 0          # RRT is random, keep the test repeatable
    mod = __import__(sim.PLANNERS[planner], fromlist=["Planner"])
    node = mod.Planner(planner)
    obs = {"blue": blue, "yellow": yellow, "orange": np.zeros((0, 2)), "big_orange": np.zeros((0, 2))}
    fake_ros.publish("/ground_truth/cones", sim.cones_msg(obs))
    traj = fake_ros.LAST.get("/trajectory")
    assert traj is not None, planner + " published nothing"
    return np.array([[w.position.x, w.position.y] for w in traj.waypoints])


# ------------------------------------------------------------------ planning
@pytest.mark.parametrize("planner", sorted(sim.PLANNERS))
def test_planner_follows_straight(planner):
    blue, yellow = straight_track()
    path = plan_once(planner, blue, yellow)
    assert len(path) >= 5
    assert np.linalg.norm(path[0]) < 0.5, "path should start at the car"
    assert path[-1][0] > 10.0, "path should go forward"
    assert np.max(np.abs(path[:, 1])) < 0.6, "path should stay near the centre"


def test_delaunay_planner_ignores_message_order():
    """The old bug: blue[i] was paired with yellow[i]. Shuffling must not matter."""
    blue, yellow = straight_track()
    rng = np.random.default_rng(1)
    a = plan_once("delaunay_planner", blue, yellow)
    b = plan_once("delaunay_planner", blue[rng.permutation(len(blue))], yellow[rng.permutation(len(yellow))])
    n = min(len(a), len(b))
    assert np.max(np.abs(a[:n] - b[:n])) < 1e-6


def test_delaunay_planner_left_turn():
    """Quarter circle to the left: the path must curve left and stay between the cones."""
    th = np.linspace(-np.pi / 2, 0.0, 8)
    centre, r = np.array([0.0, 8.0]), 8.0
    blue = centre + np.column_stack([(r - 1.75) * np.cos(th), (r - 1.75) * np.sin(th)])
    yellow = centre + np.column_stack([(r + 1.75) * np.cos(th), (r + 1.75) * np.sin(th)])
    path = plan_once("delaunay_planner", blue, yellow)
    assert path[-1][1] > 3.0
    d = np.linalg.norm(path[3:] - centre, axis=1)
    assert np.all(np.abs(d - r) < 1.0)


# ------------------------------------------------------------------ control
@pytest.mark.parametrize("controller", sorted(sim.CONTROLLERS))
def test_controller_steers_towards_path(controller):
    fake_ros.reset()
    mod, cls = sim.CONTROLLERS[controller]
    node = getattr(__import__(mod, fromlist=[cls]), cls)()
    out = {}
    fake_ros.BUS["/cmd"].append(lambda m: out.update(steer=m.drive.steering_angle, acc=m.drive.acceleration))
    for side in (+1.0, -1.0):          # path 1 m to the left, then to the right
        msg = sys.modules["eufs_msgs.msg"].WaypointArrayStamped()
        for x in np.arange(0.0, 15.0, 0.25):
            w = fake_ros.Msg()
            w.position.x, w.position.y = float(x), side * min(x / 3.0, 1.0)
            msg.waypoints.append(w)
        fake_ros.publish("/trajectory", msg)
        assert out["steer"] * side > 0.0, f"{controller} steers the wrong way"
        assert out["acc"] > 0.0, "car at standstill should accelerate"


# ------------------------------------------------------------------ closed loop
def test_full_lap_small_track():
    r = sim.run("small_track")
    assert r["lap"]
    assert r["cones_hit"] == 0
    assert r["min_clearance"] > 1.0


# ------------------------------------------------------------------ EKF
def test_ekf_tracks_noisy_odometry():
    from state_estimation.ekf_node import EKF
    rng = np.random.default_rng(0)
    dt, v, w = 0.05, 3.0, 0.2
    ekf = EKF(dt)
    truth = np.zeros(3)
    ekf.x = np.array([[0.0], [0.0], [0.0], [v]])
    raw_err, ekf_err = [], []
    for _ in range(400):
        truth = truth + [v * dt * math.cos(truth[2]), v * dt * math.sin(truth[2]), w * dt]
        z = np.array([[truth[0]], [truth[1]], [truth[2]]]) + np.array([[0.5], [0.5], [0.03]]) * rng.normal(size=(3, 1))
        u = np.array([[v], [w]]) + np.array([[0.1], [0.01]]) * rng.normal(size=(2, 1))
        x = ekf.step(z, u)
        raw_err.append(np.hypot(z[0, 0] - truth[0], z[1, 0] - truth[1]))
        ekf_err.append(np.hypot(x[0, 0] - truth[0], x[1, 0] - truth[1]))
    assert np.mean(ekf_err[50:]) < 0.5 * np.mean(raw_err[50:]), "EKF should beat the raw measurement"


# ------------------------------------------------------------------ mapping
def test_cone_mapping_rebuilds_track():
    from slam.cone_mapping_node import ConeMapper
    r = sim.run("small_track", record_every=1)
    mapper = ConeMapper()
    for f in r["frames"][::2]:
        obs = sim.visible_cones(r["cones"], (f["x"], f["y"], f["yaw"]))
        mapper.add_frame({"blue": obs["blue"], "yellow": obs["yellow"], "orange": obs["big_orange"]},
                         (f["x"], f["y"], f["yaw"]))
    m = mapper.update_map()
    for colour in ("blue", "yellow"):
        true = r["cones"][colour]
        err = np.array([np.min(np.linalg.norm(m[colour] - c, axis=1)) for c in true])
        assert np.mean(err < 0.3) > 0.9, f"{colour}: most cones should be mapped within 30 cm"
    assert mapper.laps == 1, "one lap driven, one lap counted"


# ------------------------------------------------------------------ ROS node wiring
def _car_state(x, y, yaw, v, w):
    m = sys.modules["eufs_msgs.msg"].CarState()
    m.pose.pose.position.x, m.pose.pose.position.y = x, y
    o = m.pose.pose.orientation
    o.x, o.y, o.z, o.w = 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)
    m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.angular.z = v, 0.0, w
    return m


def test_ekf_node_publishes_filtered_state():
    from state_estimation.ekf_node import StateEstimation
    fake_ros.reset()
    node = StateEstimation()
    x = y = yaw = 0.0
    for i in range(100):
        fake_ros.SIM["t"] = i * 0.05
        x, y, yaw = x + 3.0 * 0.05 * math.cos(yaw), y + 3.0 * 0.05 * math.sin(yaw), yaw + 0.1 * 0.05
        fake_ros.publish("/odometry_integration/car_state", _car_state(x, y, yaw, 3.0, 0.1))
        imu = fake_ros.Msg()
        imu.orientation.x, imu.orientation.y = 0.0, 0.0
        imu.orientation.z, imu.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        fake_ros.publish("/imu/data", imu)
        node.spin_timers()
    out = fake_ros.LAST["/odometry_filtered/car_state"]
    assert abs(out.pose.pose.position.x - x) < 0.5
    assert abs(out.pose.pose.position.y - y) < 0.5


def test_cone_mapping_node_publishes_map():
    from slam.cone_mapping_node import ConeMappingNode
    fake_ros.reset()
    node = ConeMappingNode()
    blue, yellow = straight_track(length=12.0)     # node ignores cones further than 12 m
    for i in range(60):
        fake_ros.SIM["t"] = i * 0.1
        fake_ros.publish("/odometry_filtered/car_state", _car_state(0.0, 0.0, 0.0, 0.0, 0.0))
        obs = {"blue": blue, "yellow": yellow, "orange": np.zeros((0, 2)), "big_orange": np.zeros((0, 2))}
        fake_ros.publish("/ground_truth/cones", sim.cones_msg(obs))
        node.spin_timers()
    m = fake_ros.LAST["/slam/map"]
    assert len(m.blue_cones) >= len(blue) - 1
    assert len(m.yellow_cones) >= len(yellow) - 1
