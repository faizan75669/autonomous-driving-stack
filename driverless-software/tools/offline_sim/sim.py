"""
Offline closed-loop simulator for the driverless stack.

    EUFS track csv -> /ground_truth/cones -> planner node -> /trajectory
                   -> controller node -> /cmd -> car model -> next frame

The car model is a kinematic bicycle with the EUFS "DryTrack" limits
(wheelbase 1.58 m, steering +-0.52 rad, full lock in 1 s, acc -10..3 m/s^2).
Cones are "seen" the same way the EUFS ground-truth cone plugin does it:
coloured cones within 15 m and +-60 deg of the car, in the car frame.

It is not Gazebo, but it runs the real node code, so a broken planner or
controller shows up here before we waste time in the full simulator.

Usage:
    python3 sim.py small_track peanut --planner delaunay_planner --controller pure_pursuit
"""

import argparse
import csv
import importlib
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import fake_ros  # noqa: E402

fake_ros.install()
for pkg in ("planning", "control", "slam", "state_estimation"):
    sys.path.insert(0, os.path.join(ROOT, pkg))

TRACK_DIR = os.path.join(HERE, "tracks")
LOOP_TRACKS = ["small_track", "peanut", "bone", "boa_constrictor", "its_a_mess", "rectangle",
               "garden_light", "comp_2021", "rand", "hairpins_increasing_difficulty"]

PLANNERS = {"delaunay_planner": "planning.delaunay_planner_node",
            "midpoint_planner": "planning.midpoint_planner_node",
            "rrt_planner": "planning.rrt_planner_node"}
CONTROLLERS = {"pure_pursuit": ("control.pure_pursuit_node", "PurePursuit"),
               "stanley": ("control.stanley_node", "Stanley")}


# ----------------------------------------------------------------- world
def load_track(name):
    cones = {"blue": [], "yellow": [], "orange": [], "big_orange": []}
    start = (0.0, 0.0, 0.0)
    with open(os.path.join(TRACK_DIR, name + ".csv")) as f:
        for row in csv.DictReader(f):
            tag = row["tag"]
            if tag == "car_start":
                start = (float(row["x"]), float(row["y"]), float(row["direction"]))
            elif tag in cones:
                cones[tag].append([float(row["x"]), float(row["y"])])
    return {k: np.array(v).reshape(-1, 2) for k, v in cones.items()}, start


def to_car_frame(points, x, y, yaw):
    d = points - [x, y]
    c, s = math.cos(yaw), math.sin(yaw)
    return np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]])


def visible_cones(cones, pose, view=15.0, fov=2.09):
    out = {}
    for k, v in cones.items():
        if len(v) == 0:
            out[k] = v
            continue
        pc = to_car_frame(v, *pose)
        r = np.hypot(pc[:, 0], pc[:, 1])
        a = np.abs(np.arctan2(pc[:, 1], pc[:, 0]))
        out[k] = pc[(r > 0.5) & (r < view) & (a < fov / 2)]
    return out


def cones_msg(obs):
    M = fake_ros.Msg
    msg = sys.modules["eufs_msgs.msg"].ConeArrayWithCovariance()
    for k in ("blue", "yellow", "orange", "big_orange"):
        setattr(msg, k + "_cones", [M(point=M(x=float(a), y=float(b), z=0.0)) for a, b in obs[k]])
    return msg


class Car(object):
    """Kinematic bicycle at the EUFS DryTrack limits."""
    L, lr = 1.58, 0.711

    def __init__(self, x, y, yaw):
        self.x, self.y, self.yaw, self.v, self.delta = x, y, yaw, 0.0, 0.0

    def step(self, acc, delta_cmd, dt):
        delta_cmd = float(np.clip(delta_cmd, -0.52, 0.52))
        self.delta += float(np.clip(delta_cmd - self.delta, -1.04 * dt, 1.04 * dt))
        acc = float(np.clip(acc, -10.0, 3.0)) - self.v ** 2 / 225.0      # + aero drag
        beta = math.atan(self.lr / self.L * math.tan(self.delta))
        self.x += self.v * math.cos(self.yaw + beta) * dt
        self.y += self.v * math.sin(self.yaw + beta) * dt
        self.yaw_rate = self.v / self.L * math.cos(beta) * math.tan(self.delta)
        self.yaw += self.yaw_rate * dt
        self.v = max(0.0, self.v + acc * dt)


# ----------------------------------------------------------------- run
def run(track, planner="delaunay_planner", controller="pure_pursuit", T=300.0, dt=0.02,
        record_every=5, params=None, seed=0):
    """Drive one lap. Returns a dict with the lap result and recorded frames."""
    fake_ros.reset()
    np.random.seed(seed)
    fake_ros.PARAMS["seed"] = seed       # used by the RRT planner
    if params:
        fake_ros.PARAMS.update(params)
    mod = importlib.reload(importlib.import_module(PLANNERS[planner]))
    planner_node = mod.Planner(planner)
    cmod, cname = CONTROLLERS[controller]
    controller_node = getattr(importlib.reload(importlib.import_module(cmod)), cname)()

    cones, start = load_track(track)
    car = Car(*start)
    cmd = {"acc": 0.0, "delta": 0.0}

    def on_cmd(m):
        cmd["acc"], cmd["delta"] = m.drive.acceleration, m.drive.steering_angle

    fake_ros.BUS["/cmd"].append(on_cmd)
    boundary = np.vstack([cones["blue"], cones["yellow"]])
    start_xy = np.array(start[:2])
    frames, left_start, lap, t = [], False, False, 0.0
    hit_cones = set()
    for i in range(int(T / dt)):
        state = fake_ros.Msg()
        state.twist.twist.linear.x = car.v
        fake_ros.publish("/ground_truth/state", state)
        fake_ros.LAST.pop("/trajectory", None)
        obs = visible_cones(cones, (car.x, car.y, car.yaw))
        fake_ros.publish("/ground_truth/cones", cones_msg(obs))

        car.step(cmd["acc"], cmd["delta"], dt)
        t += dt
        fake_ros.SIM["t"] = t
        dists = np.linalg.norm(boundary - [car.x, car.y], axis=1)
        clearance = float(np.min(dists))
        # car is 1.4 m wide: a cone closer than 0.5 m to its centre is under a wheel
        hit_cones.update(np.nonzero(dists < 0.5)[0].tolist())
        if i % record_every == 0:
            traj = fake_ros.LAST.get("/trajectory")
            path = np.array([[w.position.x, w.position.y] for w in traj.waypoints]).reshape(-1, 2) \
                if traj is not None else np.zeros((0, 2))
            c, s = math.cos(car.yaw), math.sin(car.yaw)
            path_map = path @ np.array([[c, s], [-s, c]]) + [car.x, car.y] if len(path) else path
            frames.append(dict(t=t, x=car.x, y=car.y, yaw=car.yaw, v=car.v, delta=car.delta,
                               clearance=clearance, path=path_map))
        dist_start = math.hypot(car.x - start_xy[0], car.y - start_xy[1])
        if dist_start > 20.0:
            left_start = True
        if left_start and dist_start < 3.0:
            lap = True
            break
        if clearance > 6.0 and i > 100:     # way off the track
            break
    return dict(track=track, planner=planner, controller=controller, lap=lap, time=t,
                cones_hit=len(hit_cones),
                min_clearance=min(f["clearance"] for f in frames),
                mean_speed=float(np.mean([f["v"] for f in frames])),
                cones=cones, start=start, frames=frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tracks", nargs="*", default=LOOP_TRACKS)
    ap.add_argument("--planner", default="delaunay_planner", choices=sorted(PLANNERS))
    ap.add_argument("--controller", default="pure_pursuit", choices=sorted(CONTROLLERS))
    ap.add_argument("--max-speed", type=float, default=None)
    a = ap.parse_args()
    params = {"max_speed": a.max_speed} if a.max_speed else None
    print(f"planner={a.planner} controller={a.controller}")
    print(f"{'track':32s} {'lap':>4s} {'time[s]':>8s} {'v_avg':>6s} {'min clear[m]':>13s} {'cones hit':>10s}")
    for track in a.tracks:
        r = run(track, a.planner, a.controller, params=params)
        print(f"{track:32s} {'yes' if r['lap'] else 'NO':>4s} {r['time']:8.1f} {r['mean_speed']:6.2f} "
              f"{r['min_clearance']:13.2f} {r['cones_hit']:10d}", flush=True)


if __name__ == "__main__":
    main()
