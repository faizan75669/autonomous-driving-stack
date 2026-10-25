"""
Evaluate state estimation + cone mapping together, closed loop.

The car is driven by the normal stack (planner + controller). On top of
that we simulate the sensors the localisation/mapping side gets in EUFS:

  wheel odometry   /odometry_integration/car_state  integrated speed + yaw rate,
                   with a speed scale error and a gyro bias -> it DRIFTS
  IMU              /imu/data   yaw with small noise
  GPS (optional)   /gps        lat/lon with 0.5 m noise, 10 Hz
  cones            noisy cone detections in the car frame, 10 Hz, using the
                   EUFS perception noise model (grows with distance)

and run the REAL EKF node and the REAL cone mapping node on them.

    python3 slam_eval.py                       # default tracks, with and without GPS
    python3 slam_eval.py peanut --laps 2 --no-gps
"""

import argparse
import math
import sys
import time

import numpy as np

import sim
import fake_ros
from state_estimation.geonav_conversions import xy2ll

COLOURS = ("blue", "yellow")


class Sensors(object):
    """Simulated car sensors. All noise values are parameters."""

    def __init__(self, start, rng, speed_scale_err=0.01, gyro_bias=0.004, speed_noise=0.05,
                 gyro_noise=0.01, imu_yaw_noise=0.01, gps_noise=0.5, use_gps=True,
                 lat0=55.94448258581324, lon0=3.1864915020968305):
        self.rng = rng
        self.odom = list(start)                  # integrated odometry pose x, y, yaw
        self.prev_yaw = start[2]
        self.scale = 1.0 + speed_scale_err
        self.gyro_bias = gyro_bias
        self.speed_noise, self.gyro_noise = speed_noise, gyro_noise
        self.imu_yaw_noise, self.gps_noise = imu_yaw_noise, gps_noise
        self.use_gps = use_gps
        self.lat0, self.lon0 = lat0, lon0
        self.next_gps = 0.0
        self.next_cones = 0.0

    def step(self, t, car, cones, dt):
        rng = self.rng
        yaw_rate = sim.wrap_angle(car.yaw - self.prev_yaw) / dt
        self.prev_yaw = car.yaw
        v_meas = car.v * self.scale + rng.normal(0, self.speed_noise) * (car.v > 0.05)
        w_meas = yaw_rate + self.gyro_bias + rng.normal(0, self.gyro_noise)
        # odometry integration (this is what drifts)
        b = math.atan(sim.Car.lr / sim.Car.L * math.tan(car.delta))
        self.odom[0] += v_meas * dt * math.cos(self.odom[2] + b)
        self.odom[1] += v_meas * dt * math.sin(self.odom[2] + b)
        self.odom[2] += w_meas * dt

        odom = fake_ros.Msg()
        odom.pose.pose.position.x, odom.pose.pose.position.y = self.odom[0], self.odom[1]
        beta = math.atan(sim.Car.lr / sim.Car.L * math.tan(car.delta))      # slip angle at the COG
        odom.twist.twist.linear.x = v_meas * math.cos(beta)
        odom.twist.twist.linear.y = v_meas * math.sin(beta)
        odom.twist.twist.angular.z = w_meas
        fake_ros.publish("/odometry_integration/car_state", odom)

        imu = fake_ros.Msg()
        yaw = car.yaw + rng.normal(0, self.imu_yaw_noise)
        imu.orientation.x, imu.orientation.y = 0.0, 0.0
        imu.orientation.z, imu.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        fake_ros.publish("/imu/data", imu)

        if self.use_gps and t >= self.next_gps:
            self.next_gps = t + 0.1
            gx, gy = car.x + rng.normal(0, self.gps_noise), car.y + rng.normal(0, self.gps_noise)
            lat, lon = xy2ll(-gx, gy, self.lat0, self.lon0)   # inverse of the EKF's conversion
            gps = fake_ros.Msg(latitude=lat, longitude=-lon, altitude=0.0)
            fake_ros.publish("/gps", gps)

        if t >= self.next_cones:
            self.next_cones = t + 0.1
            obs = sim.visible_cones(cones, (car.x, car.y, car.yaw))
            noisy = {k: eufs_cone_noise(v, rng) for k, v in obs.items()}
            fake_ros.publish("/slam_eval/cones", sim.cones_msg(noisy))


def eufs_cone_noise(points, rng, a=0.01, b=0.5, camera_share=0.4, lidar=0.03):
    """Same model as the EUFS cone plugin: noise along the line of sight grows
    exponentially with distance, sideways noise is 5x smaller."""
    if len(points) == 0:
        return points
    d = np.hypot(points[:, 0], points[:, 1])
    cam = a * np.minimum(70.0, np.exp(b * d))
    along = camera_share * cam + (1 - camera_share) * lidar
    across = camera_share * cam / 5.0 + (1 - camera_share) * lidar
    u = points / d[:, None]
    perp = np.column_stack([-u[:, 1], u[:, 0]])
    return points + u * (rng.normal(size=len(d)) * along)[:, None] + perp * (rng.normal(size=len(d)) * across)[:, None]


def map_metrics(mapped, true_cones, match_dist=0.5, false_dist=1.0):
    errs, missed, false = [], 0, 0
    for c in COLOURS:
        m, tr = mapped[c], true_cones[c]
        for p in tr:
            d = np.min(np.linalg.norm(m - p, axis=1)) if len(m) else np.inf
            if d < match_dist:
                errs.append(d)
            else:
                missed += 1
        for p in m:
            if np.min(np.linalg.norm(tr - p, axis=1)) > false_dist:
                false += 1
    n_true = sum(len(true_cones[c]) for c in COLOURS)
    return dict(mean_err=float(np.mean(errs)) if errs else np.nan,
                max_err=float(np.max(errs)) if errs else np.nan,
                matched=len(errs), n_true=n_true, missed=missed, false=false)


def aligned_metrics(mapped, true_cones):
    """Map error after the best rigid alignment (shape of the map, ignoring a global shift/rotation)."""
    from slam.cone_mapping_node import kabsch_2d
    src, dst = [], []
    for c in COLOURS:
        m, tr = mapped[c], true_cones[c]
        if len(m) == 0:
            continue
        for p in m:
            d = np.linalg.norm(tr - p, axis=1)
            j = int(np.argmin(d))
            if d[j] < 1.5:
                src.append(p)
                dst.append(tr[j])
    if len(src) < 3:
        return dict(aligned_matched=0, aligned_mean_err=np.nan)
    src, dst = np.array(src), np.array(dst)
    th, t = kabsch_2d(src, dst)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    moved = {c: (mapped[c] @ R.T + t) if len(mapped[c]) else mapped[c] for c in COLOURS}
    m = map_metrics(moved, true_cones)
    return dict(aligned_matched=m["matched"], aligned_mean_err=m["mean_err"])


def evaluate(track, laps=1, use_gps=True, seed=0, verbose=False, mapping_params=None):
    rng = np.random.default_rng(seed)
    state = {}
    pose_err, slam_err, ekf_hist = [], [], []

    def on_step(t, car, cones):
        if "sensors" not in state:          # first step: build nodes now, after the planner exists
            from state_estimation.ekf_node import StateEstimation
            from slam.cone_mapping_node import ConeMappingNode
            saved = dict(fake_ros.PARAMS)
            fake_ros.PARAMS["cones_topic"] = "/slam_eval/cones"
            fake_ros.PARAMS.update(mapping_params or {})
            state["ekf"] = StateEstimation()
            state["map"] = ConeMappingNode()
            fake_ros.PARAMS.clear()
            fake_ros.PARAMS.update(saved)
            state["sensors"] = Sensors((car.x, car.y, car.yaw), rng, use_gps=use_gps)
            state["update_times"] = []
            orig = state["map"].mapper.update_map

            def timed():
                t0 = time.perf_counter()
                out = orig()
                state["update_times"].append(time.perf_counter() - t0)
                return out
            state["map"].mapper.update_map = timed
        state["cone_pose"] = (car.x, car.y)
        state["sensors"].step(t, car, cones, 0.02)
        state["ekf"].spin_timers()
        state["map"].spin_timers()
        est = fake_ros.LAST.get("/odometry_filtered/car_state")
        if est is not None:
            ex, ey = est.pose.pose.position.x, est.pose.pose.position.y
            pose_err.append(math.hypot(ex - car.x, ey - car.y))
            sp = fake_ros.LAST.pop("/slam/pose", None)
            sx, sy = (sp.pose.pose.position.x, sp.pose.pose.position.y) if sp is not None else (ex, ey)
            if sp is not None:            # error at the moment the cones were seen
                slam_err.append(math.hypot(sx - state["cone_pose"][0], sy - state["cone_pose"][1]))
            ekf_hist.append((t, car.x, car.y, ex, ey, sx, sy))

    r = sim.run(track, laps=laps, on_step=on_step, record_every=10)
    node = state["map"]
    final_map = node.mapper.update_map()
    lc_time = node.mapper.loop_closure_time
    res = dict(track=track, gps=use_gps, laps_driven=r["laps"], lap_time=r["time"],
               laps_counted=node.mapper.laps, loop_closure=lc_time is not None, loop_closure_time=lc_time,
               ekf_err_mean=float(np.mean(pose_err)), ekf_err_max=float(np.max(pose_err)),
               slam_err_mean=float(np.mean(slam_err)), slam_err_max=float(np.max(slam_err)),
               update_ms_max=1000 * max(state["update_times"]) if state["update_times"] else 0.0,
               map=final_map, cones=r["cones"], ekf_hist=np.array(ekf_hist), frames=r["frames"],
               detections=sum(len(node.mapper.detections[c]) for c in node.mapper.detections))
    res.update(map_metrics(final_map, r["cones"]))
    res.update(aligned_metrics(final_map, r["cones"]))
    # true number of start-line passes (same rule as the mapper, but with the real car path)
    line = r["cones"]["big_orange"].mean(axis=0)
    passes, armed = 0, False
    for (_, x, y, *_rest) in ekf_hist:
        d = math.hypot(x - line[0], y - line[1])
        if d > 15.0:
            armed = True
        elif armed and d < 4.0:
            passes, armed = passes + 1, False
    res["line_passes"] = passes
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", nargs="*", default=["small_track", "peanut", "bone", "boa_constrictor",
                                                  "rectangle", "its_a_mess", "rand"])
    ap.add_argument("--laps", type=int, default=2)
    ap.add_argument("--no-gps", action="store_true")
    ap.add_argument("--gps-only", action="store_true")
    a = ap.parse_args()
    modes = [True] if a.gps_only else [False] if a.no_gps else [False, True]
    print(f"{'track':18s} {'gps':>4s} {'laps':>5s} {'lap1 [s]':>8s} {'EKF err':>13s} {'SLAM pose err':>14s} "
          f"{'map err':>13s} {'matched':>9s} {'false':>6s} {'loop cl.':>9s} {'upd ms':>7s}")
    for gps in modes:
        for t in a.tracks:
            r = evaluate(t, laps=a.laps, use_gps=gps)
            lc = f"{r['loop_closure_time']:.0f}s" if r["loop_closure_time"] else "no"
            print(f"{t:18s} {'yes' if gps else 'no':>4s} {r['laps_counted']}/{r['line_passes']:<3d} "
                  f"{r['lap_time'] / max(1, r['laps_driven']):8.1f} "
                  f"{r['ekf_err_mean']:5.2f}/{r['ekf_err_max']:5.2f} m {r['slam_err_mean']:6.2f}/{r['slam_err_max']:5.2f} m "
                  f"{r['mean_err']:5.2f}/{r['max_err']:5.2f} m "
                  f"{r['matched']:3d}/{r['n_true']:<3d} {r['aligned_matched']:3d}/{r['n_true']:<3d} "
                  f"{r['false']:6d} {lc:>9s} {r['update_ms_max']:7.0f}",
                  flush=True)


if __name__ == "__main__":
    main()
