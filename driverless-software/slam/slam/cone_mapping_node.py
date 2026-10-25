"""
Cone mapping + pose correction + loop closure node.

Our localisation and mapping is split in two nodes:
  * state_estimation/ekf_node.py  -> car pose from odometry, IMU and GPS
  * this node                      -> cone map, and a pose corrected against that map

Every cone detection (car frame) is moved into the map frame with the pose.
Then:
  1. pose correction: the cones seen now are matched with cones already in
     the map and a small rigid transform (Kabsch) pulls the pose back onto
     the map. This stops odometry drift from smearing the map, and when the
     car comes back to the start it lines up with the first lap again.
  2. LocalOutlierFactor keeps single wrong detections out of the clustering
  3. DBSCAN groups detections of the same cone; the cone position is the
     weighted mean (near detections get much more weight, camera depth
     noise grows fast with distance)
  4. loop closure: the car sees cones it first saw more than 30 s ago
  5. lap counting: the car comes back to the orange start line in the map

Pose correction makes this a simple SLAM, but not a full EKF-SLAM (the cones
are not in the filter state and there is no global optimisation).

Subscribes : cones (car frame), /odometry_filtered/car_state
Publishes  : /slam/map (ConeArrayWithCovariance, map frame), /slam/pose (CarState)
Services   : /slam/reset (std_srvs/Trigger) clears the map
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from eufs_msgs.msg import ConeArrayWithCovariance, ConeWithCovariance, CarState
from std_srvs.srv import Trigger
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor     # was imported from sklearn.cluster (crash)
from scipy.spatial import KDTree

COLOURS = ("blue", "yellow", "orange")


def quaternion_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def car_to_map(points, x, y, yaw):
    if len(points) == 0:
        return np.zeros((0, 2))
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.asarray(points, dtype=float)
    return np.column_stack([x + c * p[:, 0] - s * p[:, 1], y + s * p[:, 0] + c * p[:, 1]])


def detection_weight(rng):
    """1 / sigma^2 with sigma growing with range (camera depth noise)."""
    sigma = 0.03 + 0.004 * rng ** 2
    return 1.0 / sigma ** 2


def kabsch_2d(src, dst):
    """Rigid transform (theta, t) with dst ~ R(theta) src + t."""
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    H = (src - cs).T @ (dst - cd)
    theta = math.atan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    return theta, cd - R @ cs


def split_wide_clusters(det, labels, eps, min_samples, max_std=0.25, levels=2):
    """Two cones about 1 m apart can merge into one DBSCAN cluster (their
    detection clouds touch). A single cone's detections spread only ~0.1-0.2 m,
    so a wider cluster is split again with a smaller eps."""
    next_label = labels.max() + 1
    for _ in range(levels):
        eps *= 0.6
        changed = False
        for k in np.unique(labels[labels >= 0]):
            idx = np.nonzero(labels == k)[0]
            if len(idx) < 2 * min_samples:
                continue
            pts = det[idx, :2]
            if np.sqrt(np.max(np.linalg.eigvalsh(np.cov(pts.T)))) <= max_std:
                continue
            sub = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
            if len(set(sub[sub >= 0])) < 2:
                continue
            for j in np.unique(sub):
                if j == -1:
                    labels[idx[sub == -1]] = -1
                else:
                    labels[idx[sub == j]] = next_label
                    next_label += 1
            changed = True
        if not changed:
            break
    return labels


def cluster_cones(det, eps=0.7, min_samples=3, lof_neighbors=10, keep_per_cone=60, max_noise_age=60.0):
    """det: (N, 4) [x, y, weight, time] -> (cones (M, 2), cov (M, 2, 2), det_kept, counts (M,)).

    LOF only decides what goes into the clustering this time; it does not
    delete anything (deleting LOF outliers removed cones that were only seen
    a few times). Detections that never joined a cone are dropped after
    max_noise_age seconds, and big clusters keep only their best detections,
    so memory stays bounded on long runs."""
    empty = (np.zeros((0, 2)), np.zeros((0, 2, 2)), det, np.zeros(0, int))
    if len(det) < min_samples:
        return empty
    inlier = np.ones(len(det), bool)
    if len(det) > 3 * lof_neighbors:
        inlier = LocalOutlierFactor(n_neighbors=lof_neighbors).fit_predict(det[:, :2]) == 1
    labels = np.full(len(det), -1)
    labels[inlier] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(det[inlier, :2])
    labels = split_wide_clusters(det, labels, eps, min_samples)
    now = det[:, 3].max()
    cones, covs, counts, kept = [], [], [], []
    for k in np.unique(labels):
        g = det[labels == k]
        if k == -1:
            kept.append(g[g[:, 3] > now - max_noise_age])     # may still become a cone
            continue
        w = g[:, 2] / g[:, 2].sum()
        mean = w @ g[:, :2]
        d = g[:, :2] - mean
        cones.append(mean)
        counts.append(len(g))
        covs.append((d * w[:, None]).T @ d / max(1.0, len(g) - 1) + np.eye(2) * 1e-4)
        if len(g) > keep_per_cone:      # keep the best (closest) detections
            g = g[np.argsort(-g[:, 2])[:keep_per_cone]]
        kept.append(g)
    kept = np.vstack(kept) if kept else np.zeros((0, 4))
    if not cones:
        return np.zeros((0, 2)), np.zeros((0, 2, 2)), kept, np.zeros(0, int)
    return np.array(cones), np.array(covs), kept, np.array(counts)


class ConeMapper(object):
    """ROS free part, so it can be tested offline."""

    def __init__(self, max_range=10.0, pose_correction=True, loop_closure_age=30.0,
                 correction_gain=0.1, match_range=6.0, min_detections=15, local_radius=25.0):
        self.max_range = max_range              # far detections are too noisy to map
        self.pose_correction = pose_correction
        self.loop_closure_age = loop_closure_age
        self.gain = correction_gain
        self.match_range = match_range
        self.min_detections = min_detections    # only trust cones that were seen many times
        self.local_radius = local_radius        # only re-cluster around the car, rest is frozen
        self.trusted_std = 0.4                  # EKF std below this = GPS-quality pose
        self.leak = 0.1
        self.reset()

    def reset(self):
        self.detections = {c: np.zeros((0, 4)) for c in COLOURS}
        self.map = {c: np.zeros((0, 2)) for c in COLOURS}
        self.cov = {c: np.zeros((0, 2, 2)) for c in COLOURS}
        self.counts = {c: np.zeros(0, int) for c in COLOURS}
        self.corr = (0.0, np.zeros(2))         # map <- EKF frame: R(theta), t
        self.frame = 0
        self.laps = 0
        self.lap_armed = False                  # armed once the car is far from the start line
        self.start_line = None
        self.loop_closed = False
        self.loop_closure_time = None
        self._old_tree = None
        self.last_pose = None

    # ------------------------------------------------------------------ pose
    def corrected_pose(self, pose):
        th, t = self.corr
        c, s = math.cos(th), math.sin(th)
        x = c * pose[0] - s * pose[1] + t[0]
        y = s * pose[0] + c * pose[1] + t[1]
        return x, y, wrap(pose[2] + th)

    def _correct(self, cones_car, pose):
        """Match near cones with well established map cones and nudge the
        correction. Translation only: the IMU heading is already good, and
        rotating around a few noisy matches did more harm than good."""
        src, dst = [], []
        cp = self.corrected_pose(pose)
        for c in ("blue", "yellow"):
            pts = cones_car[c]
            good = self.counts[c] >= self.min_detections
            if len(pts) == 0 or not np.any(good):
                continue
            ref = self.map[c][good]
            pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < self.match_range]
            if len(pts) == 0:
                continue
            world = car_to_map(pts, *cp)
            d, j = KDTree(ref).query(world)
            ok = d < 0.8
            src.append(world[ok])
            dst.append(ref[j[ok]])
        if not src:
            return
        src, dst = np.vstack(src), np.vstack(dst)
        if len(src) < 3:
            return
        offset = np.median(dst - src, axis=0)          # median: robust to one wrong match
        if np.linalg.norm(offset) > 0.6:
            return
        th0, t0 = self.corr
        self.corr = (th0, t0 + self.gain * offset)

    # ------------------------------------------------------------------ frames
    def add_frame(self, cones_car, pose, t=None, pose_std=None):
        """cones_car: dict colour -> (N, 2) in car frame, pose: EKF (x, y, yaw), t: time [s],
        pose_std: EKF position std [m] if known."""
        self.frame += 1
        # When the EKF is confident (GPS available) it has no long-term drift,
        # so the correction should not build up: let it decay back to zero.
        if pose_std is not None and pose_std < self.trusted_std:
            th0, t0 = self.corr
            self.corr = (th0 * (1 - self.leak), t0 * (1 - self.leak))
        t = float(self.frame) * 0.1 if t is None else t
        cones_car = {c: np.asarray(cones_car.get(c, np.zeros((0, 2))), dtype=float).reshape(-1, 2)
                     for c in COLOURS}
        if self.pose_correction:
            self._correct(cones_car, pose)
        cp = self.corrected_pose(pose)
        self.last_pose = cp
        current = []
        for c in COLOURS:
            pts = cones_car[c]
            rng = np.hypot(pts[:, 0], pts[:, 1])
            pts, rng = pts[rng < self.max_range], rng[rng < self.max_range]
            world = car_to_map(pts, *cp)
            new = np.column_stack([world, detection_weight(rng), np.full(len(rng), t)])
            self.detections[c] = np.vstack([self.detections[c], new])
            if c != "orange":
                current.append(world)
        self._check_loop_closure(np.vstack(current), t)
        self._count_laps(cp)
        return cp

    def update_map(self):
        """Re-cluster detections near the car; cones further away are frozen,
        so the cost does not grow with the size of the track."""
        for c in COLOURS:
            det = self.detections[c]
            if self.last_pose is None or len(det) == 0:
                self.map[c], self.cov[c], self.detections[c], self.counts[c] = cluster_cones(det)
                continue
            centre = np.array(self.last_pose[:2])
            near_det = np.hypot(*(det[:, :2] - centre).T) < self.local_radius
            # frozen part: old cones outside (radius - 2 m), so cones on the border are redone
            far_cone = np.hypot(*(self.map[c] - centre).T) >= self.local_radius - 2.0 \
                if len(self.map[c]) else np.zeros(0, bool)
            new_map, new_cov, kept, new_cnt = cluster_cones(det[near_det])
            if len(new_map):
                # drop new clusters that duplicate a frozen cone (border region)
                frozen = self.map[c][far_cone]
                if len(frozen):
                    d, _ = KDTree(frozen).query(new_map)
                    ok = d > 0.5
                    new_map, new_cov, new_cnt = new_map[ok], new_cov[ok], new_cnt[ok]
            self.map[c] = np.vstack([self.map[c][far_cone], new_map]) if len(new_map) else self.map[c][far_cone]
            self.cov[c] = np.concatenate([self.cov[c][far_cone], new_cov]) if len(new_map) else self.cov[c][far_cone]
            self.counts[c] = np.concatenate([self.counts[c][far_cone], new_cnt]) if len(new_map) else self.counts[c][far_cone]
            self.detections[c] = np.vstack([det[~near_det], kept])
        if len(self.map["orange"]) >= 2:
            self.start_line = self.map["orange"].mean(axis=0)
        det = np.vstack([self.detections["blue"], self.detections["yellow"]])
        if len(det):
            # the newest timestamp is the current time
            old = det[det[:, 3] < det[:, 3].max() - self.loop_closure_age]
            self._old_tree = KDTree(old[:, :2]) if len(old) >= 10 else None
        return self.map

    # ------------------------------------------------------------------ events
    def _check_loop_closure(self, seen, t, radius=0.7, min_matches=5, fraction=0.7):
        """Loop closure = most cones we see now were first seen long ago."""
        if self.loop_closed or self._old_tree is None or len(seen) < min_matches:
            return
        d, _ = self._old_tree.query(seen)
        matched = int(np.sum(d < radius))
        if matched >= min_matches and matched >= fraction * len(seen):
            self.loop_closed = True
            self.loop_closure_time = t

    def _count_laps(self, pose, arm_dist=15.0, lap_dist=4.0):
        """Lap = coming back within 4 m of the start line after being 15 m away."""
        if self.start_line is None:
            return
        d = math.hypot(pose[0] - self.start_line[0], pose[1] - self.start_line[1])
        if d > arm_dist:
            self.lap_armed = True
        elif self.lap_armed and d < lap_dist:
            self.laps += 1
            self.lap_armed = False


class ConeMappingNode(Node):

    def __init__(self):
        super().__init__("cone_mapping")
        cones_topic = self.declare_parameter("cones_topic", "/ground_truth/cones").value
        pose_topic = self.declare_parameter("pose_topic", "/odometry_filtered/car_state").value
        self.plot = self.declare_parameter("plot", False).value      # matplotlib window
        self.mapper = ConeMapper(
            max_range=self.declare_parameter("max_range", 10.0).value,
            pose_correction=self.declare_parameter("pose_correction", True).value,
            loop_closure_age=self.declare_parameter("loop_closure_age", 30.0).value)
        self.pose = None
        self.pose_time = None
        self.vel = (0.0, 0.0, 0.0)               # vx, vy (map frame), yaw rate
        self.pose_cov = np.eye(2) * 1e-4
        self.have_cov = False
        self.path = []
        self.laps_reported = 0
        self.loop_reported = False

        self.create_subscription(ConeArrayWithCovariance, cones_topic, self.cones_cb, 10)
        self.create_subscription(CarState, pose_topic, self.pose_cb, 10)
        self.map_pub = self.create_publisher(ConeArrayWithCovariance, "/slam/map", 1)
        self.pose_pub = self.create_publisher(CarState, "/slam/pose", 10)
        self.create_service(Trigger, "/slam/reset", self.reset_cb)
        self.create_timer(1.0, self.map_timer)
        if self.plot:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.fig, self.ax = plt.subplots()
            plt.ion()

    def reset_cb(self, request, response):
        self.mapper.reset()
        self.path.clear()
        self.laps_reported = 0
        self.loop_reported = False
        response.success = True
        response.message = "SLAM map cleared"
        return response

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def pose_cb(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, quaternion_to_yaw(p.orientation))
        self.pose_time = self.now()
        tw = msg.twist.twist
        self.vel = (float(tw.linear.x), float(tw.linear.y), float(tw.angular.z))
        try:
            cov = list(msg.pose.covariance)
        except TypeError:
            cov = []
        if len(cov) == 36 and cov[0] > 0.0:
            self.pose_cov = np.array([[cov[0], cov[1]], [cov[6], cov[7]]])
            self.have_cov = True

    def cones_cb(self, msg):
        if self.pose is None:
            return
        frame = {
            "blue": [[c.point.x, c.point.y] for c in msg.blue_cones],
            "yellow": [[c.point.x, c.point.y] for c in msg.yellow_cones],
            "orange": [[c.point.x, c.point.y] for c in list(msg.orange_cones) + list(msg.big_orange_cones)],
        }
        pose_std = math.sqrt(max(self.pose_cov[0, 0] + self.pose_cov[1, 1], 0.0) / 2.0) \
            if self.have_cov else None
        cp = self.mapper.add_frame(frame, self.pose_at(self.now()), self.now(), pose_std)
        self.path.append(cp[:2])
        self.publish_pose(cp, msg)
        if self.mapper.laps != self.laps_reported:
            self.laps_reported = self.mapper.laps
            self.get_logger().info(f"Lap {self.mapper.laps} finished")
        if self.mapper.loop_closed and not self.loop_reported:
            self.loop_reported = True
            self.get_logger().info(f"Loop closure detected at t={self.mapper.loop_closure_time:.1f} s")

    def pose_at(self, t):
        """The pose arrives a bit before the cones; move it forward to time t
        with the velocity from the EKF (otherwise every detection is smeared
        by speed * delay along the track)."""
        dt = 0.0 if self.pose_time is None else min(max(t - self.pose_time, 0.0), 0.2)
        x, y, yaw = self.pose
        vx, vy, w = self.vel
        return x + vx * dt, y + vy * dt, wrap(yaw + w * dt)

    def map_timer(self):
        m = self.mapper.update_map()
        self.publish_map(m)
        if self.plot:
            self.draw(m)

    def publish_pose(self, cp, cones_msg):
        msg = CarState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_footprint"
        msg.pose.pose.position.x, msg.pose.pose.position.y = float(cp[0]), float(cp[1])
        o = msg.pose.pose.orientation
        o.x, o.y, o.z, o.w = 0.0, 0.0, math.sin(cp[2] / 2), math.cos(cp[2] / 2)
        self.pose_pub.publish(msg)

    def publish_map(self, m):
        msg = ConeArrayWithCovariance()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        for colour, field in (("blue", msg.blue_cones), ("yellow", msg.yellow_cones),
                              ("orange", msg.big_orange_cones)):
            for (x, y), cov in zip(m[colour], self.mapper.cov[colour]):
                cone = ConeWithCovariance()
                cone.point.x, cone.point.y = float(x), float(y)
                cone.covariance = [float(cov[0, 0]), float(cov[0, 1]), float(cov[1, 0]), float(cov[1, 1])]
                field.append(cone)
        self.map_pub.publish(msg)

    def draw(self, m):
        from matplotlib.patches import Ellipse
        ax = self.ax
        ax.clear()
        for colour, c in (("blue", "tab:blue"), ("yellow", "gold"), ("orange", "tab:orange")):
            if len(m[colour]):
                ax.plot(*m[colour].T, "^", color=c, label=colour)
        if len(self.path) > 1:
            ax.plot(*np.array(self.path).T, "r--", lw=1, label="SLAM path")
            # 95% uncertainty ellipse of the EKF position (chi-square 2 dof = 5.991)
            vals, vecs = np.linalg.eigh(self.pose_cov)
            vals = np.maximum(vals, 1e-6)
            ang = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
            w, h = 2 * np.sqrt(5.991 * vals[::-1])
            ax.add_patch(Ellipse(self.path[-1], w, h, angle=ang, fill=False, color="r", lw=1.5))
        ax.set_aspect("equal")
        ax.set_title(f"Cone map - laps: {self.mapper.laps}" + ("  (loop closed)" if self.mapper.loop_closed else ""))
        ax.legend(loc="upper right")
        self.plt.pause(0.001)


def main(args=None):
    rclpy.init(args=args)
    node = ConeMappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
