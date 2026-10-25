"""
Cone mapping + loop closure node.

Our localisation and mapping is split in two nodes:
  * state_estimation/ekf_node.py  -> where is the car (EKF pose)
  * this node                      -> where are the cones (the map)

Every cone detection (car frame) is moved into the map frame using the EKF
pose. All detections are kept and then:
  1. LocalOutlierFactor throws away single wrong detections
  2. DBSCAN groups detections of the same cone, the group mean is the cone
  3. a KDTree of the cones mapped earlier is used to detect loop closure
     (the car is seeing cones it already mapped long ago)
  4. passing the big orange cones counts laps

It is not a full joint EKF-SLAM yet (the cone positions do not correct the
car pose), that is the next thing on my list.

Subscribes : /ground_truth/cones (car frame), /odometry_filtered/car_state
Publishes  : /slam/map (ConeArrayWithCovariance, map frame)
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from eufs_msgs.msg import ConeArrayWithCovariance, ConeWithCovariance, CarState
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor     # was imported from sklearn.cluster (crash)
from scipy.spatial import KDTree

COLOURS = ("blue", "yellow", "orange")


def quaternion_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def car_to_map(points, x, y, yaw):
    if len(points) == 0:
        return np.zeros((0, 2))
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.asarray(points, dtype=float)
    return np.column_stack([x + c * p[:, 0] - s * p[:, 1], y + s * p[:, 0] + c * p[:, 1]])


def cluster_cones(detections, eps=0.8, min_samples=3, lof_neighbors=10):
    """Detections (N, 2) -> mapped cone positions (M, 2)."""
    if len(detections) < min_samples:
        return np.zeros((0, 2))
    pts = np.asarray(detections)
    if len(pts) > lof_neighbors + 1:
        keep = LocalOutlierFactor(n_neighbors=lof_neighbors).fit_predict(pts) == 1
        pts = pts[keep]
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    groups = [pts[labels == k] for k in np.unique(labels) if k != -1]
    return np.array([g.mean(axis=0) for g in groups]) if groups else np.zeros((0, 2))


class ConeMapper(object):
    """ROS free part, so it can be tested offline."""

    def __init__(self, max_range=12.0, max_detections=20000):
        self.max_range = max_range              # far detections are less accurate
        self.max_detections = max_detections
        self.detections = {c: [] for c in COLOURS}
        self.map = {c: np.zeros((0, 2)) for c in COLOURS}
        self.map_age = []                       # frame index when each map cone was first seen
        self.frame = 0
        self.laps = 0
        self.orange_dist = []
        self.lap_armed = True                   # must get away from the start line before next lap
        self.loop_closed = False

    def add_frame(self, cones_car, pose):
        """cones_car: dict colour -> (N, 2) in car frame, pose: (x, y, yaw)."""
        self.frame += 1
        for c in COLOURS:
            pts = np.asarray(cones_car.get(c, np.zeros((0, 2)))).reshape(-1, 2)
            if len(pts):
                pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < self.max_range]
            self.detections[c].extend(car_to_map(pts, *pose).tolist())
            self.detections[c] = self.detections[c][-self.max_detections:]
        self.count_laps(cones_car.get("orange", np.zeros((0, 2))))

    def update_map(self):
        for c in COLOURS:
            self.map[c] = cluster_cones(self.detections[c])
        return self.map

    def count_laps(self, orange_car):
        """Lap = closest approach (< 3 m) to the orange start-line cones.
        After a lap is counted the car has to be > 8 m away (or not see the
        orange cones) before the next one counts, so detection noise near
        the line cannot count the same pass twice."""
        orange_car = np.asarray(orange_car).reshape(-1, 2)
        if len(orange_car) < 2:
            self.orange_dist.clear()
            self.lap_armed = True
            return
        dist = float(np.linalg.norm(orange_car.mean(axis=0)))
        if dist > 8.0:
            self.lap_armed = True
        self.orange_dist.append(dist)
        d = self.orange_dist
        if self.lap_armed and len(d) >= 3 and d[-2] < d[-3] and d[-2] <= d[-1] and d[-2] < 3.0:
            self.laps += 1
            self.lap_armed = False
            self.orange_dist.clear()

    def detect_loop_closure(self, observed_map_frame, old_cones, threshold=1.0, min_matches=4):
        """True if most of what we see now matches cones mapped long ago."""
        if len(old_cones) < 10 or len(observed_map_frame) < min_matches:
            return False
        dist, _ = KDTree(old_cones).query(observed_map_frame)
        matched = int(np.sum(dist < threshold))
        if matched >= max(min_matches, len(observed_map_frame) - 2):
            self.loop_closed = True
            return True
        return False


class ConeMappingNode(Node):

    def __init__(self):
        super().__init__("cone_mapping")
        cones_topic = self.declare_parameter("cones_topic", "/ground_truth/cones").value
        pose_topic = self.declare_parameter("pose_topic", "/odometry_filtered/car_state").value
        self.plot = self.declare_parameter("plot", False).value      # matplotlib window
        self.mapper = ConeMapper(max_range=self.declare_parameter("max_range", 12.0).value)
        self.pose = None
        self.path = []

        self.create_subscription(ConeArrayWithCovariance, cones_topic, self.cones_cb, 10)
        self.create_subscription(CarState, pose_topic, self.pose_cb, 10)
        self.map_pub = self.create_publisher(ConeArrayWithCovariance, "/slam/map", 1)
        self.create_timer(1.0, self.map_timer)
        if self.plot:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.fig, self.ax = plt.subplots()
            plt.ion()

    def pose_cb(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, quaternion_to_yaw(p.orientation))
        self.path.append(self.pose[:2])

    def cones_cb(self, msg):
        if self.pose is None:
            return
        frame = {
            "blue": [[c.point.x, c.point.y] for c in msg.blue_cones],
            "yellow": [[c.point.x, c.point.y] for c in msg.yellow_cones],
            "orange": [[c.point.x, c.point.y] for c in list(msg.orange_cones) + list(msg.big_orange_cones)],
        }
        laps_before = self.mapper.laps
        self.mapper.add_frame(frame, self.pose)
        if self.mapper.laps != laps_before:
            self.get_logger().info(f"Lap {self.mapper.laps} finished")

    def map_timer(self):
        old = {c: m.copy() for c, m in self.mapper.map.items()}
        m = self.mapper.update_map()
        if not self.mapper.loop_closed and self.pose is not None:
            seen = np.vstack([m["blue"], m["yellow"]])
            older = np.vstack([old["blue"], old["yellow"]])
            near = seen[np.hypot(*(seen - np.array(self.pose[:2])).T) < 8.0] if len(seen) else seen
            if self.mapper.frame > 200 and self.mapper.detect_loop_closure(near, older):
                self.get_logger().info("Loop closure detected, map is complete")
        self.publish_map(m)
        if self.plot:
            self.draw(m)

    def publish_map(self, m):
        msg = ConeArrayWithCovariance()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        for colour, field in (("blue", msg.blue_cones), ("yellow", msg.yellow_cones),
                              ("orange", msg.big_orange_cones)):
            for x, y in m[colour]:
                cone = ConeWithCovariance()
                cone.point.x, cone.point.y = float(x), float(y)
                field.append(cone)
        self.map_pub.publish(msg)

    def draw(self, m):
        ax = self.ax
        ax.clear()
        for colour, style in (("blue", "b^"), ("yellow", "y^"), ("orange", "^")):
            if len(m[colour]):
                ax.plot(*m[colour].T, style, color=None if colour != "orange" else "orange", label=colour)
        if len(self.path) > 1:
            ax.plot(*np.array(self.path).T, "r--", label="EKF path")
        ax.set_aspect("equal")
        ax.set_title(f"Cone map - laps: {self.mapper.laps}")
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
