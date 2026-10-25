"""
RRT planner with Delaunay triangulation (my second planner, experimental).

1. Delaunay triangulation of the visible cones, keep the blue-yellow edges
   and take their midpoints. Chain them starting from the car (nearest
   neighbour, max 4 m jump) so midpoints from other parts of the track
   are dropped. This chain is a rough centre line.
2. Grow an RRT from the car. Random samples are drawn around those
   midpoints so the tree grows along the track and not into the grass.
3. A new branch is only added if it stays at least `clearance` metres
   away from every cone.
4. The branch that gets furthest along the track is the path, then a
   smoothing spline takes out most of the RRT zig-zag.

This is plain RRT (no rewiring), so the path is a bit wobbly and changes
from frame to frame. For the real runs we used the Delaunay planner.

Subscribes : /ground_truth/cones (car frame)
Publishes  : /trajectory, /planner/viz
"""

from typing import List, Optional

import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import splprep, splev

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from eufs_msgs.msg import WaypointArrayStamped, ConeArrayWithCovariance, Waypoint

from planning.midpoint_planner_node import sort_from_car


def delaunay_midpoints(blue: np.ndarray, yellow: np.ndarray,
                       min_width: float = 2.0, max_width: float = 7.0) -> np.ndarray:
    """Midpoints of every blue-yellow edge of the Delaunay triangulation."""
    if len(blue) == 0 or len(yellow) == 0 or len(blue) + len(yellow) < 3:
        return np.zeros((0, 2))
    pts = np.vstack([blue, yellow])
    is_blue = np.r_[np.ones(len(blue), bool), np.zeros(len(yellow), bool)]
    try:
        tri = Delaunay(pts)
    except Exception:
        return np.zeros((0, 2))
    mids = set()
    for s in tri.simplices:
        for a, b in ((s[0], s[1]), (s[1], s[2]), (s[0], s[2])):
            if is_blue[a] != is_blue[b]:
                w = np.linalg.norm(pts[a] - pts[b])
                if min_width < w < max_width:
                    m = 0.5 * (pts[a] + pts[b])
                    mids.add((round(m[0], 2), round(m[1], 2)))
    return np.array(sorted(mids)) if mids else np.zeros((0, 2))


class RRT:
    """Simple RRT in the car frame. Nodes are stored as arrays + parent index."""

    def __init__(self, guide: np.ndarray, obstacles: np.ndarray, max_samples: int = 400,
                 step: float = 1.0, clearance: float = 0.9, sample_sigma: float = 0.6,
                 rng: Optional[np.random.Generator] = None):
        self.guide = guide                  # midpoints used to bias sampling
        self.obstacles = obstacles          # all cones (N, 2)
        self.max_samples = max_samples
        self.step = step
        self.clearance = clearance
        self.sigma = sample_sigma
        self.rng = rng if rng is not None else np.random.default_rng()
        self.nodes = [np.zeros(2)]
        self.parents = [-1]

    def sample(self) -> np.ndarray:
        g = self.guide[self.rng.integers(len(self.guide))]
        return g + self.rng.normal(0.0, self.sigma, 2)

    def collides(self, a: np.ndarray, b: np.ndarray) -> bool:
        if len(self.obstacles) == 0:
            return False
        for t in np.linspace(0.0, 1.0, 5):
            p = a + t * (b - a)
            if np.min(np.linalg.norm(self.obstacles - p, axis=1)) < self.clearance:
                return True
        return False

    def plan(self) -> np.ndarray:
        nodes = np.array(self.nodes)
        for _ in range(self.max_samples):
            target = self.sample()
            nearest = int(np.argmin(np.linalg.norm(nodes - target, axis=1)))
            direction = target - nodes[nearest]
            dist = np.linalg.norm(direction)
            if dist < 1e-6:
                continue
            new = nodes[nearest] + direction / dist * min(self.step, dist)
            if self.collides(nodes[nearest], new):
                continue
            self.nodes.append(new)
            self.parents.append(nearest)
            nodes = np.vstack([nodes, new])
        return self.best_branch(nodes)

    def best_branch(self, nodes: np.ndarray) -> np.ndarray:
        """Branch whose end is closest to the last point of the guide chain."""
        far = self.guide[-1]
        i = int(np.argmin(np.linalg.norm(nodes - far, axis=1)))
        branch = []
        while i != -1:
            branch.append(nodes[i])
            i = self.parents[i]
        return np.array(branch[::-1])


def spline_path(points: np.ndarray, n: int = 60) -> np.ndarray:
    """Smoothing spline through the RRT branch (raw RRT branches zig-zag a lot)."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.r_[True, seg > 1e-3]]
    if len(points) < 4:
        return points
    try:
        w = np.ones(len(points))
        w[0] = 10.0                                   # keep the start at the car
        tck, _ = splprep([points[:, 0], points[:, 1]], w=w, k=3, s=0.4 * len(points))
        x, y = splev(np.linspace(0.0, 1.0, n), tck)
        return np.column_stack([x, y])
    except Exception:
        return points


class Planner(Node):
    def __init__(self, name: str = "rrt_planner"):
        super().__init__(name)
        cones_topic = self.declare_parameter("cones_topic", "/ground_truth/cones").value
        self.max_samples = self.declare_parameter("max_samples", 400).value
        self.step = self.declare_parameter("step_size", 1.0).value
        self.clearance = self.declare_parameter("cone_clearance", 0.9).value
        seed = self.declare_parameter("seed", -1).value          # -1 = different every run
        self.rng = np.random.default_rng(None if seed < 0 else seed)

        self.cones_sub = self.create_subscription(ConeArrayWithCovariance, cones_topic, self.cones_callback, 1)
        self.track_line_pub = self.create_publisher(WaypointArrayStamped, "/trajectory", 1)
        self.visualization_pub = self.create_publisher(Marker, "/planner/viz", 1)

    def cones_callback(self, msg: ConeArrayWithCovariance):
        blue = self.convert(msg.blue_cones)
        yellow = self.convert(msg.yellow_cones)
        orange = self.convert(list(msg.orange_cones) + list(msg.big_orange_cones))
        if len(orange):
            blue = np.vstack([blue, orange[orange[:, 1] > 0]])
            yellow = np.vstack([yellow, orange[orange[:, 1] <= 0]])

        guide = sort_from_car(delaunay_midpoints(blue, yellow), max_jump=4.0)
        if len(guide) == 0:
            return
        try:
            rrt = RRT(guide, np.vstack([blue, yellow]), max_samples=self.max_samples,
                      step=self.step, clearance=self.clearance, rng=self.rng)
            branch = rrt.plan()
        except Exception as e:  # one bad frame should not kill the node
            self.get_logger().warn(f"RRT failed this frame: {e}")
            return
        # tree did not get close to the goal -> use the midpoint chain instead
        if len(branch) < 2 or np.linalg.norm(branch[-1] - guide[-1]) > 2.0:
            branch = np.vstack([[0.0, 0.0], guide])
        path = spline_path(branch)
        self.publish_path(path)
        self.publish_visualisation(path)

    @staticmethod
    def convert(cones) -> np.ndarray:
        if len(cones) == 0:
            return np.zeros((0, 2))
        return np.array([[c.point.x, c.point.y] for c in cones], dtype=float)

    def publish_path(self, points: np.ndarray):
        msg = WaypointArrayStamped()
        msg.header.frame_id = "base_footprint"
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in points:
            msg.waypoints.append(Waypoint(position=Point(x=float(x), y=float(y), z=0.0)))
        self.track_line_pub.publish(msg)

    def publish_visualisation(self, points: np.ndarray):
        marker = Marker()
        marker.header.frame_id = "base_footprint"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.ns = "rrt_path"
        marker.id = 0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.15
        marker.color.a = 1.0
        marker.color.g = 1.0
        for x, y in points:
            marker.points.append(Point(x=float(x), y=float(y), z=0.0))
        self.visualization_pub.publish(marker)


def main():
    rclpy.init()
    node = Planner()
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
