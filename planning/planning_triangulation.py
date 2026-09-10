"""Cone-based local planner with Delaunay pairing and genuine RRT* detours."""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay, QhullError
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from eufs_msgs.msg import WaypointArrayStamped, ConeArrayWithCovariance, Waypoint

try:
    from .rrt_star import RRTStar
except ImportError:
    from rrt_star import RRTStar


class ConePlanner(Node):
    """Build a local centerline and invoke RRT* when cone obstacles block it."""
    def __init__(self, name: str = "local_planner") -> None:
        super().__init__(name)
        self.declare_parameter("cone_topic", "/ground_truth/cones")
        self.declare_parameter("trajectory_topic", "/trajectory")
        self.declare_parameter("cone_radius", 0.20)
        self.declare_parameter("safety_margin", 0.30)
        self.declare_parameter("rrt_step", 0.50)
        self.declare_parameter("rrt_neighbors", 1.50)
        self.declare_parameter("rrt_iterations", 1000)
        self.cone_radius = float(self.get_parameter("cone_radius").value)
        self.margin = float(self.get_parameter("safety_margin").value)
        self.create_subscription(ConeArrayWithCovariance, self.get_parameter("cone_topic").value, self._cones_cb, 10)
        self.path_pub = self.create_publisher(WaypointArrayStamped, self.get_parameter("trajectory_topic").value, 10)
        self.viz_pub = self.create_publisher(Marker, "/planner/viz", 10)

    @staticmethod
    def _xy(cones) -> np.ndarray:
        return np.asarray([[c.point.x, c.point.y] for c in cones], dtype=float) if cones else np.empty((0, 2))

    def find_midpoints(self, blue: np.ndarray, yellow: np.ndarray) -> np.ndarray:
        """Use cross-color Delaunay edges, then greedy one-to-one pairing."""
        if len(blue) == 0 or len(yellow) == 0:
            return np.empty((0, 2))
        n = min(len(blue), len(yellow))
        combined = np.vstack((blue[:n], yellow[:n]))
        if len(combined) < 3:
            candidates = [(i, n + i) for i in range(n)]
        else:
            try:
                tri = Delaunay(combined)
                candidates = set()
                for simplex in tri.simplices:
                    for i in simplex:
                        for j in simplex:
                            if i < n <= j:
                                candidates.add((int(i), int(j)))
            except QhullError:
                candidates = set()
        candidates = sorted(candidates, key=lambda p: np.linalg.norm(combined[p[0]] - combined[p[1]]))
        available_b, available_y, selected = set(range(n)), set(range(n)), []
        for bi, yg in candidates:
            yi = yg - n
            if bi in available_b and yi in available_y:
                selected.append((bi, yi))
                available_b.remove(bi)
                available_y.remove(yi)
        if not selected:
            selected = [(i, i) for i in range(n)]
        points = np.asarray([(blue[i] + yellow[j]) * 0.5 for i, j in selected], dtype=float)
        return self._order(points)

    @staticmethod
    def _order(points: np.ndarray) -> np.ndarray:
        if len(points) < 2:
            return points
        remaining = [p.copy() for p in points]
        ordered = [remaining.pop(int(np.argmin([np.linalg.norm(p) for p in remaining])))]
        while remaining:
            idx = int(np.argmin([np.linalg.norm(p - ordered[-1]) for p in remaining]))
            ordered.append(remaining.pop(idx))
        return np.asarray(ordered)

    def _rrt_path(self, start: np.ndarray, goal: np.ndarray, blue: np.ndarray, yellow: np.ndarray) -> np.ndarray:
        obstacles = [(p, self.cone_radius + self.margin) for p in np.vstack((blue, yellow))]
        extent = max(3.0, float(np.max(np.linalg.norm(np.vstack((start, goal)), axis=1)) + 2.0))
        planner = RRTStar(
            start, goal, obstacles=obstacles,
            bounds=(np.array([-1.0, -extent]), np.array([extent, extent])),
            step_size=float(self.get_parameter("rrt_step").value),
            neighbor_radius=float(self.get_parameter("rrt_neighbors").value),
            max_iterations=int(self.get_parameter("rrt_iterations").value),
        )
        path = planner.plan()
        return np.asarray(path, dtype=float) if path else np.empty((0, 2))

    def _cones_cb(self, msg: ConeArrayWithCovariance) -> None:
        blue, yellow = self._xy(msg.blue_cones), self._xy(msg.yellow_cones)
        midpoints = self.find_midpoints(blue, yellow)
        if len(midpoints) < 2:
            return
        obstacles = [(p, self.cone_radius + self.margin) for p in np.vstack((blue, yellow))]
        direct_free = RRTStar(midpoints[0], midpoints[-1], obstacles=obstacles).collision_free(midpoints[0], midpoints[-1])
        path = midpoints if direct_free else self._rrt_path(midpoints[0], midpoints[-1], blue, yellow)
        if len(path) == 0:
            self.get_logger().warn("No collision-free trajectory found; no path published")
            return
        self._publish(path)

    def _publish(self, points: np.ndarray) -> None:
        msg = WaypointArrayStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_footprint"
        for p in points:
            msg.waypoints.append(Waypoint(position=Point(x=float(p[0]), y=float(p[1]))))
        self.path_pub.publish(msg)
        marker = Marker()
        marker.header = msg.header
        marker.ns, marker.id = "planner", 0
        marker.type, marker.action = Marker.LINE_STRIP, Marker.ADD
        marker.scale.x, marker.color.a = 0.08, 1.0
        marker.color.g = 1.0
        for p in points:
            marker.points.append(Point(x=float(p[0]), y=float(p[1])))
        self.viz_pub.publish(marker)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ConePlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
