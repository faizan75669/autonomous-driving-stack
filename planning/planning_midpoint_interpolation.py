"""Centerline planner using one-to-one blue/yellow cone pairing.

This is the simple baseline planner in the repository. It does not claim
obstacle avoidance; the RRT* implementation is provided separately.
"""
from __future__ import annotations

from typing import List
import numpy as np
from scipy.interpolate import splprep, splev
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from eufs_msgs.msg import WaypointArrayStamped, ConeArrayWithCovariance, ConeWithCovariance, Waypoint


class MidpointPlanner(Node):
    def __init__(self, name: str = "midpoint_planner") -> None:
        super().__init__(name)
        self.declare_parameter("cone_topic", "/fusion/cones")
        self.declare_parameter("trajectory_topic", "/trajectory")
        self.declare_parameter("interpolation_points", 100)
        self.declare_parameter("smoothing", 0.0)
        self.create_subscription(ConeArrayWithCovariance, self.get_parameter("cone_topic").value, self._cones_cb, 10)
        self.path_pub = self.create_publisher(WaypointArrayStamped, self.get_parameter("trajectory_topic").value, 10)
        self.viz_pub = self.create_publisher(Marker, "/planner/viz", 10)

    @staticmethod
    def convert(cones: List[ConeWithCovariance]) -> np.ndarray:
        if not cones:
            return np.empty((0, 2))
        return np.asarray([[c.point.x, c.point.y] for c in cones], dtype=float)

    @staticmethod
    def pair_midpoints(blue: np.ndarray, yellow: np.ndarray) -> np.ndarray:
        """Greedy one-to-one nearest-neighbor pairing, starting with closest pairs."""
        if len(blue) == 0 or len(yellow) == 0:
            return np.empty((0, 2))
        candidates = sorted(
            (float(np.linalg.norm(b - y)), i, j)
            for i, b in enumerate(blue) for j, y in enumerate(yellow)
        )
        used_b, used_y, mids = set(), set(), []
        for _, i, j in candidates:
            if i in used_b or j in used_y:
                continue
            used_b.add(i)
            used_y.add(j)
            mids.append((blue[i] + yellow[j]) * 0.5)
        return MidpointPlanner.order_forward(np.asarray(mids, dtype=float))

    @staticmethod
    def order_forward(points: np.ndarray) -> np.ndarray:
        """Order local points by walking to the nearest next point from the vehicle."""
        if len(points) < 2:
            return points
        remaining = [p.copy() for p in points]
        ordered = [remaining.pop(int(np.argmin([np.linalg.norm(p) for p in remaining])))]
        while remaining:
            idx = int(np.argmin([np.linalg.norm(p - ordered[-1]) for p in remaining]))
            ordered.append(remaining.pop(idx))
        return np.asarray(ordered)

    def interpolate(self, points: np.ndarray) -> np.ndarray:
        if len(points) < 3:
            return points
        try:
            k = min(3, len(points) - 1)
            tck, _ = splprep([points[:, 0], points[:, 1]], s=float(self.get_parameter("smoothing").value), k=k)
            u = np.linspace(0.0, 1.0, int(self.get_parameter("interpolation_points").value))
            x, y = splev(u, tck)
            return np.column_stack((x, y))
        except (ValueError, TypeError):
            return points

    def _cones_cb(self, msg: ConeArrayWithCovariance) -> None:
        blue = self.convert(msg.blue_cones)
        yellow = self.convert(msg.yellow_cones)
        path = self.interpolate(self.pair_midpoints(blue, yellow))
        if len(path) == 0:
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
        marker.ns = "midpoints"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.08
        marker.color.a = 1.0
        for p in points:
            marker.points.append(Point(x=float(p[0]), y=float(p[1])))
        self.viz_pub.publish(marker)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MidpointPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
