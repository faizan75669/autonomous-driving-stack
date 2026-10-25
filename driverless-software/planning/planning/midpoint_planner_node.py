"""
Midpoint planner (my first planner, kept as a baseline).

Idea: take the middle point between every blue cone and the yellow cone
across from it, sort the middle points starting from the car, and fit a
spline through them.

My first version paired blue[i] with yellow[i] straight from the message.
But the message order is not the track order, so in corners it joined
cones from different parts of the track and the path went zig-zag.
Now every blue cone is paired with the closest yellow cone (and the other
way round) and pairs wider than a normal track are thrown away.

Subscribes : /ground_truth/cones (car frame)
Publishes  : /trajectory, /planner/viz
"""

from typing import List

import numpy as np
from scipy import interpolate

import rclpy
from rclpy.node import Node
from eufs_msgs.msg import WaypointArrayStamped, Waypoint, ConeArrayWithCovariance, ConeWithCovariance
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker


def pair_midpoints(blue, yellow, min_width=2.0, max_width=7.0):
    """Midpoint of every cone with its nearest cone of the other colour."""
    mids = []
    if len(blue) == 0 or len(yellow) == 0:
        return np.zeros((0, 2))
    for own, other in ((blue, yellow), (yellow, blue)):
        for c in own:
            d = np.linalg.norm(other - c, axis=1)
            j = int(np.argmin(d))
            if min_width < d[j] < max_width:
                mids.append(0.5 * (c + other[j]))
    if not mids:
        return np.zeros((0, 2))
    mids = np.array(mids)
    # both directions give the same pair twice, remove duplicates
    return np.unique(np.round(mids, 2), axis=0)


def sort_from_car(points, max_jump=6.0):
    """Greedy nearest-neighbour ordering that starts at the car (origin)."""
    remaining = [p for p in points if p[0] > 0.0]       # only points in front
    path = []
    current = np.zeros(2)
    while remaining:
        d = [np.linalg.norm(p - current) for p in remaining]
        i = int(np.argmin(d))
        if d[i] > max_jump:
            break
        current = remaining.pop(i)
        path.append(current)
    return np.array(path) if path else np.zeros((0, 2))


def smooth_path(points, n=60):
    pts = np.vstack([[0.0, 0.0], points])
    if len(pts) < 3:
        return pts
    try:
        tck, _ = interpolate.splprep([pts[:, 0], pts[:, 1]], s=0.5 * len(pts), k=min(3, len(pts) - 1))
        x, y = interpolate.splev(np.linspace(0, 1, n), tck)
        return np.column_stack([x, y])
    except Exception:
        return pts


class Planner(Node):
    def __init__(self, name: str):
        super().__init__(name)
        cones_topic = self.declare_parameter("cones_topic", "/ground_truth/cones").value
        self.max_width = self.declare_parameter("max_track_width", 7.0).value

        self.cones_sub = self.create_subscription(ConeArrayWithCovariance, cones_topic, self.cones_callback, 1)
        self.track_line_pub = self.create_publisher(WaypointArrayStamped, "/trajectory", 1)
        self.visualization_pub = self.create_publisher(Marker, "/planner/viz", 1)

    def cones_callback(self, msg: ConeArrayWithCovariance):
        blue = self.convert(msg.blue_cones)
        yellow = self.convert(msg.yellow_cones)
        orange = self.convert(list(msg.orange_cones) + list(msg.big_orange_cones))
        # orange cones are on the start line, put them on the side they are on
        if len(orange):
            blue = np.vstack([blue, orange[orange[:, 1] > 0]])
            yellow = np.vstack([yellow, orange[orange[:, 1] <= 0]])

        mids = sort_from_car(pair_midpoints(blue, yellow, max_width=self.max_width))
        if len(mids) == 0:
            return
        path = smooth_path(mids)
        self.publish_path(path)
        self.publish_visualization(path)

    def publish_path(self, points: np.ndarray):
        msg = WaypointArrayStamped()
        msg.header.frame_id = "base_footprint"
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in points:
            msg.waypoints.append(Waypoint(position=Point(x=float(x), y=float(y), z=0.0)))
        self.track_line_pub.publish(msg)

    def publish_visualization(self, points: np.ndarray):
        marker = Marker()
        marker.header.frame_id = "base_footprint"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "midpoints"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = 0.35
        marker.color.a = 1.0
        marker.color.g = 1.0
        for x, y in points:
            marker.points.append(Point(x=float(x), y=float(y), z=0.0))
        self.visualization_pub.publish(marker)

    @staticmethod
    def convert(cones: List[ConeWithCovariance]) -> np.ndarray:
        if len(cones) == 0:
            return np.zeros((0, 2))
        return np.array([[c.point.x, c.point.y] for c in cones], dtype=float)


def main():
    rclpy.init(args=None)
    node = Planner("midpoint_planner")
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
