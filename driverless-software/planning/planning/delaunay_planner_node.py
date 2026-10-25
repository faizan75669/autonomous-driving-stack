"""
Delaunay planner node (the one we use for trackdrive).

Subscribes : /ground_truth/cones  (eufs_msgs/ConeArrayWithCovariance, car frame)
Publishes  : /trajectory          (eufs_msgs/WaypointArrayStamped, car frame)
             /planner/viz         (Marker, LINE_STRIP of the path)
             /planner/viz_centers (Marker, the raw centre points)

The actual maths is in centerline_planner.py so it can be tested without ROS.
How it works, in short:
  1. Delaunay triangulation of all the cones we can see
  2. keep only blue-yellow edges with a sensible width ("gates")
  3. walk from gate to gate starting in front of the car
  4. fit a smoothing spline from the car through the gate midpoints

I moved to this after the midpoint planner, because the midpoint planner
went zig-zag in corners. The RRT planner in this folder came after this one
and uses these Delaunay midpoints to guide its sampling.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from eufs_msgs.msg import WaypointArrayStamped, ConeArrayWithCovariance, Waypoint
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

from planning.centerline_planner import CenterlinePlanner


class Planner(Node):

    def __init__(self, name):
        super().__init__(name)

        cones_topic = self.declare_parameter("cones_topic", "/ground_truth/cones").value
        self.verbose = self.declare_parameter("verbose", False).value

        self.planner = CenterlinePlanner(
            max_view_distance=self.declare_parameter("max_view_distance", 16.0).value,
            max_cone_spacing=self.declare_parameter("max_cone_spacing", 6.5).value,
            default_track_width=self.declare_parameter("track_width", 3.5).value,
            path_spacing=self.declare_parameter("path_spacing", 0.25).value,
            smoothing=self.declare_parameter("smoothing", 0.3).value,
            optimize=self.declare_parameter("racing_line", False).value,
            clearance=self.declare_parameter("cone_clearance", 1.0).value,
        )

        self.cones_sub = self.create_subscription(
            ConeArrayWithCovariance, cones_topic, self.cones_callback, 1)
        self.track_line_pub = self.create_publisher(WaypointArrayStamped, "/trajectory", 1)
        self.visualization_pub = self.create_publisher(Marker, "/planner/viz", 1)
        self.centers_pub = self.create_publisher(Marker, "/planner/viz_centers", 1)
        self.get_logger().info(f"Centerline planner listening on {cones_topic}")

    # ------------------------------------------------------------------
    def cones_callback(self, msg):
        blue = self.convert(msg.blue_cones)
        yellow = self.convert(msg.yellow_cones)
        orange = self.convert(msg.orange_cones)
        big_orange = self.convert(msg.big_orange_cones)

        try:
            path, debug = self.planner.plan(blue, yellow, orange, big_orange)
        except Exception as e:  # never kill the node because of one bad frame
            self.get_logger().warn(f"Planning failed: {e}")
            return

        if len(path) < 2:
            if self.verbose:
                self.get_logger().info("No path this frame (not enough cones)")
            return

        self.publish_path(path)
        self.publish_visualisation(path, debug["centers"])
        if self.verbose:
            self.get_logger().info(
                f"path: {len(path)} pts, L={len(debug['left'])} R={len(debug['right'])}")

    # ------------------------------------------------------------------
    def publish_path(self, path):
        waypoint_array = WaypointArrayStamped()
        waypoint_array.header.frame_id = "base_footprint"
        waypoint_array.header.stamp = self.get_clock().now().to_msg()
        for x, y in path:
            waypoint_array.waypoints.append(
                Waypoint(position=Point(x=float(x), y=float(y), z=0.0)))
        self.track_line_pub.publish(waypoint_array)

    def _marker(self, ns, mtype, rgb, scale):
        m = Marker()
        m.header.frame_id = "base_footprint"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = 0
        m.type = mtype
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = scale
        m.scale.y = scale
        m.color.a = 1.0
        m.color.r, m.color.g, m.color.b = rgb
        return m

    def publish_visualisation(self, path, centers):
        line = self._marker("trajectory", Marker.LINE_STRIP, (0.0, 1.0, 0.0), 0.15)
        for x, y in path:
            line.points.append(Point(x=float(x), y=float(y), z=0.0))
        self.visualization_pub.publish(line)

        pts = self._marker("centers", Marker.POINTS, (1.0, 0.0, 1.0), 0.3)
        for x, y in centers:
            pts.points.append(Point(x=float(x), y=float(y), z=0.0))
        self.centers_pub.publish(pts)

    @staticmethod
    def convert(cones):
        """ConeWithCovariance[] -> (N, 2) array."""
        if len(cones) == 0:
            return np.zeros((0, 2))
        return np.array([[c.point.x, c.point.y] for c in cones], dtype=float)


def main():
    rclpy.init(args=None)
    node = Planner("delaunay_planner")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass  # context may already be shut down by Ctrl-C


if __name__ == '__main__':
    main()
