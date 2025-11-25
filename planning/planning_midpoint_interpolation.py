"""
Basic Planning code using midpoints between yellow and blue cones with path interpolation. 
Note: The algorithm was producing irregular trajectories around corners.

"""

from eufs_msgs.msg import WaypointArrayStamped, Waypoint, ConeArrayWithCovariance, ConeWithCovariance
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from rclpy.node import Node
import rclpy
import numpy as np
from scipy import interpolate
from typing import List

class Planner(Node):
    def __init__(self, name: str):
        super().__init__(name)
        self.threshold = self.declare_parameter("threshold", 6.0).value

        self.cones_sub = self.create_subscription(
            ConeArrayWithCovariance, 
            "/fusion/cones", 
            self.cones_callback, 
            1
        )

        self.track_line_pub = self.create_publisher(WaypointArrayStamped, "/trajectory", 1)
        self.visualization_pub = self.create_publisher(Marker, "/planner/viz", 1)

    def cones_callback(self, msg: ConeArrayWithCovariance):
        blue_cones = self.convert(msg.blue_cones)
        yellow_cones = self.convert(msg.yellow_cones)
        orange_cones = np.concatenate(
            (self.convert(msg.orange_cones), self.convert(msg.big_orange_cones))
        )

        midpoints = self.compute_midpoints(blue_cones, yellow_cones, orange_cones)
        midpoints = self.sort_midpoints(midpoints)

        if len(midpoints) == 0:
            return

        try:
            tck, _ = interpolate.splprep([midpoints[:, 0], midpoints[:, 1]], s=100, k=min(3, len(midpoints) - 1))
            midpoints_interp = interpolate.splev(np.linspace(0, 1, 100), tck)
            midpoints = np.vstack(midpoints_interp).T
        except Exception:
            pass

        self.publish_path(midpoints)
        self.publish_visualization(midpoints)

    def compute_midpoints(self, blue_cones: np.ndarray, yellow_cones: np.ndarray, orange_cones: np.ndarray = None):
        if len(blue_cones) == 0 or len(yellow_cones) == 0:
            return np.array([])

        min_len = min(len(blue_cones), len(yellow_cones))
        midpoints = [(blue_cones[i] + yellow_cones[i]) / 2.0 for i in range(min_len)]

        if orange_cones is not None and len(orange_cones) >= 2:
            start_orange = np.mean(orange_cones[:2], axis=0)
            end_orange = np.mean(orange_cones[-2:], axis=0)
            midpoints.insert(0, start_orange)
            midpoints.append(end_orange)

        return np.array(midpoints)

    def sort_midpoints(self, midpoints: np.ndarray):
        if len(midpoints) < 2:
            return midpoints

        midpoints = midpoints.tolist()
        sorted_points = [midpoints.pop(0)]

        while midpoints:
            last_point = np.array(sorted_points[-1])
            distances = [np.linalg.norm(last_point - np.array(p)) for p in midpoints]
            nearest_index = int(np.argmin(distances))
            sorted_points.append(midpoints.pop(nearest_index))

        return np.array(sorted_points)

    def publish_path(self, midpoints: np.ndarray):
        waypoint_array = WaypointArrayStamped()
        waypoint_array.header.frame_id = "base_footprint"
        waypoint_array.header.stamp = self.get_clock().now().to_msg()

        for p in midpoints:
            waypoint = Waypoint(position=Point(x=p[0], y=p[1]))
            waypoint_array.waypoints.append(waypoint)

        self.track_line_pub.publish(waypoint_array)

    def publish_visualization(self, midpoints: np.ndarray):
        marker = Marker()
        marker.header.frame_id = "base_footprint"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.id = 0
        marker.scale.x = 0.35
        marker.scale.y = 0.35
        marker.ns = "midpoints"

        for midpoint in midpoints:
            marker.points.append(Point(x=midpoint[0], y=midpoint[1]))

        self.visualization_pub.publish(marker)

    def convert(self, cones: List[ConeWithCovariance], struct: str = '') -> np.ndarray:
        if struct == "complex":
            return np.array([c.point.x + 1j * c.point.y for c in cones])
        return np.array([[c.point.x, c.point.y] for c in cones])

def main():
    rclpy.init(args=None)
    node = Planner("local_planner")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

