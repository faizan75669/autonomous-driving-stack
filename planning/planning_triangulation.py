"""
RRT Path Planning with Triangulation

"""

import time
from operator import add
from typing import List

import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import CubicSpline

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from eufs_msgs.msg import WaypointArrayStamped, ConeArrayWithCovariance, Waypoint


class RRT:
    """
    RRT (Rapidly-exploring Random Tree) planner class.
    """

    def __init__(self, midpoints: np.ndarray, start: np.ndarray, goal: np.ndarray,
                 obstacles_blue: np.ndarray, obstacles_yellow: np.ndarray, max_samples: int = 20):
        self.midpoints = midpoints
        self.start = start
        self.goal = goal
        self.obstacles_blue = obstacles_blue
        self.obstacles_yellow = obstacles_yellow
        self.max_samples = max_samples
        self.tree = [start.tolist()]

    def plan(self) -> List[List[float]]:
        """
        Generate RRT path.
        """
        for _ in range(self.max_samples):
            random_point = self.generate_random_point()
            nearest_point = self.find_nearest_point(random_point)
            steered_points = self.steer(nearest_point, random_point)

            for point in steered_points:
                if not self.collides(point):
                    self.tree.append(point)

        path = self.reconstruct_path()
        return path

    def generate_random_point(self) -> np.ndarray:
        return np.random.uniform(low=self.start, high=self.goal)

    def find_nearest_point(self, target_point: np.ndarray) -> List[float]:
        distances = [np.linalg.norm(target_point - np.array(p)) for p in self.tree]
        nearest_index = np.argmin(distances)
        return self.tree[nearest_index]

    def steer(self, from_point: np.ndarray, to_point: np.ndarray, step_size: float = 0.15, n_steps: int = 10) -> List[List[float]]:
        steered_points = []
        for _ in range(n_steps):
            direction = to_point - from_point
            if np.linalg.norm(direction) != 0:
                direction /= np.linalg.norm(direction)
            from_point = (from_point + step_size * direction).tolist()
            steered_points.append(from_point)
        return steered_points

    def collides(self, point: List[float]) -> bool:
        return point in self.tree

    def reconstruct_path(self) -> List[List[float]]:
        """
        Reconstruct path by projecting tree points onto midpoints line segments.
        """
        if len(self.midpoints) < 2:
            return self.midpoints.tolist()

        path = []
        tolerance = 0.025
        for i in range(len(self.midpoints) - 1):
            start = self.midpoints[i]
            end = self.midpoints[i + 1]
            line_vector = end - start
            line_length = np.linalg.norm(line_vector)
            line_direction = line_vector / line_length if line_length != 0 else np.zeros_like(start)

            for p in self.tree:
                p_vec = np.array(p) - start
                dot = np.dot(line_direction, p_vec)
                projected_point = start + dot * line_direction
                if 0 <= dot <= line_length and np.linalg.norm(np.array(p) - projected_point) <= tolerance:
                    path.append(p)
        return path


class Planner(Node):
    """
    ROS2 Planner Node for midpoints path planning using RRT.
    """

    def __init__(self, name: str = "local_planner"):
        super().__init__(name)
        # Subscribers
        self.cones_sub = self.create_subscription(
            ConeArrayWithCovariance,
            "/ground_truth/cones",
            self.cones_callback,
            1
        )

        # Publishers
        self.track_line_pub = self.create_publisher(WaypointArrayStamped, "/trajectory", 1)
        self.visualization_pub = self.create_publisher(Marker, "/planner/viz", 1)

    def cones_callback(self, msg: ConeArrayWithCovariance):
        """
        Callback function for cone data.
        """
        blue_cones = self.to_2d_list(self.convert(msg.blue_cones))
        yellow_cones = self.to_2d_list(self.convert(msg.yellow_cones))
        orange_cones = np.concatenate((self.convert(msg.orange_cones), self.convert(msg.big_orange_cones)))
        orange_cones = self.to_2d_list(orange_cones)

        orange_mid = self.orange_midpoints(orange_cones)
        midpoints = self.find_midpoints(blue_cones, yellow_cones)

        if not midpoints.any():
            return

        try:
            rrt = RRT(
                midpoints=midpoints,
                start=midpoints[0],
                goal=midpoints[-1],
                obstacles_blue=blue_cones,
                obstacles_yellow=yellow_cones,
                max_samples=20
            )
            path = rrt.plan()

            if len(path) > 1:
                path = self.interpolate_path(path)

            for i in range(len(orange_mid) - 1, -1, -1):
                path = np.concatenate(([orange_mid[i].tolist()], path), axis=0)

            self.publish_path(path)
            self.publish_visualisation(path)

        except Exception as e:
            self.get_logger().info(f"Failed to plan path with RRT: {e}")
            self.publish_path(midpoints)
            self.publish_visualisation(midpoints)
            raise

    def interpolate_path(self, path: List[List[float]]) -> List[List[float]]:
        x_vals = [p[0] for p in path]
        y_vals = [p[1] for p in path]

        cs_x = CubicSpline(np.arange(len(x_vals)), x_vals, bc_type='natural')
        cs_y = CubicSpline(np.arange(len(y_vals)), y_vals, bc_type='natural')

        interp_indices = np.linspace(0, len(x_vals) - 1, num=100)
        interpolated_points = [[cs_x[i], cs_y[i]] for i in interp_indices]
        return interpolated_points

    def orange_midpoints(self, orange_cones: np.ndarray) -> np.ndarray:
        midpoints = []
        if len(orange_cones) >= 4:
            try:
                temp_combined = [i + j for i, j in zip(orange_cones[:2], orange_cones[2:4])]
                for pair in temp_combined:
                    mid = 0.5 * np.array(list(map(add, [pair[0], pair[1]], [pair[2], pair[3]])))
                    midpoints.append(mid)
            except Exception:
                pass
        return np.array(midpoints)

    def find_midpoints(self, blue_cones: np.ndarray, yellow_cones: np.ndarray) -> np.ndarray:
        """
        Compute midpoints between blue and yellow cones using Delaunay triangulation.
        """
        midpoint = []
        min_len = min(len(blue_cones), len(yellow_cones))
        combined = []
        for i in range(min_len):
            combined.append(blue_cones[i])
            combined.append(yellow_cones[i])
        mid_array = np.array(combined)

        tri = Delaunay(mid_array)
        final_pairs = []
        for simplex in tri.simplices:
            has_even = any(n % 2 == 0 for n in simplex)
            has_odd = any(n % 2 != 0 for n in simplex)
            if has_even and has_odd:
                for i in range(len(simplex)):
                    for j in range(i + 1, len(simplex)):
                        if simplex[i] % 2 != simplex[j] % 2:
                            final_pairs.append([simplex[i], simplex[j]])

        for pair in final_pairs:
            x_blue, y_blue = mid_array[pair[0]] if pair[0] % 2 == 0 else mid_array[pair[1]]
            x_yellow, y_yellow = mid_array[pair[1]] if pair[1] % 2 != 0 else mid_array[pair[0]]
            midpoint.append([(x_blue + x_yellow) / 2, (y_blue + y_yellow) / 2])

        return np.array(midpoint)

    @staticmethod
    def to_2d_list(arr: np.ndarray) -> List[List[float]]:
        return [[c.real, c.imag] for c in arr]

    @staticmethod
    def convert(cones):
        return np.array([c.point.x + 1j * c.point.y for c in cones])

    def publish_path(self, points: np.ndarray):
        msg = WaypointArrayStamped()
        msg.header.frame_id = "base_footprint"
        msg.header.stamp = self.get_clock().now().to_msg()
        for p in points:
            msg.waypoints.append(Waypoint(position=Point(x=p[0], y=p[1])))
        self.track_line_pub.publish(msg)

    def publish_visualisation(self, points: np.ndarray):
        marker = Marker()
        marker.header.frame_id = "base_footprint"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.scale.x = marker.scale.y = 0.35
        marker.id = 0
        marker.ns = "midpoints"
        for p in points:
            marker.points.append(Point(x=p[0], y=p[1]))
        self.visualization_pub.publish(marker)


def main():
    rclpy.init()
    node = Planner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

