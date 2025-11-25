
"""
pure_pursuit.py

ROS2 node implementing a Pure Pursuit-style controller with:
 - dynamic look-ahead based on cone-curvature detection,
 - pure-pursuit steering,
 - simple PID acceleration control,
 - RViz visualization (MarkerArray).

"""
from __future__ import annotations

import math
import time
from typing import Tuple, Optional

import numpy as np
from scipy.optimize import curve_fit

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from eufs_msgs.msg import WaypointArrayStamped, ConeArrayWithCovariance, CarState
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


# ----------------- Utility functions ----------------- #
def poly2(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Quadratic polynomial y = a*x^2 + b*x + c (vectorized)."""
    return a * x**2 + b * x + c


def curvature_quad(a: float, b: float, x: float) -> float:
    """
    Curvature for quadratic y = a*x^2 + b*x + c at x:
      k(x) = |y''| / (1 + (y')^2)^(3/2)
      y' = 2*a*x + b, y'' = 2*a
    Returns large number for near-zero denominator handled safely.
    """
    y_prime = 2.0 * a * x + b
    y_double = 2.0 * a
    denom = (1.0 + y_prime**2) ** 1.5
    if denom == 0:
        return float("inf")
    return abs(y_double) / denom


def safe_to_2d_array(arr) -> np.ndarray:
    """Ensure arr becomes a Nx2 numpy array or empty Nx2 array if input empty."""
    a = np.array(arr)
    if a.size == 0:
        return np.empty((0, 2))
    return a.reshape(-1, 2)


# ----------------- Node ----------------- #
class ConePursuitNode(Node):
    """ROS2 node that subscribes to cones/path/state and publishes Ackermann commands."""

    def __init__(self):
        super().__init__("cone_pursuit")

        # QoS for camera-like topics
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ----- Parameters (tune in params.yaml instead of editing code) -----
        self.declare_parameter("cone_topic_cam0", "/camera_0/cones")
        self.declare_parameter("cone_topic_cam1", "/camera_1/cones")
        self.declare_parameter("path_topic", "/filtered_path")
        self.declare_parameter("state_topic", "/ground_truth/state")
        self.declare_parameter("odom_topic", "/ground_truth/odom")
        self.declare_parameter("cmd_topic", "/cmd")

        self.declare_parameter("L", 1.53)  # wheelbase-ish parameter
        self.declare_parameter("min_lookahead", 2.5)
        self.declare_parameter("max_lookahead", 6.0)
        self.declare_parameter("curvature_threshold", 20.0)
        self.declare_parameter("safe_speed", 2.5)
        self.declare_parameter("max_speed", 4.5)

        # PID params for acceleration
        self.declare_parameter("pid_kp", 2.0)
        self.declare_parameter("pid_ki", 0.06)
        self.declare_parameter("pid_kd", 0.08)

        # Read parameters
        self.topic_cam0 = self.get_parameter("cone_topic_cam0").value
        self.topic_cam1 = self.get_parameter("cone_topic_cam1").value
        self.path_topic = self.get_parameter("path_topic").value
        self.state_topic = self.get_parameter("state_topic").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.cmd_topic = self.get_parameter("cmd_topic").value

        self.L = float(self.get_parameter("L").value)
        self.min_lookahead = float(self.get_parameter("min_lookahead").value)
        self.max_lookahead = float(self.get_parameter("max_lookahead").value)
        self.curvature_threshold = float(self.get_parameter("curvature_threshold").value)
        self.safe_speed = float(self.get_parameter("safe_speed").value)
        self.max_speed = float(self.get_parameter("max_speed").value)

        self.kp = float(self.get_parameter("pid_kp").value)
        self.ki = float(self.get_parameter("pid_ki").value)
        self.kd = float(self.get_parameter("pid_kd").value)

        # ----- Internal state -----
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.speed = 0.0
        self.yaw = 0.0

        self.blue_cones = np.empty((0, 2))
        self.yellow_cones = np.empty((0, 2))

        # Simple PID internals
        self._pid_prev_error = 0.0
        self._pid_integral = 0.0
        self._pid_last_time = None

        # ----- Publishers / Subscribers -----
        self.create_subscription(ConeArrayWithCovariance, self.topic_cam0, self._cones0_cb, qos)
        self.create_subscription(ConeArrayWithCovariance, self.topic_cam1, self._cones1_cb, qos)
        self.create_subscription(WaypointArrayStamped, self.path_topic, self._path_cb, 10)
        self.create_subscription(CarState, self.state_topic, self._state_cb, 10)
        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, qos)

        self._cmd_pub = self.create_publisher(AckermannDriveStamped, self.cmd_topic, 10)
        self._viz_pub = self.create_publisher(MarkerArray, "visualization_marker_array", 10)

        self.get_logger().info("cone_pursuit node started")

    # ----------------- Callbacks ----------------- #
    def _cones0_cb(self, msg: ConeArrayWithCovariance) -> None:
        """Blue cones callback (camera 0)."""
        self.blue_cones = safe_to_2d_array([[c.point.x, c.point.y] for c in msg.blue_cones])

    def _cones1_cb(self, msg: ConeArrayWithCovariance) -> None:
        """Yellow cones callback (camera 1)."""
        self.yellow_cones = safe_to_2d_array([[c.point.x, c.point.y] for c in msg.yellow_cones])

    def _state_cb(self, msg: CarState) -> None:
        """Car state callback (position and speed)."""
        self.speed = float(msg.twist.twist.linear.x)
        self.pos_x = float(msg.pose.pose.position.x)
        self.pos_y = float(msg.pose.pose.position.y)

    def _odom_cb(self, msg: Odometry) -> None:
        """Odometry callback for yaw extraction (quaternion -> yaw)."""
        q = msg.pose.pose.orientation
        # yaw (z) from quaternion
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))

    def _path_cb(self, msg: WaypointArrayStamped) -> None:
        """Main path callback: compute look-ahead, steering, acceleration, publish commands."""
        path = self._convert_waypoints(msg)
        if path.size == 0:
            self.get_logger().debug("Received empty path; skipping control update.")
            return

        # compute look-ahead index
        lookahead_idx = self._get_look_ahead_index(path)

        # steering via pure pursuit
        steering = self._compute_steering(path, lookahead_idx)

        # speed target from curvature on cones
        speed_target = self._get_speed_target()

        # PID acceleration command
        acceleration = self._pid_acceleration(speed_target)

        # publish command
        self._publish_cmd(steering, acceleration)

        # publish visualization markers
        self._publish_visualization(path, lookahead_idx)

    # ----------------- Core algorithms ----------------- #
    def _convert_waypoints(self, msg: WaypointArrayStamped) -> np.ndarray:
        """Convert WaypointArrayStamped to Nx2 numpy array in global frame (adds current pos)."""
        try:
            pts = np.array([[p.position.x + self.pos_x, p.position.y + self.pos_y] for p in msg.waypoints])
            if pts.size == 0:
                return np.empty((0, 2))
            return pts.reshape(-1, 2)
        except Exception as e:
            self.get_logger().warn(f"Failed to convert waypoints: {e}")
            return np.empty((0, 2))

    def _get_look_ahead_index(self, path: np.ndarray) -> int:
        """
        Determine a look-ahead index on `path` using cone curvature:
         - Fit quadratics to both cone sets (if enough points).
         - Compute min curvature and map to look-ahead distance.
         - Find first path point with distance >= look-ahead distance.
        """
        px, py = self.pos_x, self.pos_y

        # Ensure proper shapes
        blue = safe_to_2d_array(self.blue_cones)
        yellow = safe_to_2d_array(self.yellow_cones)

        # Compute minimal curvature from fitted quadratic curves
        min_curv = float("inf")
        try:
            if blue.shape[0] >= 3 and yellow.shape[0] >= 3:
                bx, by = blue[:, 0], blue[:, 1]
                yx, yy = yellow[:, 0], yellow[:, 1]
                bparams, _ = curve_fit(poly2, bx, by)
                yparams, _ = curve_fit(poly2, yx, yy)
                bcurv = [curvature_quad(bparams[0], bparams[1], x) for x in bx]
                ycurv = [curvature_quad(yparams[0], yparams[1], x) for x in yx]
                min_curv = min(min(bcurv), min(ycurv))
        except Exception as e:
            self.get_logger().debug(f"Curve fitting failed or unstable: {e}")
            min_curv = float("inf")

        # Map curvature to lookahead distance
        if min_curv < self.curvature_threshold:
            dynamic_lookahead = max(self.min_lookahead, min(self.max_lookahead, 1.0 / (min_curv + 1e-6)))
        else:
            dynamic_lookahead = self.max_lookahead

        # Find first path point >= dynamic_lookahead away from car
        for i, (x, y) in enumerate(path):
            d = math.hypot(x - px, y - py)
            if d >= dynamic_lookahead:
                return i

        # fallback to last index
        return len(path) - 1

    def _compute_steering(self, path: np.ndarray, lookahead_idx: int) -> float:
        """Pure pursuit steering calculation to compute steering angle (radians)."""
        if lookahead_idx >= len(path):
            lookahead_idx = len(path) - 1
        target_x, target_y = float(path[lookahead_idx][0]), float(path[lookahead_idx][1])

        # relative vector from car to look-ahead point (in global frame)
        dx = target_x - self.pos_x
        dy = target_y - self.pos_y
        alpha = math.atan2(dy, dx)  # heading to target (global)

        # distance to the target point
        distance = math.hypot(dx, dy)
        dynamic_lookahead = max(0.001, distance + 0.1 * self.speed)

        steering = math.atan2(2.0 * self.L * math.sin(alpha), dynamic_lookahead)

        # clip to reasonable steering (radians)
        return float(np.clip(steering, -0.5, 0.5))

    def _get_speed_target(self) -> float:
        """Return safe_speed if curvature detected, else max_speed."""
        # try to detect curvature using available cones
        blue = safe_to_2d_array(self.blue_cones)
        yellow = safe_to_2d_array(self.yellow_cones)
        try:
            if blue.shape[0] >= 3 and yellow.shape[0] >= 3:
                bx, by = blue[:, 0], blue[:, 1]
                yx, yy = yellow[:, 0], yellow[:, 1]
                bparams, _ = curve_fit(poly2, bx, by)
                yparams, _ = curve_fit(poly2, yx, yy)
                bcurv = [curvature_quad(bparams[0], bparams[1], x) for x in bx]
                ycurv = [curvature_quad(yparams[0], yparams[1], x) for x in yx]
                min_curv = min(min(bcurv), min(ycurv))
            else:
                min_curv = float("inf")
        except Exception:
            min_curv = float("inf")

        if min_curv < self.curvature_threshold:
            self.get_logger().debug("High curvature detected -> using safe_speed")
            return self.safe_speed
        return self.max_speed

    def _pid_acceleration(self, speed_target: float) -> float:
        """Simple PID controller for acceleration with basic anti-windup and dt handling."""
        now = time.time()
        error = speed_target - self.speed
        if self._pid_last_time is None:
            dt = 1e-3
        else:
            dt = max(1e-3, now - self._pid_last_time)

        # integral with simple anti-windup via clamping
        self._pid_integral += error * dt
        # derivative
        derivative = (error - self._pid_prev_error) / dt

        # PID output
        acc = self.kp * error + self.ki * self._pid_integral + self.kd * derivative

        # update state
        self._pid_prev_error = error
        self._pid_last_time = now

        # clamp acceleration to reasonable bounds
        acc = float(max(min(acc, 5.0), -5.0))
        return acc

    # ----------------- Publishing / Visualization ----------------- #
    def _publish_cmd(self, steering: float, acceleration: float) -> None:
        """Publish AckermannDriveStamped command."""
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "pure_pursuit"
        msg.drive.steering_angle = steering
        msg.drive.acceleration = acceleration
        self._cmd_pub.publish(msg)

    def _publish_visualization(self, path: np.ndarray, lookahead_index: int) -> None:
        """Publish MarkerArray for RViz (path line and look-ahead sphere)."""
        markers = MarkerArray()

        # path line strip
        path_marker = Marker()
        path_marker.header.stamp = self.get_clock().now().to_msg()
        path_marker.header.frame_id = "base_footprint"
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.scale.x = 0.05
        path_marker.color.a = 1.0
        path_marker.color.r = 1.0
        path_marker.color.g = 0.0
        path_marker.color.b = 0.0

        for px, py in path:
            p = Point()
            p.x = float(px)
            p.y = float(py)
            path_marker.points.append(p)

        markers.markers.append(path_marker)

        # look-ahead sphere
        if 0 <= lookahead_index < len(path):
            look = Marker()
            look.header.stamp = self.get_clock().now().to_msg()
            look.header.frame_id = "base_footprint"
            look.id = 1
            look.type = Marker.SPHERE
            look.action = Marker.ADD
            look.scale.x = look.scale.y = look.scale.z = 0.6
            look.color.a = 1.0
            look.color.g = 1.0
            p = Point()
            p.x = float(path[lookahead_index][0])
            p.y = float(path[lookahead_index][1])
            look.pose.position = p
            markers.markers.append(look)

        self._viz_pub.publish(markers)


# ----------------- Entry point ----------------- #
def main(args=None):
    rclpy.init(args=args)
    node = ConePursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down cone_pursuit node")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
