"""
stanley_node.py

ROS2 node implementing a Stanley lateral controller with a simple PID speed controller.

Subscriptions:
 - /trajectory            (WaypointArrayStamped)  : planned path (list of waypoints)
 - /ground_truth/state    (CarState)             : pose + twist (used for position and velocity)
 - /imu/data              (sensor_msgs/Imu)      : optional, used for yaw if available
 - /cmd                   (AckermannDriveStamped): optional, listens to external commands

Publications:
 - /cmd                   (AckermannDriveStamped) : steering & acceleration commands
 - /control/viz           (visualization_msgs/Marker) : simple text marker showing current controls

"""

import math
import time
from typing import List, Tuple, Optional

import numpy as np
import rclpy
from rclpy.node import Node

from eufs_msgs.msg import WaypointArrayStamped, CarState
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry  # optional if you want to switch to odom for yaw



def normalize_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Convert quaternion to yaw (Z-axis rotation)."""
    # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between 2D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def project_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> Tuple[float, float, float]:
    """
    Project point P(px,py) onto line segment A(ax,ay)-B(bx,by).
    Returns (closest_x, closest_y, t) where t in [0,1] is fraction along AB.
    """
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    seg_len2 = vx * vx + vy * vy
    if seg_len2 == 0.0:
        return ax, ay, 0.0
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len2))
    cx = ax + t * vx
    cy = ay + t * vy
    return cx, cy, t


# ---------------- Stanley Controller Node ----------------- #


class StanleyNode(Node):
    def __init__(self):
        super().__init__("stanley_controller")

        # --- Topics (make configurable) ---
        self.declare_parameter("path_topic", "/trajectory")
        self.declare_parameter("state_topic", "/ground_truth/state")
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("cmd_topic", "/cmd")
        self.declare_parameter("viz_topic", "/control/viz")

        path_topic = self.get_parameter("path_topic").value
        state_topic = self.get_parameter("state_topic").value
        imu_topic = self.get_parameter("imu_topic").value
        cmd_topic = self.get_parameter("cmd_topic").value
        viz_topic = self.get_parameter("viz_topic").value

        # --- Controller parameters (tune as needed) ---
        self.declare_parameter("stanley_k", 2.5)  # control gain (k)
        self.declare_parameter("stanley_soft_k", 1.0)  # softening gain to avoid division by zero
        self.declare_parameter("max_steer", 0.418879)  # 24 degrees in radians
        self.declare_parameter("wheelbase", 1.53)  # m, set appropriately for your vehicle

        # PID params for speed control
        self.declare_parameter("pid_kp", 0.13)
        self.declare_parameter("pid_ki", 0.02)
        self.declare_parameter("pid_kd", 0.001)
        self.declare_parameter("pid_dt", 0.1)  # used for initial integrator scaling if needed

        # Read parameters
        self.k = float(self.get_parameter("stanley_k").value)
        self.k_soft = float(self.get_parameter("stanley_soft_k").value)
        self.max_steer = float(self.get_parameter("max_steer").value)
        self.wheelbase = float(self.get_parameter("wheelbase").value)

        self.kp = float(self.get_parameter("pid_kp").value)
        self.ki = float(self.get_parameter("pid_ki").value)
        self.kd = float(self.get_parameter("pid_kd").value)
        self.pid_dt = float(self.get_parameter("pid_dt").value)

        # --- State ---
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0  # vehicle heading (rad)
        self.vx = 0.0  # body-frame forward velocity (m/s)
        self.vy = 0.0

        # PID internals
        self._prev_speed_error = 0.0
        self._integral_speed_error = 0.0
        self._last_pid_time = None

        # --- Trajectory storage ---
        self.path_x: List[float] = []
        self.path_y: List[float] = []
        self.path_yaw: List[float] = []  # precomputed path yaw between consecutive waypoints

        # Last target index used (for continuity)
        self._last_target_idx = 0

        # Publishers / Subscribers
        qos = 10
        self.path_sub = self.create_subscription(WaypointArrayStamped, path_topic, self._path_cb, qos)
        self.state_sub = self.create_subscription(CarState, state_topic, self._state_cb, qos)
        self.imu_sub = self.create_subscription(Imu, imu_topic, self._imu_cb, qos)
        # optional external commands listener
        self.cmd_in_sub = self.create_subscription(AckermannDriveStamped, cmd_topic, self._cmd_in_cb, qos)

        self.cmd_pub = self.create_publisher(AckermannDriveStamped, cmd_topic, qos)
        self.viz_pub = self.create_publisher(Marker, viz_topic, qos)

        self.get_logger().info("Stanley controller node started")

    # ---------------- Callbacks ---------------- #
    def _cmd_in_cb(self, msg: AckermannDriveStamped) -> None:
        # If you want to monitor external commands, handle here (optional)
        pass

    def _imu_cb(self, msg: Imu) -> None:
        """Extract yaw from IMU quaternion if present (overwrites yaw if used)."""
        q = msg.orientation
        # keep yaw if quaternion is invalid
        try:
            self.yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        except Exception:
            pass

    def _state_cb(self, msg: CarState) -> None:
        """Update vehicle pose and velocity from CarState (ground truth or estimator)."""
        # Position
        self.x = float(msg.pose.pose.position.x)
        self.y = float(msg.pose.pose.position.y)

        # NOTE: CarState pose.orientation may be quaternion; if yaw not available from IMU you can compute
        # But original code used yaw_state from somewhere else; we'll keep IMU or compute from msg if provided.
        # Velocities
        self.vx = float(msg.twist.twist.linear.x)
        self.vy = float(msg.twist.twist.linear.y)

    def _path_cb(self, msg: WaypointArrayStamped) -> None:
        """Receive planned waypoints and run control (if enough points)."""
        if len(msg.waypoints) < 2:
            self.get_logger().debug("Received path with <2 waypoints — ignoring")
            return

        # Replace path (instead of appending) to avoid unbounded growth
        self.path_x = [float(p.position.x) for p in msg.waypoints]
        self.path_y = [float(p.position.y) for p in msg.waypoints]

        # Precompute path yaw (heading between consecutive waypoints)
        self.path_yaw = []
        for i in range(len(self.path_x) - 1):
            dx = self.path_x[i + 1] - self.path_x[i]
            dy = self.path_y[i + 1] - self.path_y[i]
            self.path_yaw.append(math.atan2(dy, dx))
        # keep last yaw equal to the previous (or duplicate last)
        if self.path_yaw:
            self.path_yaw.append(self.path_yaw[-1])
        else:
            self.path_yaw = [0.0] * len(self.path_x)

        # Run controller
        accel_cmd, steer_cmd = self._control_main()
        self.get_logger().debug(f"Control outputs -> accel: {accel_cmd:.3f}, steer: {steer_cmd:.3f}")

        # Publish
        self._publish_cmd(accel_cmd, steer_cmd)
        self._publish_viz(accel_cmd, steer_cmd)

    # ---------------- Core computations ---------------- #
    def _calc_target_index(self) -> Tuple[int, float]:
        """
        Return index of nearest path point to the vehicle's front axle projection
        and the signed cross-track error (positive = vehicle is to the left of path).
        """
        if not self.path_x:
            return 0, 0.0

        # front axle position (approximate) using wheelbase
        fx = self.x + self.wheelbase * math.cos(self.yaw)
        fy = self.y + self.wheelbase * math.sin(self.yaw)

        # distances to all waypoints
        dists = [math.hypot(fx - px, fy - py) for px, py in zip(self.path_x, self.path_y)]
        target_idx = int(np.argmin(dists))

        # Project front axle onto segment between target_idx and target_idx+1 (if possible)
        if target_idx < len(self.path_x) - 1:
            ax, ay = self.path_x[target_idx], self.path_y[target_idx]
            bx, by = self.path_x[target_idx + 1], self.path_y[target_idx + 1]
            cx, cy, t = project_point_to_segment(fx, fy, ax, ay, bx, by)
            # cross track error: signed distance from path to vehicle
            # compute sign using path normal (rotate path tangent by +90 deg)
            tangent_x = bx - ax
            tangent_y = by - ay
            normal_x = -tangent_y
            normal_y = tangent_x
            # vector from path point to front axle
            vx = fx - cx
            vy = fy - cy
            cross_err = (vx * normal_x + vy * normal_y) / math.hypot(normal_x, normal_y)
        else:
            # if last point, signed distance to that waypoint
            px_last, py_last = self.path_x[-1], self.path_y[-1]
            dx = fx - px_last
            dy = fy - py_last
            cross_err = math.hypot(dx, dy)
            # sign by comparing heading
            path_yaw = self.path_yaw[-1] if self.path_yaw else 0.0
            cross_err *= math.copysign(1.0, math.sin(self.yaw - path_yaw))

        return target_idx, cross_err

    def _stanley_steer(self, last_idx: int) -> Tuple[float, int]:
        """Compute steering using Stanley method. Returns (steering, used_target_idx)."""
        target_idx, cross_err = self._calc_target_index()

        # ensure monotonic target index (avoid going backwards)
        if last_idx >= target_idx:
            target_idx = last_idx

        # heading error between path and vehicle
        path_heading = self.path_yaw[target_idx] if target_idx < len(self.path_yaw) else self.path_yaw[-1]
        heading_error = normalize_angle(path_heading - self.yaw)

        # speed: use magnitude of velocity
        v = math.sqrt(self.vx ** 2 + self.vy ** 2)
        # softening to avoid division by zero
        denom = v + self.k_soft

        # cross-track correction
        cross_term = math.atan2(self.k * cross_err, denom)

        steering = heading_error + cross_term
        steering = normalize_angle(steering)
        # clip to limits
        steering = float(np.clip(steering, -self.max_steer, self.max_steer))

        return steering, target_idx

    def _pid_speed(self, target_speed: float) -> float:
        """Simple PID controller for longitudinal acceleration command (not throttle)."""
        now = time.time()
        if self._last_pid_time is None:
            dt = max(1e-3, self.pid_dt)
        else:
            dt = max(1e-3, now - self._last_pid_time)

        current_speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
        error = target_speed - current_speed

        # integral with basic anti-windup by clamping integral
        self._integral_speed_error += error * dt
        # derivative
        derivative = (error - self._prev_speed_error) / dt

        acc = self.kp * error + self.ki * self._integral_speed_error + self.kd * derivative

        # update state
        self._prev_speed_error = error
        self._last_pid_time = now

        # clamp acceleration to reasonable limits
        acc = float(np.clip(acc, -5.0, 5.0))
        return acc

    # ---------------- Main control loop wrapper ---------------- #
    def _control_main(self) -> Tuple[float, float]:
        """
        Returns (acceleration_command, steering_command).
        If path is empty, returns zeros.
        """
        if len(self.path_x) < 2:
            return 0.0, 0.0

        # example target speed (tune or derive from path)
        target_speed = 15.0  # m/s, change as needed or compute per waypoint

        # compute steering
        steering, idx = self._stanley_steer(self._last_target_idx)
        self._last_target_idx = idx

        # compute acceleration via PID
        accel = self._pid_speed(target_speed)

        return accel, steering

    # ---------------- Publish & Viz ---------------- #
    def _publish_cmd(self, acceleration: float, steering: float) -> None:
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "stanley_controller"
        msg.drive.steering_angle = float(steering)
        msg.drive.acceleration = float(acceleration)
        # optional: set speed or velocity field if available in your Ackermann message
        self.cmd_pub.publish(msg)

    def _publish_viz(self, accel: float, steer: float) -> None:
        """Simple text marker to show speed and steering in RViz."""
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "base_footprint"
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.ns = "controls"
        marker.id = 0
        marker.pose.position.x = 3.0
        marker.pose.position.y = 4.0
        marker.pose.position.z = 1.0
        marker.scale.x = 0.35
        marker.scale.y = 0.35
        marker.scale.z = 0.5
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        current_speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
        marker.text = f"Speed: {current_speed:.2f} m/s\nAccel(cmd): {accel:.2f}\nSteer(cmd): {steer:.3f} rad"
        self.viz_pub.publish(marker)


# ----------------- Node entrypoint ----------------- #
def main(args=None):
    rclpy.init(args=args)
    node = StanleyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Stanley controller")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
