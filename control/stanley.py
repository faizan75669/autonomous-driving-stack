"""ROS 2 Stanley controller for a local ``base_footprint`` trajectory."""
from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from eufs_msgs.msg import WaypointArrayStamped, CarState
from ackermann_msgs.msg import AckermannDriveStamped


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def project_to_segment(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    d2 = vx * vx + vy * vy
    if d2 < 1e-12:
        return ax, ay
    t = float(np.clip(((px - ax) * vx + (py - ay) * vy) / d2, 0.0, 1.0))
    return ax + t * vx, ay + t * vy


def stanley_steering(path: np.ndarray, speed: float, wheelbase: float,
                     gain: float, softening: float, max_steer: float) -> float:
    """Stanley steering for a path whose coordinates are vehicle-local."""
    if len(path) < 2:
        return 0.0
    # Stanley convention: evaluate error at the front axle.
    fx, fy = wheelbase, 0.0
    dists = np.linalg.norm(path - np.array([fx, fy]), axis=1)
    idx = int(np.argmin(dists))
    j = min(idx + 1, len(path) - 1)
    if j == idx:
        j = max(0, idx - 1)
    ax, ay = path[idx]
    bx, by = path[j]
    cx, cy = project_to_segment(fx, fy, ax, ay, bx, by)
    tx, ty = bx - ax, by - ay
    length = math.hypot(tx, ty)
    if length < 1e-9:
        return 0.0

    path_heading = math.atan2(ty, tx)
    # Vehicle heading is zero in its local frame.
    heading_error = normalize_angle(path_heading)
    # Positive means front axle is left of the path tangent.
    cross_track = ((fx - cx) * (-ty) + (fy - cy) * tx) / length
    correction = math.atan2(gain * cross_track, abs(speed) + softening)
    return float(np.clip(normalize_angle(heading_error + correction), -max_steer, max_steer))


class StanleyNode(Node):
    def __init__(self) -> None:
        super().__init__("stanley_controller")
        self.declare_parameter("path_topic", "/trajectory")
        self.declare_parameter("state_topic", "/ground_truth/state")
        self.declare_parameter("cmd_topic", "/cmd")
        self.declare_parameter("stanley_k", 2.5)
        self.declare_parameter("stanley_softening", 1.0)
        self.declare_parameter("wheelbase", 1.53)
        self.declare_parameter("max_steer", 0.418879)
        self.declare_parameter("target_speed", 4.5)
        self.declare_parameter("kp_speed", 1.0)
        self.declare_parameter("ki_speed", 0.0)
        self.declare_parameter("kd_speed", 0.05)
        self.declare_parameter("max_accel", 5.0)

        self.k = float(self.get_parameter("stanley_k").value)
        self.softening = float(self.get_parameter("stanley_softening").value)
        self.L = float(self.get_parameter("wheelbase").value)
        self.max_steer = float(self.get_parameter("max_steer").value)
        self.target_speed = float(self.get_parameter("target_speed").value)
        self.kp = float(self.get_parameter("kp_speed").value)
        self.ki = float(self.get_parameter("ki_speed").value)
        self.kd = float(self.get_parameter("kd_speed").value)
        self.max_accel = float(self.get_parameter("max_accel").value)
        self.speed = 0.0
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time: Optional[float] = None

        self.create_subscription(WaypointArrayStamped, self.get_parameter("path_topic").value, self._path_cb, 10)
        self.create_subscription(CarState, self.get_parameter("state_topic").value, self._state_cb, 10)
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, self.get_parameter("cmd_topic").value, 10)
        self.get_logger().info("Stanley controller started (local-frame path)")

    def _state_cb(self, msg: CarState) -> None:
        self.speed = float(msg.twist.twist.linear.x)

    def _speed_control(self) -> float:
        now = time.monotonic()
        error = self.target_speed - self.speed
        dt = 0.01 if self.previous_time is None else max(1e-3, now - self.previous_time)
        self.integral = float(np.clip(self.integral + error * dt, -10.0, 10.0))
        derivative = (error - self.previous_error) / dt
        acceleration = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error, self.previous_time = error, now
        return float(np.clip(acceleration, -self.max_accel, self.max_accel))

    def _path_cb(self, msg: WaypointArrayStamped) -> None:
        path = np.asarray([[p.position.x, p.position.y] for p in msg.waypoints], dtype=float)
        if len(path) < 2:
            return
        if str(msg.header.frame_id) not in ("", "base_footprint"):
            self.get_logger().warn_once("Expected base_footprint trajectory; frame mismatch may invalidate control")
        steering = stanley_steering(path, self.speed, self.L, self.k, self.softening, self.max_steer)
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_footprint"
        cmd.drive.steering_angle = steering
        cmd.drive.acceleration = self._speed_control()
        self.cmd_pub.publish(cmd)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StanleyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
