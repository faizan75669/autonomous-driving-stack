"""ROS 2 Pure Pursuit controller for a local ``base_footprint`` path."""
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


def pure_pursuit_steering(target_x: float, target_y: float, wheelbase: float, lookahead: float,
                          max_steer: float = 0.5) -> float:
    """Return Ackermann steering for a target expressed in vehicle-local coordinates."""
    if lookahead <= 1e-6:
        return 0.0
    alpha = math.atan2(target_y, target_x)
    delta = math.atan2(2.0 * wheelbase * math.sin(alpha), lookahead)
    return float(np.clip(delta, -max_steer, max_steer))


class PurePursuitNode(Node):
    def __init__(self) -> None:
        super().__init__("pure_pursuit")
        self.declare_parameter("path_topic", "/trajectory")
        self.declare_parameter("state_topic", "/ground_truth/state")
        self.declare_parameter("cmd_topic", "/cmd")
        self.declare_parameter("wheelbase", 1.53)
        self.declare_parameter("min_lookahead", 2.5)
        self.declare_parameter("max_lookahead", 6.0)
        self.declare_parameter("lookahead_gain", 0.4)
        self.declare_parameter("max_steer", 0.5)
        self.declare_parameter("target_speed", 4.5)
        self.declare_parameter("kp_speed", 1.0)
        self.declare_parameter("ki_speed", 0.0)
        self.declare_parameter("kd_speed", 0.05)
        self.declare_parameter("max_accel", 5.0)
        self.L = float(self.get_parameter("wheelbase").value)
        self.min_ld = float(self.get_parameter("min_lookahead").value)
        self.max_ld = float(self.get_parameter("max_lookahead").value)
        self.ld_gain = float(self.get_parameter("lookahead_gain").value)
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
        self.frame_warning_sent = False
        self.create_subscription(WaypointArrayStamped, self.get_parameter("path_topic").value, self._path_cb, 10)
        self.create_subscription(CarState, self.get_parameter("state_topic").value, self._state_cb, 10)
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, self.get_parameter("cmd_topic").value, 10)
        self.get_logger().info("Pure Pursuit controller started (local-frame path)")

    def _state_cb(self, msg: CarState) -> None:
        self.speed = float(msg.twist.twist.linear.x)

    @staticmethod
    def _path(msg: WaypointArrayStamped) -> np.ndarray:
        return np.asarray([[p.position.x, p.position.y] for p in msg.waypoints], dtype=float)

    def _select_target(self, path: np.ndarray) -> tuple[np.ndarray, float]:
        lookahead = float(np.clip(self.min_ld + self.ld_gain * abs(self.speed), self.min_ld, self.max_ld))
        distances = np.linalg.norm(path, axis=1)
        candidates = np.flatnonzero(distances >= lookahead)
        idx = int(candidates[0]) if len(candidates) else len(path) - 1
        return path[idx], max(float(distances[idx]), 1e-3)

    def _speed_pid(self) -> float:
        now = time.monotonic()
        error = self.target_speed - self.speed
        dt = 0.01 if self.previous_time is None else max(1e-3, now - self.previous_time)
        self.integral = float(np.clip(self.integral + error * dt, -10.0, 10.0))
        derivative = (error - self.previous_error) / dt
        acc = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error, self.previous_time = error, now
        return float(np.clip(acc, -self.max_accel, self.max_accel))

    def _path_cb(self, msg: WaypointArrayStamped) -> None:
        path = self._path(msg)
        if len(path) == 0:
            return
        frame = str(msg.header.frame_id)
        if frame not in ("", "base_footprint") and not self.frame_warning_sent:
            self.get_logger().warn("Expected a base_footprint trajectory; received another frame")
            self.frame_warning_sent = True
        target, ld = self._select_target(path)
        steering = pure_pursuit_steering(target[0], target[1], self.L, ld, self.max_steer)
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_footprint"
        cmd.drive.steering_angle = steering
        cmd.drive.acceleration = self._speed_pid()
        self.cmd_pub.publish(cmd)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PurePursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
