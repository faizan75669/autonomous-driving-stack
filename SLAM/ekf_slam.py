"""ROS 2 adapter for the repository's EKF-SLAM implementation.

The filter itself lives in :mod:`ekf_slam_core` and is ROS-independent.
This node accepts cone detections in the vehicle frame and velocity/yaw-rate
odometry, performs range/bearing EKF-SLAM, and publishes the estimated pose.

This implementation deliberately does not use ground-truth pose as an EKF
measurement. Ground truth can be used externally for evaluation only.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from eufs_msgs.msg import ConeArrayWithCovariance

try:
    from .ekf_slam_core import EKFConfig, EKFSLAMCore
except ImportError:
    from ekf_slam_core import EKFConfig, EKFSLAMCore


def quaternion_to_yaw(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class EKFSLAMNode(Node):
    def __init__(self) -> None:
        super().__init__("ekf_slam_node")
        self.declare_parameter("cones_topic", "/cones")
        self.declare_parameter("odom_topic", "/odometry_integration/odom")
        self.declare_parameter("pose_topic", "/robot_pose")
        self.declare_parameter("association_gate", 9.21)
        self.declare_parameter("min_range", 0.2)
        self.declare_parameter("max_range", 30.0)
        self.declare_parameter("max_detections", 30)

        cfg = EKFConfig(
            association_gate=float(self.get_parameter("association_gate").value)
        )
        self.filter = EKFSLAMCore(cfg)
        self.min_range = float(self.get_parameter("min_range").value)
        self.max_range = float(self.get_parameter("max_range").value)
        self.max_detections = int(self.get_parameter("max_detections").value)

        self.last_stamp = None
        self.last_odom_yaw: Optional[float] = None
        self.last_odom_time = None

        cones_topic = self.get_parameter("cones_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        pose_topic = self.get_parameter("pose_topic").value

        self.create_subscription(ConeArrayWithCovariance, cones_topic, self._cones_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 20)
        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)
        self.get_logger().info("EKF-SLAM node started")

    def _odom_cb(self, msg: Odometry) -> None:
        now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_odom_time is None:
            self.last_odom_time = now
            self.last_odom_yaw = quaternion_to_yaw(msg.pose.pose.orientation)
            return

        dt = now - self.last_odom_time
        if dt <= 0.0 or dt > 1.0:
            self.last_odom_time = now
            self.last_odom_yaw = quaternion_to_yaw(msg.pose.pose.orientation)
            return

        yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        previous_yaw = self.last_odom_yaw if self.last_odom_yaw is not None else yaw
        yaw_rate = self._angle_diff(yaw, previous_yaw) / dt
        speed = float(msg.twist.twist.linear.x)
        self.filter.predict(speed, yaw_rate, dt)
        self.last_odom_time = now
        self.last_odom_yaw = yaw
        self._publish_pose(msg.header.stamp)

    def _cones_cb(self, msg: ConeArrayWithCovariance) -> None:
        detections = []
        # Color is not part of the geometric measurement model. It can be
        # incorporated later as a landmark-classification/data-association cue.
        for cone in list(msg.blue_cones) + list(msg.yellow_cones) + list(msg.big_orange_cones):
            x = float(cone.point.x)
            y = float(cone.point.y)
            r = math.hypot(x, y)
            if self.min_range <= r <= self.max_range:
                detections.append((r, math.atan2(y, x)))
            if len(detections) >= self.max_detections:
                break

        for measurement in detections:
            idx = self.filter.associate(np.asarray(measurement))
            if idx is None:
                self.filter.add_landmark(np.asarray(measurement))
            else:
                self.filter.update(np.asarray(measurement), idx)

        self._publish_pose(msg.header.stamp)

    def _publish_pose(self, stamp) -> None:
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.pose.position.x = float(self.filter.state[0])
        msg.pose.position.y = float(self.filter.state[1])
        msg.pose.orientation.z = math.sin(self.filter.state[2] / 2.0)
        msg.pose.orientation.w = math.cos(self.filter.state[2] / 2.0)
        self.pose_pub.publish(msg)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EKFSLAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
