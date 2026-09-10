"""ROS 2 adapter for the repository's ROS-independent EKF-SLAM core."""
from __future__ import annotations

import math
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
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class EKFSLAMNode(Node):
    """Cone EKF-SLAM node. Ground truth is intentionally not used by the filter."""
    def __init__(self) -> None:
        super().__init__("ekf_slam_node")
        self.declare_parameter("cones_topic", "/cones")
        self.declare_parameter("odom_topic", "/odometry_integration/odom")
        self.declare_parameter("pose_topic", "/robot_pose")
        self.declare_parameter("association_gate", 9.21)
        self.declare_parameter("min_range", 0.2)
        self.declare_parameter("max_range", 30.0)
        self.declare_parameter("max_detections", 30)
        self.filter = EKFSLAMCore(EKFConfig(
            association_gate=float(self.get_parameter("association_gate").value)))
        self.min_range = float(self.get_parameter("min_range").value)
        self.max_range = float(self.get_parameter("max_range").value)
        self.max_detections = int(self.get_parameter("max_detections").value)
        self.last_time = None
        self.last_yaw = None

        self.create_subscription(ConeArrayWithCovariance, self.get_parameter("cones_topic").value, self._cones_cb, 10)
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._odom_cb, 20)
        self.pose_pub = self.create_publisher(PoseStamped, self.get_parameter("pose_topic").value, 10)
        self.get_logger().info("EKF-SLAM node started")

    def _odom_cb(self, msg: Odometry) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        if self.last_time is None:
            self.last_time, self.last_yaw = stamp, yaw
            return
        dt = stamp - self.last_time
        if not 0.0 < dt <= 1.0:
            self.last_time, self.last_yaw = stamp, yaw
            return
        yaw_rate = self._angle_diff(yaw, self.last_yaw) / dt
        self.filter.predict(float(msg.twist.twist.linear.x), yaw_rate, dt)
        self.last_time, self.last_yaw = stamp, yaw
        self._publish(msg.header.stamp)

    def _cones_cb(self, msg: ConeArrayWithCovariance) -> None:
        detections = []
        for landmark_type, cones in (
            ("blue", msg.blue_cones),
            ("yellow", msg.yellow_cones),
            ("orange", msg.big_orange_cones),
        ):
            for cone in cones:
                x, y = float(cone.point.x), float(cone.point.y)
                r = math.hypot(x, y)
                if self.min_range <= r <= self.max_range:
                    detections.append((np.array([r, math.atan2(y, x)]), landmark_type))
                if len(detections) >= self.max_detections:
                    break
            if len(detections) >= self.max_detections:
                break

        for measurement, landmark_type in detections:
            idx = self.filter.associate(measurement, landmark_type)
            if idx is None:
                self.filter.add_landmark(measurement, landmark_type)
            else:
                self.filter.update(measurement, idx)
        self._publish(msg.header.stamp)

    def _publish(self, stamp) -> None:
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "map"
        pose.pose.position.x = float(self.filter.state[0])
        pose.pose.position.y = float(self.filter.state[1])
        pose.pose.orientation.z = math.sin(self.filter.state[2] / 2.0)
        pose.pose.orientation.w = math.cos(self.filter.state[2] / 2.0)
        self.pose_pub.publish(pose)

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
