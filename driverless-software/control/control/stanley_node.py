"""
Stanley controller (lateral) + PID speed controller.

Stanley steers using the front axle:
    delta = heading_error + atan(k * e / (v + k_soft))
where e is the cross-track error of the front axle to the path.

Everything is in the car frame because /trajectory comes in base_footprint:
the car is at (0, 0), heading 0, and the front axle is at (front_axle_x, 0).

Sign convention: e > 0 means the path is to the LEFT of the front axle,
so the car has to steer left (positive steering in EUFS).

Subscribes : /trajectory (WaypointArrayStamped), /ground_truth/state (CarState)
Publishes  : /cmd (AckermannDriveStamped), /control/viz (Marker text)
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from eufs_msgs.msg import WaypointArrayStamped, CarState
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker

from control.pure_pursuit_node import path_curvature


def normalize_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def nearest_on_path(path, p):
    """Closest point on the polyline to p. Returns (point, segment index)."""
    best, best_i, best_d = path[0], 0, np.inf
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 < 1e-12 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
        q = a + t * ab
        d = float(np.linalg.norm(p - q))
        if d < best_d:
            best, best_i, best_d = q, i, d
    return best, best_i


class Stanley(Node):

    def __init__(self, name="stanley_controller"):
        super().__init__(name)
        p = self.declare_parameter
        path_topic = p("path_topic", "/trajectory").value
        state_topic = p("state_topic", "/ground_truth/state").value

        self.k = p("stanley_k", 1.5).value
        self.k_soft = p("stanley_soft_k", 1.0).value
        self.front_axle_x = p("front_axle_x", 0.87).value     # COG -> front axle (EUFS b_F)
        self.max_steering = p("max_steering", 0.52).value
        self.max_speed = p("max_speed", 3.0).value
        self.safe_speed = p("safe_speed", 1.5).value
        self.max_lat_acc = p("max_lat_acc", 4.5).value
        self.max_decel = p("max_decel", 4.0).value
        self.kp = p("pid_kp", 2.0).value
        self.ki = p("pid_ki", 0.08).value
        self.kd = p("pid_kd", 0.06).value
        self.max_acc = p("max_acc", 3.0).value

        self.speed = 0.0
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = None

        self.create_subscription(WaypointArrayStamped, path_topic, self.path_callback, 1)
        self.create_subscription(CarState, state_topic, self.state_callback, 1)
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, "/cmd", 1)
        self.viz_pub = self.create_publisher(Marker, "/control/viz", 1)

    def state_callback(self, msg):
        self.speed = float(msg.twist.twist.linear.x)

    def path_callback(self, msg):
        path = np.array([[w.position.x, w.position.y] for w in msg.waypoints]).reshape(-1, 2)
        if len(path) < 2:
            return
        steering = self.stanley(path)
        acc = self.pid(self.speed_target(path))
        self.publish_cmd(acc, steering)
        self.publish_viz(acc, steering)

    def stanley(self, path):
        front = np.array([self.front_axle_x, 0.0])
        q, i = nearest_on_path(path, front)
        tangent = path[i + 1] - path[i]
        path_heading = math.atan2(tangent[1], tangent[0])
        heading_error = normalize_angle(path_heading - 0.0)       # car heading is 0 in its own frame
        # signed cross-track error: + when the path is left of the front axle
        # (path left normal) . (q - front) is + when the axle is right of the path
        t = tangent / (np.linalg.norm(tangent) + 1e-9)
        left_normal = np.array([-t[1], t[0]])
        e = float((q - front) @ left_normal)
        delta = heading_error + math.atan2(self.k * e, self.speed + self.k_soft)
        return float(np.clip(normalize_angle(delta), -self.max_steering, self.max_steering))

    def speed_target(self, path):
        if len(path) < 3:
            return self.safe_speed
        s, kappa = path_curvature(path)
        v_curve = np.sqrt(self.max_lat_acc / np.maximum(kappa, 1e-4))
        v_allowed = np.sqrt(v_curve ** 2 + 2.0 * self.max_decel * s[1:-1])
        v = float(min(self.max_speed, np.min(v_allowed)))
        if s[-1] < 4.0:
            v = min(v, self.safe_speed)
        return max(v, self.safe_speed)

    def pid(self, target):
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.02 if self.last_time is None else min(max(now - self.last_time, 1e-3), 0.2)
        self.last_time = now
        error = target - self.speed
        self.integral = float(np.clip(self.integral + error * dt, -5.0, 5.0))
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        acc = self.kp * error + self.ki * self.integral + self.kd * derivative
        return float(np.clip(acc, -self.max_acc * 3.0, self.max_acc))

    def publish_cmd(self, acc, steering):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_footprint"
        msg.drive.steering_angle = float(steering)
        msg.drive.acceleration = float(acc)
        self.cmd_pub.publish(msg)

    def publish_viz(self, acc, steering):
        m = Marker()
        m.header.frame_id = "base_footprint"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id = "stanley", 0
        m.type, m.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = 3.0, 4.0, 1.0
        m.pose.orientation.w = 1.0
        m.scale.z = 0.5
        m.color.a, m.color.g = 1.0, 1.0
        m.text = f"v {self.speed:.2f} m/s | acc {acc:.2f} | steer {steering:.3f} rad"
        self.viz_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = Stanley()
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
