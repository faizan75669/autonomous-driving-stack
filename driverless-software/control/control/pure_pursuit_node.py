"""
Pure pursuit controller (lateral) + PID speed controller (longitudinal).

The planner publishes /trajectory in the car frame (base_footprint), so the
car is always at (0, 0) looking along +x. That makes pure pursuit simple:
  - pick the first waypoint that is at least `look_ahead` metres away
  - steering = atan(2 * L * sin(alpha) / l_d)
Look-ahead grows with speed: l_d = look_ahead_min + look_ahead_gain * v.

Target speed comes from the path itself: v = sqrt(a_lat_max / curvature),
also limited so the car can brake in time for the next corner.

Subscribes : /trajectory (WaypointArrayStamped), /ground_truth/state (CarState)
Publishes  : /cmd (AckermannDriveStamped), visualization_marker_array
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from eufs_msgs.msg import WaypointArrayStamped, CarState
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


def path_curvature(path):
    """Arc length s (N) and curvature kappa (N-2) of an (N, 2) path."""
    seg = np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1]))
    s = np.concatenate(([0.0], np.cumsum(seg)))
    heading = np.unwrap(np.arctan2(np.diff(path[:, 1]), np.diff(path[:, 0])))
    kappa = np.abs(np.diff(heading)) / np.maximum(seg[1:], 1e-3)
    if len(kappa) >= 5:     # finite differences are noisy, smooth a bit
        kappa = np.convolve(kappa, np.ones(5) / 5.0, mode="same")
    return s, kappa


class PurePursuit(Node):

    def __init__(self, name="pure_pursuit"):
        super().__init__(name)
        p = self.declare_parameter
        path_topic = p("path_topic", "/trajectory").value
        state_topic = p("state_topic", "/ground_truth/state").value

        self.L = p("wheelbase", 1.53).value                  # ADS-DV (the FS-AI car) wheelbase [m]
        self.look_ahead_min = p("look_ahead_min", 2.5).value
        self.look_ahead_gain = p("look_ahead_gain", 0.5).value
        self.look_ahead_max = p("look_ahead_max", 7.0).value
        self.max_steering = p("max_steering", 0.52).value    # EUFS steering lock [rad]
        self.max_speed = p("max_speed", 3.0).value
        self.safe_speed = p("safe_speed", 1.5).value
        self.max_lat_acc = p("max_lat_acc", 4.5).value
        self.max_decel = p("max_decel", 4.0).value
        self.kp = p("pid_kp", 2.0).value
        self.ki = p("pid_ki", 0.08).value
        self.kd = p("pid_kd", 0.06).value
        self.max_acc = p("max_acc", 3.0).value

        self.speed = 0.0
        self.look_ahead = self.look_ahead_min
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = None

        self.create_subscription(WaypointArrayStamped, path_topic, self.path_callback, 1)
        self.create_subscription(CarState, state_topic, self.state_callback, 1)
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, "/cmd", 1)
        self.viz_pub = self.create_publisher(MarkerArray, "visualization_marker_array", 10)

    # ------------------------------------------------------------ callbacks
    def state_callback(self, msg):
        self.speed = float(msg.twist.twist.linear.x)

    def path_callback(self, msg):
        path = np.array([[w.position.x, w.position.y] for w in msg.waypoints]).reshape(-1, 2)
        if len(path) < 2:
            return
        idx = self.look_ahead_index(path)
        steering = self.steering(path[idx])
        acc = self.pid(self.speed_target(path))
        self.publish_cmd(acc, steering)
        self.publish_viz(path, idx)

    # ------------------------------------------------------------ control
    def look_ahead_index(self, path):
        self.look_ahead = float(np.clip(self.look_ahead_min + self.look_ahead_gain * self.speed,
                                        self.look_ahead_min, self.look_ahead_max))
        dist = np.hypot(path[:, 0], path[:, 1])
        idx = np.nonzero(dist >= self.look_ahead)[0]
        return int(idx[0]) if len(idx) else len(path) - 1

    def steering(self, target):
        l_d = max(math.hypot(target[0], target[1]), 0.5)
        alpha = math.atan2(target[1], target[0])
        delta = math.atan2(2.0 * self.L * math.sin(alpha), l_d)
        return float(np.clip(delta, -self.max_steering, self.max_steering))

    def speed_target(self, path):
        if len(path) < 3:
            return self.safe_speed
        s, kappa = path_curvature(path)
        v_curve = np.sqrt(self.max_lat_acc / np.maximum(kappa, 1e-4))
        # v^2 <= v_curve^2 + 2*a*s  -> we can still brake for corners ahead
        v_allowed = np.sqrt(v_curve ** 2 + 2.0 * self.max_decel * s[1:-1])
        v = float(min(self.max_speed, np.min(v_allowed)))
        if s[-1] < 4.0:          # very short path = few cones visible, go slow
            v = min(v, self.safe_speed)
        return max(v, self.safe_speed)

    def pid(self, target):
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.02 if self.last_time is None else min(max(now - self.last_time, 1e-3), 0.2)
        self.last_time = now
        error = target - self.speed
        self.integral = float(np.clip(self.integral + error * dt, -5.0, 5.0))   # anti wind-up
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        acc = self.kp * error + self.ki * self.integral + self.kd * derivative
        return float(np.clip(acc, -self.max_acc * 3.0, self.max_acc))

    # ------------------------------------------------------------ output
    def publish_cmd(self, acc, steering):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_footprint"
        msg.drive.steering_angle = float(steering)
        msg.drive.acceleration = float(acc)
        self.cmd_pub.publish(msg)

    def publish_viz(self, path, idx):
        markers = MarkerArray()
        line = Marker()
        line.header.frame_id = "base_footprint"
        line.header.stamp = self.get_clock().now().to_msg()
        line.ns, line.id = "pp_path", 0
        line.type, line.action = Marker.LINE_STRIP, Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.08
        line.color.a, line.color.r = 1.0, 1.0
        for x, y in path:
            line.points.append(Point(x=float(x), y=float(y), z=0.0))
        markers.markers.append(line)

        target = Marker()
        target.header = line.header
        target.ns, target.id = "pp_target", 1
        target.type, target.action = Marker.SPHERE, Marker.ADD
        target.pose.position.x = float(path[idx][0])
        target.pose.position.y = float(path[idx][1])
        target.pose.orientation.w = 1.0
        target.scale.x = target.scale.y = target.scale.z = 0.6
        target.color.a, target.color.g = 1.0, 1.0
        markers.markers.append(target)
        self.viz_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
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
