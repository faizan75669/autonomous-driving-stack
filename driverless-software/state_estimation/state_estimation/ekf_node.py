"""
EKF state estimation node.

Fuses wheel odometry (speed + yaw rate) with a position measurement
(GPS converted to x/y, or the odometry position if there is no GPS) to
estimate the car state [x, y, yaw, v].

Motion model (unicycle, with the slip angle beta from body-frame vx, vy):
    x'   = x + v*dt*cos(yaw + beta)
    y'   = y + v*dt*sin(yaw + beta)
    yaw' = yaw + w*dt
    v'   = v_measured
Measurements: x, y from GPS (each fix used once) and yaw from the IMU.
Without GPS it dead-reckons; the integrated odometry position is not used by
default because it drifts without bound.

The EKF maths started from the PythonRobotics example by Atsushi Sakai
(https://github.com/AtsushiSakai/PythonRobotics), then we changed it for the
EUFS topics and our car.

Subscribes : /odometry_integration/car_state (CarState), /imu/data (Imu), /gps (NavSatFix)
Publishes  : /odometry_filtered/car_state (CarState, map frame)
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from eufs_msgs.msg import CarState
from sensor_msgs.msg import NavSatFix, Imu

from state_estimation.geonav_conversions import ll2xy


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class EKF(object):
    """Plain numpy EKF, no ROS inside so it can be unit tested.

    q_diag: process noise std per step [x, y, yaw, v]
    Each update can use any subset of the measurements x, y, yaw.
    """

    def __init__(self, dt, q_diag=(0.02, 0.02, np.deg2rad(0.3), 0.1)):
        self.dt = dt
        self.Q = np.diag(q_diag) ** 2
        self.x = np.zeros((4, 1))              # x, y, yaw, v
        self.P = np.diag([1.0, 1.0, 0.1, 1.0])

    @staticmethod
    def _slip(u):
        # u = [v, w] or [v, w, beta]; beta = direction of travel relative to heading
        return float(u[2, 0]) if len(u) > 2 else 0.0

    def motion_model(self, x, u):
        v, w = float(u[0, 0]), float(u[1, 0])
        course = float(x[2, 0]) + self._slip(u)
        return np.array([[x[0, 0] + v * self.dt * math.cos(course)],
                         [x[1, 0] + v * self.dt * math.sin(course)],
                         [x[2, 0] + w * self.dt],
                         [v]])

    def jacobian_f(self, x, u):
        yaw, v = float(x[2, 0]) + self._slip(u), float(u[0, 0])
        return np.array([[1.0, 0.0, -self.dt * v * math.sin(yaw), self.dt * math.cos(yaw)],
                         [0.0, 1.0, self.dt * v * math.cos(yaw), self.dt * math.sin(yaw)],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])

    def predict(self, u):
        F = self.jacobian_f(self.x, u)
        self.x = self.motion_model(self.x, u)
        self.P = F @ self.P @ F.T + self.Q
        self.x[2, 0] = wrap(self.x[2, 0])

    def update(self, z, idx, r_std):
        """z: measured values, idx: state indices they measure, r_std: their noise std."""
        z = np.asarray(z, dtype=float).reshape(-1, 1)
        H = np.eye(4)[list(idx)]
        R = np.diag(np.asarray(r_std, dtype=float) ** 2)
        y = z - H @ self.x
        for k, i in enumerate(idx):
            if i == 2:
                y[k, 0] = wrap(y[k, 0])        # yaw difference must be wrapped
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.x[2, 0] = wrap(self.x[2, 0])

    def step(self, z, u, r_std=(0.5, 0.5, np.deg2rad(1.0))):
        """Predict + update with [x, y] or [x, y, yaw] (kept for the tests/demo)."""
        self.predict(u)
        n = len(z)
        self.update(np.asarray(z).reshape(-1)[:n], list(range(n)), list(r_std)[:n])
        return self.x


def quaternion_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def yaw_to_quaternion(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class StateEstimation(Node):

    def __init__(self):
        super().__init__("state_estimation")
        dt = self.declare_parameter("dt", 0.05).value
        # lat/lon of the map origin (only used when GPS is available)
        self.lat0 = self.declare_parameter("gps_origin_lat", 55.94448258581324).value
        self.lon0 = self.declare_parameter("gps_origin_lon", 3.1864915020968305).value
        self.gps_std = self.declare_parameter("gps_noise", 0.5).value              # [m]
        self.imu_yaw_std = self.declare_parameter("imu_yaw_noise", 0.02).value     # [rad]
        # Integrated odometry position drifts without bound, so by default it is
        # NOT used as a measurement. Without GPS the filter dead-reckons with
        # wheel speed + yaw rate and corrects the heading with the IMU.
        self.use_odom_position = self.declare_parameter("use_odom_position", False).value
        self.odom_pos_std = self.declare_parameter("odom_position_noise", 1.0).value
        self.use_imu_yaw = self.declare_parameter("use_imu_yaw", True).value

        self.ekf = EKF(dt)
        self.u = None            # [v, w]
        self.odom_xy = None      # position from odometry
        self.gps_xy = None       # latest GPS fix, used once
        self.imu_yaw = None
        self.initialised = False
        self.prev = None
        self.last_time = None

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CarState, "/odometry_integration/car_state", self.odom_cb, qos)
        self.create_subscription(NavSatFix, "/gps", self.gps_cb, qos)
        self.create_subscription(Imu, "/imu/data", self.imu_cb, qos)
        # create the publisher ONCE (old code made a new one every timer tick)
        self.pub = self.create_publisher(CarState, "/odometry_filtered/car_state", 10)
        self.create_timer(dt, self.update)

    def odom_cb(self, msg):
        # body-frame velocity: vy != 0 when the car slides a bit in corners,
        # so the car moves along yaw + beta, not exactly along yaw
        vx, vy = msg.twist.twist.linear.x, msg.twist.twist.linear.y
        beta = math.atan2(vy, vx) if abs(vx) > 0.1 else 0.0
        self.u = np.array([[math.hypot(vx, vy)], [msg.twist.twist.angular.z], [beta]])
        self.odom_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def gps_cb(self, msg):
        x, y = ll2xy(msg.latitude, -msg.longitude, self.lat0, self.lon0)
        self.gps_xy = (-x, y)

    def imu_cb(self, msg):
        self.imu_yaw = quaternion_to_yaw(msg.orientation)

    def update(self):
        if self.u is None or self.imu_yaw is None:
            return
        if not self.initialised:
            # start where the car is: first GPS fix, else the odometry origin
            xy = self.gps_xy if self.gps_xy is not None else self.odom_xy
            if xy is None:
                return
            self.ekf.x = np.array([[xy[0]], [xy[1]], [self.imu_yaw], [float(self.u[0, 0])]])
            self.initialised = True
        # integrate over the time that really passed (timers jitter)
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_time is not None:
            self.ekf.dt = min(max(now - self.last_time, 1e-3), 0.5)
        self.last_time = now
        self.ekf.predict(self.u)
        if self.gps_xy is not None:
            self.ekf.update(self.gps_xy, (0, 1), (self.gps_std, self.gps_std))
            self.gps_xy = None                       # every fix is used only once
        elif self.use_odom_position and self.odom_xy is not None:
            self.ekf.update(self.odom_xy, (0, 1), (self.odom_pos_std, self.odom_pos_std))
        if self.use_imu_yaw:
            self.ekf.update([self.imu_yaw], (2,), (self.imu_yaw_std,))
        self.publish(self.ekf.x)

    def publish(self, x):
        dt = self.ekf.dt
        px, py, yaw, v = (float(a) for a in x[:, 0])
        vx, vy = v * math.cos(yaw), v * math.sin(yaw)
        yaw_rate = ax = ay = 0.0
        if self.prev is not None:
            yaw_rate = wrap(yaw - self.prev[0]) / dt
            ax, ay = (vx - self.prev[1]) / dt, (vy - self.prev[2]) / dt
        self.prev = (yaw, vx, vy)

        msg = CarState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_footprint"
        msg.pose.pose.position.x, msg.pose.pose.position.y = px, py
        # x, y, yaw covariance into the 6x6 pose covariance (row-major x y z roll pitch yaw)
        P = self.ekf.P
        cov = [0.0] * 36
        for (i, j), (a, b) in {(0, 0): (0, 0), (0, 1): (0, 1), (1, 0): (1, 0), (1, 1): (1, 1),
                               (0, 5): (0, 2), (5, 0): (2, 0), (1, 5): (1, 2), (5, 1): (2, 1),
                               (5, 5): (2, 2)}.items():
            cov[6 * i + j] = float(P[a, b])
        msg.pose.covariance = cov
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        o = msg.pose.pose.orientation
        o.x, o.y, o.z, o.w = qx, qy, qz, qw
        msg.twist.twist.linear.x, msg.twist.twist.linear.y = vx, vy
        msg.twist.twist.angular.z = yaw_rate
        msg.linear_acceleration.x, msg.linear_acceleration.y = ax, ay
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = StateEstimation()
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
