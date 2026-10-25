"""
EKF state estimation node.

Fuses wheel odometry (speed + yaw rate) with a position measurement
(GPS converted to x/y, or the odometry position if there is no GPS) to
estimate the car state [x, y, yaw, v].

Motion model (unicycle):
    x'   = x + v*dt*cos(yaw)
    y'   = y + v*dt*sin(yaw)
    yaw' = yaw + w*dt
    v'   = v_measured
Observation model: z = [x, y] from GPS/odometry, plus yaw from the IMU

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
    """Plain numpy EKF, no ROS inside so it can be unit tested."""

    def __init__(self, dt, q_diag=(0.1, 0.1, np.deg2rad(1.0), 0.2), r_diag=(1.0, 1.0, np.deg2rad(2.0))):
        self.dt = dt
        self.Q = np.diag(q_diag) ** 2          # process noise
        self.R_full = np.diag(r_diag) ** 2     # measurement noise (x, y, yaw)
        self.x = np.zeros((4, 1))              # x, y, yaw, v
        self.P = np.eye(4)

    def motion_model(self, x, u):
        v, w = float(u[0, 0]), float(u[1, 0])
        yaw = float(x[2, 0])
        return np.array([[x[0, 0] + v * self.dt * math.cos(yaw)],
                         [x[1, 0] + v * self.dt * math.sin(yaw)],
                         [yaw + w * self.dt],
                         [v]])

    def jacobian_f(self, x, u):
        yaw, v = float(x[2, 0]), float(u[0, 0])
        return np.array([[1.0, 0.0, -self.dt * v * math.sin(yaw), self.dt * math.cos(yaw)],
                         [0.0, 1.0, self.dt * v * math.cos(yaw), self.dt * math.sin(yaw)],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])

    def step(self, z, u):
        """z is [x, y] or [x, y, yaw] (column vector), u is [v, w]."""
        # predict
        x_pred = self.motion_model(self.x, u)
        F = self.jacobian_f(self.x, u)
        P_pred = F @ self.P @ F.T + self.Q
        # update
        n = len(z)
        H = np.eye(4)[:n]
        R = self.R_full[:n, :n]
        y = z - H @ x_pred
        if n == 3:
            y[2, 0] = wrap(y[2, 0])            # yaw difference must be wrapped
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        self.P = (np.eye(4) - K @ H) @ P_pred
        self.x[2, 0] = wrap(self.x[2, 0])
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
        # the original PythonRobotics demo adds noise on purpose; keep it optional
        self.add_sim_noise = self.declare_parameter("add_sim_noise", False).value
        self.use_imu_yaw = self.declare_parameter("use_imu_yaw", True).value

        self.ekf = EKF(dt)
        self.u = None            # [v, w]
        self.odom_xy = None      # position from odometry
        self.gps_xy = None       # position from GPS
        self.imu_yaw = None
        self.initialised = False
        self.prev = None

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CarState, "/odometry_integration/car_state", self.odom_cb, qos)
        self.create_subscription(NavSatFix, "/gps", self.gps_cb, qos)
        self.create_subscription(Imu, "/imu/data", self.imu_cb, qos)
        # create the publisher ONCE (old code made a new one every timer tick)
        self.pub = self.create_publisher(CarState, "/odometry_filtered/car_state", 10)
        self.create_timer(dt, self.update)

    def odom_cb(self, msg):
        vx, vy = msg.twist.twist.linear.x, msg.twist.twist.linear.y
        self.u = np.array([[math.hypot(vx, vy)], [msg.twist.twist.angular.z]])
        self.odom_xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def gps_cb(self, msg):
        x, y = ll2xy(msg.latitude, -msg.longitude, self.lat0, self.lon0)
        self.gps_xy = (-x, y)

    def imu_cb(self, msg):
        self.imu_yaw = quaternion_to_yaw(msg.orientation)

    def update(self):
        if self.u is None or self.imu_yaw is None:
            return
        xy = self.gps_xy if self.gps_xy is not None else self.odom_xy
        if xy is None:
            return
        if not self.initialised:   # start the filter where the car actually is
            self.ekf.x = np.array([[xy[0]], [xy[1]], [self.imu_yaw], [self.u[0, 0]]])
            self.initialised = True
        if self.use_imu_yaw:
            z = np.array([[xy[0]], [xy[1]], [self.imu_yaw]])
        else:
            z = np.array([[xy[0]], [xy[1]]])
        u = self.u.copy()
        if self.add_sim_noise:
            z[:2] = z[:2] + np.diag([0.05, 0.05]) @ np.random.randn(2, 1)
            u = u + np.diag([0.5, np.deg2rad(1.0)]) @ np.random.randn(2, 1)
        x = self.ekf.step(z, u)
        self.publish(x)

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
