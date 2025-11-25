"""
EKF SLAM Node for Autonomous Vehicle Simulation

This ROS2 node implements an Extended Kalman Filter (EKF) based SLAM system for tracking
a vehicle and mapping cone landmarks in a simulated environment. Key functionalities include:

1. Subscribing to cone detections, vehicle odometry, and ground truth for yaw.
2. Estimating vehicle position and orientation using EKF and publishing the estimated pose.
3. Filtering cone observations using DBSCAN clustering, Mahalanobis distance, and LOF 
   to remove noise and outliers.
4. Detecting loop closure using KDTree and orange cone landmarks.
5. Maintaining a history of observed cones, paths, and estimated positions.
6. Visualizing vehicle trajectory, detected cones, and uncertainty ellipses using matplotlib.
7. Providing a reset service to clear state and restart the SLAM process.
"""

import math
import numpy as np
import rclpy
import time
import os
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose as GeoPose
from nav_msgs.msg import Odometry
from eufs_msgs.msg import ConeArrayWithCovariance, CarState, WaypointArrayStamped
from std_srvs.srv import Trigger
from sklearn.cluster import DBSCAN, LocalOutlierFactor
from scipy.spatial import KDTree
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

class EKFSLAM(Node):
    def __init__(self):
        super().__init__('ekf_slam_node')
        self.cones_sub = self.create_subscription(ConeArrayWithCovariance, "/cones", self.cones_callback, 10)
        self.car_state_sub = self.create_subscription(CarState, "/odometry_integration/car_state", self.car_state_callback, 10)
        self.yaw_sub = self.create_subscription(Odometry, "/ground_truth/odom", self.yaw_callback, 10)
        self.create_subscription(Twist, '/cmd_vel', self.velocity_callback, 10)
        self.estimated_pose_sub = self.create_subscription(CarState, "/odometry_filtered/car_state", self.estimated_pose_callback, 10)
        self.pose_pub = self.create_publisher(GeoPose, '/robot_pose', 10)

        self.yaw = 0.0
        self.yaw_est = 0.0
        self.car_position = np.array([0.0, 0.0])
        self.array_est_position = np.array([0.0, 0.0])
        self.estimated_positions = []
        self.cones = np.array([0.0, 0.0])
        self.est_cones = []
        self.est_pairs = []
        self.cones_pairs = []
        self.global_cones = []
        self.all_blue_cones, self.all_yellow_cones, self.all_orange_cones = [], [], []
        self.est_blue_cones, self.est_yellow_cones, self.est_orange_cones = [], [], []
        self.orange_cones_index = []

        self.distance_list, self.path, self.car_post = [], [], []
        self.flag, self.flag_two = False, True
        self.fig, self.ax = plt.subplots()
        plt.ion()
        self.create_timer(0.1, self.update_plot)
        self.coneee, self.previous_cones, self.current_detection = [], [], []
        self.seen_coords, self.unique_coordinates, self.loop_closure_indices, self.buffer = set(), [], [], []
        self.threshold, self.persistence_frames = 0.5, 5
        self.last_iterated_list, self.filtered_list, self.final_list = [], [], []
        self.cones_storage, self.cone_timestamps = [], {}
        self.trajectory_timeout, self.last_trajectory_time = 5.0, time.time()
        self.reset_service = self.create_service(Trigger, '/ros_can/reset', self.reset_callback)

    def reset_callback(self, request, response):
        response.success = True
        response.message = "System Reset Successfully"
        os.system("ros2 lifecycle set /ekf_slam_node unconfigured")
        os.system("ros2 lifecycle set /ekf_slam_node active")
        self.reset_simulation()
        return response

    def yaw_callback(self, msg):
        q_x, q_y, q_z, q_w = msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
        self.yaw = math.atan2(2.0 * (q_w * q_z + q_x * q_y), 1.0 - 2.0 * (q_y ** 2 + q_z ** 2))

    def car_state_callback(self, msg: CarState):
        self.car_position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        self.car_post.append(self.car_position)
        self.path.append(self.car_position.copy())

    def estimated_pose_callback(self, msg):
        try:
            estimated_x, estimated_y = msg.pose.pose.position.x, msg.pose.pose.position.y
            self.estimated_positions.append([estimated_x, estimated_y])
            self.array_est_position = np.array([estimated_x, estimated_y])
            x, y, z, w = msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
            self.yaw_est = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y ** 2 + z ** 2))
        except AttributeError:
            pass

    def cones_callback(self, msg):
        blue_cones = [[c.point.x, c.point.y] for c in msg.blue_cones]
        yellow_cones = [[c.point.x, c.point.y] for c in msg.yellow_cones]
        orange_cones = [[c.point.x, c.point.y] for c in msg.big_orange_cones]
        self.orange_cones_index = np.array(orange_cones)
        self.global_cones = np.array(list(zip(blue_cones + yellow_cones)))

        self.all_blue_cones, self.all_yellow_cones, self.all_orange_cones = [], [], []
        self.est_blue_cones, self.est_yellow_cones, self.est_orange_cones = [], [], []

        for cone_list, store_list, pos_ref, yaw_ref in [
            (blue_cones, self.all_blue_cones, self.car_position, self.yaw),
            (yellow_cones, self.all_yellow_cones, self.car_position, self.yaw),
            (orange_cones, self.all_orange_cones, self.car_position, self.yaw)
        ]:
            for cone in cone_list:
                x_rot = cone[0]*math.cos(yaw_ref) - cone[1]*math.sin(yaw_ref)
                y_rot = cone[0]*math.sin(yaw_ref) + cone[1]*math.cos(yaw_ref)
                store_list.append((pos_ref[0]+x_rot, pos_ref[1]+y_rot))

        self.cones = (self.all_blue_cones, self.all_yellow_cones)
        self.cones_pairs = np.array(list(zip(self.all_blue_cones + self.all_yellow_cones)))

        for cone_list, store_list, pos_ref, yaw_ref in [
            (blue_cones, self.est_blue_cones, self.array_est_position, self.yaw_est),
            (yellow_cones, self.est_yellow_cones, self.array_est_position, self.yaw_est),
            (orange_cones, self.est_orange_cones, self.array_est_position, self.yaw_est)
        ]:
            for cone in cone_list:
                x_rot = cone[0]*math.cos(yaw_ref) - cone[1]*math.sin(yaw_ref)
                y_rot = cone[0]*math.sin(yaw_ref) + cone[1]*math.cos(yaw_ref)
                store_list.append((pos_ref[0]+x_rot, pos_ref[1]+y_rot))

        self.est_cones = (self.est_blue_cones, self.est_yellow_cones)
        self.est_pairs = np.array(list(zip(self.est_blue_cones + self.est_yellow_cones)))

    def velocity_callback(self, msg: Twist):
        pass

    def loop_closure_trigger(self):
        if len(self.orange_cones_index) > 0:
            x_coords = [p[0] for p in self.orange_cones_index]
            y_coords = [p[1] for p in self.orange_cones_index]
            Mx, My = sum(x_coords)/4, sum(y_coords)/4
            car_x, car_y = self.car_position
            distance = np.sqrt((car_x - Mx)**2 + (car_y - My)**2)
            self.distance_list.append(distance)
            for i in range(1, len(self.distance_list)-1):
                if self.distance_list[i]<self.distance_list[i-1] and self.distance_list[i]<self.distance_list[i+1] and self.distance_list[i]<2.0:
                    self.count += 1
                    self.distance_list.clear()
                    if self.count>1:
                        print("loop closure detected")
        else:
            self.distance_list.clear()

    def mahalanobis_filter(self):
        cones = np.array(self.cones, dtype=object)
        try:
            dim_cones = np.vstack(cones)
        except ValueError:
            return np.array([])
        dim_cones = np.array(dim_cones, dtype=np.float64)
        if dim_cones.shape[0] < 3:
            return dim_cones

        covariance_matrix = np.cov(dim_cones, rowvar=False)
        inv_cov = np.linalg.pinv(covariance_matrix)
        mean_cones = np.mean(dim_cones, axis=0)
        distances = np.array([mahalanobis(p, mean_cones, inv_cov) for p in dim_cones])

        lof = LocalOutlierFactor(n_neighbors=5)
        inliers = lof.fit_predict(dim_cones)
        filtered_cones = dim_cones[inliers==1]

        dbscan = DBSCAN(eps=1.0, min_samples=2)
        cluster_labels = dbscan.fit_predict(filtered_cones)
        unique_clusters = np.unique(cluster_labels[cluster_labels!=-1])
        final_cones = np.array([np.mean(filtered_cones[cluster_labels==c], axis=0) for c in unique_clusters])
        stable_coordinates = np.round(final_cones, 1)

        cones_buffer = []
        for point in stable_coordinates:
            rounded_point = tuple(np.round(point, 4))
            self.cones_storage.append(rounded_point)
            if rounded_point not in self.seen_coords:
                cones_buffer.append(rounded_point)
                self.seen_coords.add(rounded_point)
                self.unique_coordinates.append(point)

        self.coneee.clear()
        self.coneee.extend(final_cones)
        x = np.array(self.collect_cones(self.s_cones))
        self.detect_loop_closure_kdtree_excluding_recent(x)
        return final_cones

    def collect_cones(self, s_cones):
        if not s_cones or len(s_cones)!=2:
            return np.array([])
        blue_cones, yellow_cones = s_cones
        return np.array(blue_cones + yellow_cones)

    def should_consider_cone(self, cone_id):
        cone_id = tuple(map(tuple, cone_id))
        current_time = time.time()
        if cone_id in self.cone_timestamps:
            return False
        self.cone_timestamps[cone_id] = current_time
        return True

    def detect_loop_closure_kdtree_excluding_recent(self, observed_cones):
        threshold = 1.7
        if not self.unique_coordinates or len(self.unique_coordinates)<10:
            return False
        older_cones = self.unique_coordinates[:-10]
        if not older_cones:
            return False
        tree = KDTree(np.array(older_cones))
        count = 0
        for cone in observed_cones:
            dist, idx = tree.query(cone)
            nearest_cone = tree.data[idx]
            if dist<threshold:
                count+=1
                if count>=len(observed_cones)-2 and self.should_consider_cone(observed_cones):
                    self.get_logger().info(f"Loop closure detected! Observed cone: {cone}, Nearest: {nearest_cone}, Distance: {dist:.2f}")
                    return True
        return False

    def computing_covariance_matrix(self):
        window_size = 20
        recent_positions = np.array(self.estimated_positions[-window_size:])
        if recent_positions.shape[0]<2:
            return 0, 0, 0
        errors = recent_positions - self.car_position
        mean_error = np.mean(errors, axis=0)
        centered_errors = errors - mean_error
        cov_matrix = np.dot(centered_errors.T, centered_errors)/(len(centered_errors)-1)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        eigenvalues = np.abs(eigenvalues)
        angle = np.arctan2(eigenvectors[1,0], eigenvectors[0,0])
        scale_factor = math.sqrt(5.991)
        axis_x, axis_y = scale_factor*math.sqrt(eigenvalues[0]), scale_factor*math.sqrt(eigenvalues[1])
        axis_x = min(axis_x, 2.0)
        if np.linalg.norm(mean_error)<0.05:
            return 0.01, 0.01, np.degrees(angle)
        return axis_x, axis_y, np.degrees(angle)

    def reset_simulation(self):
        self.unique_coordinates.clear()
        self.count = 0
        self.blue_cones, self.yellow_cones = [], []
        self.path.clear()
        self.estimated_positions.clear()
        self.ax.clear()
        self.ax.set_xlim([-10,10])
        self.ax.set_ylim([-10,10])
        self.ax.set_title("Updating SLAM Simulation")
        plt.draw()

    def update_plot(self):
        self.ax.clear()
        self.loop_closure_trigger()
        dim_cones = self.mahalanobis_filter()
        if self.unique_coordinates:
            x, y = zip(*self.unique_coordinates)
            self.ax.scatter(x, y, c='g', marker='x', label='Cones')

        if self.s_cones:
            blue, yellow = self.s_cones
            if blue: self.ax.scatter(*zip(*blue), c='b', marker='x', label='Blue Cones')
            if yellow: self.ax.scatter(*zip(*yellow), c='y', marker='o', label='Yellow Cones')

        arrow = FancyArrowPatch(
            posA=(self.car_position[0], self.car_position[1]),
            posB=(self.car_position[0]+0.5*math.cos(self.yaw),
                  self.car_position[1]+0.5*math.sin(self.yaw)),
            color='r', arrowstyle='->', mutation_scale=10
        )
        self.ax.add_patch(arrow)

        if len(self.path)>1:
            self.ax.plot(*zip(*self.path), 'r--')
        if self.estimated_positions:
            est_np = np.array(self.estimated_positions)
            self.ax.plot(est_np[:,0], est_np[:,1], 'g--', label='Estimated Path')

        x_pad, y_pad = 2, 2
        if self.all_blue_cones or self.all_yellow_cones:
            all_cones = self.all_blue_cones + self.all_yellow_cones
            max_dist = max(np.sqrt((x-self.car_position[0])**2+(y-self.car_position[1])**2) for x,y in all_cones)
            self.ax.set_xlim(self.car_position[0]-max_dist-x_pad, self.car_position[0]+max_dist+x_pad)
            self.ax.set_ylim(self.car_position[1]-max_dist-y_pad, self.car_position[1]+max_dist+y_pad)

        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.legend()
        plt.draw()
        plt.pause(0.001)


def main(args=None):
    rclpy.init(args=args)
    ekf_slam_node = EKFSLAM()
    try:
        rclpy.spin(ekf_slam_node)
    except KeyboardInterrupt:
        pass
    ekf_slam_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

