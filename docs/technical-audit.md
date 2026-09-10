# Technical audit

## Baseline issues found

The original repository contained useful project experiments, but several parts were not technically safe to present as a complete autonomous-driving stack.

### SLAM

The original `SLAM/ekf_slam.py` mixed ground-truth state, estimated state, visualization, filtering experiments, and loop-closure logic in one node. It also referenced state such as `self.s_cones` without defining it and did not contain a complete EKF prediction/update implementation despite being described as EKF-SLAM.

The repository therefore distinguishes between the **project concept/experiment** and a mathematically complete EKF implementation. The code should only be described as EKF-SLAM after the state vector, motion model, Jacobians, covariance prediction, measurement model, data association, and update equations are explicitly implemented and tested.

### Planning

The original RRT class stored blue/yellow cones as `obstacles_*`, but `collides()` only tested whether a candidate point was already in the tree. Therefore it did not perform obstacle collision checking.

The original sampler used `np.random.uniform(low=start, high=goal)`, which is not a general 2-D sampling strategy when coordinates have different ordering and does not represent a proper planning search space. The tree also had no parent/cost structure, so it was **RRT, not RRT***, and even that RRT implementation was incomplete.

The corrected documentation therefore avoids calling this code RRT* until nearest-neighbour selection, steering, collision checking, parent selection, path cost, and rewiring are actually implemented.

### Control

The original controllers contained an important frame-consistency problem. The planning nodes publish waypoints with `base_footprint` as their frame, while the Pure Pursuit implementation added the vehicle's global position to those waypoints. That mixes local and global coordinates and can produce incorrect target locations.

The control equations themselves should operate on a consistent coordinate frame. For a local `base_footprint` path, the controller should use the waypoint coordinates directly. For a global `map` path, the path should first be transformed into the controller frame using TF2.

The Stanley implementation also used a hard-coded `target_speed = 15.0` m/s, which conflicts with the stated project speed limits and should not be presented as a validated vehicle parameter.

## Validation policy

Changes in this repository should be tested in the target ROS 2/EUFS environment before being described as experimentally validated. A clean Python syntax check is not equivalent to a successful ROS 2 runtime test.

## Portfolio policy

The repository should show what was actually implemented, what was tested, and what remains experimental. It should not claim full autonomy, production readiness, RRT*, EKF-SLAM, or successful hardware deployment unless the corresponding implementation and evidence exist in the repository.
