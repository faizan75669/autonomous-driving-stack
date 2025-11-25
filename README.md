# 🚘 Autonomous Driving Stack

A complete autonomous driving software stack integrating **SLAM**, **Path Planning**, and **Control Algorithms** — developed for an FSAE-style autonomous vehicle.

---

##  Modules Overview

###  SLAM
- **EKF-SLAM** implementation for localization and mapping using cone landmarks.
- ROS 2 node subscribes to vehicle odometry and cone detections.
- ![planning](https://github.com/user-attachments/assets/05966335-7578-44c8-ae3a-ff3e20fbdd1e)


###  Planning
- **Triangulation-based planner** for smooth local paths.
- **Midpoint planner** for efficient cone-to-cone path generation.
- 

###  Control
- **Stanley Controller** for lateral control and stability.
- **Pure Pursuit** for trajectory following.

---

##  Visualization
- Real-time trajectory and cone visualization using Matplotlib.
- Displays the car’s live position and estimated map.
