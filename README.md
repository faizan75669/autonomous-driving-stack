# 🚘 Autonomous Driving Stack

A complete autonomous driving software stack integrating **SLAM**, **Path Planning**, and **Control Algorithms** — developed for an FSAE-style autonomous vehicle.

---

##  Modules Overview

###  SLAM
- **EKF-SLAM** implementation for localization and mapping using cone landmarks.
- ROS 2 node subscribes to vehicle odometry and cone detections.


###  Planning
- **Triangulation-based planner** for smooth local paths.
- **Midpoint planner** for efficient cone-to-cone path generation.
- ![planning](https://github.com/user-attachments/assets/0979890b-9f5c-451b-b5ba-a89a9136889a)


###  Control
- **Stanley Controller** for lateral control and stability.
- **Pure Pursuit** for trajectory following.
- ![pure](https://github.com/user-attachments/assets/34008751-45e6-4714-ba5a-92c365adb5bd)


---

##  Visualization
- Real-time trajectory and cone visualization using Matplotlib.
- Displays the car’s live position and estimated map.
