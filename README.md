# Autonomous Driving Stack

Hi, I am Faizan. This repo has the driverless software I worked on as a member of the
driverless software team of **Team FERN**, our Formula Student team. Team FERN took part in
**Formula Student AI (FS-AI) 2025 at Silverstone, UK**, where the car has to drive itself
around a track marked only by cones: blue on the left, yellow on the right, orange at the
start line.

My work was mostly on the software side of the car: estimating where the car is, building
a map of the cones, planning a path between them and controlling the steering and speed to
follow that path. We developed and tested everything in the
[EUFS simulator](https://gitlab.com/eufs/eufs_sim) on ROS 2.

<p align="center">
  <img src="driverless-software/docs/media/trackdrive.gif" width="560" alt="Car driving the peanut track, planned trajectory in green"><br>
  <em>The stack driving the EUFS "peanut" track. Green is the path the planner publishes every frame.</em>
</p>

## What is inside

Everything lives in one folder, [`driverless-software/`](driverless-software), split the way
the car's software is split (each one is a ROS 2 package):

| Folder | What it does |
|---|---|
| [`state_estimation/`](driverless-software/state_estimation) | EKF that fuses wheel odometry, IMU yaw and GPS into the car's pose |
| [`slam/`](driverless-software/slam) | Builds the cone map from detections + EKF pose, counts laps, detects loop closure |
| [`planning/`](driverless-software/planning) | Three local planners: midpoint (my first one), RRT, and the Delaunay planner we use |
| [`control/`](driverless-software/control) | Pure pursuit and Stanley steering controllers with PID speed control |
| [`bringup/`](driverless-software/bringup) | One launch file + one params file to run the whole stack with the simulator |
| [`tools/offline_sim/`](driverless-software/tools/offline_sim) | Small Python simulator to test the real nodes without Gazebo |
| [`tests/`](driverless-software/tests) | Automated tests for every module |

The detailed write-up (how each part works, results, what is still not good) is in
[`driverless-software/README.md`](driverless-software/README.md). I will keep adding more
projects next to this folder in future.

## Screenshots from the EUFS simulator

| SLAM | Planning | Pure pursuit |
|---|---|---|
| ![slam](https://github.com/user-attachments/assets/9062e8ed-6b6e-4bbc-a171-da7c7ab0b250) | ![planning](https://github.com/user-attachments/assets/0979890b-9f5c-451b-b5ba-a89a9136889a) | ![pure](https://github.com/user-attachments/assets/34008751-45e6-4714-ba5a-92c365adb5bd) |

## Quick start

```bash
# test everything without ROS (needs numpy, scipy, scikit-learn, pytest)
cd driverless-software
python3 -m pytest tests -q

# drive one lap of every EUFS track in the offline simulator
python3 tools/offline_sim/sim.py
```

To run it with ROS 2 and the EUFS simulator, see
[Running with ROS 2](driverless-software/README.md#running-with-ros-2).

## Credits

- [EUFS simulator](https://gitlab.com/eufs/eufs_sim) (MIT) by Edinburgh University Formula Student, the track files in `tools/offline_sim/tracks` are theirs.
- The EKF started from Atsushi Sakai's [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics), the GPS conversion is from [geonav_transform](https://github.com/bsb808/geonav_transform).
- My teammates at Team FERN, a lot of this was built and debugged together.
- Some of the refactoring (the Delaunay gate planner, the controller clean-up) and the offline test setup were done with help from an AI coding assistant (Claude).
