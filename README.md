# Autonomous Driving Stack

Hi, I am Faizan. This repo has the autonomy work I have done so far: a driverless race car
for Formula Student AI, and GPS-denied navigation for a hexacopter drone. Each project has
its own folder.

| Project | Folder |
|---|---|
| **Driverless race car**, FS-AI 2025 at Silverstone (Team FERN, PNEC): path planning, EKF, cone mapping, pure pursuit and Stanley control | [`driverless-software/`](driverless-software) |
| **GPS-denied hexacopter navigation** at Prodigy Airworks: 4 solid-state LiDARs, time sync, FAST-LIO2 SLAM, OctoMap and RRT* on a Jetson AGX Orin | [`gps-denied-uav-navigation/`](gps-denied-uav-navigation) |

---

## 1. Driverless race car (FS-AI 2025)

I was the autonomous software developer of **Team FERN** from **Pakistan Navy Engineering
College (PNEC)**. We took part in **Formula Student AI (FS-AI) 2025 at Silverstone, UK**,
and as far as we know it was the first time a team from Pakistan ran a driverless car
there. We finished **17th overall (290.7 points)** in the Dynamic Driving Task class, where
teams write the autonomous software for the IMechE ADS-DV car.

In FS-AI the car drives itself around a track marked only by cones: blue on the left,
yellow on the right, orange at the start line. All the autonomous software in
`driverless-software/` (planning, state estimation, mapping and control) is my work; the
rest of the team worked on the other parts of the project. I developed and tested it in the
[EUFS simulator](https://gitlab.com/eufs/eufs_sim) on ROS 2.

<p align="center">
  <img src="driverless-software/docs/media/fsai_2025_results.png" width="620" alt="FS-AI 2025 overall results, PNEC 17th with 290.7 points"><br>
  <em>FS-AI 2025 overall results (IMechE). Team FERN / PNEC is car 146, 17th place.</em>
</p>

### What I worked on, in order

1. **Path planning from cone positions.** I started with a simple midpoint planner (middle
   point between blue and yellow cones). It went zig-zag in corners, so I moved to
   **Delaunay triangulation** of the cones, then worked on **RRT / RRT\***, and in the end
   combined the two: the RRT samples around the Delaunay midpoints so the tree grows along
   the track.
2. **EKF** for the car's position, using wheel odometry, IMU and GPS.
3. **Localisation and cone mapping on top of the EKF** (the SLAM part): putting cone
   detections into a map, clustering them, counting laps and detecting loop closure.
4. **Control**: pure pursuit for steering and PID for speed first, then I implemented
   **Stanley** as well to compare the two.

All of these are in this repo, so you can see the planners side by side and compare the
controllers on the same tracks.

<p align="center">
  <img src="driverless-software/docs/media/trackdrive.gif" width="560" alt="Car driving the peanut track, planned trajectory in green"><br>
  <em>The stack driving the EUFS "peanut" track. Green is the path the planner publishes every frame.</em>
</p>

### What is inside

Everything lives in one folder, [`driverless-software/`](driverless-software), split the way
the car's software is split (each one is a ROS 2 package):

| Folder | What it does |
|---|---|
| [`state_estimation/`](driverless-software/state_estimation) | EKF that fuses wheel odometry, IMU yaw and GPS into the car's pose |
| [`slam/`](driverless-software/slam) | Builds the cone map, corrects the pose against it, counts laps, detects loop closure |
| [`planning/`](driverless-software/planning) | Midpoint planner, Delaunay planner, and RRT guided by Delaunay midpoints |
| [`control/`](driverless-software/control) | Pure pursuit and Stanley steering controllers with PID speed control |
| [`bringup/`](driverless-software/bringup) | One launch file + one params file to run the whole stack with the simulator |
| [`tools/offline_sim/`](driverless-software/tools/offline_sim) | Small Python simulator to test the real nodes without Gazebo |
| [`tests/`](driverless-software/tests) | Automated tests for every module |

The detailed write-up (how each part works, results, what is still not good) is in
[`driverless-software/README.md`](driverless-software/README.md).

### Screenshots from the EUFS simulator

| SLAM | Planning | Pure pursuit |
|---|---|---|
| ![slam](https://github.com/user-attachments/assets/9062e8ed-6b6e-4bbc-a171-da7c7ab0b250) | ![planning](https://github.com/user-attachments/assets/0979890b-9f5c-451b-b5ba-a89a9136889a) | ![pure](https://github.com/user-attachments/assets/34008751-45e6-4714-ba5a-92c365adb5bd) |

### Quick start

```bash
# test everything without ROS (needs numpy, scipy, scikit-learn, pytest)
cd driverless-software
python3 -m pytest tests -q

# drive one lap of every EUFS track in the offline simulator
python3 tools/offline_sim/sim.py
```

To run it with ROS 2 and the EUFS simulator, see
[Running with ROS 2](driverless-software/README.md#running-with-ros-2).

## 2. GPS-denied hexacopter navigation (Prodigy Airworks)

At Prodigy Airworks I worked on making the company's hexacopter navigate without GPS. I
built the navigation stack starting from perception: four RoboSense E1R solid-state LiDARs
mounted 90° apart for 360° coverage, time synchronisation so the overlapping LiDARs do not
create double objects, a node that merges the four clouds into one virtual LiDAR for
FAST-LIO2 (with the IMU from the CUAV X25 EVO flight controller over UART), then OctoMap
for a 3D occupancy map and OMPL RRT* for path planning, all on a Jetson AGX Orin.

The code belongs to the company, so this folder has the write-up, photos and recordings
instead: [`gps-denied-uav-navigation/README.md`](gps-denied-uav-navigation/README.md).

| | |
|---|---|
| ![Hexacopter](gps-denied-uav-navigation/media/hexacopter.jpg) | ![3D map](gps-denied-uav-navigation/media/map_height_coloured.jpg) |
| The hexacopter in the lab | 3D LiDAR map of the lab |

---

## Credits

Driverless race car:

- [EUFS simulator](https://gitlab.com/eufs/eufs_sim) (MIT) by Edinburgh University Formula Student, the track files in `tools/offline_sim/tracks` are theirs.
- The EKF started from Atsushi Sakai's [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics), the GPS conversion is from [geonav_transform](https://github.com/bsb808/geonav_transform).
- The rest of Team FERN, who worked on the other parts of the project and made the competition happen.
- Some of the refactoring (the Delaunay gate planner, the controller clean-up) and the offline test setup were done with help from an AI coding assistant (Claude).

GPS-denied navigation: [FAST-LIO2](https://github.com/hku-mars/FAST_LIO) by the HKU MARS lab,
[OctoMap](https://octomap.github.io/), [OMPL](https://ompl.kavrakilab.org/), and Prodigy
Airworks for the hardware and for letting me share the photos and recordings.
