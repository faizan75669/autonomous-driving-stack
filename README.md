# Autonomous Vehicle Stack — SLAM, Planning & Control
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#license)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](#requirements)
[![CI](https://github.com/<your-username>/autonomous-vehicle-stack/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/autonomous-vehicle-stack/actions)

> End-to-end autonomous navigation stack including EKF-SLAM, Path Planning, and steering controllers (Pure Pursuit & Stanley). Built for simulation & real-robot integration (ROS2-ready).

---

## 🚀 Highlights
- EKF-SLAM for robust localization and mapping  
- Path planner for smooth trajectories and obstacle avoidance  
- Pure Pursuit and Stanley controllers for trajectory tracking  
- ROS2 launch files for easy integration with simulators (Gazebo/ignition)  
- Dockerized environment + CI for reproducible experiments

---

## Repo structure
See the `slam/`, `planning/`, and `control/` folders for details. Each module includes:
- code (`src/`)  
- notebooks (`notebooks/`) where applicable  
- tests (`tests/`) and CI checks

---

## Quick start (local)
1. Clone:
```bash
git clone https://github.com/<your-username>/autonomous-vehicle-stack.git
cd autonomous-vehicle-stack
