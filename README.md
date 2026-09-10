# Autonomous Driving Stack

A ROS 2 autonomous-driving research stack organized around **localization / mapping → planning → control**. The repository contains the corrected implementations of the algorithms that were previously mixed together in larger experimental scripts.

> **Portfolio note:** this is an engineering/research repository, not a claim of production-ready autonomous driving. ROS runtime and vehicle-level validation still require the original simulation/hardware environment.

## Architecture

```text
             Cone / odometry observations
                       │
                       ▼
              ┌─────────────────┐
              │   EKF-SLAM      │
              │ pose + landmarks│
              └────────┬────────┘
                       │ estimated pose / map
                       ▼
              ┌─────────────────┐
              │    Planning     │
              │ cones → path    │
              │ midpoint / RRT* │
              └────────┬────────┘
                       │ local trajectory
                       ▼
              ┌─────────────────┐
              │     Control     │
              │ Pure Pursuit or │
              │ Stanley + speed │
              └────────┬────────┘
                       │
                       ▼
                Ackermann command
```

## Repository layout

```text
.
├── SLAM/
│   ├── ekf_slam.py              # ROS 2 adapter
│   └── ekf_slam_core.py         # ROS-independent EKF-SLAM math
├── planning/
│   ├── planning_midpoint_interpolation.py  # simple centerline baseline
│   ├── planning_triangulation.py           # cone pairing + RRT* integration
│   └── rrt_star.py                         # ROS-independent RRT* core
├── control/
│   ├── control_core.py           # ROS-independent steering equations
│   ├── pure_pursuit.py            # ROS 2 Pure Pursuit node
│   ├── stanley.py                 # ROS 2 Stanley node
│   └── controller.gif             # existing visualization asset
├── tests/
│   └── test_core_algorithms.py
├── docs/
│   ├── architecture.md
│   └── technical-audit.md
├── requirements.txt
└── README.md
```

## 1. EKF-SLAM

The SLAM state is

```text
[x, y, yaw, landmark_1_x, landmark_1_y, ...]
```

The filter performs:

1. **Prediction:** propagate vehicle pose using forward velocity and yaw rate.
2. **Measurement model:** convert each local cone observation `(range, bearing)` into an expected measurement.
3. **Data association:** compare observations against existing landmarks using normalized innovation squared (Mahalanobis / chi-square gating).
4. **Update:** apply the EKF correction and Joseph-form covariance update.
5. **Initialization:** add previously unseen landmarks from their range/bearing measurement.

Cone class is retained during association so blue, yellow and orange landmarks are not freely matched to one another.

The ROS node does **not** feed ground-truth pose into the filter. Ground truth should be used separately when evaluating estimator error.

## 2. Planning

### Midpoint baseline

`planning_midpoint_interpolation.py` provides the simple centerline baseline:

```text
blue cones + yellow cones
          ↓
one-to-one geometric pairing
          ↓
       midpoints
          ↓
 optional spline interpolation
          ↓
     local trajectory
```

The trajectory is published in `base_footprint`.

### Triangulation + RRT*

`planning_triangulation.py` uses cross-color Delaunay edges to generate plausible cone pairs. If the direct centerline route is blocked, it invokes the independent `RRTStar` implementation.

`rrt_star.py` contains the actual RRT* operations:

- state-space sampling
- nearest-node selection
- steering
- circular cone safety regions
- segment collision checking
- nearby-node search
- minimum-cost parent selection
- rewiring
- goal connection
- path reconstruction

A random tree without collision checking, parent-cost optimization and rewiring is **not** RRT*, so those operations are implemented explicitly here.

The current planner is 2-D. It does not claim full 3-D vehicle planning or guarantee a track-valid solution for every cone arrangement.

## 3. Control

Both controllers consume a local `base_footprint` trajectory.

### Pure Pursuit

For a look-ahead target `(x_t, y_t)` in the vehicle frame:

```text
alpha = atan2(y_t, x_t)
steering = atan2(2 L sin(alpha), L_d)
```

Look-ahead distance is speed-dependent and bounded by configurable minimum/maximum values.

### Stanley

The controller uses the front axle reference point and combines heading error with signed cross-track error:

```text
steering = heading_error
         + atan2(k * cross_track_error, speed + softening)
```

Steering is bounded by the configured vehicle limit.

### Coordinate-frame rule

The planner publishes local points in `base_footprint`. Therefore the controller **must not add global vehicle position to those points**. If a future planner publishes a global `map` path, it must first be transformed into the controller's local frame using TF2.

## Testing

The mathematical components are separated from ROS message plumbing so they can be tested with ordinary Python tooling:

```bash
pip install -r requirements.txt
pytest -q
```

ROS nodes additionally require the ROS 2 / EUFS message environment used by the original project.

## Honest scope

This repository demonstrates implemented algorithmic building blocks and their interfaces. It should not be presented as proof of a fully validated autonomous vehicle unless simulation/hardware experiments and quantitative results are added.
