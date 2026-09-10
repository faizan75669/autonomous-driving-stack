# Autonomous Driving Stack

A ROS 2 software stack for an FSAE-style autonomous vehicle, organized around the three core autonomy layers: **localization & mapping, path planning, and vehicle control**.

> This repository documents and consolidates the algorithms developed for the driverless vehicle project. Simulation-specific topics and message types are kept configurable where practical.

## Architecture

```text
                 Cone / Sensor Observations
                           |
                           v
                 +-------------------+
                 |      SLAM         |
                 | EKF-SLAM / Map    |
                 +---------+---------+
                           |
                 estimated pose + map
                           |
                           v
                 +-------------------+
                 |     Planning      |
                 | cone pairing /    |
                 | midpoint / RRT*   |
                 +---------+---------+
                           |
                     trajectory
                           |
                           v
                 +-------------------+
                 |     Control       |
                 | Pure Pursuit /    |
                 | Stanley + speed   |
                 +---------+---------+
                           |
                           v
                 AckermannDrive command
```

## Repository layout

```text
.
├── SLAM/
│   └── ekf_slam.py
├── planning/
│   ├── planning_midpoint_interpolation.py
│   └── planning_triangulation.py
├── control/
│   ├── pure_pursuit.py
│   ├── stanley.py
│   └── controller.gif
├── docs/
│   ├── architecture.md
│   └── technical-audit.md
├── requirements.txt
└── README.md
```

## Modules

### 1. SLAM

The SLAM module is intended to estimate the vehicle pose while building a map from observed cone landmarks.

Key concepts:

- vehicle state: \(x, y, \theta\)
- landmark state: \(l_x, l_y\)
- motion prediction from vehicle velocity / yaw rate
- landmark measurement update using range and bearing
- covariance propagation and measurement uncertainty
- data association before updating an existing landmark
- loop-closure handling must be based on actual landmark correspondence, not only proximity to an arbitrary cluster

### 2. Planning

The planning layer converts cone observations into a drivable centerline and, where required, searches for an obstacle-free path.

The intended pipeline is:

```text
Blue + Yellow cones
        |
        v
Geometric pairing / triangulation
        |
        v
Track-center midpoints
        |
        +----> direct centerline / interpolation
        |
        +----> RRT* local detour when an obstacle blocks the nominal path
        |
        v
Trajectory
```

RRT* is only useful when a genuine search problem exists. A random tree whose collision test never checks the cone obstacles is **not** an RRT planner.

### 3. Control

Two lateral controllers are retained for comparison:

- **Pure Pursuit** — geometric tracking using a look-ahead point.
- **Stanley** — heading-error plus signed cross-track-error correction.

Longitudinal control is separated conceptually from lateral control and uses a configurable target speed and PID acceleration command.

## Coordinate-frame rule

The planner and controller must agree on the waypoint frame.

If `/trajectory` is published in `base_footprint`, the controller must **not add the vehicle's global position again**. If the trajectory is in a global frame such as `map`, it must first be transformed into the controller's working frame using TF2.

This distinction is important because mixing local and global coordinates can produce apparently unstable steering even when the controller equation itself is correct.

## Dependencies

The current project targets the ROS 2 / EUFS simulation environment and uses:

- ROS 2
- Python 3
- NumPy
- SciPy
- scikit-learn where required by the legacy filtering experiments
- `eufs_msgs`
- `ackermann_msgs`
- standard ROS 2 geometry/navigation/visualization messages

See `docs/technical-audit.md` for known simulation-specific assumptions and corrections.

## Important scope note

This is a research/engineering project, not a production autonomous-driving system. Numerical parameters such as steering limits, wheelbase, planner clearance, covariance values, and speed limits must be validated against the actual vehicle and simulator before deployment.
