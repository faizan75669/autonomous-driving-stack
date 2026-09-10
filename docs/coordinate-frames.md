# Coordinate frames

The current planning/control interface uses a **local vehicle frame**.

## `base_footprint`

- Origin: vehicle reference point.
- `+x`: forward.
- `+y`: left.
- Planner waypoints are published in this frame.
- Pure Pursuit and Stanley consume the coordinates directly in this frame.

## `map`

EKF-SLAM publishes the estimated vehicle pose in `map`.

A global path should not be passed directly to the current controllers. If a future planner produces `map` coordinates, transform every waypoint into `base_footprint` with TF2 before running the local control equations.

## Why this matters

For a local waypoint `(x, y)`, the controller already has the displacement from the vehicle. Adding the global vehicle position again would double-count the translation and produce an incorrect steering target.
