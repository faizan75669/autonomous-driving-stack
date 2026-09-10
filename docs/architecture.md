# Stack architecture

```text
             Perception / Cone Detection
                       |
                       v
              +------------------+
              |       SLAM       |
              | pose + landmarks |
              +--------+---------+
                       |
                 pose / map
                       |
                       v
              +------------------+
              |     Planning     |
              | pairing + center |
              | line + detours   |
              +--------+---------+
                       |
                  trajectory
                       |
                       v
              +------------------+
              |     Control      |
              | Pure Pursuit /   |
              | Stanley + speed  |
              +--------+---------+
                       |
                       v
               Ackermann command
```

## Data contracts

### SLAM input

Cone detections provide landmark observations in the vehicle/sensor frame. Vehicle motion information supplies the prediction input. A real implementation must define the exact state, process model, measurement model, and covariance conventions.

### Planning input

Planning consumes the currently observed track boundaries and/or a mapped cone set. Blue and yellow cones represent the two track boundaries; orange cones can represent start/finish or special track markers depending on the simulator convention.

### Planning output

The planner publishes an ordered sequence of waypoints. The message header's `frame_id` defines whether these coordinates are local or global. All downstream nodes must respect that declaration.

### Control input

The controller consumes an ordered trajectory plus vehicle pose and velocity. It computes lateral steering and longitudinal acceleration commands and publishes an Ackermann command.

## Algorithms represented

- EKF-SLAM experiment for cone landmarks
- Delaunay-based cone pairing experiment
- midpoint centerline generation
- path interpolation
- Pure Pursuit
- Stanley
- PID-style longitudinal control

The repository intentionally separates **algorithm experiments** from claims of a fully validated end-to-end stack.
