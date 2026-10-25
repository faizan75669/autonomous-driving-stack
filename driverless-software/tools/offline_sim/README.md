# Offline simulator

A small closed-loop simulator so the nodes can be tested without ROS or Gazebo.

| File | What it is |
|---|---|
| `fake_ros.py` | Minimal stand-in for `rclpy` and the message types our nodes use |
| `sim.py` | Track loading, EUFS-style cone visibility, kinematic bicycle car, lap runner |
| `results_table.py` | Runs every planner/controller pair on every track, prints the README table |
| `make_figures.py` | Makes the figures in `docs/media` |
| `tracks/` | Track files from the [EUFS simulator](https://gitlab.com/eufs/eufs_sim) (MIT, see `LICENSE_EUFS`) |

```bash
python3 sim.py                                   # every loop track, default stack
python3 sim.py peanut rectangle --controller stanley
python3 sim.py comp_2021 --planner midpoint_planner --max-speed 4
```

Car model: kinematic bicycle, wheelbase 1.58 m, steering limit 0.52 rad (full lock in 1 s),
acceleration between -10 and 3 m/s^2, plus aero drag. Cones are visible when they are
0.5-15 m away and within 60 degrees of the heading, like the EUFS ground-truth cone plugin.

Limitations: no tyre slip, no sensor noise on the cones, no latency. So passing here does
not mean it will pass in Gazebo, but failing here almost always means it will fail there.
