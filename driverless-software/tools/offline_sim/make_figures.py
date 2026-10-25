"""
Make the figures used in the README (saved to docs/media).

    python3 make_figures.py            # all figures
    python3 make_figures.py gif        # only the animation
"""

import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from scipy.spatial import Delaunay  # noqa: E402

import sim  # noqa: E402
import fake_ros  # noqa: E402

OUT = os.path.join(sim.ROOT, "docs", "media")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, YELLOW, ORANGE, PATH, RED = "#2a78d6", "#eda100", "#eb6834", "#1baf7a", "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "lines.linewidth": 2.0,
})


def draw_cones(ax, cones, size=10, alpha=1.0):
    ax.scatter(*cones["blue"].T, s=size, color=BLUE, alpha=alpha, zorder=3, linewidths=0)
    ax.scatter(*cones["yellow"].T, s=size, color=YELLOW, alpha=alpha, zorder=3, linewidths=0)
    orange = np.vstack([cones.get("orange", np.zeros((0, 2))), cones.get("big_orange", np.zeros((0, 2)))])
    if len(orange):
        ax.scatter(*orange.T, s=size * 1.5, color=ORANGE, alpha=alpha, zorder=3, linewidths=0)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ----------------------------------------------------------------- 1. laps
def fig_laps():
    tracks = sim.LOOP_TRACKS
    fig, axs = plt.subplots(2, 5, figsize=(17, 7.2))
    for ax, t in zip(axs.flat, tracks):
        r = sim.run(t)
        draw_cones(ax, r["cones"], size=5)
        F = r["frames"]
        ax.plot([f["x"] for f in F], [f["y"] for f in F], color=PATH, lw=1.6, zorder=4)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        hit = "no cones hit" if r["cones_hit"] == 0 else f"{r['cones_hit']} cones hit"
        ax.set_title(f"{t.replace('_', ' ')}\n{r['time']:.1f} s  |  {hit}", fontsize=10, linespacing=1.5)
    fig.suptitle("One lap on each EUFS track: Delaunay planner + pure pursuit (offline sim, 3 m/s)",
                 fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    save(fig, "laps_overview.png")


# ----------------------------------------------------------------- 2. planners
def fig_planners():
    cones, _ = sim.load_track("peanut")
    pose = (13.6, -2.6, -0.15)                   # peanut, entering the big left-hand loop
    obs = sim.visible_cones(cones, pose)
    fig, axs = plt.subplots(1, 3, figsize=(12, 5.6), sharex=True, sharey=True)
    titles = {"midpoint_planner": "1. Midpoint planner (baseline)",
              "rrt_planner": "2. RRT guided by Delaunay midpoints",
              "delaunay_planner": "3. Delaunay gate planner (used)"}
    blue = np.vstack([obs["blue"], obs["big_orange"][obs["big_orange"][:, 1] > 0]])
    yellow = np.vstack([obs["yellow"], obs["big_orange"][obs["big_orange"][:, 1] <= 0]])
    for ax, planner in zip(axs, ["midpoint_planner", "rrt_planner", "delaunay_planner"]):
        fake_ros.reset()
        fake_ros.PARAMS["seed"] = 3
        mod = __import__(sim.PLANNERS[planner], fromlist=["Planner"])
        node = mod.Planner(planner)
        if planner == "rrt_planner":            # show the tree too
            from planning.rrt_planner_node import RRT, delaunay_midpoints
            from planning.midpoint_planner_node import sort_from_car
            guide = sort_from_car(delaunay_midpoints(blue, yellow), max_jump=4.0)
            rrt = RRT(guide, np.vstack([blue, yellow]), rng=np.random.default_rng(3))
            rrt.plan()
            nodes = np.array(rrt.nodes)
            for i, p in enumerate(rrt.parents):
                if p >= 0:
                    ax.plot(*np.array([nodes[p], nodes[i]]).T, color=INK2, lw=0.5, alpha=0.35, zorder=1)
        if planner == "delaunay_planner":       # show the triangulation
            pts = np.vstack([blue, yellow])
            tri = Delaunay(pts)
            ax.triplot(pts[:, 0], pts[:, 1], tri.simplices, color=INK2, lw=0.5, alpha=0.35, zorder=1)
        fake_ros.publish("/ground_truth/cones", sim.cones_msg(obs))
        traj = fake_ros.LAST["/trajectory"]
        path = np.array([[w.position.x, w.position.y] for w in traj.waypoints])
        draw_cones(ax, {"blue": blue, "yellow": yellow}, size=40)
        ax.plot(*path.T, color=PATH, lw=2.5, zorder=4)
        ax.scatter([0], [0], marker=">", s=120, color=INK, zorder=5)
        ax.set_title(titles[planner], loc="left")
        ax.set_aspect("equal")
        ax.set_xlabel("x, car frame [m]")
    axs[0].set_ylabel("y, car frame [m]")
    fig.tight_layout()
    save(fig, "planner_comparison.png")


# ----------------------------------------------------------------- 3. EKF
def fig_ekf():
    from state_estimation.ekf_node import EKF
    rng = np.random.default_rng(4)
    dt = 0.05
    ekf = EKF(dt)
    truth = np.array([0.0, 0.0, 0.0])
    ekf.x = np.array([[0.0], [0.0], [0.0], [3.0]])
    T, Z, X, err_z, err_x, ts = [], [], [], [], [], []
    for i in range(700):
        t = i * dt
        v, w = 3.0, 0.35 * math.sin(0.25 * t)
        truth = truth + [v * dt * math.cos(truth[2]), v * dt * math.sin(truth[2]), w * dt]
        z = np.array([[truth[0]], [truth[1]], [truth[2]]]) + np.array([[0.6], [0.6], [0.03]]) * rng.normal(size=(3, 1))
        u = np.array([[v], [w]]) + np.array([[0.15], [0.02]]) * rng.normal(size=(2, 1))
        x = ekf.step(z, u)
        T.append(truth[:2].copy()), Z.append(z[:2, 0]), X.append(x[:2, 0].copy()), ts.append(t)
        err_z.append(np.linalg.norm(z[:2, 0] - truth[:2]))
        err_x.append(np.linalg.norm(x[:2, 0] - truth[:2]))
    T, Z, X = np.array(T), np.array(Z), np.array(X)
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    a.scatter(*Z.T, s=5, color=INK2, alpha=0.35, linewidths=0, label="noisy position (GPS, sigma 0.6 m)")
    a.plot(*T.T, color=INK, lw=1.2, ls="--", label="true path")
    a.plot(*X.T, color=PATH, lw=2.0, label="EKF estimate")
    a.set_aspect("equal")
    a.set_xlabel("x [m]")
    a.set_ylabel("y [m]")
    a.set_title("EKF: odometry + IMU yaw + noisy position", loc="left")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
    b.plot(ts, err_z, color=INK2, lw=1.0, alpha=0.6, label=f"raw measurement (mean {np.mean(err_z):.2f} m)")
    b.plot(ts, err_x, color=PATH, lw=2.0, label=f"EKF (mean {np.mean(err_x):.2f} m)")
    b.set_xlabel("time [s]")
    b.set_ylabel("position error [m]")
    b.set_ylim(0, None)
    b.set_title("Position error", loc="left")
    b.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "ekf_estimate.png")


# ----------------------------------------------------------------- 4. map
def fig_map(track="peanut"):
    """Closed loop: drifting odometry + IMU (+ GPS) -> EKF node -> mapping node."""
    import slam_eval
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.6), sharex=True, sharey=True)
    for ax, gps in zip(axs, (True, False)):
        r = slam_eval.evaluate(track, laps=2, use_gps=gps)
        h = r["ekf_hist"]
        for colour, c in (("blue", BLUE), ("yellow", YELLOW)):
            ax.scatter(*r["cones"][colour].T, s=70, facecolors="none", edgecolors=INK, linewidths=0.8, zorder=3)
            ax.scatter(*r["map"][colour].T, s=24, color=c, zorder=4)
        ax.plot(h[:, 1], h[:, 2], color=INK2, lw=1.0, ls="--", label="true path")
        ax.plot(h[:, 5], h[:, 6], color=PATH, lw=1.4, label="SLAM pose")
        ax.scatter([], [], s=70, facecolors="none", edgecolors=INK, linewidths=0.8, label="true cone")
        ax.scatter([], [], s=24, color=BLUE, label="mapped cone")
        title = "with GPS (0.5 m noise)" if gps else "without GPS (dead reckoning + map)"
        ax.set_title(f"{title}\nmean cone error {100 * r['mean_err']:.0f} cm, "
                     f"{r['matched']}/{r['n_true']} cones within 0.5 m, {r['false']} fake cones",
                     loc="left", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
    axs[0].set_ylabel("y [m]")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Cone map after 2 laps of {track}: EKF + cone mapping, noisy detections",
                 fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "cone_map.png")


# ----------------------------------------------------------------- 5. gif
def fig_gif(track="peanut"):
    r = sim.run(track, record_every=4)
    F = r["frames"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    draw_cones(ax, r["cones"], size=14)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    driven, = ax.plot([], [], color=INK2, lw=1.2)
    planned, = ax.plot([], [], color=PATH, lw=3.0, zorder=5)
    car, = ax.plot([], [], marker="o", ms=8, color=RED, zorder=6)
    txt = ax.text(0.01, 0.98, "", transform=ax.transAxes, va="top", fontsize=9, color=INK)
    ax.set_title(f"{track}: planned trajectory (green) updated every frame", loc="left", fontsize=10)
    xs, ys = [f["x"] for f in F], [f["y"] for f in F]

    def update(i):
        f = F[i]
        driven.set_data(xs[:i + 1], ys[:i + 1])
        if len(f["path"]):
            planned.set_data(*f["path"].T)
        car.set_data([f["x"]], [f["y"]])
        txt.set_text(f"t = {f['t']:4.1f} s   v = {f['v']:.1f} m/s   steer = {math.degrees(f['delta']):5.1f} deg")
        return driven, planned, car, txt

    anim = FuncAnimation(fig, update, frames=range(0, len(F), 2), blit=True)
    path = os.path.join(OUT, "trackdrive.gif")
    anim.save(path, writer=PillowWriter(fps=15), dpi=70)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1:] or ["laps", "planners", "ekf", "map", "gif"]
    for w in which:
        {"laps": fig_laps, "planners": fig_planners, "ekf": fig_ekf, "map": fig_map, "gif": fig_gif}[w]()
