"""
Run every planner / controller combination on every track and print the
markdown results table used in the README.

    python3 results_table.py
"""

from multiprocessing import Pool

import sim

COMBOS = [("delaunay_planner", "pure_pursuit"), ("delaunay_planner", "stanley"),
          ("midpoint_planner", "pure_pursuit"), ("rrt_planner", "pure_pursuit")]


def job(args):
    track, planner, controller = args
    r = sim.run(track, planner, controller, record_every=10)
    return track, planner, controller, r["lap"], r["time"], r["cones_hit"], r["min_clearance"]


def cell(lap, t, hit, clear):
    if not lap:
        return "did not finish"
    return f"{t:.1f} s, {hit} hit, {clear:.2f} m"


if __name__ == "__main__":
    jobs = [(t, p, c) for t in sim.LOOP_TRACKS for p, c in COMBOS]
    with Pool(2) as pool:
        res = {(t, p, c): (lap, tt, h, cl) for t, p, c, lap, tt, h, cl in pool.map(job, jobs)}
    head = ["Track"] + [f"{p.replace('_planner', '')} + {c.replace('_', ' ')}" for p, c in COMBOS]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for t in sim.LOOP_TRACKS:
        print("| " + " | ".join([f"`{t}`"] + [cell(*res[(t, p, c)]) for p, c in COMBOS]) + " |")
    print()
    for p, c in COMBOS:
        laps = sum(res[(t, p, c)][0] for t in sim.LOOP_TRACKS)
        clean = sum(res[(t, p, c)][0] and res[(t, p, c)][2] == 0 for t in sim.LOOP_TRACKS)
        print(f"{p} + {c}: {laps}/10 laps, {clean}/10 without touching a cone")
