"""
Run every planner / controller combination on the EUFS tracks and print the
markdown results table used in the README.

    python3 results_table.py
"""

from multiprocessing import Pool

import sim

COMBOS = [("delaunay_planner", "pure_pursuit"), ("delaunay_planner", "stanley"),
          ("midpoint_planner", "pure_pursuit"), ("rrt_planner", "pure_pursuit")]
# RRT re-plans 400 samples every frame, too slow for the long tracks
RRT_TRACKS = ["small_track", "peanut", "bone", "boa_constrictor", "its_a_mess", "rectangle"]


def job(args):
    track, planner, controller = args
    r = sim.run(track, planner, controller, record_every=10)
    return track, planner, controller, r["lap"], r["time"], r["cones_hit"], r["min_clearance"]


def cell(lap, t, hit, clear):
    if lap is None:
        return "not run"
    if not lap:
        return "did not finish"
    return f"{t:.1f} s, {hit} hit, {clear:.2f} m"


if __name__ == "__main__":
    jobs = [(t, p, c) for t in sim.LOOP_TRACKS for p, c in COMBOS
            if p != "rrt_planner" or t in RRT_TRACKS]
    res = {(t, p, c): (None, 0, 0, 0) for t in sim.LOOP_TRACKS for p, c in COMBOS}
    with Pool(2) as pool:
        for t, p, c, lap, tt, h, cl in pool.imap_unordered(job, jobs):
            res[(t, p, c)] = (lap, tt, h, cl)
            print(f"# done {t} {p} {c}: {cell(lap, tt, h, cl)}", flush=True)

    head = ["Track"] + [f"{p.replace('_planner', '')} + {c.replace('_', ' ')}" for p, c in COMBOS]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for t in sim.LOOP_TRACKS:
        print("| " + " | ".join([f"`{t}`"] + [cell(*res[(t, p, c)]) for p, c in COMBOS]) + " |")
    print()
    for p, c in COMBOS:
        ran = [t for t in sim.LOOP_TRACKS if res[(t, p, c)][0] is not None]
        laps = sum(bool(res[(t, p, c)][0]) for t in ran)
        clean = sum(bool(res[(t, p, c)][0]) and res[(t, p, c)][2] == 0 for t in ran)
        print(f"{p} + {c}: {laps}/{len(ran)} laps, {clean}/{len(ran)} without touching a cone")
