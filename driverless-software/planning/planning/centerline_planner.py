"""
Cone-based centerline path planner (ROS independent).

All coordinates are in the CAR frame (base_footprint): x forward, y left,
exactly how EUFS publishes /ground_truth/cones.

Pipeline
--------
1. Pre-filter cones (range, behind-the-car) and put orange cones on a side.
2. Order each boundary (blue = left, yellow = right) into a chain that starts
   next to the car and walks forward, rejecting jumps that are too long or
   turn too sharply (this removes cones from other parts of the track).
3. Build centre candidates:
     * every cone is projected onto the opposite boundary -> midpoint
     * where the opposite boundary is not visible, the cone is offset
       inwards by half the (measured) track width.
4. Sort candidates by arc length along the longer boundary, merge
   duplicates, prepend the car origin (the controller treats path[0] as the
   car position) and fit a smoothing spline.
5. Resample at a fixed spacing and return an (N, 2) array.

Only numpy and scipy are used, compatible with Python 3.8 / ROS 2 Galactic.
"""

import numpy as np
from scipy.interpolate import splprep, splev


class CenterlinePlanner(object):

    def __init__(self,
                 max_view_distance=16.0,
                 max_cone_spacing=6.5,
                 max_turn_angle=np.deg2rad(75.0),
                 default_track_width=3.5,
                 min_track_width=2.0,
                 max_track_width=7.0,
                 path_spacing=0.25,
                 smoothing=0.3,
                 min_path_length=3.0,
                 optimize=False,
                 clearance=1.0,
                 opt_spacing=1.0):
        self.max_view_distance = max_view_distance
        self.max_cone_spacing = max_cone_spacing
        self.max_turn_angle = max_turn_angle
        self.default_track_width = default_track_width
        self.min_track_width = min_track_width
        self.max_track_width = max_track_width
        self.path_spacing = path_spacing
        self.smoothing = smoothing
        self.min_path_length = min_path_length
        self.optimize = optimize          # min-curvature line instead of pure centreline
        self.clearance = clearance        # car half width 0.7 + cone + margin [m]
        self.opt_spacing = opt_spacing    # resampling step for the optimisation [m]
        # track width is remembered between calls (low-pass filtered)
        self.track_width = default_track_width
        self.last_path = None

    # ------------------------------------------------------------------ API
    def plan(self, blue, yellow, orange=None, big_orange=None):
        """Return (path Nx2, debug dict). path is empty (0,2) if impossible."""
        blue = self._as_xy(blue)
        yellow = self._as_xy(yellow)
        oranges = np.vstack([self._as_xy(orange), self._as_xy(big_orange)])

        blue = self._prefilter(blue)
        yellow = self._prefilter(yellow)
        oranges = self._prefilter(oranges)

        # Orange cones: assign to the side they are on in the car frame.
        # (start/finish area is a straight, so this is reliable)
        if len(oranges):
            blue = np.vstack([blue, oranges[oranges[:, 1] > 0]])
            yellow = np.vstack([yellow, oranges[oranges[:, 1] <= 0]])

        half = 0.5 * self.track_width
        left = self._order_boundary(blue, start=np.array([0.0, half]))
        right = self._order_boundary(yellow, start=np.array([0.0, -half]))

        # 1st choice: Delaunay gate walk (robust against neighbouring lanes)
        self._gate_bounds = np.zeros((0, 2, 2))
        centers, method = self._gate_walk(blue, yellow), "delaunay"
        # extend with one-sided offsets if the far end is only seen on one side
        if len(centers) >= 2:
            n_gates = len(centers)
            centers = self._extend_with_boundary(centers, left, right)
            bounds = self._bounds_for(centers, self._gate_bounds, n_gates)
        # fallback: projection / offset method (one side visible, very few cones)
        if len(centers) < 2 or self._length(centers) < 3.0:
            centers, method = self._centre_candidates(left, right), "projection"
            bounds = self._bounds_for(centers, np.zeros((0, 2, 2)), 0)
        racing = self._min_curvature(centers, bounds) if self.optimize and len(centers) >= 2 else centers
        path = self._smooth(racing, self.smoothing if racing is centers else 0.03)

        if len(path) < 2:
            # nothing usable this frame: keep the previous plan if we have one
            path = self.last_path if self.last_path is not None else np.zeros((0, 2))
        else:
            self.last_path = path

        debug = {"left": left, "right": right, "centers": centers, "method": method,
                 "racing": racing}
        return path, debug

    # ------------------------------------------- minimum-curvature line
    def _bounds_for(self, centers, gate_bounds, n_gates):
        """(N,2,2) [left_pt, right_pt] corridor for every centre point. Real
        gates use their cones; synthetic points get +-half track width."""
        out = np.zeros((len(centers), 2, 2))
        if len(centers) < 2:
            return out
        t = np.gradient(centers, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        nrm = np.column_stack([-t[:, 1], t[:, 0]])          # points left
        half = 0.5 * self.track_width
        for i, c in enumerate(centers):
            if i < n_gates and i < len(gate_bounds):
                out[i] = gate_bounds[i]
            else:
                # one side not visible: the corridor is unknown, so this point
                # gets zero lateral freedom (never cut towards unseen cones)
                out[i] = (c, c)
        return out

    def _min_curvature(self, centers, bounds):
        """Shift every point along its gate inside the corridor so that the
        sum of squared second differences (~curvature^2) is minimal.
        Bounded linear least squares (scipy.optimize.lsq_linear)."""
        from scipy.optimize import lsq_linear
        # resample the corridor at uniform spacing so 2nd differences ~ curvature
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(centers, axis=0), axis=1))])
        if s[-1] < 2.0:
            return centers
        ds = self.opt_spacing
        s_u = np.arange(0.0, s[-1] + 1e-6, ds)
        if len(s_u) < 3:
            return centers
        L = np.column_stack([np.interp(s_u, s, bounds[:, 0, k]) for k in (0, 1)])
        R = np.column_stack([np.interp(s_u, s, bounds[:, 1, k]) for k in (0, 1)])
        M = 0.5 * (L + R)
        width = np.linalg.norm(L - R, axis=1)
        n = (L - R) / np.maximum(width, 1e-6)[:, None]      # unit vector right -> left
        # point i = M_i + a_i * n_i ;  |a_i| <= width/2 - clearance
        lim = np.maximum(0.5 * width - self.clearance, 0.0)

        # prepend the car: fixed point (0,0) plus a virtual point behind it so
        # the line leaves the car along its current heading
        P0 = np.array([[-ds, 0.0], [0.0, 0.0]])
        N = len(M)
        rows_A, rows_b = [], []
        pts_const = np.vstack([P0, M])                       # constant part
        # second differences over the full sequence (P0 has no variables)
        for i in range(1, N + 1):                            # centre index in full seq
            A = np.zeros((2, N))
            b = pts_const[i - 1] - 2 * pts_const[i] + pts_const[i + 1]
            for j, coef in ((i - 1, 1.0), (i, -2.0), (i + 1, 1.0)):
                k = j - 2                                    # variable index
                if 0 <= k < N:
                    A[:, k] += coef * n[k]
            w = 3.0 if i <= 2 else 1.0                       # respect car heading
            rows_A.append(w * A)
            rows_b.append(-w * b)
        A = np.vstack(rows_A)
        b = np.concatenate(rows_b)
        # tiny regularisation towards the centre keeps the problem well posed
        A = np.vstack([A, 0.05 * np.eye(N)])
        b = np.concatenate([b, np.zeros(N)])
        lo, hi = -lim, lim
        hi = np.where(hi - lo < 1e-6, lo + 1e-6, hi)
        try:
            res = lsq_linear(A, b, bounds=(lo, hi), method="bvls")
            a = res.x
        except Exception:
            return centers
        return M + a[:, None] * n

    # ------------------------------------------------- Delaunay gate walk
    @staticmethod
    def _length(pts):
        if len(pts) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def _gate_walk(self, blue, yellow):
        """Centre points from blue-yellow Delaunay edges, walked triangle to
        triangle. Every triangle with both colours has exactly two blue-yellow
        edges, so gates form simple chains -> no combinatorial search."""
        if len(blue) < 1 or len(yellow) < 1 or len(blue) + len(yellow) < 3:
            return np.zeros((0, 2))
        from scipy.spatial import Delaunay
        pts = np.vstack([blue, yellow])
        is_blue = np.r_[np.ones(len(blue), bool), np.zeros(len(yellow), bool)]
        try:
            tri = Delaunay(pts)
        except Exception:
            return np.zeros((0, 2))

        gate_id = {}          # (i, j) sorted -> gate index
        gates = []            # (blue_idx, yellow_idx)
        tri_gates = []        # per triangle: list of gate indices
        for simplex in tri.simplices:
            a, b, c = pts[simplex]
            area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
            g_here = []
            for u, v in ((0, 1), (1, 2), (0, 2)):
                i, j = simplex[u], simplex[v]
                if is_blue[i] == is_blue[j]:
                    continue
                w = np.linalg.norm(pts[i] - pts[j])
                if not (self.min_track_width < w < self.max_track_width):
                    continue
                key = (min(i, j), max(i, j))
                if key not in gate_id:
                    gate_id[key] = len(gates)
                    gates.append((i, j) if is_blue[i] else (j, i))
                g_here.append(gate_id[key])
            tri_gates.append(g_here if area > 0.05 else [])
        if not gates:
            return np.zeros((0, 2))

        adj = [[] for _ in gates]
        for g_here in tri_gates:
            if len(g_here) == 2:
                g1, g2 = g_here
                adj[g1].append(g2)
                adj[g2].append(g1)

        mids = np.array([0.5 * (pts[b] + pts[y]) for b, y in gates])

        def oriented_ok(g):
            # blue must be on the LEFT when crossing forward (+x direction)
            b, y = gates[g]
            return pts[b][1] - pts[y][1] > -0.5

        # start gate: the gate the car is about to drive through. It must be
        # close, in front, and "face" the car: the yellow->blue vector points
        # to the car's left (+y). Gates of the track further around a hairpin
        # are close too, but face the other way.
        start, best = None, np.inf
        for g in range(len(gates)):
            m = mids[g]
            if m[0] < -0.5 or np.linalg.norm(m) > 8.0:
                continue
            b, y = gates[g]
            gv = pts[b] - pts[y]
            gv = gv / (np.linalg.norm(gv) + 1e-9)
            if gv[1] < 0.3:        # not facing the car
                continue
            score = np.linalg.norm(m) * (2.0 - gv[1]) + 2.0 * abs(np.arctan2(m[1], max(m[0], 0.1)))
            if score < best:
                start, best = g, score
        if start is None:
            return np.zeros((0, 2))

        # walk: from start choose neighbour that leads away from the car
        chain = [start]
        visited = {start}
        prev_dir = np.array([1.0, 0.0])
        cur = start
        while True:
            cand = [n for n in adj[cur] if n not in visited]
            if not cand:
                break
            best, best_score = None, -np.inf
            for n in cand:
                step = mids[n] - mids[cur]
                d = np.linalg.norm(step)
                if d < 1e-6:
                    continue
                # blue must stay on the left of the direction of travel
                b, y = gates[n]
                bl = pts[b] - mids[n]
                cross = step[0] * bl[1] - step[1] * bl[0]
                score = np.dot(step / d, prev_dir) + (0.5 if cross > 0 else -2.0)
                if score > best_score:
                    best, best_score = n, score
            if best is None or best_score < -0.5:
                break
            step = mids[best] - mids[cur]
            prev_dir = step / np.linalg.norm(step)
            visited.add(best)
            chain.append(best)
            cur = best

        # merge near-duplicates (consecutive gates sharing a cone are close)
        out, bnd = [mids[chain[0]]], [(pts[gates[chain[0]][0]], pts[gates[chain[0]][1]])]
        for g in chain[1:]:
            if np.linalg.norm(mids[g] - out[-1]) > 0.5:
                out.append(mids[g])
                bnd.append((pts[gates[g][0]], pts[gates[g][1]]))
        out = np.array(out)
        bnd = np.array(bnd)                      # (N, 2[blue,yellow], 2[xy])
        # learn the track width from the gates used
        widths = [np.linalg.norm(pts[gates[g][0]] - pts[gates[g][1]]) for g in chain]
        self.track_width = 0.8 * self.track_width + 0.2 * float(np.median(widths))
        keep = out[:, 0] > 0.3
        self._gate_bounds = bnd[keep]
        return out[keep] if np.any(keep) else np.zeros((0, 2))

    def _extend_with_boundary(self, centers, left, right):
        """After the last gate, continue along whichever boundary is longer
        (outer side of a corner is often seen further ahead) using an
        inward offset of half the track width."""
        ext = []
        last = centers[-1]
        last_dir = centers[-1] - centers[-2]
        last_dir /= np.linalg.norm(last_dir) + 1e-9
        for own, sign in ((left, -1.0), (right, 1.0)):
            if len(own) < 2:
                continue
            nrm = self._normals(own)
            pts = own + sign * nrm * 0.5 * self.track_width
            # keep only the part clearly beyond the last centre point
            ahead = [p for p in pts if np.dot(p - last, last_dir) > 1.0]
            if len(ahead) > len(ext):
                ext = ahead
        if not ext:
            return centers
        # accept extension points only while they continue smoothly:
        # step <= 6 m and heading change <= 40 deg per step
        out = []
        prev, prev_dir = last, last_dir
        for p in ext:
            step = p - prev
            d = np.linalg.norm(step)
            if d < 0.5:
                continue
            if d > 6.0 or np.dot(step / d, prev_dir) < np.cos(np.deg2rad(40.0)):
                break
            out.append(p)
            prev, prev_dir = p, step / d
            if len(out) >= 4:
                break
        if not out:
            return centers
        return np.vstack([centers, np.array(out)])

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _as_xy(arr):
        if arr is None:
            return np.zeros((0, 2))
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return np.zeros((0, 2))
        if np.iscomplexobj(arr):
            arr = np.column_stack([arr.real, arr.imag])
        return arr.reshape(-1, 2)

    def _prefilter(self, cones):
        if len(cones) == 0:
            return cones
        d = np.linalg.norm(cones, axis=1)
        keep = (d < self.max_view_distance) & (cones[:, 0] > -2.0)
        cones = cones[keep]
        # remove exact duplicates
        if len(cones):
            cones = np.unique(np.round(cones, 3), axis=0)
        return cones

    def _order_boundary(self, cones, start):
        """Greedy chain from `start` heading +x through the cones."""
        if len(cones) == 0:
            return np.zeros((0, 2))
        remaining = list(range(len(cones)))
        chain = []
        pos = start.copy()
        heading = np.array([1.0, 0.0])
        first = True
        while remaining:
            best, best_cost = None, np.inf
            for i in remaining:
                v = cones[i] - pos
                d = np.linalg.norm(v)
                if d < 1e-6:
                    continue
                max_d = self.max_cone_spacing + (2.0 if first else 0.0)
                if d > max_d:
                    continue
                cosang = np.clip(np.dot(v, heading) / d, -1.0, 1.0)
                ang = np.arccos(cosang)
                # first cone may be beside the car, so be lenient
                max_ang = np.deg2rad(100.0) if first else self.max_turn_angle
                if ang > max_ang:
                    continue
                cost = d * (1.0 + 1.5 * ang)
                if cost < best_cost:
                    best, best_cost = i, cost
            if best is None:
                break
            new = cones[best]
            step = new - pos
            if not first or np.linalg.norm(step) > 0.5:
                h = step / np.linalg.norm(step)
                # smooth the heading slightly so one noisy cone can't flip it
                heading = h if first else (0.7 * h + 0.3 * heading)
                heading /= np.linalg.norm(heading)
            pos = new
            chain.append(new)
            remaining.remove(best)
            first = False
        return np.array(chain) if chain else np.zeros((0, 2))

    @staticmethod
    def _project_on_polyline(p, poly):
        """Closest point of p on polyline. Returns (point, s, dist, t_end)
        where t_end flags projection beyond the first/last vertex."""
        if len(poly) == 1:
            q = poly[0]
            return q, 0.0, np.linalg.norm(p - q), True
        seg = np.diff(poly, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        seg_len[seg_len < 1e-9] = 1e-9
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        best = (None, 0.0, np.inf, False)
        for i in range(len(seg)):
            t_raw = np.dot(p - poly[i], seg[i]) / seg_len[i] ** 2
            t = np.clip(t_raw, 0.0, 1.0)
            q = poly[i] + t * seg[i]
            d = np.linalg.norm(p - q)
            if d < best[2]:
                # beyond the ends by more than ~1.5 m -> not a real match
                over = 1.5 / seg_len[i]
                outside = (i == 0 and t_raw < -over) or \
                          (i == len(seg) - 1 and t_raw > 1.0 + over)
                best = (q, cum[i] + t * seg_len[i], d, outside)
        return best

    @staticmethod
    def _normals(poly):
        """Left-pointing unit normals of an ordered polyline."""
        if len(poly) < 2:
            return np.array([[0.0, 1.0]] * len(poly))
        t = np.gradient(poly, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        return np.column_stack([-t[:, 1], t[:, 0]])

    def _centre_candidates(self, left, right):
        cands = []
        widths = []
        # left cones -> right boundary, right cones -> left boundary
        for own, other, inward_sign in ((left, right, -1.0), (right, left, +1.0)):
            if len(own) == 0:
                continue
            normals = self._normals(own) if len(own) >= 2 else None
            for k, c in enumerate(own):
                used = False
                if len(other):
                    q, _, d, outside = self._project_on_polyline(c, other)
                    if (not outside) and self.min_track_width < d < self.max_track_width:
                        cands.append(0.5 * (c + q))
                        widths.append(d)
                        used = True
                if not used and normals is not None:
                    # opposite side not visible here: offset inwards
                    cands.append(c + inward_sign * normals[k] * 0.5 * self.track_width)
        # single cone each side, no chain normals available
        if len(left) == 1 and len(right) == 1 and not cands:
            cands.append(0.5 * (left[0] + right[0]))

        if widths:
            w = float(np.median(widths))
            self.track_width = 0.8 * self.track_width + 0.2 * w
        if not cands:
            return np.zeros((0, 2))
        cands = np.array(cands)

        # sort by arc length along the longer boundary
        ref = left if len(left) >= len(right) else right
        if len(ref) >= 2:
            s = np.array([self._project_on_polyline(c, ref)[1] for c in cands])
            # candidates projecting before the first cone: order by x
            cands = cands[np.argsort(s + 1e-3 * cands[:, 0])]
        else:
            cands = cands[np.argsort(np.linalg.norm(cands, axis=1))]

        # merge points closer than 0.8 m
        merged = [cands[0]]
        for c in cands[1:]:
            if np.linalg.norm(c - merged[-1]) < 0.8:
                merged[-1] = 0.5 * (merged[-1] + c)
            else:
                merged.append(c)
        merged = np.array(merged)
        # drop points behind the car and enforce monotone forward progress
        merged = merged[merged[:, 0] > 0.5]
        out = []
        for c in merged:
            if out:
                v = c - out[-1]
                if len(out) >= 2:
                    h = out[-1] - out[-2]
                    if np.dot(v, h) < 0:      # going backwards -> skip
                        continue
            out.append(c)
        return np.array(out) if out else np.zeros((0, 2))

    def _smooth(self, centers, smoothing=None):
        if smoothing is None:
            smoothing = self.smoothing
        if len(centers) == 0:
            return np.zeros((0, 2))
        pts = np.vstack([[0.0, 0.0], centers])
        # remove consecutive duplicates (splprep fails on them)
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        pts = np.vstack([pts[:1], pts[1:][d > 1e-3]])
        if len(pts) < 2:
            return np.zeros((0, 2))
        if len(pts) == 2:
            length = np.linalg.norm(pts[1] - pts[0])
            n = max(2, int(length / self.path_spacing) + 1)
            return np.linspace(pts[0], pts[1], n)
        k = min(3, len(pts) - 1)
        # weight the origin strongly so the path really starts at the car
        w = np.ones(len(pts))
        w[0] = 10.0
        try:
            tck, u = splprep([pts[:, 0], pts[:, 1]], w=w, k=k,
                             s=smoothing * len(pts))
        except Exception:
            return pts
        fine = np.linspace(0, 1, 400)
        x, y = splev(fine, tck)
        dense = np.column_stack([x, y])
        seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        if cum[-1] < 1e-6:
            return pts
        s_new = np.arange(0.0, cum[-1], self.path_spacing)
        path = np.column_stack([np.interp(s_new, cum, dense[:, 0]),
                                np.interp(s_new, cum, dense[:, 1])])
        path[0] = [0.0, 0.0]
        return path
