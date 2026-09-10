"""ROS-independent 2-D RRT* planner.

Obstacles are circular safety regions around detected cones. The planner has
proper sampling, nearest-node search, steering, collision checking, best-parent
selection, rewiring, goal connection, and path extraction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import math
import numpy as np


@dataclass
class Node:
    point: np.ndarray
    parent: Optional[int]
    cost: float


def segment_circle_collision(a: np.ndarray, b: np.ndarray, center: np.ndarray, radius: float) -> bool:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(a - center)) <= radius
    t = float(np.clip(np.dot(center - a, ab) / denom, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(closest - center)) <= radius


class RRTStar:
    def __init__(self, start, goal, obstacles=None, bounds=None, step_size=0.5,
                 neighbor_radius=1.5, max_iterations=1000, goal_tolerance=0.5,
                 rng_seed=7):
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.obstacles = [(np.asarray(c, dtype=float), float(r)) for c, r in (obstacles or [])]
        if bounds is None:
            lo = np.minimum(self.start, self.goal) - 5.0
            hi = np.maximum(self.start, self.goal) + 5.0
            bounds = (lo, hi)
        self.low = np.asarray(bounds[0], dtype=float)
        self.high = np.asarray(bounds[1], dtype=float)
        self.step_size = float(step_size)
        self.neighbor_radius = float(neighbor_radius)
        self.max_iterations = int(max_iterations)
        self.goal_tolerance = float(goal_tolerance)
        self.rng = np.random.default_rng(rng_seed)
        self.nodes: List[Node] = [Node(self.start.copy(), None, 0.0)]

    def collision_free(self, a, b) -> bool:
        return not any(segment_circle_collision(np.asarray(a), np.asarray(b), c, r)
                        for c, r in self.obstacles)

    def sample(self) -> np.ndarray:
        if self.rng.random() < 0.10:
            return self.goal.copy()
        return self.rng.uniform(self.low, self.high)

    def nearest(self, point) -> int:
        p = np.asarray(point)
        return int(np.argmin([np.linalg.norm(n.point - p) for n in self.nodes]))

    def steer(self, source, target) -> np.ndarray:
        source, target = np.asarray(source), np.asarray(target)
        delta = target - source
        dist = float(np.linalg.norm(delta))
        if dist <= self.step_size:
            return target.copy()
        return source + delta / dist * self.step_size

    def near(self, point) -> List[int]:
        return [i for i, n in enumerate(self.nodes)
                if np.linalg.norm(n.point - point) <= self.neighbor_radius]

    def _rewire(self, new_idx: int, neighbors: List[int]) -> None:
        new = self.nodes[new_idx]
        for i in neighbors:
            if i == new_idx or i == 0:
                continue
            candidate = new.cost + np.linalg.norm(self.nodes[i].point - new.point)
            if candidate + 1e-9 < self.nodes[i].cost and self.collision_free(new.point, self.nodes[i].point):
                self.nodes[i].parent = new_idx
                self.nodes[i].cost = candidate
                self._propagate_cost(i)

    def _propagate_cost(self, parent_idx: int) -> None:
        for i, n in enumerate(self.nodes):
            if n.parent == parent_idx:
                n.cost = self.nodes[parent_idx].cost + np.linalg.norm(n.point - self.nodes[parent_idx].point)
                self._propagate_cost(i)

    def _path(self, idx: int) -> List[List[float]]:
        result = []
        while idx is not None:
            result.append(self.nodes[idx].point.tolist())
            idx = self.nodes[idx].parent
        return result[::-1]

    def plan(self) -> List[List[float]]:
        best_goal_idx = None
        best_goal_cost = float("inf")
        for _ in range(self.max_iterations):
            sample = self.sample()
            nearest_idx = self.nearest(sample)
            new_point = self.steer(self.nodes[nearest_idx].point, sample)
            if not self.collision_free(self.nodes[nearest_idx].point, new_point):
                continue

            neighbors = self.near(new_point)
            parent = nearest_idx
            best_cost = self.nodes[parent].cost + np.linalg.norm(new_point - self.nodes[parent].point)
            for i in neighbors:
                cost = self.nodes[i].cost + np.linalg.norm(new_point - self.nodes[i].point)
                if cost < best_cost and self.collision_free(self.nodes[i].point, new_point):
                    parent, best_cost = i, cost
            self.nodes.append(Node(new_point, parent, best_cost))
            new_idx = len(self.nodes) - 1
            self._rewire(new_idx, neighbors)

            if np.linalg.norm(new_point - self.goal) <= self.goal_tolerance and self.collision_free(new_point, self.goal):
                goal_cost = self.nodes[new_idx].cost + np.linalg.norm(self.goal - new_point)
                if goal_cost < best_goal_cost:
                    self.nodes.append(Node(self.goal.copy(), new_idx, goal_cost))
                    best_goal_idx = len(self.nodes) - 1
                    best_goal_cost = goal_cost

        if best_goal_idx is None:
            # A valid fallback is the direct path only if it is collision-free.
            if self.collision_free(self.start, self.goal):
                return [self.start.tolist(), self.goal.tolist()]
            return []
        return self._path(best_goal_idx)
