"""Synthetic sequence task for learning ordered second-order graph relations.

Each example is a self-avoiding nearest-neighbor path embedded in a 2D grid.
The model-facing graph is given only by predecessor indices; ``path`` is
retained in the bank for provenance and checks.  Coordinates are stored as
``(row, column)`` pairs and flattened node indices use row-major order.
"""
from __future__ import annotations

from collections.abc import Sequence
import operator

import numpy as np


FEATURES = 8
TEACHER_SEED = 170001


def teacher(seed: int = TEACHER_SEED) -> dict[str, np.ndarray | float]:
    """Return one seeded, normalized orthogonal pair and its fixed scale."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(FEATURES)
    a /= np.linalg.norm(a)
    b = rng.standard_normal(FEATURES)
    b -= np.dot(a, b) * a
    b /= np.linalg.norm(b)
    return {"a": a.astype(np.float32), "b": b.astype(np.float32), "gamma": 1.0}


def direct_second_order(tokens: np.ndarray) -> tuple[np.float64, np.float64]:
    """Compute first and ordered second sums from ``[alpha_i, beta_i]`` tokens.

    Returns ``(sum_i alpha_i, sum_{i<j} alpha_i * beta_j)``.  The explicit
    pair accumulation is useful as a small reference for the streaming target
    calculation used by :func:`make_bank`.
    """
    values = np.asarray(tokens, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("tokens must have shape [length, 2]")
    if not np.all(np.isfinite(values)):
        raise ValueError("tokens must be finite")

    alpha = values[:, 0]
    beta = values[:, 1]
    sum1 = np.float64(0.0)
    sum2 = np.float64(0.0)
    for i in range(len(values)):
        sum1 = np.float64(sum1 + alpha[i])
        for j in range(i + 1, len(values)):
            sum2 = np.float64(sum2 + alpha[i] * beta[j])
    return sum1, sum2


def _length_options(lengths: int | range | Sequence[int] | np.ndarray) -> np.ndarray:
    if isinstance(lengths, (int, np.integer)):
        choices = np.asarray([operator.index(lengths)], dtype=np.int64)
    elif isinstance(lengths, range):
        choices = np.fromiter(lengths, dtype=np.int64)
    elif isinstance(lengths, tuple) and len(lengths) == 2:
        lo, hi = (operator.index(value) for value in lengths)
        if hi < lo:
            raise ValueError("length bounds must satisfy low <= high")
        choices = np.arange(lo, hi + 1, dtype=np.int64)
    else:
        try:
            choices = np.asarray(
                [operator.index(value) for value in lengths], dtype=np.int64
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("lengths must be an integer or a nonempty sequence") from exc
    if choices.ndim != 1 or choices.size == 0:
        raise ValueError("lengths must contain at least one integer")
    if np.any(choices < 1):
        raise ValueError("path lengths must be positive")
    return choices


def _neighbors(node: int, side: int) -> tuple[int, ...]:
    row, col = divmod(node, side)
    out = []
    if row > 0:
        out.append(node - side)
    if row + 1 < side:
        out.append(node + side)
    if col > 0:
        out.append(node - 1)
    if col + 1 < side:
        out.append(node + 1)
    return tuple(out)


def _ordered_unvisited(
    node: int,
    seen: np.ndarray,
    side: int,
    rng: np.random.Generator,
) -> list[int]:
    """Randomize choices, preferring moves with fewer onward options."""
    candidates = [next_node for next_node in _neighbors(node, side) if not seen[next_node]]
    # Stable sorting after random tie keys keeps paths varied while helping
    # backtracking avoid consuming a narrow corridor too early.
    tie_keys = rng.random(len(candidates))
    candidates_with_score = []
    for next_node, tie_key in zip(candidates, tie_keys):
        onward = sum(
            not seen[neighbor] and neighbor != node
            for neighbor in _neighbors(next_node, side)
        )
        candidates_with_score.append((onward, float(tie_key), next_node))
    candidates_with_score.sort()
    return [item[2] for item in candidates_with_score]


def _self_avoiding_path(
    side: int,
    length: int,
    rng: np.random.Generator,
) -> list[int]:
    """Find an exact-length path using bounded randomized DFS with restarts."""
    cells = side * side
    # A good branch usually reaches 96 cells quickly on a 16x16 grid.  The
    # explicit cap prevents unlucky backtracking from making bank creation
    # unbounded; independent randomized restarts keep the search robust.
    max_edge_attempts = max(20_000, 250 * length)
    restarts = 12
    for _ in range(restarts):
        start = int(rng.integers(cells))
        path = [start]
        seen = np.zeros(cells, dtype=np.bool_)
        seen[start] = True
        # Each frame stores its node, its randomized candidates, and the next
        # candidate index.  A candidate is consumed at most once per frame.
        frames: list[tuple[int, list[int], int]] = [
            (start, _ordered_unvisited(start, seen, side, rng), 0)
        ]
        attempts = 0
        while path and attempts < max_edge_attempts:
            if len(path) == length:
                return path
            node, candidates, next_index = frames[-1]
            if next_index >= len(candidates):
                frames.pop()
                removed = path.pop()
                seen[removed] = False
                continue

            next_node = candidates[next_index]
            frames[-1] = (node, candidates, next_index + 1)
            attempts += 1
            # A frame's candidates were legal when it was created.  Nodes
            # below it are removed during backtracking, so this remains the
            # same legal choice set for each parent branch.
            if seen[next_node]:
                continue
            seen[next_node] = True
            path.append(next_node)
            frames.append(
                (next_node, _ordered_unvisited(next_node, seen, side, rng), 0)
            )
        if len(path) == length:
            return path
    raise RuntimeError(
        f"could not generate a simple path of length {length} on a {side}x{side} grid"
    )


def _teacher_parameters(
    specification: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.float64]:
    try:
        a = np.asarray(specification["a"], dtype=np.float64)
        b = np.asarray(specification["b"], dtype=np.float64)
        gamma = np.float64(specification["gamma"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("teacher must provide finite a, b, and gamma values") from exc
    if a.shape != (FEATURES,) or b.shape != (FEATURES,):
        raise ValueError(f"teacher vectors a and b must each have shape ({FEATURES},)")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or not np.isfinite(gamma):
        raise ValueError("teacher must provide finite a, b, and gamma values")
    if not np.isclose(np.linalg.norm(a), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("teacher vector a must be normalized")
    if not np.isclose(np.linalg.norm(b), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("teacher vector b must be normalized")
    if not np.isclose(np.dot(a, b), 0.0, atol=1e-5, rtol=0.0):
        raise ValueError("teacher vectors a and b must be orthogonal")
    return a, b, gamma


def make_bank(
    count: int,
    side: int,
    lengths: int | range | Sequence[int] | np.ndarray,
    seed: int,
    teacher: dict[str, object],
) -> dict[str, np.ndarray]:
    """Create a deterministic bank of path-ordered second-order examples.

    ``lengths`` may be one fixed integer, an inclusive ``(low, high)`` tuple,
    a ``range``, or an explicit sequence of allowed lengths.  One allowed
    length is sampled uniformly for each example.  Each graph exposes only
    the immediate predecessor index; no grid-adjacency structure is returned.
    """
    try:
        count = operator.index(count)
        side = operator.index(side)
    except TypeError as exc:
        raise ValueError("count and side must be integers") from exc
    if count < 1:
        raise ValueError("count must be positive")
    if side < 2 or side > np.iinfo(np.int16).max + 1:
        raise ValueError("side must be between 2 and 32768 for int16 path coordinates")

    choices = _length_options(lengths)
    if np.any(choices > side * side):
        raise ValueError("every path length must be at most side squared")
    a, b, gamma = _teacher_parameters(teacher)
    rng = np.random.default_rng(seed)
    selected_lengths = rng.choice(choices, size=count, replace=True).astype(np.int64)

    nodes = side * side
    max_length = int(selected_lengths.max())
    x = np.zeros((count, nodes, FEATURES), dtype=np.float32)
    pred = np.full((count, nodes), -1, dtype=np.int64)
    mask = np.zeros((count, nodes), dtype=np.bool_)
    endpoint = np.empty(count, dtype=np.int64)
    path_array = np.full((count, max_length, 2), -1, dtype=np.int16)
    y = np.empty(count, dtype=np.float32)

    for batch_index, length_value in enumerate(selected_lengths):
        length = int(length_value)
        node_path = _self_avoiding_path(side, length, rng)
        coords = [(node // side, node % side) for node in node_path]
        path_array[batch_index, :length] = np.asarray(coords, dtype=np.int16)
        active = np.asarray(node_path, dtype=np.int64)
        mask[batch_index, active] = True
        endpoint[batch_index] = active[-1]

        projected_a_sum = np.float64(0.0)
        ordered_pair_sum = np.float64(0.0)
        for position, node in enumerate(node_path):
            if position:
                pred[batch_index, node] = node_path[position - 1]
            token = rng.standard_normal(FEATURES).astype(np.float32)
            x[batch_index, node] = token
            token64 = token.astype(np.float64)
            alpha = np.float64(np.dot(a, token64))
            beta = np.float64(np.dot(b, token64))
            # Stream the ordered second-order sum in float64, then cast only
            # the final target to float32.
            ordered_pair_sum = np.float64(ordered_pair_sum + projected_a_sum * beta)
            projected_a_sum = np.float64(projected_a_sum + alpha)
        scaled = np.float64(gamma * ordered_pair_sum / np.float64(length))
        y[batch_index] = np.float32(np.tanh(scaled))

    return {
        "x": x,
        "pred": pred,
        "mask": mask,
        "endpoint": endpoint,
        "lengths": selected_lengths,
        "y": y,
        "path": path_array,
    }


def bank_to_torch(bank: dict[str, np.ndarray], device: object) -> dict[str, object]:
    """Convert only model-facing bank fields to tensors on ``device``."""
    import torch

    keys = ("x", "pred", "mask", "endpoint", "lengths", "y")
    missing = [key for key in keys if key not in bank]
    if missing:
        raise KeyError(f"bank is missing required fields: {', '.join(missing)}")
    return {key: torch.as_tensor(bank[key], device=device) for key in keys}
