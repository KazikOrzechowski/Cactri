from __future__ import annotations

import numpy as np


def tree_vertex_levels(n_levels: int) -> np.ndarray:
    return np.concatenate(
        [np.full(2**level, level, dtype=np.int64) for level in range(n_levels + 1)]
    )


def vertex_clone_presence(n_levels: int, *, include_reference: bool = True) -> np.ndarray:
    n_leaf = 2**n_levels
    n_vertices = 2 ** (n_levels + 1) - 1
    n_clones = n_leaf + int(include_reference)
    out = np.zeros((n_vertices, n_clones), dtype=np.int8)
    cursor = 0
    for level in range(n_levels + 1):
        leaves_per_node = 2 ** (n_levels - level)
        for offset in range(2**level):
            start = offset * leaves_per_node
            stop = start + leaves_per_node
            shift = 1 if include_reference else 0
            out[cursor, shift + start : shift + stop] = 1
            cursor += 1
    return out


def tree_edge_distance_matrix(n_vertices: int) -> np.ndarray:
    """Return undirected graph distances for heap-indexed full-binary vertices."""
    if n_vertices < 1 or ((n_vertices + 1) & n_vertices) != 0:
        raise ValueError("n_vertices must equal 2^(levels+1)-1.")
    ancestors: list[list[int]] = []
    for node in range(n_vertices):
        path = [node]
        while node:
            node = (node - 1) // 2
            path.append(node)
        ancestors.append(path)
    dist = np.zeros((n_vertices, n_vertices), dtype=np.int64)
    for i in range(n_vertices):
        ai = {node: depth for depth, node in enumerate(ancestors[i])}
        for j in range(n_vertices):
            for depth_j, node in enumerate(ancestors[j]):
                if node in ai:
                    dist[i, j] = ai[node] + depth_j
                    break
    return dist


def inverse_distance_transition(n_vertices: int, power: float = 1.0) -> np.ndarray:
    if power <= 0:
        raise ValueError("power must be positive.")
    distance = tree_edge_distance_matrix(n_vertices).astype(float)
    weights = np.zeros_like(distance)
    mask = distance > 0
    weights[mask] = distance[mask] ** (-power)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def edge_assignments_to_profiles(
    assignments: np.ndarray,
    *,
    n_levels: int,
    include_reference: bool = True,
) -> np.ndarray:
    assignment = np.asarray(assignments, dtype=np.int64)
    presence = vertex_clone_presence(n_levels, include_reference=include_reference)
    if np.any((assignment < 0) | (assignment >= presence.shape[0])):
        raise ValueError("mutation edge assignment contains an invalid vertex.")
    return presence[assignment].T.astype(np.int8, copy=False)


def profiles_to_edge_assignments(
    genotype_matrix: np.ndarray,
    *,
    n_levels: int | None = None,
    reference_clone: bool = True,
    strict: bool = True,
) -> np.ndarray:
    g = np.asarray(genotype_matrix).astype(bool)
    if g.ndim != 2:
        raise ValueError("genotype_matrix must be 2D.")
    leaves = g[1:] if reference_clone else g
    if reference_clone and np.any(g[0]):
        raise ValueError("reference clone must be all zero.")
    n_leaf = leaves.shape[0]
    if n_levels is None:
        n_levels = int(round(np.log2(n_leaf)))
    if 2**n_levels != n_leaf:
        raise ValueError("number of leaf clones must be 2**n_levels.")
    patterns = vertex_clone_presence(n_levels, include_reference=False).astype(bool)
    lookup = {tuple(row.tolist()): i for i, row in enumerate(patterns)}
    out = np.full(leaves.shape[1], -1, dtype=np.int64)
    for j in range(leaves.shape[1]):
        key = tuple(leaves[:, j].tolist())
        if key in lookup:
            out[j] = lookup[key]
        elif strict:
            raise ValueError(f"mutation {j} is not compatible with the full binary tree.")
    return out
