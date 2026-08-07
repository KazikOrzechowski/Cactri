"""Compact export helpers for downstream Cactri analyses.

The default export deliberately avoids the retained MCMC arrays whose size
scales as ``n_draws * n_cells`` or ``n_draws * n_clones * n_snv``.  Instead it
materializes coherent posterior summaries and the posterior-medoid-aligned
state needed by downstream pruning/refinement code.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .omega import CactriOmega
from .tree import CactriTree
from .utils.validation import normalize_dirichlet_prior

ModelKind = Literal["tree", "omega"]


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return {str(k): _json_value(v) for k, v in asdict(value).items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _selected_indices(n_draws: int, burn_in: int | float, thin: int) -> np.ndarray:
    if n_draws <= 0:
        return np.empty(0, dtype=np.int64)
    if thin < 1:
        raise ValueError("thin must be at least 1.")
    if isinstance(burn_in, float):
        if not 0.0 <= burn_in < 1.0:
            raise ValueError("fractional burn_in must lie in [0,1).")
        start = int(math.floor(n_draws * burn_in))
    else:
        start = int(burn_in)
        if start < 0:
            raise ValueError("burn_in must be nonnegative.")
    idx = np.arange(start, n_draws, thin, dtype=np.int64)
    if idx.size == 0:
        raise ValueError("no posterior draws remain after burn-in and thinning.")
    return idx


def _trace_mean(trace: list[Any], burn_in: int | float, thin: int, fallback: Any) -> np.ndarray:
    if not trace:
        return np.asarray(fallback).copy()
    arr = np.asarray(trace)
    idx = _selected_indices(len(arr), burn_in, thin)
    return np.asarray(arr[idx].mean(axis=0))


def _clone_genotype_probability(model: CactriTree | CactriOmega, burn_in: int | float, thin: int) -> np.ndarray:
    if model.genotype_state_trace_:
        arr = np.asarray(model.genotype_state_trace_, dtype=float)
        idx = _selected_indices(len(arr), burn_in, thin)
        return arr[idx].mean(axis=0)
    return np.asarray(model._genotype_matrix(), dtype=float)


def _medoid_payload(
    model: CactriTree | CactriOmega,
    *,
    kind: Literal["hypercluster", "clone"],
    burn_in: int | float,
    thin: int,
) -> tuple[np.ndarray, int, np.ndarray]:
    trace = model.assignment_trace_ if kind == "hypercluster" else model.cell_clone_assignment_trace_
    if trace:
        result = model.posterior_partition_medoid(kind, burn_in=burn_in, thin=thin)
        return (
            np.asarray(result["assignment"], dtype=np.int64),
            int(result["trace_index"]),
            np.asarray(result["losses"], dtype=float),
        )
    assignment = (
        np.asarray(model.assignments_, dtype=np.int64)
        if kind == "hypercluster"
        else np.asarray(model.current_cell_clone_assignment(), dtype=np.int64)
    )
    return assignment.copy(), -1, np.empty(0, dtype=float)


def _medoid_aligned_state(
    model: CactriTree | CactriOmega,
    clone_trace_index: int,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if clone_trace_index >= 0 and clone_trace_index < len(model.genotype_state_trace_):
        out["clone_medoid_genotype_state"] = np.asarray(
            model.genotype_state_trace_[clone_trace_index]
        ).copy()
    else:
        out["clone_medoid_genotype_state"] = np.asarray(model._genotype_matrix()).copy()

    if clone_trace_index >= 0 and clone_trace_index < len(model.p_obs_by_mutation_trace_):
        out["clone_medoid_p_obs_by_mutation"] = np.asarray(
            model.p_obs_by_mutation_trace_[clone_trace_index], dtype=float
        ).copy()
    else:
        out["clone_medoid_p_obs_by_mutation"] = np.asarray(
            model.p_obs_by_mutation_, dtype=float
        ).copy()

    if clone_trace_index >= 0 and clone_trace_index < len(model.p_unobs_trace_):
        p0 = float(model.p_unobs_trace_[clone_trace_index])
    else:
        p0 = float(model.p_unobs_)
    out["clone_medoid_p_unobs"] = np.asarray(p0, dtype=float)

    if isinstance(model, CactriTree):
        if clone_trace_index >= 0 and clone_trace_index < len(model.mutation_tree_assignment_trace_):
            edge = np.asarray(model.mutation_tree_assignment_trace_[clone_trace_index], dtype=np.int64)
        else:
            edge = np.asarray(model.mutation_tree_assignment_, dtype=np.int64)
        out["clone_medoid_mutation_tree_assignment"] = edge.copy()
    return out


def _tree_edge_posterior(model: CactriTree, burn_in: int | float, thin: int) -> np.ndarray:
    if not model.mutation_tree_assignment_trace_:
        out = np.zeros((model.n_snv, model.n_tree_vertices), dtype=float)
        out[np.arange(model.n_snv), np.asarray(model.mutation_tree_assignment_, dtype=np.int64)] = 1.0
        return out
    draws = np.asarray(model.mutation_tree_assignment_trace_, dtype=np.int64)
    idx = _selected_indices(len(draws), burn_in, thin)
    draws = draws[idx]
    out = np.zeros((model.n_snv, model.n_tree_vertices), dtype=float)
    for vertex in range(model.n_tree_vertices):
        out[:, vertex] = np.mean(draws == vertex, axis=0)
    return out


def _scalar_diagnostics(model: CactriTree | CactriOmega) -> dict[str, np.ndarray]:
    names = (
        "log_likelihood_trace_",
        "bcr_log_likelihood_trace_",
        "mutation_log_likelihood_trace_",
        "crp_logprior_trace_",
        "clone_assignment_logprior_trace_",
        "genotype_logprior_trace_",
        "log_posterior_trace_",
        "n_hyperclusters_trace_",
        "alpha_trace_",
        "p_unobs_trace_",
        "p_obs_mean_trace_",
    )
    out: dict[str, np.ndarray] = {}
    for name in names:
        value = getattr(model, name, None)
        if value:
            out[name.removesuffix("_")] = np.asarray(value)
    if isinstance(model, CactriTree):
        if model.edge_error_rate_trace_:
            out["edge_error_rate_trace"] = np.asarray(model.edge_error_rate_trace_, dtype=float)
        if model.edge_error_component_trace_:
            out["edge_error_component_trace"] = np.asarray(model.edge_error_component_trace_, dtype=np.int64)
    else:
        if model.relax_rate_trace_:
            out["relax_rate_trace"] = np.asarray(model.relax_rate_trace_, dtype=float)
    return out


def _raw_trace_arrays(model: CactriTree | CactriOmega) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    names = (
        "assignment_trace_",
        "cell_clone_assignment_trace_",
        "genotype_state_trace_",
        "p_obs_by_mutation_trace_",
        "dominant_clone_trace_",
        "hypercluster_clone_proportions_trace_",
        "admixture_mass_trace_",
        "mixture_active_trace_",
        "effective_clone_count_trace_",
        "admixture_entropy_trace_",
        "dominant_fraction_trace_",
    )
    if isinstance(model, CactriTree):
        names += ("mutation_tree_assignment_trace_",)
    for name in names:
        value = getattr(model, name, None)
        if value:
            # Variable-hypercluster traces may be ragged; object storage is used
            # only in the explicit opt-in raw-trace file.
            try:
                arr = np.asarray(value)
            except ValueError:
                arr = np.asarray(value, dtype=object)
            out[name.removesuffix("_")] = arr
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_analysis_artifacts(
    model: CactriTree | CactriOmega,
    path: str | Path,
    *,
    model_kind: ModelKind,
    burn_in: int | float = 0,
    thin: int = 1,
    include_sequences: bool = True,
    include_scalar_traces: bool = True,
    include_coassignments: bool = False,
    include_raw_traces: bool = False,
    overwrite: bool = False,
) -> Path:
    """Save a compact, portable artifact set for downstream analysis.

    Parameters
    ----------
    model:
        A fitted or prefitted :class:`CactriTree` or :class:`CactriOmega`.
    path:
        Output directory.  Five compact files are written by default:
        ``manifest.json``, ``data.npz``, ``state.npz``, ``posterior.npz``, and
        ``diagnostics.npz``.
    model_kind:
        Explicitly choose ``"tree"`` or ``"omega"``.  The argument is checked
        against the supplied model to avoid silently exporting the wrong
        subclass-specific state.
    burn_in, thin:
        Applied while converting retained posterior traces into compact
        summaries. They do not alter the fitted model.
    include_sequences:
        Store the BCR sequence matrix. Mutation read matrices are always stored
        because genotype/refinement analyses require them.
    include_scalar_traces:
        Retain inexpensive one-dimensional convergence traces. Enabled by
        default because they are useful for diagnostics and typically tiny.
    include_coassignments:
        Also materialize the two dense cell-by-cell posterior co-assignment
        matrices. Disabled by default because each is O(N^2).
    include_raw_traces:
        Write ``raw_traces.npz`` with the memory-heavy retained arrays. Disabled
        by default; the normal artifact set is designed to make this unnecessary
        for most downstream genotype, pruning, and mixture analyses.
    overwrite:
        Permit writing into an existing non-empty directory.

    Notes
    -----
    The default posterior artifact contains the summaries that otherwise need
    the large cell/genotype traces: posterior cell-clone probabilities,
    coherent posterior cell-genotype probabilities, clone-level genotype
    probabilities, both sampled partition medoids, a genotype/observation state
    aligned to the clone-partition medoid, and posterior means of observation
    probabilities. Tree exports additionally contain mutation-edge posterior
    probabilities.
    """

    expected = CactriTree if model_kind == "tree" else CactriOmega if model_kind == "omega" else None
    if expected is None:
        raise ValueError("model_kind must be 'tree' or 'omega'.")
    if not isinstance(model, expected):
        raise TypeError(
            f"model_kind={model_kind!r} requires {expected.__name__}, "
            f"got {type(model).__name__}."
        )
    model._require_prefit()

    destination = Path(path)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory {destination} is not empty; pass overwrite=True to replace files."
        )
    destination.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for stale_name in (
            "manifest.json",
            "data.npz",
            "state.npz",
            "posterior.npz",
            "diagnostics.npz",
            "raw_traces.npz",
        ):
            stale = destination / stale_name
            if stale.exists():
                stale.unlink()

    data: dict[str, np.ndarray] = {
        "alt_counts": np.asarray(model.alt_counts_).copy(),
        "total_counts": np.asarray(model.total_counts_).copy(),
    }
    if include_sequences:
        data["sequences"] = np.asarray(model.sequences_).copy()

    state: dict[str, np.ndarray] = {
        "assignments": np.asarray(model.assignments_, dtype=np.int64).copy(),
        "bcr_profiles": np.asarray(model.bcr_profiles_, dtype=float).copy(),
        "hypercluster_to_clone": np.asarray(model.hypercluster_to_clone_, dtype=np.int64).copy(),
        "dominant_clones": np.asarray(model.dominant_clones_, dtype=np.int64).copy(),
        "hypercluster_clone_proportions": np.asarray(model.hypercluster_clone_proportions_, dtype=float).copy(),
        "admixture_mass": np.asarray(model.admixture_mass_, dtype=float).copy(),
        "residual_clone_proportions": np.asarray(model.residual_clone_proportions_, dtype=float).copy(),
        "mixture_active": np.asarray(model.mixture_active_, dtype=bool).copy(),
        "mixture_presence_rate": np.asarray(float(model.mixture_presence_rate_), dtype=float),
        "cell_clone_assignment": np.asarray(model.current_cell_clone_assignment(), dtype=np.int64).copy(),
        "genotype_matrix": np.asarray(model._genotype_matrix()).copy(),
        "p_obs_by_mutation": np.asarray(model.p_obs_by_mutation_, dtype=float).copy(),
        "p_unobs": np.asarray(float(model.p_unobs_), dtype=float),
        "alpha": np.asarray(float(model.alpha_), dtype=float),
        "clone_prior": np.asarray(model.clone_prior, dtype=float).copy(),
    }
    if isinstance(model, CactriTree):
        state.update(
            {
                "mutation_tree_assignment": np.asarray(model.mutation_tree_assignment_, dtype=np.int64).copy(),
                "mutation_profile": np.asarray(model.mutation_profile_, dtype=np.int8).copy(),
                "vertex_clone_presence": np.asarray(model._vertex_clone_presence, dtype=np.int8).copy(),
                "base_edge_prior": np.asarray(model._base_edge_prior, dtype=float).copy(),
                "current_edge_prior": np.asarray(model._current_edge_prior(), dtype=float).copy(),
            }
        )
        if model.observed_edge_assignment_ is not None:
            state["observed_edge_assignment"] = np.asarray(model.observed_edge_assignment_, dtype=np.int64).copy()
        if model.observed_edge_probabilities_ is not None:
            state["observed_edge_probabilities"] = np.asarray(model.observed_edge_probabilities_, dtype=float).copy()
        if model.edge_error_rate_ is not None:
            state["edge_error_rate"] = np.asarray(float(model.edge_error_rate_), dtype=float)
    else:
        state["genotype_prior"] = np.asarray(model._current_genotype_prior(), dtype=float).copy()
        if model.omega_prior_ is not None:
            state["omega_prior"] = np.asarray(model.omega_prior_, dtype=np.int8).copy()
        if model.relax_rate_ is not None:
            state["relax_rate"] = np.asarray(float(model.relax_rate_), dtype=float)

    clone_medoid, clone_medoid_idx, clone_losses = _medoid_payload(
        model, kind="clone", burn_in=burn_in, thin=thin
    )
    hyper_medoid, hyper_medoid_idx, hyper_losses = _medoid_payload(
        model, kind="hypercluster", burn_in=burn_in, thin=thin
    )

    has_clone_trace = bool(model.cell_clone_assignment_trace_)
    has_coherent_genotype_trace = bool(
        model.cell_clone_assignment_trace_
        and len(model.cell_clone_assignment_trace_) == len(model.genotype_state_trace_)
    )
    posterior: dict[str, np.ndarray] = {
        "cell_clone_probability": np.asarray(
            model.posterior_cell_clone_probabilities(
                burn_in=burn_in, thin=thin, use_trace=True if has_clone_trace else False
            ),
            dtype=float,
        ),
        "cell_genotype_probability": np.asarray(
            model.posterior_cell_genotype_probabilities(
                burn_in=burn_in,
                thin=thin,
                use_trace=True if has_coherent_genotype_trace else False,
            ),
            dtype=float,
        ),
        "clone_genotype_probability": _clone_genotype_probability(model, burn_in, thin),
        "clone_partition_medoid_assignment": clone_medoid,
        "clone_partition_medoid_trace_index": np.asarray(clone_medoid_idx, dtype=np.int64),
        "clone_partition_medoid_losses": clone_losses,
        "hypercluster_partition_medoid_assignment": hyper_medoid,
        "hypercluster_partition_medoid_trace_index": np.asarray(hyper_medoid_idx, dtype=np.int64),
        "hypercluster_partition_medoid_losses": hyper_losses,
        "posterior_mean_p_obs_by_mutation": _trace_mean(
            model.p_obs_by_mutation_trace_, burn_in, thin, model.p_obs_by_mutation_
        ).astype(float),
        "posterior_mean_p_unobs": np.asarray(
            float(_trace_mean(model.p_unobs_trace_, burn_in, thin, model.p_unobs_)),
            dtype=float,
        ),
    }
    posterior.update(_medoid_aligned_state(model, clone_medoid_idx))

    # Medoid-aligned BCR and clone summaries avoid retaining the full assignment
    # and variable-dimensional hypercluster-parameter traces.
    hyper_labels, hyper_reindexed = np.unique(hyper_medoid, return_inverse=True)
    n_medoid_h = len(hyper_labels)
    prior = normalize_dirichlet_prior(model.dirichlet_prior, model.L)
    counts = np.zeros((n_medoid_h, model.L, 4), dtype=float)
    positions = np.arange(model.L, dtype=np.int64)
    np.add.at(
        counts,
        (hyper_reindexed[:, None], positions[None, :], np.asarray(model.sequences_, dtype=np.int64)),
        1.0,
    )
    bcr_mean = counts + prior[None, :, :]
    bcr_mean /= bcr_mean.sum(axis=-1, keepdims=True)
    posterior["hypercluster_medoid_bcr_profile_probability"] = bcr_mean
    medoid_clone_prob = np.zeros((n_medoid_h, model.n_clones), dtype=float)
    for h in range(n_medoid_h):
        medoid_clone_prob[h] = posterior["cell_clone_probability"][hyper_reindexed == h].mean(axis=0)
    posterior["hypercluster_medoid_clone_probability"] = medoid_clone_prob

    if model.clone_mixture.enabled:
        posterior.update(
            {
                "posterior_hypercluster_clone_proportions": np.asarray(
                    model.posterior_hypercluster_clone_proportions(burn_in=burn_in, thin=thin),
                    dtype=float,
                ),
                "posterior_dominant_clone_probabilities": np.asarray(
                    model.posterior_dominant_clone_probabilities(burn_in=burn_in, thin=thin),
                    dtype=float,
                ),
                "posterior_admixture_probabilities": np.asarray(
                    model.posterior_admixture_probabilities(burn_in=burn_in, thin=thin),
                    dtype=float,
                ),
                "posterior_effective_clone_counts": np.asarray(
                    model.posterior_effective_clone_counts(burn_in=burn_in, thin=thin),
                    dtype=float,
                ),
            }
        )

    if isinstance(model, CactriTree):
        posterior["mutation_tree_assignment_probability"] = _tree_edge_posterior(
            model, burn_in, thin
        )
    else:
        posterior["posterior_mean_relax_rate"] = np.asarray(
            float(_trace_mean(model.relax_rate_trace_, burn_in, thin, model.relax_rate_ if model.relax_rate_ is not None else np.nan)),
            dtype=float,
        )

    if include_coassignments:
        if model.assignment_trace_:
            posterior["hypercluster_coassignment_probability"] = np.asarray(
                model.posterior_hypercluster_coassignment(burn_in=burn_in, thin=thin), dtype=float
            )
        if model.cell_clone_assignment_trace_:
            posterior["cell_clone_coassignment_probability"] = np.asarray(
                model.posterior_cell_clone_coassignment(burn_in=burn_in, thin=thin), dtype=float
            )

    diagnostics = _scalar_diagnostics(model) if include_scalar_traces else {}

    files: list[Path] = []
    for filename, payload in (
        ("data.npz", data),
        ("state.npz", state),
        ("posterior.npz", posterior),
    ):
        target = destination / filename
        np.savez_compressed(target, **payload)
        files.append(target)
    diag_target = destination / "diagnostics.npz"
    np.savez_compressed(diag_target, **diagnostics)
    files.append(diag_target)

    if include_raw_traces:
        raw_target = destination / "raw_traces.npz"
        np.savez_compressed(raw_target, **_raw_trace_arrays(model))
        files.append(raw_target)

    common_config = {
        "n_clones": model.n_clones,
        "alpha": model.alpha,
        "alpha_prior": model.alpha_prior,
        "sample_alpha": model.sample_alpha,
        "dirichlet_prior": model.dirichlet_prior,
        "p_obs_beta_prior": model.p_obs_beta_prior,
        "p_unobs_beta_prior": model.p_unobs_beta_prior,
        "p_obs_init": model.p_obs_init,
        "p_unobs_init": model.p_unobs_init,
        "clone_prior": model.clone_prior,
        "clone_mixture": model.clone_mixture,
        "assignment_likelihood": model.assignment_likelihood,
        "new_cluster_likelihood": model.new_cluster_likelihood,
        "eps": model.eps,
    }
    if isinstance(model, CactriTree):
        subclass_config = {
            "n_levels": model.n_levels,
            "tree_prior": model.tree_prior,
            "learn_edge_error_rate": model.learn_edge_error_rate,
            "edge_error_rate_prior": model.edge_error_rate_prior,
            "distance_power": model.distance_power,
            "fixed_edge_confidence": model.fixed_edge_confidence,
        }
    else:
        subclass_config = {
            "genotype_prior": model.genotype_prior,
            "relax_rate_prior": model.relax_rate_prior,
            "sample_relax_rate": model.sample_relax_rate,
            "fix_reference_clone": model.fix_reference_clone,
        }

    trace_lengths = {
        name.removesuffix("_"): len(value)
        for name, value in vars(model).items()
        if name.endswith("_trace_") and isinstance(value, list)
    }
    manifest = {
        "format": "cactri-analysis-artifacts",
        "format_version": 1,
        "model_kind": model_kind,
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "dimensions": {
            "n_cells": model.n_cells,
            "n_sequence_positions": model.L,
            "n_snv": model.n_snv,
            "n_clones": model.n_clones,
            "n_hyperclusters_current": model.n_hyperclusters,
            **({
                "n_levels": model.n_levels,
                "n_leaf_clones": model.n_leaf_clones,
                "n_tree_vertices": model.n_tree_vertices,
            } if isinstance(model, CactriTree) else {}),
        },
        "summary_selection": {"burn_in": burn_in, "thin": thin},
        "posterior_sources": {
            "cell_clone_probability": "trace_mean" if has_clone_trace else "current_state",
            "cell_genotype_probability": "coherent_trace_mean" if has_coherent_genotype_trace else "current_state",
            "clone_partition_medoid": "trace_medoid" if model.cell_clone_assignment_trace_ else "current_state",
            "hypercluster_partition_medoid": "trace_medoid" if model.assignment_trace_ else "current_state",
        },
        "trace_lengths_in_model": trace_lengths,
        "raw_traces_exported": bool(include_raw_traces),
        "dense_coassignments_exported": bool(include_coassignments),
        "sequences_exported": bool(include_sequences),
        "iterations_completed": int(model._iterations_completed_),
        "accelerator": model.accelerator_name,
        "deterministic": model.deterministic,
        "config": {"common": common_config, model_kind: subclass_config},
        "split_merge_diagnostics": model.split_merge_diagnostics(),
        "files": {},
    }
    for file in files:
        manifest["files"][file.name] = {
            "sha256": _sha256(file),
            "size_bytes": file.stat().st_size,
        }

    manifest_target = destination / "manifest.json"
    with manifest_target.open("w", encoding="utf-8") as handle:
        json.dump(_json_value(manifest), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return destination
