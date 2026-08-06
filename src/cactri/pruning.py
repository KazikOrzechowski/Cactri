"""Tree-pruning, split-back, and post-pruning refinement utilities.

This module is the canonical home of Cactri's post-fit tree-refinement API:
:class:`GreedyTreePruner`, :class:`ReadImpuritySplitBack`, and
:class:`PrunedTreeRefiner`, together with their immutable result records.

The legacy modules ``cactri.greedy_pruning``,
``cactri.read_impurity_splitback``, and ``cactri.pruned_refinement`` remain as
compatibility shims and re-export the same class objects.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal

import numpy as np

Interval = tuple[int, int]
Cut = tuple[Interval, ...]
StateMode = Literal["final", "clone_medoid"]
PriorMode = Literal["global", "hypercluster"]
@dataclass(frozen=True)
class GreedyCollapseStep:
    left: Interval
    right: Interval
    merged: Interval
    bic_before: float
    bic_after: float
    mutation_log_likelihood_before: float
    mutation_log_likelihood_after: float
    parameter_saving: int

@dataclass(frozen=True)
class GreedyTreePruningResult:
    intervals: Cut
    bic: float
    mutation_log_likelihood: float
    n_parameters: int
    n_observations: int
    history: tuple[GreedyCollapseStep,...]
    state_mode: StateMode

class GreedyTreePruner:
    """Select a dyadic cut using the benchmarked split-Hamming BIC.

    BCR and clone-mixture likelihoods are deliberately excluded.  The expanded
    benchmark found that adding collapsed mixture evidence reduced structural
    recovery; mixtures should instead be conditionally refit after cut selection.
    """
    def __init__(self,*,state_mode:StateMode='final',bic_tolerance:float=0.0,min_occupancy:float=1.0,eps:float=1e-12):
        if state_mode not in {'final','clone_medoid'}:raise ValueError('invalid state_mode')
        if eps<=0:raise ValueError('eps must be positive')
        if min_occupancy < 0: raise ValueError('min_occupancy must be nonnegative')
        self.state_mode=state_mode;self.bic_tolerance=float(bic_tolerance);self.min_occupancy=float(min_occupancy);self.eps=float(eps)

    @staticmethod
    def _eligible(cut:Cut):
        for i in range(len(cut)-1):
            a,b=cut[i],cut[i+1];la=a[1]-a[0];lb=b[1]-b[0]
            if a[1]==b[0] and la==lb and a[0]%(2*la)==0:yield i

    def select(self,model)->GreedyTreePruningResult:
        if not hasattr(model,'n_leaf_clones'):raise TypeError('CactriTree-like model required')
        n=int(model.n_leaf_clones);m=int(model.n_snv)
        if self.state_mode=='clone_medoid':
            med=model.posterior_clone_partition_medoid();idx=int(med['trace_index']);y=np.asarray(model.cell_clone_assignment_trace_[idx],int);g=np.asarray(model.genotype_state_trace_[idx],np.int8)
            t1=np.asarray(model.p_obs_by_mutation_trace_[idx] if model.p_obs_by_mutation_trace_ else model.p_obs_by_mutation_,float);t0=float(model.p_unobs_trace_[idx] if model.p_unobs_trace_ else model.p_unobs_)
        else:
            y=np.asarray(model.cell_clone_assignments_,int);g=np.asarray(model.mutation_profile_,np.int8);t1=np.asarray(model.p_obs_by_mutation_,float);t0=float(model.p_unobs_)
        alt=np.nan_to_num(np.asarray(model.alt_counts_,float),nan=0.0);total=np.nan_to_num(np.asarray(model.total_counts_,float),nan=0.0);ref=total-alt
        p1=np.clip(t1,self.eps,1-self.eps);p0=float(np.clip(t0,self.eps,1-self.eps));lp=np.log(p1);l1p=np.log1p(-p1);la=math.log(p0);l1a=math.log1p(-p0)
        A=np.zeros((n+1,m));R=np.zeros_like(A);occ=np.bincount(y,minlength=n+1).astype(float)
        for k in range(n+1):
            ix=y==k
            if np.any(ix):A[k]=alt[ix].sum(0);R[k]=ref[ix].sum(0)
        profile_cache={};split_cache={}
        def profile(node):
            if node not in profile_cache:
                a,b=node;profile_cache[node]=np.any(g[1+a:1+b].astype(bool),axis=0)
            return profile_cache[node]
        def subtree_diff(node):
            if node in split_cache:return split_cache[node]
            a,b=node
            if b-a<=1:v=0
            else:
                mid=(a+b)//2;l=(a,mid);r=(mid,b);v=int(np.count_nonzero(profile(l)!=profile(r)))+subtree_diff(l)+subtree_diff(r)
            split_cache[node]=v;return v
        root=(0,n);full_params=int(np.count_nonzero(profile(root))+subtree_diff(root));nobs=int(model.n_cells*model.n_snv)
        ref_ll=float(np.sum(A[0]*la+R[0]*l1a))
        def score(cut):
            ll=ref_ll
            for a,b in cut:
                aa=A[1+a:1+b].sum(0);rr=R[1+a:1+b].sum(0);pr=profile((a,b));ll+=float(np.sum(np.where(pr,aa*lp+rr*l1p,aa*la+rr*l1a)))
            k=full_params-sum(subtree_diff(x) for x in cut);bic=2*ll-k*math.log(max(nobs,1));return bic,ll,int(k)
        cut=tuple((i,i+1) for i in range(n));bic,ll,k=score(cut);hist=[]
        while True:
            candidates=[]
            for i in self._eligible(cut):
                c=cut[:i]+((cut[i][0],cut[i+1][1]),)+cut[i+2:]
                if min(float(occ[1+a:1+b].sum()) for a,b in c) < self.min_occupancy: continue
                cb,cl,ck=score(c);candidates.append((cb,cl,ck,i,c))
            if not candidates:break
            cb,cl,ck,i,c=max(candidates,key=lambda x:x[0])
            if cb<=bic+self.bic_tolerance:break
            left,right=cut[i],cut[i+1];merged=(left[0],right[1]);saving=int(np.count_nonzero(profile(left)!=profile(right)))
            hist.append(GreedyCollapseStep(left,right,merged,bic,cb,ll,cl,saving));cut,bic,ll,k=c,cb,cl,ck
        return GreedyTreePruningResult(cut,float(bic),float(ll),int(k),nobs,tuple(hist),self.state_mode)

@dataclass(frozen=True)
class SplitBackDecision:
    interval: Interval
    score: float
    read_gain: float
    mutation_difference_count: int
    effective_observations: float
    posterior_mass: float
    accepted: bool

@dataclass(frozen=True)
class ReadImpuritySplitBackResult:
    initial_intervals: Cut
    intervals: Cut
    decisions: tuple[SplitBackDecision, ...]
    penalty_scale: float
    prior_mode: PriorMode
    min_posterior_mass: float

class ReadImpuritySplitBack:
    """Recursively split collapsed dyadic blocks when reads support heterogeneity.

    The benchmark-supported experimental defaults are ``penalty_scale=0.5``
    and ``prior_mode='global'``.  The score is

    ``2 * read_gain - penalty_scale * d * log(n_eff)``,

    where ``d`` is the Hamming distance between the two child union genotypes.
    """
    def __init__(self, *, penalty_scale: float = 0.5,
                 prior_mode: PriorMode = "global",
                 min_posterior_mass: float = 0.0,
                 smoothing: float = 0.5,
                 eps: float = 1e-12) -> None:
        if penalty_scale < 0: raise ValueError("penalty_scale must be nonnegative")
        if prior_mode not in {"global", "hypercluster"}: raise ValueError("invalid prior_mode")
        if not 0 <= min_posterior_mass <= 1: raise ValueError("min_posterior_mass must lie in [0,1]")
        if smoothing <= 0 or eps <= 0: raise ValueError("smoothing and eps must be positive")
        self.penalty_scale=float(penalty_scale);self.prior_mode=prior_mode
        self.min_posterior_mass=float(min_posterior_mass);self.smoothing=float(smoothing);self.eps=float(eps)

    @staticmethod
    def _logsumexp2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        m=np.maximum(a,b);return m+np.log(np.exp(a-m)+np.exp(b-m))

    @staticmethod
    def _validate_cut(intervals: Iterable[Interval], n_leaf: int) -> Cut:
        cut=tuple((int(a),int(b)) for a,b in intervals);cursor=0
        for a,b in cut:
            length=b-a
            if a!=cursor or length<=0 or b>n_leaf or (length & (length-1)) or a%length:
                raise ValueError("intervals must be an ordered dyadic partition of all leaves")
            cursor=b
        if cursor!=n_leaf: raise ValueError("intervals must cover all leaves")
        return cut

    def select(self, model, intervals: Iterable[Interval], *, clone_trace_index: int | None = None, hypercluster_assignment: np.ndarray | None = None) -> ReadImpuritySplitBackResult:
        if not getattr(model,"cell_clone_assignment_trace_",None):
            raise RuntimeError("cell-clone posterior tracking is required")
        if not getattr(model,"genotype_state_trace_",None):
            raise RuntimeError("genotype-state posterior tracking is required")
        n_leaf=int(model.n_leaf_clones);cut=self._validate_cut(intervals,n_leaf)
        Y=np.asarray(model.cell_clone_assignment_trace_,dtype=np.int64)
        leaf_prob=np.stack([np.mean(Y==k,axis=0) for k in range(n_leaf+1)],axis=1)
        idx=(int(clone_trace_index) if clone_trace_index is not None else int(model.posterior_clone_partition_medoid()["trace_index"]))
        if idx < 0 or idx >= len(model.genotype_state_trace_): raise IndexError("clone_trace_index is out of range")
        genotype=np.asarray(model.genotype_state_trace_[idx],dtype=np.int8)
        theta1=np.asarray(model.p_obs_by_mutation_trace_[idx] if model.p_obs_by_mutation_trace_ else model.p_obs_by_mutation_,dtype=float)
        theta0=float(model.p_unobs_trace_[idx] if model.p_unobs_trace_ else model.p_unobs_)
        alt=np.nan_to_num(np.asarray(model.alt_counts_,dtype=float),nan=0.0)
        total=np.nan_to_num(np.asarray(model.total_counts_,dtype=float),nan=0.0);ref=total-alt
        z=np.asarray(hypercluster_assignment if hypercluster_assignment is not None else model.posterior_hypercluster_partition_medoid()["assignment"],dtype=np.int64)
        _,z=np.unique(z,return_inverse=True);n_h=int(z.max())+1
        decisions=[]

        def profile(node: Interval) -> np.ndarray:
            a,b=node;return np.any(genotype[1+a:1+b].astype(bool),axis=0)

        def score(node: Interval) -> SplitBackDecision:
            a,b=node;m=(a+b)//2;left=(a,m);right=(m,b)
            lp=profile(left);rp=profile(right);union=lp|rp;different=lp!=rp;d=int(np.count_nonzero(different))
            w=leaf_prob[:,1+a:1+b].sum(axis=1);mass=float(np.mean(w))
            if d==0 or mass<self.min_posterior_mass:
                return SplitBackDecision(node,float("-inf"),0.0,d,0.0,mass,False)
            aa=alt[:,different];rr=ref[:,different];p1=np.clip(theta1[different],self.eps,1-self.eps);p0=float(np.clip(theta0,self.eps,1-self.eps))
            def ll(p):
                return np.sum(np.where(p[different][None,:],aa*np.log(p1)[None,:]+rr*np.log1p(-p1)[None,:],aa*math.log(p0)+rr*math.log1p(-p0)),axis=1)
            ll_l,ll_r,ll_u=ll(lp),ll(rp),ll(union)
            wl=leaf_prob[:,1+a:1+m].sum(axis=1);wr=leaf_prob[:,1+m:1+b].sum(axis=1)
            if self.prior_mode=="global":
                q=(wl.sum()+self.smoothing)/(wl.sum()+wr.sum()+2*self.smoothing);q=np.full(len(w),q)
            else:
                q=np.empty(len(w))
                for h in range(n_h):
                    mask=z==h;sl=wl[mask].sum();sr=wr[mask].sum();q[mask]=(sl+self.smoothing)/(sl+sr+2*self.smoothing)
            q=np.clip(q,self.eps,1-self.eps)
            split=self._logsumexp2(np.log(q)+ll_l,np.log1p(-q)+ll_r)
            gain=float(np.dot(w,split-ll_u));n_eff=float(np.sum(w[:,None]*(total[:,different]>0)))
            stat=float(2*gain-self.penalty_scale*d*math.log(max(n_eff,2.0)))
            return SplitBackDecision(node,stat,gain,d,n_eff,mass,stat>0)

        def recurse(node: Interval) -> list[Interval]:
            if node[1]-node[0]<=1:return [node]
            dec=score(node);decisions.append(dec)
            if not dec.accepted:return [node]
            a,b=node;m=(a+b)//2
            return recurse((a,m))+recurse((m,b))
        selected=[]
        for node in cut:selected.extend(recurse(node))
        return ReadImpuritySplitBackResult(cut,tuple(selected),tuple(decisions),self.penalty_scale,self.prior_mode,self.min_posterior_mass)

@dataclass(frozen=True)
class PrunedTreeRefinement:
    intervals: Cut
    original_to_pruned_clone: np.ndarray
    pruned_clone_genotype_probability: np.ndarray
    cell_clone_probability: np.ndarray
    cell_genotype_probability: np.ndarray
    hard_cell_clone_assignment: np.ndarray
    dominant_clones: np.ndarray
    hypercluster_clone_proportions: np.ndarray
    admixture_mass: np.ndarray
    residual_clone_proportions: np.ndarray
    anchor_weight: float

class PrunedTreeRefiner:
    """Conditionally refit a selected pruned Cactri tree.

    Parameters
    ----------
    anchor_weight:
        Fraction of the one-step mixture/read responsibility update used in the
        final cell-clone probabilities.  The remaining mass comes from the
        original posterior leaf allocation mapped to the pruned cut.  The
        benchmark-supported default is 0.10.
    """

    def __init__(self, *, anchor_weight: float = 0.10, eps: float = 1e-12) -> None:
        if not 0.0 <= anchor_weight <= 1.0:
            raise ValueError("anchor_weight must lie in [0,1].")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")
        self.anchor_weight = float(anchor_weight)
        self.eps = float(eps)

    @staticmethod
    def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
        m = np.max(x, axis=axis, keepdims=True)
        out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
        return np.squeeze(out, axis=axis)

    @staticmethod
    def _mapping(cut: Cut, n_leaf: int) -> np.ndarray:
        out = np.zeros(n_leaf + 1, dtype=np.int64)
        cursor = 0
        for index, (start, stop) in enumerate(cut, 1):
            if start != cursor or stop <= start or stop > n_leaf:
                raise ValueError("intervals must be ordered, disjoint, and cover all leaves")
            out[1 + start : 1 + stop] = index
            cursor = stop
        if cursor != n_leaf:
            raise ValueError("intervals must cover all non-reference leaves")
        return out

    @staticmethod
    def _posterior_leaf_probabilities(draws: np.ndarray, n_clone: int) -> np.ndarray:
        draws = np.asarray(draws, dtype=np.int64)
        out = np.empty((draws.shape[1], n_clone), dtype=float)
        for clone in range(n_clone):
            out[:, clone] = np.mean(draws == clone, axis=0)
        return out

    def _collapsed_patterns(self, model, cut: Cut) -> tuple[np.ndarray, np.ndarray]:
        vertex_leaf = np.asarray(model._vertex_clone_presence[:, 1:], dtype=np.int8)
        collapsed = np.stack(
            [
                [int(np.any(row[start:stop])) for start, stop in cut]
                for row in vertex_leaf
            ],
            axis=0,
        ).astype(np.int8)
        patterns, inverse = np.unique(collapsed, axis=0, return_inverse=True)
        edge_prior = np.asarray(model._current_edge_prior(), dtype=float)
        prior = np.zeros((model.n_snv, patterns.shape[0]), dtype=float)
        for vertex, pattern_index in enumerate(inverse):
            prior[:, pattern_index] += edge_prior[:, vertex]
        prior = np.clip(prior, self.eps, None)
        prior /= prior.sum(axis=1, keepdims=True)
        return patterns, prior

    def _fit_genotypes(
        self,
        model,
        responsibilities: np.ndarray,
        patterns: np.ndarray,
        pattern_prior: np.ndarray,
        theta_present: np.ndarray,
        theta_absent: float,
    ) -> np.ndarray:
        alt = np.nan_to_num(np.asarray(model.alt_counts_, dtype=float), nan=0.0)
        total = np.nan_to_num(np.asarray(model.total_counts_, dtype=float), nan=0.0)
        ref = total - alt
        a = responsibilities.T @ alt
        r = responsibilities.T @ ref
        p1 = np.clip(theta_present, self.eps, 1.0 - self.eps)
        p0 = float(np.clip(theta_absent, self.eps, 1.0 - self.eps))
        absent = a * math.log(p0) + r * math.log1p(-p0)
        present = a * np.log(p1)[None, :] + r * np.log1p(-p1)[None, :]
        base = absent.sum(axis=0)
        delta = present[1:] - absent[1:]
        log_likelihood = (base[None, :] + patterns @ delta).T
        log_posterior = log_likelihood + np.log(pattern_prior)
        norm = self._logsumexp(log_posterior, axis=1)
        posterior = np.exp(log_posterior - norm[:, None])
        return (posterior @ patterns).T

    def _fit_no_spike_mixture(
        self,
        model,
        hyperclusters: np.ndarray,
        responsibilities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if model.clone_mixture.allow_pure_hyperclusters:
            raise ValueError("PrunedTreeRefiner currently supports no-spike mixtures only.")
        n_h = int(hyperclusters.max()) + 1
        n_clone = responsibilities.shape[1]
        counts = np.zeros((n_h, n_clone), dtype=float)
        np.add.at(counts, hyperclusters, responsibilities)
        proportions = np.zeros_like(counts)
        residual = np.zeros_like(counts)
        admixture = np.zeros(n_h, dtype=float)
        dominant = np.zeros(n_h, dtype=np.int64)
        a, b = model.clone_mixture.admixture_mass_prior
        concentration = float(model.clone_mixture.residual_concentration)
        for h in range(n_h):
            n = counts[h]
            n_total = float(n.sum())
            scores = np.empty(n_clone, dtype=float)
            for d in range(n_clone):
                n_d = float(n[d])
                n_r = n_total - n_d
                score = (
                    math.lgamma(a + n_r)
                    + math.lgamma(b + n_d)
                    - math.lgamma(a + b + n_total)
                    - math.lgamma(a)
                    - math.lgamma(b)
                    + math.lgamma(a + b)
                    - math.log(n_clone)
                )
                if n_clone > 1:
                    mask = np.arange(n_clone) != d
                    alpha = np.full(
                        n_clone - 1,
                        concentration / (n_clone - 1),
                        dtype=float,
                    )
                    rc = n[mask]
                    score += math.lgamma(float(alpha.sum()))
                    score -= math.lgamma(float(alpha.sum() + rc.sum()))
                    score += sum(
                        math.lgamma(float(x + c)) - math.lgamma(float(x))
                        for x, c in zip(alpha, rc, strict=True)
                    )
                scores[d] = score
            d = int(np.argmax(scores))
            dominant[h] = d
            epsilon = float((a + n_total - n[d]) / (a + b + n_total))
            admixture[h] = epsilon
            if n_clone == 1:
                proportions[h, 0] = 1.0
                continue
            mask = np.arange(n_clone) != d
            alpha = np.full(
                n_clone - 1,
                concentration / (n_clone - 1),
                dtype=float,
            )
            q = alpha + n[mask]
            q /= q.sum()
            residual[h, mask] = q
            proportions[h, d] = 1.0 - epsilon
            proportions[h, mask] = epsilon * q
        return proportions, dominant, admixture, residual

    def refine(self, model, intervals: Iterable[Interval], *, hypercluster_assignment: np.ndarray | None = None) -> PrunedTreeRefinement:
        if not getattr(model, "clone_mixture", None) or not model.clone_mixture.enabled:
            raise ValueError("a fitted clone-mixture CactriTree model is required")
        if not model.cell_clone_assignment_trace_ or not model.assignment_trace_:
            raise RuntimeError("cell-clone and hypercluster posterior traces are required")
        cut = tuple((int(a), int(b)) for a, b in intervals)
        n_leaf = model.n_leaf_clones
        mapping = self._mapping(cut, n_leaf)
        leaf_prob = self._posterior_leaf_probabilities(
            np.asarray(model.cell_clone_assignment_trace_), model.n_clones
        )
        responsibilities = np.zeros((model.n_cells, len(cut) + 1), dtype=float)
        for leaf in range(model.n_clones):
            responsibilities[:, mapping[leaf]] += leaf_prob[:, leaf]
        responsibilities = np.clip(responsibilities, self.eps, None)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        hyperclusters = np.asarray(
            hypercluster_assignment if hypercluster_assignment is not None else model.posterior_hypercluster_partition_medoid()["assignment"],
            dtype=np.int64,
        )
        _, hyperclusters = np.unique(hyperclusters, return_inverse=True)
        theta_present = (
            np.asarray(model.p_obs_by_mutation_trace_, dtype=float).mean(axis=0)
            if model.p_obs_by_mutation_trace_
            else np.asarray(model.p_obs_by_mutation_, dtype=float)
        )
        theta_absent = (
            float(np.mean(model.p_unobs_trace_))
            if model.p_unobs_trace_
            else float(model.p_unobs_)
        )
        patterns, pattern_prior = self._collapsed_patterns(model, cut)
        genotype_prob = self._fit_genotypes(
            model,
            responsibilities,
            patterns,
            pattern_prior,
            theta_present,
            theta_absent,
        )
        proportions, dominant, admixture, residual = self._fit_no_spike_mixture(
            model, hyperclusters, responsibilities
        )

        alt = np.asarray(model.alt_counts_, dtype=float)
        total = np.asarray(model.total_counts_, dtype=float)
        ref = total - alt
        p1 = np.clip(theta_present, self.eps, 1.0 - self.eps)
        p0 = float(np.clip(theta_absent, self.eps, 1.0 - self.eps))
        absent = alt * math.log(p0) + ref * math.log1p(-p0)
        delta = alt * (np.log(p1) - math.log(p0)) + ref * (
            np.log1p(-p1) - math.log1p(-p0)
        )
        base = absent.sum(axis=1)
        cell_loglik = np.empty((model.n_cells, len(cut) + 1), dtype=float)
        cell_loglik[:, 0] = base
        cell_loglik[:, 1:] = base[:, None] + delta @ genotype_prob.T
        target_log = cell_loglik + np.log(np.clip(proportions[hyperclusters], self.eps, 1.0))
        target_norm = self._logsumexp(target_log, axis=1)
        target = np.exp(target_log - target_norm[:, None])
        refined = (1.0 - self.anchor_weight) * responsibilities + self.anchor_weight * target
        refined /= refined.sum(axis=1, keepdims=True)
        cell_genotype = refined[:, 1:] @ genotype_prob
        return PrunedTreeRefinement(
            intervals=cut,
            original_to_pruned_clone=mapping,
            pruned_clone_genotype_probability=genotype_prob,
            cell_clone_probability=refined,
            cell_genotype_probability=cell_genotype,
            hard_cell_clone_assignment=np.argmax(refined, axis=1).astype(np.int64),
            dominant_clones=dominant,
            hypercluster_clone_proportions=proportions,
            admixture_mass=admixture,
            residual_clone_proportions=residual,
            anchor_weight=self.anchor_weight,
        )

__all__ = [
    "Interval",
    "Cut",
    "StateMode",
    "PriorMode",
    "GreedyCollapseStep",
    "GreedyTreePruningResult",
    "GreedyTreePruner",
    "SplitBackDecision",
    "ReadImpuritySplitBackResult",
    "ReadImpuritySplitBack",
    "PrunedTreeRefinement",
    "PrunedTreeRefiner",
]
