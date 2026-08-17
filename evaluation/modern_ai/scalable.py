from __future__ import annotations

"""Memory-bounded exact posterior evaluation for large deterministic galleries.

This module does **not** renormalize over retrieved top-k candidates.  For M=0 it
computes exactly the same mixed-prior posterior as ``MPRiskPredictiveProb`` while
streaming gallery chunks.  Compute remains O(NK), which is scientifically honest:
the method's full-gallery normalizer is expensive.  Memory is O(NB), where B is
the gallery chunk size.
"""

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from evaluation.open_set_methods.kappa_utils import (
    log_uniform_density,
    power_log_normalizer_np,
    vmf_log_normalizer_np,
)

from .embedders import l2_normalize
from .types import RetrievalScores


def _known_log_terms(
    sim: torch.Tensor,
    gallery_kappa: float,
    d: int,
    K: int,
    beta: float,
    gallery_prior: str,
    inv_temperature: float,
) -> torch.Tensor:
    k = float(max(gallery_kappa, 1e-12))
    if gallery_prior == "power":
        log_norm = float(np.asarray(power_log_normalizer_np(k, d)))
        kernel = k * torch.log1p(torch.clamp(sim, -1.0 + 1e-9, 1.0 - 1e-9))
    elif gallery_prior == "vMF":
        log_norm = float(np.asarray(vmf_log_normalizer_np(k, d)))
        kernel = k * torch.clamp(sim, -1.0 + 1e-9, 1.0 - 1e-9)
    else:
        raise ValueError(f"Unknown gallery_prior={gallery_prior}")
    log_prior = math.log((1.0 - float(beta)) / int(K))
    return float(inv_temperature) * (log_norm + kernel + log_prior)


def streaming_p0_and_decision(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_kappa: float,
    beta: float,
    gallery_prior: str,
    predict_T: float,
    query_batch_size: int = 64,
    gallery_chunk_size: int = 32768,
    device: Optional[torch.device | str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-pass exact p(unknown), top gallery index and reject decision for M=0."""

    q = l2_normalize(query_embeddings)
    g = l2_normalize(gallery_embeddings)
    if q.ndim != 2 or g.ndim != 2 or q.shape[1] != g.shape[1]:
        raise ValueError("query/gallery embeddings must be compatible 2-D arrays")
    if len(g) < 1:
        raise ValueError("gallery cannot be empty")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float64
    K, d = len(g), q.shape[1]
    invT = 1.0 / max(float(predict_T), 1e-6)
    log_oog = invT * (log_uniform_density(d) + math.log(float(beta)))

    out_p0 = np.empty(len(q), dtype=np.float64)
    out_pred = np.empty(len(q), dtype=np.int64)
    out_rej = np.empty(len(q), dtype=bool)

    for qs in range(0, len(q), int(query_batch_size)):
        qe = min(qs + int(query_batch_size), len(q))
        qt = torch.as_tensor(q[qs:qe], dtype=dtype, device=dev)
        gallery_lse = torch.full((len(qt),), -torch.inf, dtype=dtype, device=dev)
        best_log = torch.full_like(gallery_lse, -torch.inf)
        best_idx = torch.zeros(len(qt), dtype=torch.long, device=dev)
        for gs in range(0, K, int(gallery_chunk_size)):
            ge = min(gs + int(gallery_chunk_size), K)
            gt = torch.as_tensor(g[gs:ge], dtype=dtype, device=dev)
            sim = qt @ gt.T
            terms = _known_log_terms(sim, gallery_kappa, d, K, beta, gallery_prior, invT)
            gallery_lse = torch.logaddexp(gallery_lse, torch.logsumexp(terms, dim=1))
            vals, inds = torch.max(terms, dim=1)
            improve = vals > best_log
            best_log = torch.where(improve, vals, best_log)
            best_idx = torch.where(improve, inds + gs, best_idx)
        log_oog_t = torch.full_like(gallery_lse, float(log_oog))
        log_den = torch.logaddexp(gallery_lse, log_oog_t)
        p0 = torch.exp(log_oog_t - log_den)
        pbest = torch.exp(best_log - log_den)
        out_p0[qs:qe] = p0.cpu().numpy()
        out_pred[qs:qe] = best_idx.cpu().numpy()
        # np.argmax([known..., p0]) rejects only when p0 is strictly larger.
        out_rej[qs:qe] = (p0 > pbest).cpu().numpy()
    return out_p0, out_pred, out_rej


def streaming_deterministic_score(
    *,
    query_ids: Sequence[str],
    corpus_ids: Sequence[str],
    query_embeddings: np.ndarray,
    query_kappa: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_kappa: float,
    beta: float,
    gallery_prior: str,
    predict_T: float,
    kappa_input_scale: float = 1.0,
    lambda_fa: float = 1.0,
    lambda_id: float = 1.0,
    lambda_fr: float = 1.0,
    lambda_ns: float = 1.0,
    softmax_temperature: float = 0.05,
    query_resultant_length: Optional[np.ndarray] = None,
    query_batch_size: int = 64,
    gallery_chunk_size: int = 32768,
    device: Optional[torch.device | str] = None,
) -> RetrievalScores:
    """Exact deterministic HolUE/GalUE/MPRisk score without an N x K matrix.

    The formula is algebraically identical to the M=0 branch of the dissertation
    implementation.  ``mean_known_probs`` is intentionally returned as an empty
    matrix; top identity, p0, KL1/2, entropy, and MPRisk are accumulated exactly.
    """

    q = l2_normalize(query_embeddings)
    g = l2_normalize(gallery_embeddings)
    qk = np.asarray(query_kappa, dtype=np.float64).reshape(-1) * float(kappa_input_scale)
    if len(qk) != len(q):
        raise ValueError("query_kappa length differs from query embeddings")
    if tuple(map(str, corpus_ids)) and len(corpus_ids) != len(g):
        raise ValueError("corpus_ids length differs from gallery embeddings")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float64
    K, d = len(g), q.shape[1]
    if K < 1:
        raise ValueError("gallery cannot be empty")
    invT = 1.0 / max(float(predict_T), 1e-6)
    log_gallery_prior = math.log((1.0 - float(beta)) / K)
    log_beta_uniform = math.log(float(beta)) + log_uniform_density(d)
    log_oog = invT * log_beta_uniform

    n = len(q)
    pred = np.empty(n, dtype=np.int64)
    rejected = np.empty(n, dtype=bool)
    p0_out = np.empty(n, dtype=np.float64)
    ns_out = np.empty(n, dtype=np.float64)
    kl1_out = np.empty(n, dtype=np.float64)
    kl2_out = np.empty(n, dtype=np.float64)
    max_sim_out = np.empty(n, dtype=np.float64)
    margin_out = np.empty(n, dtype=np.float64)
    soft_entropy_out = np.empty(n, dtype=np.float64)
    soft_msp_out = np.empty(n, dtype=np.float64)
    posterior_entropy_out = np.empty(n, dtype=np.float64)
    pbest_out = np.empty(n, dtype=np.float64)

    softT = max(float(softmax_temperature), 1e-8)

    for qs in range(0, n, int(query_batch_size)):
        qe = min(qs + int(query_batch_size), n)
        qt = torch.as_tensor(q[qs:qe], dtype=dtype, device=dev)
        b = len(qt)
        gallery_lse = torch.full((b,), -torch.inf, dtype=dtype, device=dev)
        soft_lse = torch.full((b,), -torch.inf, dtype=dtype, device=dev)
        best_log = torch.full((b,), -torch.inf, dtype=dtype, device=dev)
        best_idx = torch.zeros(b, dtype=torch.long, device=dev)
        top2 = torch.full((b, 2), -torch.inf, dtype=dtype, device=dev)

        # Pass 1: posterior denominator, retrieval maximum/margin, softmax denominator.
        for gs in range(0, K, int(gallery_chunk_size)):
            ge = min(gs + int(gallery_chunk_size), K)
            gt = torch.as_tensor(g[gs:ge], dtype=dtype, device=dev)
            sim = qt @ gt.T
            terms = _known_log_terms(sim, gallery_kappa, d, K, beta, gallery_prior, invT)
            gallery_lse = torch.logaddexp(gallery_lse, torch.logsumexp(terms, dim=1))
            soft_lse = torch.logaddexp(soft_lse, torch.logsumexp(sim / softT, dim=1))

            vals, inds = torch.max(terms, dim=1)
            improve = vals > best_log
            best_log = torch.where(improve, vals, best_log)
            best_idx = torch.where(improve, inds + gs, best_idx)

            chunk_top = torch.topk(sim, k=min(2, sim.shape[1]), dim=1).values
            if chunk_top.shape[1] == 1:
                chunk_top = torch.cat([chunk_top, torch.full_like(chunk_top, -torch.inf)], dim=1)
            top2 = torch.topk(torch.cat([top2, chunk_top], dim=1), k=2, dim=1).values

        log_oog_t = torch.full_like(gallery_lse, float(log_oog))
        log_den = torch.logaddexp(gallery_lse, log_oog_t)
        p0 = torch.exp(log_oog_t - log_den)
        pbest = torch.exp(best_log - log_den)

        # Pass 2: KL1 and exact entropy accumulators.  No probability matrix retained.
        kl1 = torch.zeros(b, dtype=dtype, device=dev)
        known_entropy = torch.zeros_like(kl1)
        soft_entropy = torch.zeros_like(kl1)
        for gs in range(0, K, int(gallery_chunk_size)):
            ge = min(gs + int(gallery_chunk_size), K)
            gt = torch.as_tensor(g[gs:ge], dtype=dtype, device=dev)
            sim = qt @ gt.T
            terms = _known_log_terms(sim, gallery_kappa, d, K, beta, gallery_prior, invT)
            logp = terms - log_den[:, None]
            pp = torch.exp(logp)
            kl1 += torch.sum(pp * (logp - log_gallery_prior), dim=1)
            known_entropy -= torch.sum(pp * logp, dim=1)

            slogp = sim / softT - soft_lse[:, None]
            sp = torch.exp(slogp)
            soft_entropy -= torch.sum(sp * slogp, dim=1)

        p0_safe = torch.clamp(p0, min=1e-300)
        posterior_entropy = known_entropy - p0 * torch.log(p0_safe)

        # Deterministic M=0 KL2: z is exactly the normalized query mean, so mu^T z=1.
        kx = np.maximum(qk[qs:qe], 1e-12)
        log_norm_x = vmf_log_normalizer_np(kx, d=d)
        log_q_at_mean = log_norm_x + kx * (1.0 - 1e-9)
        log_arg = (invT - 1.0) * log_beta_uniform + torch.as_tensor(
            log_q_at_mean, dtype=dtype, device=dev
        ) - log_den
        kl2 = p0 * log_arg

        # Exact analytic vMF collision non-specificity used by default MPRisk.
        log_norm_2x = vmf_log_normalizer_np(2.0 * kx, d=d)
        log_surface = -log_uniform_density(d)
        log_collision = np.clip(log_surface + 2.0 * log_norm_x - log_norm_2x, 0.0, 700.0)
        ns = np.exp(-log_collision)

        pred[qs:qe] = best_idx.cpu().numpy()
        p0_np = p0.cpu().numpy()
        pbest_np = pbest.cpu().numpy()
        p0_out[qs:qe] = p0_np
        pbest_out[qs:qe] = pbest_np
        rejected[qs:qe] = p0_np > pbest_np
        ns_out[qs:qe] = ns
        kl1_out[qs:qe] = kl1.cpu().numpy()
        kl2_out[qs:qe] = kl2.cpu().numpy()
        max_sim_out[qs:qe] = top2[:, 0].cpu().numpy()
        second = top2[:, 1].cpu().numpy()
        second[~np.isfinite(second)] = -1.0
        margin_out[qs:qe] = max_sim_out[qs:qe] - second
        soft_entropy_out[qs:qe] = soft_entropy.cpu().numpy()
        soft_msp_out[qs:qe] = (1.0 - torch.exp(top2[:, 0] / softT - soft_lse)).cpu().numpy()
        posterior_entropy_out[qs:qe] = posterior_entropy.cpu().numpy()

    accepted = ~rejected
    other_known = np.clip(1.0 - p0_out - pbest_out, 0.0, 1.0)
    r_fa = np.where(accepted, p0_out, 0.0)
    r_id = np.where(accepted, other_known, 0.0)
    r_fr = np.where(rejected, 1.0 - p0_out, 0.0)
    r_ns = np.where(rejected, p0_out * ns_out, 0.0)
    ordinary = float(lambda_fa) * r_fa + float(lambda_id) * r_id + float(lambda_fr) * r_fr
    mprisk = ordinary + float(lambda_ns) * r_ns

    scores: Dict[str, np.ndarray] = {
        "max_similarity": -max_sim_out,
        "margin": -margin_out,
        "softmax_entropy": soft_entropy_out,
        "softmax_msp": soft_msp_out,
        "galue_entropy": posterior_entropy_out,
        "galue_msp": 1.0 - np.maximum(pbest_out, p0_out),
        "holue_kl_sum": -(kl1_out + kl2_out),
        "mprisk": mprisk,
        "mprisk_no_ns": ordinary,
        "unknown_probability": p0_out,
    }
    if query_resultant_length is not None:
        scores["rewrite_dispersion"] = 1.0 - np.asarray(query_resultant_length, dtype=np.float64)
        scores["negative_query_kappa"] = -np.asarray(query_kappa, dtype=np.float64).reshape(-1)

    finite = [p0_out, ns_out, kl1_out, kl2_out, mprisk, posterior_entropy_out]
    if any(not np.all(np.isfinite(x)) for x in finite):
        raise FloatingPointError("Non-finite streaming posterior result")

    return RetrievalScores(
        query_ids=tuple(map(str, query_ids)),
        corpus_ids=tuple(map(str, corpus_ids)),
        predicted_indices=pred,
        was_rejected=rejected,
        mean_known_probs=np.empty((n, 0), dtype=np.float64),
        unknown_prob=p0_out,
        unknown_nonspecificity=ns_out,
        kl_1=kl1_out,
        kl_2=kl2_out,
        scores=scores,
        metadata={
            "gallery_kappa": float(gallery_kappa),
            "beta": float(beta),
            "predict_T": float(predict_T),
            "gallery_prior": str(gallery_prior),
            "mc_samples": 0,
            "streaming_exact": True,
            "gallery_chunk_size": int(gallery_chunk_size),
            "query_batch_size": int(query_batch_size),
        },
    )
