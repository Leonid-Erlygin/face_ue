from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.special import softmax

from evaluation.open_set_methods.class_prob_models import MPRiskPredictiveProb
from evaluation.open_set_methods.kappa_utils import (
    candidate_kappas_for_tau,
    solve_kappa_for_tau,
    threshold_at_far,
)
from evaluation.samplers import VonMisesFisher

from .calibration import binary_nll
from .embedders import l2_normalize
from .types import RetrievalScores


@dataclass
class PosteriorModelConfig:
    beta: float = 0.5
    target_fpir: float = 0.1
    gallery_prior: str = "power"
    predict_T: float = 200.0
    mc_samples: int = 0
    kappa_input_scale: float = 1.0
    gallery_kappa: Optional[float] = None
    gallery_kappa_strategy: str = "boundary_roots_calibrated"
    kappa_low: float = 10.0
    kappa_high: float = 1_000_000.0
    kappa_grid_size: int = 512
    nonspecificity_mode: str = "analytic_vmf"
    lambda_fa: float = 1.0
    lambda_id: float = 1.0
    lambda_fr: float = 1.0
    lambda_ns: float = 1.0
    softmax_temperature: float = 0.05
    prob_batch_size: Optional[int] = None
    max_prob_elements: int = 8_000_000
    mc_num_workers: int = 1
    streaming_threshold_elements: int = 20_000_000
    streaming_query_batch_size: int = 64
    streaming_gallery_chunk_size: int = 32768


class ModernUncertaintyModel:
    """Adapter from the dissertation posterior to modern embedding galleries.

    It reuses the exact HolUE/MPRisk probability implementation. The only new
    responsibility is fitting gallery concentration from *known-vs-unknown*
    validation queries, because modern retrieval does not use face identity IDs.
    """

    def __init__(self, config: PosteriorModelConfig):
        self.config = config
        self.gallery_kappa: Optional[float] = config.gallery_kappa
        self.kappa_candidates_: Tuple[float, ...] = ()
        self.kappa_candidate_nll_: Tuple[float, ...] = ()
        self._make_core()

    def _make_core(self) -> None:
        c = self.config
        self.core = MPRiskPredictiveProb(
            gallery_prior=c.gallery_prior,
            emb_unc_model="vMF",
            beta=c.beta,
            far=c.target_fpir,
            M=c.mc_samples,
            calibration_set=None,
            calibration_transform=None,
            gallery_kappa=c.gallery_kappa,
            kappa_input_scale=c.kappa_input_scale,
            predict_T=c.predict_T,
            lambda_fa=c.lambda_fa,
            lambda_id=c.lambda_id,
            lambda_fr=c.lambda_fr,
            lambda_ns=c.lambda_ns,
            nonspecificity_mode=c.nonspecificity_mode,
            use_calibration=False,
            tune_lambdas=False,
            prob_batch_size=c.prob_batch_size,
            max_prob_elements=c.max_prob_elements,
        )
        # Existing sampler defaults to 21 processes; modern NLP runs are often
        # already multi-process/GPU workloads, so use an explicit small worker count.
        self.core.sampler = VonMisesFisher(c.mc_samples, num_workers=c.mc_num_workers)

    @staticmethod
    def _posterior_decision(mean_probs: np.ndarray, p0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pred = np.argmax(mean_probs, axis=1)
        all_probs = np.column_stack([mean_probs, p0])
        rejected = np.argmax(all_probs, axis=1) == mean_probs.shape[1]
        return pred.astype(int), rejected.astype(bool)

    def _score_with_kappa(
        self,
        query_embeddings: np.ndarray,
        query_kappa: np.ndarray,
        gallery_embeddings: np.ndarray,
        gallery_kappa: float,
    ):
        gallery_unc = np.ones((len(gallery_embeddings), 1), dtype=np.float64)
        return self.core._compute_probs_aux(
            probe_feats=np.asarray(query_embeddings, dtype=np.float64),
            probe_unc=np.asarray(query_kappa, dtype=np.float64),
            gallery_feats=np.asarray(gallery_embeddings, dtype=np.float64),
            gallery_unc=gallery_unc,
            gallery_kappa=float(gallery_kappa),
        )

    def fit_gallery_kappa(
        self,
        query_embeddings: np.ndarray,
        query_kappa: np.ndarray,
        gallery_embeddings: np.ndarray,
        known_mask: np.ndarray,
    ) -> float:
        if self.config.gallery_kappa is not None:
            self.gallery_kappa = float(self.config.gallery_kappa)
            self.core.gallery_kappa = self.gallery_kappa
            return self.gallery_kappa

        q = l2_normalize(query_embeddings)
        g = l2_normalize(gallery_embeddings)
        known = np.asarray(known_mask, dtype=bool).reshape(-1)
        if not np.any(~known):
            raise ValueError("Gallery-kappa fitting requires at least one unknown calibration query.")
        similarities = q @ g.T
        tau = threshold_at_far(np.max(similarities[~known], axis=1), self.config.target_fpir)
        d = int(q.shape[1])
        K = int(g.shape[0])
        strategy = str(self.config.gallery_kappa_strategy)

        if strategy == "boundary_first_root":
            candidates = candidate_kappas_for_tau(
                tau=tau,
                beta=self.config.beta,
                K=K,
                d=d,
                class_model=self.config.gallery_prior,
                kappa_low=self.config.kappa_low,
                kappa_high=self.config.kappa_high,
                grid_size=self.config.kappa_grid_size,
            )
            if not candidates:
                candidates = [solve_kappa_for_tau(
                    tau, self.config.beta, K, d, self.config.gallery_prior,
                    self.config.kappa_low, self.config.kappa_high,
                    self.config.kappa_grid_size,
                )]
            chosen = float(candidates[0])
            losses = [np.nan] * len(candidates)

        elif strategy == "boundary_roots_calibrated":
            candidates = candidate_kappas_for_tau(
                tau=tau,
                beta=self.config.beta,
                K=K,
                d=d,
                class_model=self.config.gallery_prior,
                kappa_low=self.config.kappa_low,
                kappa_high=self.config.kappa_high,
                grid_size=self.config.kappa_grid_size,
            )
            if not candidates:
                candidates = [solve_kappa_for_tau(
                    tau, self.config.beta, K, d, self.config.gallery_prior,
                    self.config.kappa_low, self.config.kappa_high,
                    self.config.kappa_grid_size,
                )]
            losses = []
            y_unknown = (~known).astype(float)
            for kappa in candidates:
                if (len(q) * len(g) > self.config.streaming_threshold_elements
                        and self.config.mc_samples == 0):
                    from .scalable import streaming_p0_and_decision
                    p0, _, _ = streaming_p0_and_decision(
                        q, g, float(kappa), self.config.beta,
                        self.config.gallery_prior, self.config.predict_T,
                        query_batch_size=self.config.streaming_query_batch_size,
                        gallery_chunk_size=self.config.streaming_gallery_chunk_size,
                        device=self.core.device,
                    )
                else:
                    _, _, _, p0, _ = self._score_with_kappa(q, query_kappa, g, kappa)
                losses.append(binary_nll(y_unknown, p0))
            chosen = float(candidates[int(np.argmin(losses))])

        elif strategy == "empirical_grid":
            # Appropriate for MC posterior decisions, whose empirical acceptance
            # boundary need not exactly match the deterministic mean embedding.
            grid = np.exp(
                np.linspace(
                    np.log(self.config.kappa_low),
                    np.log(self.config.kappa_high),
                    min(self.config.kappa_grid_size, 96),
                )
            )
            y_unknown = (~known).astype(float)
            candidates = list(map(float, grid))
            losses = []
            fpir_errors = []
            for kappa in candidates:
                if (len(q) * len(g) > self.config.streaming_threshold_elements
                        and self.config.mc_samples == 0):
                    from .scalable import streaming_p0_and_decision
                    p0, _, rejected = streaming_p0_and_decision(
                        q, g, float(kappa), self.config.beta,
                        self.config.gallery_prior, self.config.predict_T,
                        query_batch_size=self.config.streaming_query_batch_size,
                        gallery_chunk_size=self.config.streaming_gallery_chunk_size,
                        device=self.core.device,
                    )
                else:
                    mean_probs, _, _, p0, _ = self._score_with_kappa(q, query_kappa, g, kappa)
                    _, rejected = self._posterior_decision(mean_probs, p0)
                fpir = float(np.mean(~rejected[~known]))
                fpir_errors.append(abs(fpir - self.config.target_fpir))
                losses.append(binary_nll(y_unknown, p0))
            best_fpir = min(fpir_errors)
            eligible = [i for i, e in enumerate(fpir_errors) if e <= best_fpir + 1e-12]
            idx = min(eligible, key=lambda i: losses[i])
            chosen = float(candidates[idx])
        else:
            raise ValueError(
                "gallery_kappa_strategy must be boundary_first_root, "
                "boundary_roots_calibrated, or empirical_grid"
            )

        self.gallery_kappa = chosen
        self.core.gallery_kappa = chosen
        self.kappa_candidates_ = tuple(map(float, candidates))
        self.kappa_candidate_nll_ = tuple(map(float, losses))
        self.fit_tau_ = float(tau)
        return chosen

    def score(
        self,
        query_ids: Sequence[str],
        corpus_ids: Sequence[str],
        query_embeddings: np.ndarray,
        query_kappa: np.ndarray,
        gallery_embeddings: np.ndarray,
        query_resultant_length: Optional[np.ndarray] = None,
    ) -> RetrievalScores:
        if self.gallery_kappa is None:
            raise RuntimeError("Call fit_gallery_kappa() or configure gallery_kappa first.")
        q = l2_normalize(query_embeddings)
        g = l2_normalize(gallery_embeddings)
        if (len(q) * len(g) > self.config.streaming_threshold_elements
                and self.config.mc_samples == 0):
            from .scalable import streaming_deterministic_score
            result = streaming_deterministic_score(
                query_ids=query_ids, corpus_ids=corpus_ids,
                query_embeddings=q, query_kappa=query_kappa,
                gallery_embeddings=g, gallery_kappa=float(self.gallery_kappa),
                beta=self.config.beta, gallery_prior=self.config.gallery_prior,
                predict_T=self.config.predict_T,
                kappa_input_scale=self.config.kappa_input_scale,
                lambda_fa=self.config.lambda_fa, lambda_id=self.config.lambda_id,
                lambda_fr=self.config.lambda_fr, lambda_ns=self.config.lambda_ns,
                softmax_temperature=self.config.softmax_temperature,
                query_resultant_length=query_resultant_length,
                query_batch_size=self.config.streaming_query_batch_size,
                gallery_chunk_size=self.config.streaming_gallery_chunk_size,
                device=self.core.device,
            )
            result.metadata.update({
                "fit_tau": getattr(self, "fit_tau_", np.nan),
                "kappa_candidates": self.kappa_candidates_,
                "kappa_candidate_nll": self.kappa_candidate_nll_,
            })
            return result
        mean_probs, kl1, kl2, p0, nonspec = self._score_with_kappa(
            q, query_kappa, g, self.gallery_kappa
        )
        pred, rejected = self._posterior_decision(mean_probs, p0)
        comps = self.core._risk_components(mean_probs, p0, nonspec)

        sim = q @ g.T
        max_sim = np.max(sim, axis=1)
        if sim.shape[1] > 1:
            part = np.partition(sim, kth=sim.shape[1] - 2, axis=1)
            second = part[:, -2]
        else:
            second = np.full(len(sim), -1.0)
        margin = max_sim - second
        sm = softmax(sim / max(float(self.config.softmax_temperature), 1e-8), axis=1)
        sm_entropy = -np.sum(np.where(sm > 0, sm * np.log(sm), 0.0), axis=1)

        all_probs = np.column_stack([mean_probs, p0])
        posterior_entropy = -np.sum(
            np.where(all_probs > 0, all_probs * np.log(np.clip(all_probs, 1e-300, 1.0)), 0.0),
            axis=1,
        )
        scores: Dict[str, np.ndarray] = {
            "max_similarity": -max_sim,
            "margin": -margin,
            "softmax_entropy": sm_entropy,
            "softmax_msp": 1.0 - np.max(sm, axis=1),
            "galue_entropy": posterior_entropy,
            "galue_msp": 1.0 - np.max(all_probs, axis=1),
            # HolUE uses two components; this raw scalar is diagnostic only.
            "holue_kl_sum": -(kl1 + kl2),
            "mprisk": np.asarray(comps["mprisk"], dtype=np.float64),
            "mprisk_no_ns": np.asarray(comps["ordinary_risk"], dtype=np.float64),
            "unknown_probability": np.asarray(p0, dtype=np.float64),
        }
        if query_resultant_length is not None:
            scores["rewrite_dispersion"] = 1.0 - np.asarray(query_resultant_length, dtype=np.float64)
            scores["negative_query_kappa"] = -np.asarray(query_kappa, dtype=np.float64).reshape(-1)

        return RetrievalScores(
            query_ids=tuple(map(str, query_ids)),
            corpus_ids=tuple(map(str, corpus_ids)),
            predicted_indices=pred,
            was_rejected=rejected,
            mean_known_probs=np.asarray(mean_probs),
            unknown_prob=np.asarray(p0),
            unknown_nonspecificity=np.asarray(nonspec),
            kl_1=np.asarray(kl1),
            kl_2=np.asarray(kl2),
            scores=scores,
            metadata={
                "gallery_kappa": float(self.gallery_kappa),
                "fit_tau": getattr(self, "fit_tau_", np.nan),
                "kappa_candidates": self.kappa_candidates_,
                "kappa_candidate_nll": self.kappa_candidate_nll_,
                "beta": float(self.config.beta),
                "predict_T": float(self.config.predict_T),
                "gallery_prior": self.config.gallery_prior,
                "mc_samples": int(self.config.mc_samples),
            },
        )

    def score_variable_galleries(
        self,
        query_embeddings: np.ndarray,
        query_kappa: np.ndarray,
        galleries: Sequence[np.ndarray],
        query_resultant_length: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Score one local evidence/tool gallery per query.

        This is used for RAGTruth and BFCL, where the candidate context/functions
        differ for every example. A fixed, externally calibrated gallery_kappa is
        strongly preferred for cross-benchmark evaluation.
        """
        if self.gallery_kappa is None:
            raise RuntimeError("Variable-gallery scoring requires a configured/transferred gallery_kappa.")
        names = [
            "max_similarity", "margin", "galue_entropy", "galue_msp",
            "holue_kl_1", "holue_kl_2", "holue_kl_sum", "mprisk",
            "mprisk_no_ns", "unknown_probability", "unknown_nonspecificity",
            "rejected", "predicted_index",
        ]
        acc = {n: [] for n in names}
        for i, gallery in enumerate(galleries):
            gallery = l2_normalize(np.asarray(gallery, dtype=np.float64))
            if gallery.ndim != 2 or len(gallery) < 1:
                for name in names:
                    acc[name].append(np.nan)
                continue
            mean_probs, kl1, kl2, p0, ns = self._score_with_kappa(
                query_embeddings[i : i + 1], query_kappa[i : i + 1], gallery, self.gallery_kappa
            )
            pred, rejected = self._posterior_decision(mean_probs, p0)
            comps = self.core._risk_components(mean_probs, p0, ns)
            sims = l2_normalize(query_embeddings[i : i + 1]) @ gallery.T
            top = np.sort(sims[0])[::-1]
            margin = top[0] - (top[1] if len(top) > 1 else -1.0)
            allp = np.r_[mean_probs[0], p0[0]]
            ent = -np.sum(np.where(allp > 0, allp * np.log(np.clip(allp, 1e-300, 1.0)), 0.0))
            values = {
                "max_similarity": -float(top[0]),
                "margin": -float(margin),
                "galue_entropy": float(ent),
                "galue_msp": float(1.0 - np.max(allp)),
                "holue_kl_1": float(kl1[0]),
                "holue_kl_2": float(kl2[0]),
                "holue_kl_sum": -float(kl1[0] + kl2[0]),
                "mprisk": float(comps["mprisk"][0]),
                "mprisk_no_ns": float(comps["ordinary_risk"][0]),
                "unknown_probability": float(p0[0]),
                "unknown_nonspecificity": float(ns[0]),
                "rejected": float(rejected[0]),
                "predicted_index": float(pred[0]),
            }
            for name in names:
                acc[name].append(values[name])
        out = {k: np.asarray(v, dtype=np.float64) for k, v in acc.items()}
        if query_resultant_length is not None:
            out["rewrite_dispersion"] = 1.0 - np.asarray(query_resultant_length, dtype=np.float64)
        return out
