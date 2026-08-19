from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.special import ive

from .embedders import BaseTextEmbedder, l2_normalize


def _vmf_mean_resultant(kappa: float, d: int) -> float:
    kappa = float(max(kappa, 1e-10))
    nu = d / 2.0 - 1.0
    den = ive(nu, kappa)
    num = ive(nu + 1.0, kappa)
    if np.isfinite(num) and np.isfinite(den) and abs(den) > 0:
        return float(num / den)
    # High-concentration asymptotic A_d(k) ~= 1 - (d-1)/(2k).
    return float(np.clip(1.0 - (d - 1.0) / (2.0 * kappa), 0.0, 1.0))


def estimate_vmf_kappa_from_embeddings(
    embeddings: np.ndarray,
    min_kappa: float = 1.0,
    max_kappa: float = 1_000_000.0,
    single_observation_kappa: float = 10_000.0,
) -> Tuple[np.ndarray, float, float]:
    """Estimate vMF mean direction and concentration from unit embeddings.

    Returns ``(mean_direction, kappa, resultant_length)``. Concentration is the
    MLE solution A_d(kappa)=Rbar, clipped to a finite interval for numerical and
    operational stability.
    """

    x = l2_normalize(np.asarray(embeddings, dtype=np.float64))
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError("embeddings must have shape [num_rewrites, dimension].")
    mean = np.mean(x, axis=0)
    rbar = float(np.linalg.norm(mean))
    direction = mean / max(rbar, 1e-12)
    if x.shape[0] == 1:
        return direction, float(single_observation_kappa), rbar
    rbar = float(np.clip(rbar, 0.0, 1.0 - 1e-12))
    if rbar < 1e-8:
        return direction, float(min_kappa), rbar

    d = int(x.shape[1])

    # Banerjee-style approximation is a useful bracket centre.
    init = rbar * (d - rbar * rbar) / max(1.0 - rbar * rbar, 1e-12)
    lo = float(min_kappa)
    hi = float(max(max_kappa, lo * 1.01))

    def f(log_k: float) -> float:
        return _vmf_mean_resultant(float(np.exp(log_k)), d) - rbar

    flo = f(np.log(lo))
    fhi = f(np.log(hi))
    if np.isfinite(flo) and np.isfinite(fhi) and flo * fhi <= 0:
        kappa = float(np.exp(brentq(f, np.log(lo), np.log(hi), maxiter=100)))
    else:
        kappa = float(np.clip(init, lo, hi))
    return direction, kappa, rbar


def load_rewrites_jsonl(path: str | Path) -> Dict[str, Tuple[str, ...]]:
    out: Dict[str, Tuple[str, ...]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rewrites = row.get("rewrites") or row.get("paraphrases") or []
            out[str(row["id"])] = tuple(str(x) for x in rewrites if str(x).strip())
    return out


def query_embeddings_and_kappa(
    query_ids: Sequence[str],
    queries: Sequence[str],
    embedder: BaseTextEmbedder,
    rewrites: Optional[Mapping[str, Sequence[str]]] = None,
    default_kappa: float = 300.0,
    include_original: bool = True,
    min_kappa: float = 1.0,
    max_kappa: float = 1_000_000.0,
    mean_mode: str = "rewrite_mean",
    kappa_source: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode queries and estimate HolUE concentration from paraphrase dispersion.

    When no rewrites are supplied for a query, the original embedding is used and
    ``default_kappa`` is returned. This makes the uncertainty source explicit
    rather than pretending a single deterministic sentence identifies kappa.
    """

    if mean_mode not in {"rewrite_mean", "original"}:
        raise ValueError("mean_mode must be 'rewrite_mean' or 'original'.")
    if kappa_source not in {"auto", "default", "rewrite", "embedder"}:
        raise ValueError("kappa_source must be auto/default/rewrite/embedder")

    qids = list(map(str, query_ids))
    queries = list(map(str, queries))
    if len(qids) != len(queries):
        raise ValueError("query_ids and queries must have equal length.")

    source = kappa_source
    if source == "auto":
        if rewrites:
            source = "rewrite"
        elif bool(getattr(embedder, "supports_query_kappa", False)):
            source = "embedder"
        else:
            source = "default"
    if source == "embedder":
        if rewrites:
            raise ValueError(
                "kappa_source=embedder uses SCF concentration from the original query; "
                "do not also supply rewrites in the same condition."
            )
        if not bool(getattr(embedder, "supports_query_kappa", False)):
            raise ValueError(f"{type(embedder).__name__} does not expose learned query kappa")
        emb, kappa = embedder.encode_queries_with_kappa(queries)
        kappa = np.asarray(kappa, dtype=np.float64).reshape(len(queries), 1)
        return l2_normalize(np.asarray(emb)), kappa, np.full(len(queries), np.nan, dtype=np.float64)

    means = []
    kappas = []
    rbars = []
    for qid, query in zip(qids, queries):
        variants = [] if source == "default" else list((rewrites or {}).get(qid, ()))
        if include_original or not variants:
            variants = [query] + variants
        emb = embedder.encode_queries(variants)
        if len(variants) <= 1:
            means.append(emb[0])
            kappas.append(float(default_kappa))
            rbars.append(1.0)
        else:
            direction, kappa, rbar = estimate_vmf_kappa_from_embeddings(
                emb, min_kappa=min_kappa, max_kappa=max_kappa
            )
            # For the clean HolUE ablation, keep the original query representation
            # fixed and use rewrites only to estimate concentration.  This avoids
            # confounding a better/worse query embedding with uncertainty quality.
            if mean_mode == "original":
                original = embedder.encode_queries([query])[0]
                means.append(original)
            else:
                means.append(direction)
            kappas.append(kappa)
            rbars.append(rbar)
    return (
        l2_normalize(np.asarray(means)),
        np.asarray(kappas, dtype=np.float64)[:, None],
        np.asarray(rbars, dtype=np.float64),
    )


class TransformersRewriteGenerator:
    """Optional local-HF paraphrase generator for rewrite-dispersion experiments."""

    def __init__(
        self,
        model_name: str,
        num_rewrites: int = 5,
        max_new_tokens: int = 64,
        temperature: float = 0.9,
        device: Optional[str] = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("transformers and torch are required for rewrite generation") from e
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if device is not None:
            self.model = self.model.to(device)
        self.num_rewrites = int(num_rewrites)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)

    def __call__(self, query: str) -> Tuple[str, ...]:
        prompt = (
            "Paraphrase the following retrieval query without adding facts. "
            "Return one concise paraphrase only.\nQuery: " + str(query) + "\nParaphrase:"
        )
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        outputs = self.model.generate(
            **encoded,
            do_sample=True,
            temperature=self.temperature,
            num_return_sequences=self.num_rewrites,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        prefix = encoded["input_ids"].shape[1]
        texts = [self.tokenizer.decode(row[prefix:], skip_special_tokens=True).strip() for row in outputs]
        return tuple(x for x in texts if x)
