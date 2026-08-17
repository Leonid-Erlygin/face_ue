from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from .calibration import IsotonicScoreCalibrator, LogisticFeatureCalibrator
from .embedders import BaseTextEmbedder
from .methods import ModernUncertaintyModel, PosteriorModelConfig
from .metrics import calibration_metrics, grouped_bootstrap_metric, risk_coverage_curve
from .query_uncertainty import query_embeddings_and_kappa
from .statistics import paired_group_bootstrap_difference, stratified_group_split
from .types import RAGRecord


def build_rag_prompt(query: str, contexts: Sequence[str]) -> str:
    joined = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return (
        "Answer the question using only the evidence below. If the evidence is "
        "insufficient, say that you do not know.\n\nEvidence:\n"
        f"{joined}\n\nQuestion: {query}\nAnswer:"
    )


def _safe_binary_metrics(y: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y, dtype=int).reshape(-1)
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(np.unique(y)) < 2:
        return {"auroc": np.nan, "auprc": np.nan}
    return {
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
    }


def _record_groups(records: Sequence[RAGRecord]):
    groups = []
    group_to_idx = {}
    for r in records:
        if r.group_id not in group_to_idx:
            group_to_idx[r.group_id] = len(groups)
            groups.append(r)
    return groups, group_to_idx


def score_rag_evidence(
    records: Sequence[RAGRecord],
    embedder: BaseTextEmbedder,
    model_config: PosteriorModelConfig,
    *,
    rewrites: Optional[Mapping[str, Sequence[str]]] = None,
    default_query_kappa: float = 300.0,
    query_mean_mode: str = "original",
) -> Dict[str, np.ndarray]:
    """Score retrieval-side uncertainty once per unique RAG source and broadcast.

    RAGTruth has several generated responses for one source. Recomputing or fitting
    on response rows would create pseudo-replication.  We therefore compute the
    evidence uncertainty at source granularity and broadcast it only afterwards.

    A transferred/fixed gallery kappa is required: RAGTruth/CRAG context sets do
    not provide the known-vs-unknown calibration population needed to identify it.
    """

    if model_config.gallery_kappa is None:
        raise ValueError(
            "RAG evidence scoring requires a fixed/OSER-transferred gallery_kappa. "
            "Do not fit it using hallucination labels."
        )
    groups, group_to_idx = _record_groups(records)
    ids = [r.group_id for r in groups]
    queries = [r.query for r in groups]
    q, kappa, rbar = query_embeddings_and_kappa(
        ids, queries, embedder, rewrites=rewrites,
        default_kappa=default_query_kappa, mean_mode=query_mean_mode,
    )
    galleries = []
    for r in groups:
        if not r.contexts:
            # Preserve row alignment; zero/empty galleries are marked NaN by the scorer.
            galleries.append(np.empty((0, q.shape[1]), dtype=np.float64))
        else:
            galleries.append(embedder.encode_documents(r.contexts))
    model = ModernUncertaintyModel(model_config)
    group_scores = model.score_variable_galleries(q, kappa, galleries, rbar)
    row_to_group = np.asarray([group_to_idx[r.group_id] for r in records], dtype=int)
    return {name: np.asarray(vals)[row_to_group] for name, vals in group_scores.items()}


def semantic_entropy_from_clusters(cluster_ids: Sequence[int]) -> Dict[str, float]:
    """Discrete semantic uncertainty from sampled meaning clusters.

    This probability-free form is useful when generation transition probabilities
    are unavailable or not comparable across decoding stacks.  Entropy is in nats.
    """

    labels = np.asarray(cluster_ids, dtype=int).reshape(-1)
    if labels.size == 0:
        return {
            "generator_semantic_entropy_discrete": np.nan,
            "generator_semantic_entropy_normalized": np.nan,
            "generator_self_consistency": np.nan,
            "generator_semantic_cluster_count": np.nan,
        }
    _, counts = np.unique(labels, return_counts=True)
    probs = counts.astype(np.float64) / float(labels.size)
    entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-300))))
    denom = np.log(float(labels.size)) if labels.size > 1 else 0.0
    return {
        "generator_semantic_entropy_discrete": entropy,
        "generator_semantic_entropy_normalized": float(entropy / denom) if denom > 0 else 0.0,
        "generator_self_consistency": float(1.0 - np.max(probs)),
        "generator_semantic_cluster_count": float(len(counts)),
    }


def cluster_semantic_samples(
    samples: Sequence[str],
    equivalent: Callable[[str, str], bool],
) -> np.ndarray:
    """Cluster sampled answers using a symmetric semantic-equivalence predicate.

    Pairwise equivalence is converted to connected components.  This makes the
    clustering deterministic and explicitly exposes the semantic equivalence
    relation used by the experiment.
    """

    n = len(samples)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i):
            # Exact duplicates should never require an NLI forward pass.
            if samples[i].strip() == samples[j].strip() or equivalent(samples[i], samples[j]):
                union(i, j)

    roots = [find(i) for i in range(n)]
    root_to_cluster = {}
    labels = []
    for r in roots:
        if r not in root_to_cluster:
            root_to_cluster[r] = len(root_to_cluster)
        labels.append(root_to_cluster[r])
    return np.asarray(labels, dtype=int)


class TransformersConditionalSequenceScorer:
    """Score existing RAG responses under a local causal LM.

    This is useful for RAGTruth when the original generation logits are not
    available. It is explicitly a *surrogate* generator confidence if the scoring
    LM differs from the model that produced the response.
    """

    def __init__(self, model_name: str, device: Optional[str] = None, max_length: int = 4096):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("Install transformers and torch for sequence scoring") from e
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if device is not None:
            self.model = self.model.to(device)
        self.model.eval()
        self.model_name = str(model_name)
        self.max_length = int(max_length)

    def score(self, records: Sequence[RAGRecord]) -> Dict[str, np.ndarray]:
        torch = self.torch
        out = {"generator_mean_nll": [], "generator_mean_entropy": [], "generator_worst_token_nll": []}
        for r in records:
            prompt = build_rag_prompt(r.query, r.contexts)
            prompt_ids = self.tokenizer(prompt, add_special_tokens=True).input_ids
            response_ids = self.tokenizer(r.response, add_special_tokens=False).input_ids
            if len(response_ids) >= self.max_length:
                response_ids = response_ids[-(self.max_length - 1):]
            if not response_ids:
                for k in out: out[k].append(np.nan)
                continue
            # Keep the response; trim context from the left when needed.
            keep_prompt = max(1, self.max_length - len(response_ids))
            prompt_ids = prompt_ids[-keep_prompt:]
            ids = prompt_ids + response_ids
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.model.device)
            with torch.no_grad():
                logits = self.model(input_ids=input_ids).logits[:, :-1, :]
            target = input_ids[:, 1:]
            start = max(len(prompt_ids) - 1, 0)
            resp_logits = logits[:, start:, :]
            resp_target = target[:, start:]
            logp = torch.log_softmax(resp_logits, dim=-1)
            tok = torch.gather(logp, -1, resp_target[..., None]).squeeze(-1)
            p = torch.softmax(resp_logits, dim=-1)
            ent = -torch.sum(p * logp, dim=-1)
            out["generator_mean_nll"].append(float((-tok.mean()).cpu()))
            out["generator_mean_entropy"].append(float(ent.mean().cpu()))
            out["generator_worst_token_nll"].append(float((-tok.min()).cpu()))  # larger = more uncertain
        return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


class TransformersRAGGenerator:
    """Local-HF RAG generator returning uncertainty from generation logits."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_input_tokens: int = 4096,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError("Install transformers and torch for RAG generation") from e
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if device is not None:
            self.model = self.model.to(device)
        self.model.eval()
        self.model_name = str(model_name)
        # Prompt ends with the question/instruction; preserve that end under truncation.
        self.tokenizer.truncation_side = "left"
        self.max_input_tokens = int(max_input_tokens)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)

    def generate_one(self, query: str, contexts: Sequence[str]) -> tuple[str, Dict[str, float]]:
        torch = self.torch
        prompt = build_rag_prompt(query, contexts)
        enc = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if self.temperature > 0:
            kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            kwargs.update(do_sample=False)
        with torch.no_grad():
            outputs = self.model.generate(**enc, **kwargs)
        prefix = enc["input_ids"].shape[1]
        generated = outputs.sequences[:, prefix:]
        response = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        if not outputs.scores:
            return response, {
                "generator_mean_nll": np.nan,
                "generator_mean_entropy": np.nan,
                "generator_worst_token_nll": np.nan,
            }
        # Official Transformers API aligns transition scores with generated tokens.
        transition = self.model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            getattr(outputs, "beam_indices", None),
            normalize_logits=True,
        )
        # compute_transition_scores may include only generated transitions depending
        # on decoding strategy; keep the last number of generated tokens.
        trans = transition[0, -generated.shape[1]:]
        entropies = []
        for step_logits in outputs.scores:
            lp = torch.log_softmax(step_logits[0], dim=-1)
            pp = torch.softmax(step_logits[0], dim=-1)
            entropies.append(float((-torch.sum(pp * lp)).cpu()))
        return response, {
            "generator_mean_nll": float((-trans.mean()).cpu()) if trans.numel() else np.nan,
            "generator_mean_entropy": float(np.mean(entropies)) if entropies else np.nan,
            "generator_worst_token_nll": float((-trans.min()).cpu()) if trans.numel() else np.nan,
        }

    def generate(self, records: Sequence[RAGRecord]) -> tuple[list[str], Dict[str, np.ndarray]]:
        responses = []
        feats = {"generator_mean_nll": [], "generator_mean_entropy": [], "generator_worst_token_nll": []}
        for r in records:
            text, f = self.generate_one(r.query, r.contexts)
            responses.append(text)
            for k in feats: feats[k].append(f[k])
        return responses, {k: np.asarray(v, dtype=np.float64) for k, v in feats.items()}


class TransformersSemanticEntropyScorer:
    """Sample free-form RAG answers and measure uncertainty over their meanings.

    Semantic equivalence is defined conservatively as *bidirectional* NLI
    entailment above ``entailment_threshold``.  The primary feature is discrete
    semantic entropy, so it remains well-defined without relying on sequence
    likelihoods.  The class is intentionally optional because it requires several
    sampled generations and O(S^2) NLI comparisons per prompt.
    """

    def __init__(
        self,
        generator: TransformersRAGGenerator,
        nli_model_name: str,
        *,
        nli_device: Optional[str] = None,
        num_samples: int = 10,
        temperature: float = 1.0,
        top_p: float = 1.0,
        entailment_threshold: float = 0.5,
        entailment_label_id: Optional[int] = None,
        max_nli_tokens: int = 512,
    ):
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            raise ImportError("Install transformers and torch for semantic entropy") from e
        if int(num_samples) < 2:
            raise ValueError("Semantic entropy requires at least two sampled generations")
        if float(temperature) <= 0:
            raise ValueError("Semantic entropy sampling requires temperature > 0")
        self.generator = generator
        self.torch = generator.torch
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        if nli_device is not None:
            self.nli_model = self.nli_model.to(nli_device)
        self.nli_model.eval()
        self.nli_model_name = str(nli_model_name)
        self.num_samples = int(num_samples)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.entailment_threshold = float(entailment_threshold)
        self.max_nli_tokens = int(max_nli_tokens)

        if entailment_label_id is None:
            id2label = getattr(self.nli_model.config, "id2label", {}) or {}
            candidates = [
                int(idx) for idx, label in id2label.items()
                if "entail" in str(label).lower()
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "Could not infer the NLI entailment class unambiguously. "
                    "Set generator.semantic_entropy.entailment_label_id explicitly."
                )
            entailment_label_id = candidates[0]
        self.entailment_label_id = int(entailment_label_id)

    @property
    def nli_device(self):
        return next(self.nli_model.parameters()).device

    def _entailment_probability(self, premise: str, hypothesis: str) -> float:
        torch = self.torch
        enc = self.nli_tokenizer(
            premise, hypothesis, return_tensors="pt", truncation=True,
            max_length=self.max_nli_tokens,
        )
        enc = {k: v.to(self.nli_device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.nli_model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1)
        return float(probs[self.entailment_label_id].cpu())

    def _equivalent(self, a: str, b: str) -> bool:
        # Mutual entailment prevents one answer that merely contains another from
        # being treated as the same proposition.
        return (
            self._entailment_probability(a, b) >= self.entailment_threshold
            and self._entailment_probability(b, a) >= self.entailment_threshold
        )

    def sample_answers(self, query: str, contexts: Sequence[str]) -> list[str]:
        torch = self.torch
        prompt = build_rag_prompt(query, contexts)
        tok = self.generator.tokenizer
        enc = tok(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.generator.max_input_tokens,
        )
        enc = {k: v.to(self.generator.model.device) for k, v in enc.items()}
        kwargs = dict(
            max_new_tokens=self.generator.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            num_return_sequences=self.num_samples,
            pad_token_id=tok.eos_token_id,
        )
        with torch.no_grad():
            out = self.generator.model.generate(**enc, **kwargs)
        prefix = enc["input_ids"].shape[1]
        generated = out[:, prefix:]
        return [tok.decode(row, skip_special_tokens=True).strip() for row in generated]

    def score_one(self, query: str, contexts: Sequence[str]) -> Dict[str, float]:
        samples = self.sample_answers(query, contexts)
        labels = cluster_semantic_samples(samples, self._equivalent)
        return semantic_entropy_from_clusters(labels)

    def score(
        self, records: Sequence[RAGRecord], *, group_by_source: bool = False
    ) -> Dict[str, np.ndarray]:
        names = list(semantic_entropy_from_clusters([0, 1]).keys())
        if group_by_source:
            groups, group_to_idx = _record_groups(records)
            grouped = self.score(groups, group_by_source=False)
            row_to_group = np.asarray([group_to_idx[r.group_id] for r in records], dtype=int)
            return {name: np.asarray(vals)[row_to_group] for name, vals in grouped.items()}

        out = {name: [] for name in names}
        for r in records:
            feats = self.score_one(r.query, r.contexts)
            for name in names:
                out[name].append(feats[name])
        return {name: np.asarray(vals, dtype=np.float64) for name, vals in out.items()}


def save_generation_cache(
    path: str | Path,
    records: Sequence[RAGRecord],
    responses: Sequence[str],
    features: Mapping[str, np.ndarray],
) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(records):
            row = {"id": r.record_id, "response": responses[i]}
            for name, vals in features.items(): row[name] = float(np.asarray(vals)[i])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_feature_cache(
    path: str | Path,
    records: Sequence[RAGRecord],
    features: Mapping[str, np.ndarray],
) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(records):
            row = {"id": r.record_id}
            for name, vals in features.items():
                row[name] = float(np.asarray(vals)[i])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_feature_cache(
    path: str | Path, records: Sequence[RAGRecord]
) -> Dict[str, np.ndarray]:
    rows = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line); rows[str(row["id"])] = row
    missing = [str(r.record_id) for r in records if str(r.record_id) not in rows]
    if missing:
        raise KeyError(f"Feature cache is missing {len(missing)} record ids; first={missing[0]}")
    names = sorted({k for row in rows.values() for k in row if k != "id"})
    return {
        name: np.asarray([float(rows[str(r.record_id)].get(name, np.nan)) for r in records], dtype=np.float64)
        for name in names
    }


def load_generation_cache(path: str | Path, records: Sequence[RAGRecord]):
    rows = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line); rows[str(row["id"])] = row
    responses = []
    names = sorted({k for row in rows.values() for k in row if k.startswith("generator_")})
    feats = {k: [] for k in names}
    for r in records:
        row = rows[str(r.record_id)]
        responses.append(str(row.get("response", "")))
        for k in names: feats[k].append(float(row.get(k, np.nan)))
    return responses, {k: np.asarray(v, dtype=np.float64) for k, v in feats.items()}


def _normalize_answer(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
    return " ".join(text.split())


def reference_answer_error(response: str, references: Sequence[str]) -> bool:
    """Conservative *heuristic* CRAG evaluator for smoke/debug use only."""
    pred = _normalize_answer(response)
    refs = [_normalize_answer(x) for x in references if _normalize_answer(x)]
    if not pred or not refs:
        return True
    return not any(pred == r or r in pred for r in refs)


def load_binary_judgments_jsonl(
    path: str | Path,
    *,
    id_field: str = "id",
    error_field: str = "is_error",
) -> Dict[str, bool]:
    out = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line); out[str(row[id_field])] = bool(row[error_field])
    return out


def _fit_and_eval_feature_models(
    y: np.ndarray,
    group_ids: Sequence[str],
    retrieval: Mapping[str, np.ndarray],
    generator: Optional[Mapping[str, np.ndarray]],
    validation_fraction: float,
    seed: int,
    n_boot: int,
):
    val_idx, test_idx = stratified_group_split(
        y, group_ids, validation_fraction=validation_fraction, seed=seed
    )
    if not len(val_idx) or not len(test_idx):
        raise ValueError("Insufficient groups for calibration/test split")

    models = {}
    predictions = {}

    # Individual theoretically motivated methods.
    for name in ["mprisk", "mprisk_no_ns", "galue_entropy", "max_similarity"]:
        if name not in retrieval: continue
        cal = IsotonicScoreCalibrator().fit(retrieval[name][val_idx], y[val_idx])
        predictions[name] = cal.predict_proba(retrieval[name][test_idx])
        models[name] = cal
    holue_x = np.column_stack([retrieval["holue_kl_1"], retrieval["holue_kl_2"]])
    holue = LogisticFeatureCalibrator().fit(holue_x[val_idx], y[val_idx])
    predictions["holue"] = holue.predict_proba(holue_x[test_idx]); models["holue"] = holue

    retrieval_names = ["mprisk", "holue_kl_1", "holue_kl_2", "max_similarity"]
    rx = np.column_stack([retrieval[n] for n in retrieval_names])
    rmodel = LogisticFeatureCalibrator().fit(rx[val_idx], y[val_idx])
    predictions["retrieval_combined"] = rmodel.predict_proba(rx[test_idx]); models["retrieval_combined"] = rmodel

    generator_names = []
    if generator:
        generator_names = [k for k, v in generator.items() if np.any(np.isfinite(v))]
        if generator_names:
            # Report every generator-side uncertainty measure on its own before
            # allowing a learned combination.  All exported generator features
            # use the convention "larger = more uncertain".
            for name in generator_names:
                gcal = IsotonicScoreCalibrator().fit(generator[name][val_idx], y[val_idx])
                predictions[name] = gcal.predict_proba(generator[name][test_idx])
                models[name] = gcal
            gx = np.column_stack([generator[n] for n in generator_names])
            gmodel = LogisticFeatureCalibrator().fit(gx[val_idx], y[val_idx])
            predictions["generator_only"] = gmodel.predict_proba(gx[test_idx]); models["generator_only"] = gmodel
            hx = np.column_stack([rx, gx])
            hmodel = LogisticFeatureCalibrator().fit(hx[val_idx], y[val_idx])
            predictions["hybrid"] = hmodel.predict_proba(hx[test_idx]); models["hybrid"] = hmodel

    metrics = {}
    test_groups = np.asarray(group_ids, dtype=object)[test_idx]
    for name, pred in predictions.items():
        ok = np.isfinite(pred)
        if not np.any(ok): continue
        yy = y[test_idx][ok]
        pp = np.asarray(pred)[ok]
        gg = test_groups[ok]
        rc = risk_coverage_curve(yy, pp)
        selective = {}
        for coverage in (0.5, 0.8, 0.9):
            idx_cov = int(np.argmin(np.abs(np.asarray(rc["coverage"]) - coverage)))
            selective[f"risk_at_{int(coverage*100)}pct_coverage"] = float(np.asarray(rc["risk"])[idx_cov])
        metrics[name] = {
            **_safe_binary_metrics(yy, pp),
            "aurc": float(rc["aurc"]),
            **selective,
            **{f"calibration_{k}": v for k, v in calibration_metrics(yy, pp).items()},
            "auroc_cluster_bootstrap": grouped_bootstrap_metric(
                yy, pp, gg, metric="auroc", n_boot=n_boot, seed=seed
            ),
            "auprc_cluster_bootstrap": grouped_bootstrap_metric(
                yy, pp, gg, metric="auprc", n_boot=n_boot, seed=seed
            ),
        }
    comparisons = {}
    if "generator_only" in predictions and "hybrid" in predictions:
        comparisons["hybrid_minus_generator"] = paired_group_bootstrap_difference(
            y[test_idx], predictions["hybrid"], predictions["generator_only"],
            test_groups, metric="auroc", n_boot=n_boot, seed=seed,
        )
    return {
        "validation_indices": val_idx,
        "test_indices": test_idx,
        "predictions": predictions,
        "models": models,
        "metrics": metrics,
        "comparisons": comparisons,
        "generator_features": generator_names,
    }


def run_ragtruth_experiment(
    records: Sequence[RAGRecord],
    embedder: BaseTextEmbedder,
    model_config: PosteriorModelConfig,
    *,
    generator_features: Optional[Mapping[str, np.ndarray]] = None,
    target: str = "hallucination",
    rewrites: Optional[Mapping[str, Sequence[str]]] = None,
    default_query_kappa: float = 300.0,
    query_mean_mode: str = "original",
    validation_fraction: float = 0.3,
    seed: int = 777,
    n_boot: int = 1000,
    output_dir: Optional[str | Path] = None,
) -> dict:
    if target not in {"hallucination", "error"}:
        raise ValueError("target must be hallucination or error")
    records = list(records)
    y = np.asarray([
        r.is_hallucination if target == "hallucination" else r.is_error for r in records
    ], dtype=int)
    retrieval = score_rag_evidence(
        records, embedder, model_config, rewrites=rewrites,
        default_query_kappa=default_query_kappa, query_mean_mode=query_mean_mode,
    )
    groups = [r.group_id for r in records]
    fit = _fit_and_eval_feature_models(
        y, groups, retrieval, generator_features,
        validation_fraction, seed, n_boot,
    )
    summary = {
        "experiment": "ragtruth_hallucination_risk",
        "target": target,
        "num_responses": len(records),
        "num_sources": len(set(groups)),
        "positive_rate": float(np.mean(y)) if len(y) else np.nan,
        "embedder": getattr(embedder, "model_name", type(embedder).__name__),
        "gallery_kappa_transferred": float(model_config.gallery_kappa),
        "generator_features": fit["generator_features"],
        "metrics": fit["metrics"],
        "paired_comparisons": fit["comparisons"],
        "note": (
            "Retrieval features are source-level and duplicated across responses. "
            "All splitting/bootstrap is clustered by source_id to avoid pseudoreplication."
        ),
    }
    if output_dir is not None:
        root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
        (root / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
        test_idx = fit["test_indices"]
        with (root / "per_response.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["id", "source_id", "target"] + list(retrieval.keys()) + list(fit["predictions"].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
            for local, i in enumerate(test_idx):
                row = {"id": records[i].record_id, "source_id": records[i].group_id, "target": int(y[i])}
                row.update({k: float(v[i]) for k, v in retrieval.items()})
                for k, v in fit["predictions"].items(): row[k] = float(v[local])
                w.writerow(row)
    return {"summary": summary, "retrieval_scores": retrieval, **fit}


def run_end_to_end_rag_experiment(
    records: Sequence[RAGRecord],
    embedder: BaseTextEmbedder,
    model_config: PosteriorModelConfig,
    *,
    responses: Sequence[str],
    generator_features: Mapping[str, np.ndarray],
    judgments: Optional[Mapping[str, bool]] = None,
    allow_reference_heuristic: bool = False,
    validation_fraction: float = 0.3,
    seed: int = 777,
    n_boot: int = 1000,
    output_dir: Optional[str | Path] = None,
) -> dict:
    """Evaluate whether retrieval risk adds information beyond generator uncertainty.

    Use official/external judgments whenever possible.  The reference-answer
    heuristic exists only to exercise the pipeline and is clearly marked so it
    cannot accidentally be reported as the official CRAG metric.
    """
    records = list(records); responses = list(map(str, responses))
    if len(records) != len(responses): raise ValueError("responses must align with records")
    evaluator = "external_judgments"
    if judgments is not None:
        missing = [r.record_id for r in records if r.record_id not in judgments]
        if missing: raise ValueError(f"Missing judgments for {len(missing)} records")
        y = np.asarray([bool(judgments[r.record_id]) for r in records], dtype=int)
    else:
        if not allow_reference_heuristic:
            raise ValueError(
                "End-to-end RAG requires external/official binary error judgments. "
                "Set allow_reference_heuristic=True only for smoke/debug runs."
            )
        evaluator = "reference_substring_heuristic_NOT_OFFICIAL"
        y = np.asarray([
            reference_answer_error(resp, r.reference_answers)
            for r, resp in zip(records, responses)
        ], dtype=int)
    retrieval = score_rag_evidence(records, embedder, model_config)
    groups = [r.group_id for r in records]
    fit = _fit_and_eval_feature_models(
        y, groups, retrieval, generator_features,
        validation_fraction, seed, n_boot,
    )
    summary = {
        "experiment": "end_to_end_rag_incremental_risk",
        "evaluator": evaluator,
        "num_examples": len(records),
        "error_rate": float(np.mean(y)) if len(y) else np.nan,
        "metrics": fit["metrics"],
        "paired_comparisons": fit["comparisons"],
        "primary_hypothesis": "hybrid retrieval+generation risk improves held-out error prediction over generator-only uncertainty",
    }
    if output_dir is not None:
        root=Path(output_dir); root.mkdir(parents=True,exist_ok=True)
        (root/"summary.json").write_text(json.dumps(summary,indent=2,default=float),encoding="utf-8")
    return {"summary": summary, "retrieval_scores": retrieval, "labels": y, **fit}
