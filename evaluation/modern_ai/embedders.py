from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


def _file_cache_token(path: str | Path) -> str:
    p = Path(path).resolve()
    st = p.stat()
    return f"{p}:{st.st_size}:{st.st_mtime_ns}"


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


class BaseTextEmbedder:
    model_name: str = "base"
    supports_query_kappa: bool = False

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode_queries(texts)

    def encode_queries_with_kappa(self, texts: Sequence[str]):
        raise NotImplementedError(f"{type(self).__name__} does not provide learned query concentration")


class SentenceTransformerEmbedder(BaseTextEmbedder):
    """SentenceTransformers adapter with asymmetric query/document support."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 64,
        device: Optional[str] = None,
        query_prompt_name: Optional[str] = None,
        document_prompt_name: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "SentenceTransformerEmbedder requires sentence-transformers. "
                "Install requirements-modern-ai.txt."
            ) from e
        self.model_name = str(model_name)
        self.batch_size = int(batch_size)
        self.query_prompt_name = query_prompt_name
        self.document_prompt_name = document_prompt_name
        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=trust_remote_code,
        )

    def _encode(self, texts: Sequence[str], kind: str) -> np.ndarray:
        texts = list(map(str, texts))
        if kind == "query" and hasattr(self.model, "encode_query"):
            arr = self.model.encode_query(
                texts,
                prompt_name=self.query_prompt_name,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > self.batch_size,
            )
        elif kind == "document" and hasattr(self.model, "encode_document"):
            arr = self.model.encode_document(
                texts,
                prompt_name=self.document_prompt_name,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > self.batch_size,
            )
        else:
            arr = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > self.batch_size,
            )
        return l2_normalize(np.asarray(arr))

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "query")

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "document")


class HashingTextEmbedder(BaseTextEmbedder):
    """Dependency-light deterministic embedder for smoke tests and CI.

    It is intentionally not a scientific baseline. The production configs use
    SentenceTransformers. Hashing makes the entire experiment stack testable
    offline.
    """

    def __init__(self, n_features: int = 128, ngram_range=(1, 2)) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self.model_name = f"hashing-{n_features}"
        self.vectorizer = HashingVectorizer(
            n_features=int(n_features),
            alternate_sign=False,
            norm=None,
            ngram_range=tuple(ngram_range),
        )

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        x = self.vectorizer.transform(list(map(str, texts))).toarray()
        return l2_normalize(x)


class RepoArcFaceTextEmbedder(BaseTextEmbedder):
    """Inference adapter for the repository's trained BERT ArcFace backbone."""

    def __init__(
        self,
        backbone_path: str,
        model_name: str = "bert-base-uncased",
        num_features: int = 768,
        bottleneck_dim: int = 768,
        proj_depth: int = 2,
        batch_size: int = 64,
        max_length: int = 192,
        device: Optional[str] = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoTokenizer
            from training.models.lightning_wrappers import BERTEmbedder
        except ImportError as e:
            raise ImportError("RepoArcFaceTextEmbedder requires torch and transformers") from e
        self.torch = torch
        token = _file_cache_token(backbone_path)
        self.model_name = f"repo-arcface:{model_name}:{token}"
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = BERTEmbedder(
            model_name=model_name,
            num_features=int(num_features),
            bottleneck_dim=int(bottleneck_dim),
            proj_depth=int(proj_depth),
            freeze_projection=True,
            freeze_backbone=True,
            backbone_path=str(backbone_path),
        ).to(self.device).eval()

    def _batches(self, texts: Sequence[str]):
        xs = list(map(str, texts))
        for start in range(0, len(xs), self.batch_size):
            tok = self.tokenizer(
                xs[start:start + self.batch_size], padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            yield {k: v.to(self.device) for k, v in tok.items()}

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        features = []
        with self.torch.no_grad():
            for batch in self._batches(texts):
                out = self.backbone(batch)
                features.append(out["feature"].detach().cpu().numpy())
        if not features:
            return np.empty((0, int(self.backbone.num_features)), dtype=np.float64)
        return l2_normalize(np.concatenate(features, axis=0))

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)


class RepoSCFTextEmbedder(BaseTextEmbedder):
    """Inference adapter for the repository's BERT ArcFace + SCF text model.

    This is intentionally implemented from the same ``BERTEmbedder`` and
    ``SCFHead`` classes used during training, so the modern tool-routing
    experiment consumes the native probabilistic text embedding rather than a
    parallel implementation.
    """

    supports_query_kappa = True

    def __init__(
        self,
        backbone_path: str,
        scf_checkpoint_path: str,
        model_name: str = "bert-base-uncased",
        num_features: int = 768,
        bottleneck_dim: int = 768,
        proj_depth: int = 2,
        scf_latent_dim: int = 512,
        batch_size: int = 64,
        max_length: int = 192,
        device: Optional[str] = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoTokenizer
            from training.models.heads import SCFHead
            from training.models.lightning_wrappers import BERTEmbedder
        except ImportError as e:
            raise ImportError("RepoSCFTextEmbedder requires torch and transformers") from e

        self.torch = torch
        backbone_token = _file_cache_token(backbone_path)
        scf_token = _file_cache_token(scf_checkpoint_path)
        self.model_name = f"repo-scf:{model_name}:{backbone_token}:{scf_token}"
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = BERTEmbedder(
            model_name=model_name, num_features=int(num_features),
            bottleneck_dim=int(bottleneck_dim), proj_depth=int(proj_depth),
            freeze_projection=True, freeze_backbone=True, backbone_path=str(backbone_path),
        ).to(self.device).eval()
        self.head = SCFHead(
            convf_dim=int(bottleneck_dim), latent_vector_size=int(scf_latent_dim)
        ).to(self.device)
        checkpoint = torch.load(scf_checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        head_state = {k[len("head."):]: v for k, v in state.items() if k.startswith("head.")}
        if not head_state:
            raise KeyError(f"No head.* SCF parameters found in {scf_checkpoint_path}")
        missing, unexpected = self.head.load_state_dict(head_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"SCF head checkpoint mismatch; missing={missing}, unexpected={unexpected}"
            )

        # SphereConfidenceFace checkpoints contain the frozen BERTEmbedder under
        # ``backbone.*``. Pairing an SCF head with a different ArcFace embedding
        # space invalidates kappa, so detect that mistake before evaluation.
        scf_backbone = {
            k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")
        }
        if scf_backbone:
            native = self.backbone.state_dict()
            if set(scf_backbone) != set(native):
                missing_keys = sorted(set(native) - set(scf_backbone))[:5]
                extra_keys = sorted(set(scf_backbone) - set(native))[:5]
                raise RuntimeError(
                    "SCF/ArcFace backbone key mismatch: "
                    f"missing={missing_keys}, extra={extra_keys}"
                )
            mismatched = [
                key for key, value in native.items()
                if not torch.equal(value.detach().cpu(), scf_backbone[key].detach().cpu())
            ]
            if mismatched:
                raise RuntimeError(
                    "SCF checkpoint was trained with a different ArcFace backbone; "
                    f"first mismatched tensor: {mismatched[0]}"
                )
        self.head.eval()
        del checkpoint, state, scf_backbone

    def _batches(self, texts: Sequence[str]):
        xs = list(map(str, texts))
        for start in range(0, len(xs), self.batch_size):
            batch = xs[start:start + self.batch_size]
            tok = self.tokenizer(
                batch, padding=True, truncation=True, max_length=self.max_length,
                return_tensors="pt",
            )
            yield {k: v.to(self.device) for k, v in tok.items()}

    def _encode(self, texts: Sequence[str], with_kappa: bool):
        features, kappas = [], []
        with self.torch.no_grad():
            for batch in self._batches(texts):
                out = self.backbone(batch)
                features.append(out["feature"].detach().cpu().numpy())
                if with_kappa:
                    log_kappa = self.head(out)
                    kappas.append(self.torch.exp(log_kappa).detach().cpu().numpy())
        emb = l2_normalize(np.concatenate(features, axis=0)) if features else np.empty((0, 0))
        if not with_kappa:
            return emb
        return emb, np.concatenate(kappas, axis=0).astype(np.float64)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, False)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        # ArcFace training includes canonical API descriptions as samples, so the
        # same text encoder is intentionally used for query and gallery sides.
        return self._encode(texts, False)

    def encode_queries_with_kappa(self, texts: Sequence[str]):
        return self._encode(texts, True)


class CachedTextEmbedder(BaseTextEmbedder):
    def __init__(self, base: BaseTextEmbedder, cache_dir: str | Path) -> None:
        self.base = base
        self.model_name = base.model_name
        self.supports_query_kappa = bool(getattr(base, "supports_query_kappa", False))
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, texts: Sequence[str], kind: str) -> str:
        h = hashlib.sha256()
        h.update(self.model_name.encode())
        h.update(kind.encode())
        for text in texts:
            h.update(str(text).encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def _encode(self, texts: Sequence[str], kind: str) -> np.ndarray:
        texts = tuple(map(str, texts))
        key = self._key(texts, kind)
        path = self.cache_dir / f"{key}.npz"
        if path.exists():
            return np.load(path)["embeddings"]
        if kind == "query":
            emb = self.base.encode_queries(texts)
        else:
            emb = self.base.encode_documents(texts)
        np.savez_compressed(path, embeddings=np.asarray(emb, dtype=np.float32))
        return emb

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "query")

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "document")

    def encode_queries_with_kappa(self, texts: Sequence[str]):
        if not self.supports_query_kappa:
            return super().encode_queries_with_kappa(texts)
        texts = tuple(map(str, texts))
        key = self._key(texts, "query_with_kappa")
        path = self.cache_dir / f"{key}.npz"
        if path.exists():
            data = np.load(path)
            return data["embeddings"], data["kappa"]
        emb, kappa = self.base.encode_queries_with_kappa(texts)
        np.savez_compressed(
            path, embeddings=np.asarray(emb, dtype=np.float32),
            kappa=np.asarray(kappa, dtype=np.float32),
        )
        return emb, kappa
