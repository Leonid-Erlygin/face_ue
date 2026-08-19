from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class ToolRoutingClassificationDataset(Dataset):
    """Prepared known-class ToolBench samples for ArcFace/SCF training."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Prepared tool-routing training file is missing: {self.path}. "
                "Run `python experiments/prepare_tool_routing.py build-toolbench` first."
            )
        with self.path.open("r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        if not self.rows:
            raise ValueError(f"No samples in {self.path}")
        bad = [r for r in self.rows if int(r.get("label", -1)) < 0]
        if bad:
            raise ValueError("ArcFace/SCF training data must contain known classes only")
        self.num_classes = 1 + max(int(r["label"]) for r in self.rows)
        counts = {}
        for row in self.rows:
            label = int(row["label"])
            counts[label] = counts.get(label, 0) + 1
        self.class_counts = counts
        self.sample_weights = torch.tensor(
            [1.0 / counts[int(row["label"])] for row in self.rows], dtype=torch.double
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        return {"text": str(row["text"]), "label": int(row["label"])}


class ToolRoutingDataModule(pl.LightningDataModule):
    """Text datamodule following the repository's existing BERT ArcFace interface."""

    def __init__(
        self,
        train_path: str,
        tokenizer_name: str = "bert-base-uncased",
        batch_size: int = 128,
        num_workers: int = 8,
        max_length: int = 192,
        class_balanced: bool = True,
    ):
        super().__init__()
        self.train_path = str(train_path)
        self.tokenizer_name = str(tokenizer_name)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.max_length = int(max_length)
        self.class_balanced = bool(class_balanced)
        self.tokenizer = None
        self.train_dataset = None
        self.num_classes: Optional[int] = None

    def setup(self, stage: Optional[str] = None):
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as e:
                raise ImportError(
                    "ToolRoutingDataModule requires transformers. "
                    "Install requirements-modern-ai.txt before training."
                ) from e
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self.train_dataset is None:
            self.train_dataset = ToolRoutingClassificationDataset(self.train_path)
            self.num_classes = int(self.train_dataset.num_classes)

    def collate_fn(self, batch: List[Dict[str, Any]]):
        tokenized = self.tokenizer(
            [x["text"] for x in batch], padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        inputs = {"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"]}
        labels = torch.tensor([int(x["label"]) for x in batch], dtype=torch.long)
        return inputs, labels

    def train_dataloader(self):
        self.setup("fit")
        sampler = None
        shuffle = True
        if self.class_balanced:
            sampler = WeightedRandomSampler(
                self.train_dataset.sample_weights,
                num_samples=len(self.train_dataset),
                replacement=True,
            )
            shuffle = False
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=shuffle, sampler=sampler,
            drop_last=True, num_workers=self.num_workers, collate_fn=self.collate_fn,
            persistent_workers=self.num_workers > 0,
        )
