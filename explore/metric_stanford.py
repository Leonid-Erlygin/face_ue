import os
from pathlib import Path
import sys

import torch
import pandas as pd
import numpy as np
import faiss

sys.path.append(".")
from explore.random_model import ResNet9
from explore.stanford_dataset import ffcv_loader_by_df, SPLIT_CLASSES


def generate_embeddings(model, loader):
    embeddings, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            preds = model(x)

            embeddings.extend(model.features.detach().cpu().numpy())
            labels.extend(y.detach().cpu())

    return np.array(embeddings), np.array(labels)


def build_embeddings(base_dir, model_name):
    data_dir = base_dir / "Stanford_Online_Products"
    small_dir = base_dir / "small"
    checkpoint_dir = base_dir / "models"

    checkpoint = torch.load(checkpoint_dir / model_name)
    num_classes = list(checkpoint.values())[-2].shape[0]  # Wow, so magick!
    print(num_classes)
    model = ResNet9(num_classes)
    model.load_state_dict(checkpoint)
    model.eval().cuda()

    test_df = pd.read_csv(
        data_dir / "Ebay_test.txt", delim_whitespace=True, index_col="image_id"
    )
    test_df = test_df[test_df.super_class_id.isin(np.arange(SPLIT_CLASSES) + 1)]
    test_df["labels"] = (test_df.class_id) - 1
    print(len(test_df))

    test_loader = ffcv_loader_by_df(
        test_df, small_dir, "/tmp/ds_test.beton", random_order=False, batch_size=500
    )

    x, y = generate_embeddings(model, test_loader)
    np.save("/tmp/stanford_x.npy", x)
    np.save("/tmp/stanford_y.npy", y)


def recall_at_k(embeddings, labels, k=3, cosine=False):
    d = embeddings.shape[-1]
    if cosine:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
        index = faiss.IndexFlat(d, faiss.METRIC_INNER_PRODUCT)
    else:
        index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    print(index.ntotal)
    distances, indices = index.search(embeddings, k + 1)
    indices = indices[:, 1:]  # as it's the same db / query
    pred_labels = labels[indices]
    recall = np.any(pred_labels == labels[:, None], axis=-1).mean()

    return recall


def multi_recall(embeddings, labels):
    for k in range(10):
        k = 2**k
        print(k)
        print(recall_at_k(embeddings, labels, k, cosine=True))


import matplotlib.pyplot as plt


def recall_with_rejection(embeddings, labels, k=5):
    # uncertainties = torch.norm(predictions, dim=-1).cpu()
    confidences = np.linalg.norm(embeddings, axis=-1)
    print(recall_at_k(embeddings, labels, 4, cosine=True))
    print(confidences)

    splits = np.arange(0, 0.3, 0.025)
    idxs = np.argsort(confidences)
    embeddings = embeddings[idxs]
    labels = labels[idxs]
    recalls = []
    for split in splits:
        start_idx = int(split * len(labels))
        recalls.append(recall_at_k(embeddings[start_idx:], labels[start_idx:], k=k))

    plt.plot(splits, recalls)
    plt.xlabel("Rejection portion")
    plt.ylabel("Recall@3")

    plt.show()


def main():
    base_dir = Path("/home/kirill/data/stanford/")
    build_embeddings(base_dir, "resnet9_arcface.pth")
    embeddings = np.load("/tmp/stanford_x.npy")
    labels = np.load("/tmp/stanford_y.npy")
    print(embeddings.shape)

    # multi_recall(embeddings, labels)
    recall_with_rejection(embeddings, labels, k=3)


if __name__ == "__main__":
    main()
