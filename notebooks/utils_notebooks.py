import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Tuple, List
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pytorch_lightning import LightningModule


def predict_features(
    model: LightningModule, test_dataloader: DataLoader, device: str = "cuda"
) -> Tuple[np.array, np.array]:
    """Transform images and get their embeddings.

    :param model: trained MetricLearningModel
    :param test_dataloader: DataLoader with images to be transformed
    :param device: 'gpu' or 'cuda', if available
    :return a tuple of:
        - numpy array with obtained features
        - true image labels (people id's)
    """
    model.to(device)

    # switch model to 'eval' mode: disable randomness, dropout, etc.
    model.eval()

    predicted_features = []
    image_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_dataloader):
            images = images.to(device)
            features, _ = model(images)
            features = features.detach().cpu().numpy()
            labels = labels.numpy()
            predicted_features.append(features)
            image_labels.append(labels)

    predicted_features = np.concatenate(predicted_features)
    image_labels = np.concatenate(image_labels)

    return predicted_features, image_labels


def compute_distance_and_visualize(
    predicted_features: np.array,
    image_labels: np.array,
    softmax_weights: np.array,
    num_features: int = 2,
) -> List[float]:
    """Compute average distance between embeddings and class centers and visualize 2D results.

    :param predicted_features - model embeddings
    :param image_labels - true image labels
    :param softmax_weights - weights of the classes
    :return - List of average distances for classes
    """
    assert num_features in [
        2,
        3,
    ], f"Cannot visualize feature space with num features = {num_features}."
    assert num_features == len(
        predicted_features[0]
    ), "Desired num_features is not equal to the actual one."

    num_people = len(np.unique(image_labels))
    dists = []
    colors = list(mcolors.TABLEAU_COLORS)[:num_people]

    if num_features == 2:
        plt.figure(figsize=(6, 6))
        for i, (center, color) in enumerate(zip(softmax_weights, colors)):
            points = predicted_features[image_labels == i]

            dists.append(((points - center) ** 2).sum(axis=1).mean().item())

            x, y = [0, center[0]], [0, center[1]]
            plt.plot(x, y, marker="", c=color)
            plt.scatter(points[:, 0], points[:, 1], color=color, s=3)
            if i == (num_people - 1):
                break
        plt.title("Feature space visualization", fontsize=14)
        plt.gca().set_aspect("equal")
        plt.axis("off")
        plt.show()

    else:
        fig = plt.figure(figsize=(6, 6))
        ax = plt.axes(projection="3d")

        u, v = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 50j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)

        ax.plot_wireframe(x, y, z, color="gray", alpha=0.2, rstride=2, cstride=2)

        for i, (center, color) in enumerate(zip(softmax_weights, colors)):
            points = predicted_features[image_labels == i]

            dists.append(((points - center) ** 2).sum(axis=1).mean().item())

            x, y, z = [0, center[0]], [0, center[1]], [0, center[2]]
            ax.plot3D(x, y, z, marker="", c=color)
            ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], color=color, s=3)
            if i == (num_people - 1):
                break
        plt.title("Feature space visualization", fontsize=14)
        plt.gca().set_aspect("equal")
        plt.axis("off")
        plt.show()

    return dists
