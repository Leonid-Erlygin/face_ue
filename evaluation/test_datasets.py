from .data_tools import extract_meta_data, extract_gallery_prob_data
import numpy as np
from pathlib import Path


class FaceRecogntionDataset:
    def __init__(self, dataset_name: str, dataset_path: str) -> None:
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        (
            self.templates,
            self.medias,
            self.p1,
            self.p2,
            self.label,
            _,
            _,
            self.face_scores,
        ) = extract_meta_data(dataset_path, dataset_name)
        (
            self.g1_templates,
            self.g1_ids,
            self.g2_templates,
            self.g2_ids,
            self.probe_templates,
            self.probe_ids,
        ) = extract_gallery_prob_data(dataset_path, dataset_name)


class HQVerifivationDataset:
    def __init__(self, dataset_name: str, dataset_path: str) -> None:
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        issame = np.load(Path(dataset_path) / "meta" / "issame.npy")

        index = np.arange(issame.shape[0])
        self.templates = index
        self.medias = index
        self.p1 = index[0::2]
        self.p2 = index[1::2]
        self.label = issame[0::2]
        self.face_scores = None
        # (
        #     self.templates,
        #     self.medias,
        #     self.p1,
        #     self.p2,
        #     self.label,
        #     _,
        #     _,
        #     self.face_scores,
        # ) = extract_meta_data(dataset_path, dataset_name)
