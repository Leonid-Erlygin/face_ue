from .data_tools import extract_meta_data, extract_gallery_prob_data


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
