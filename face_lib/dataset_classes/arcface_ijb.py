from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from pathlib import Path
from evaluation.embeddings import face_align_landmark
from evaluation.data_tools import extract_meta_data
from torchvision import transforms


class IJB_aligned_images(Dataset):
    def __init__(
        self, dataset_path: str, dataset_name: str, norm_image: bool = False
    ) -> None:
        super().__init__()
        self.norm_image = norm_image
        img_list_path = (
            Path(dataset_path) / "meta" / f"{dataset_name.lower()}_name_5pts_score.txt"
        )
        img_path = Path(dataset_path) / "loose_crop"
        with open(img_list_path, "r") as ff:
            # 1.jpg 46.060 62.026 87.785 60.323 68.851 77.656 52.162 99.875 86.450 98.648 0.999
            img_records = np.array([ii.strip().split(" ") for ii in ff.readlines()])
        self.img_names = np.array([img_path / ii for ii in img_records[:, 0]])
        self.landmarks = img_records[:, 1:-1].astype("float32").reshape(-1, 5, 2)
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        (
            self.templates,
            self.medias,
            self.p1,
            self.p2,
            self.label,
            self.img_names,
            self.landmarks,
            face_scores,
        ) = extract_meta_data(dataset_path, dataset_name)

    def __getitem__(self, index):
        img = self.img_names[index]
        landmark = self.landmarks[index]
        img = face_align_landmark(cv2.imread(str(img)), landmark)
        if self.norm_image:
            sample = self.transform(img)
        return sample

    def __len__(self):
        return len(self.img_names)


if __name__ == "__main__":
    pass
    ds = IJB_aligned_images("/app/datasets/interpolation", "inter", norm_image=True)
    t = ds[0]
    x = 1
    # ds = IJB_aligned_images("/app/datasets/arcface_ijb/IJBC", "IJBC", norm_image=True)
    # t = ds[0]
    # x = 1
