import torch

# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import cv2

# import albumentations as A
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import mxnet as mx
import numbers
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
import pandas as pd


import sys

# sys.path.append("/app/sandbox/happy_whale/kaggle-happywhale-1st-place")
# from config.config import load_config
# from src.dataset import load_df


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, test=False, num_classes=0):
        """
        ArcFace loader
        https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/dataset.py
        """
        super(MXFaceDataset, self).__init__()
        self.num_classes = num_classes
        self.test = test
        if self.test:
            self.transform = v2.Compose(
                [
                    v2.ToPILImage(),
                    v2.ToTensor(),
                    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.ToPILImage(),
                    v2.RandomHorizontalFlip(),
                    v2.ToTensor(),
                    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    # v2.RandomApply(
                    #     [v2.RandomChoice(
                    #         transforms = [
                    #                       v2.GaussianBlur(kernel_size=(21, 21), sigma=(1.5, 1.5)),
                    #                       v2.GaussianBlur(kernel_size=(21, 21), sigma=(2.3, 2.3)),
                    #                       v2.GaussianBlur(kernel_size=(21, 21), sigma=(3.0, 3.0)),
                    #                       v2.GaussianBlur(kernel_size=(21, 21), sigma=(4.0, 4.0)),
                    #                       v2.GaussianBlur(kernel_size=(21, 21), sigma=(6.0, 6.0)),
                    #                       v2.GaussianBlur(kernel_size=(21, 21), sigma=(10.0, 10.0)),
                    #                       v2.GaussianBlur(kernel_size=(21, 21), sigma=(15.0, 15.0)),
                    #                       v2.GaussianBlur(kernel_size=(21, 21), sigma=(20.0, 20.0))],
                    #         p = [1/8 for _ in range(0, 8)]
                    #     )],
                    #     p=0.3,
                    # ),
                    # v2.RandomApply(
                    #     torch.nn.ModuleList([v2.ElasticTransform(alpha=250.0)]), p=0.05
                    # ),
                    # v2.RandomApply(torch.nn.ModuleList([v2.AugMix()]), p=0.05),
                ]
            )

            # load pictures
        self.root_dir = root_dir
        path_imgrec = os.path.join(root_dir, "train.rec")
        path_imgidx = os.path.join(root_dir, "train.idx")
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)

        self.imgidx = np.array(range(1, int(header.label[0])))

        # load or create labels
        labels_path = Path(root_dir) / "labels.npy"
        if labels_path.is_file():
            self.labels = np.load(labels_path)
        else:
            print("Listing labels...")
            labels = []
            for i in range(len(self.imgidx)):
                idx = self.imgidx[i]
                s = self.imgrec.read_idx(idx)
                header, img = mx.recordio.unpack(s)
                label = header.label
                labels.append(int(label))
            self.labels = np.array(labels)
            # save labels
            np.save(labels_path, self.labels)

        if num_classes > 0:
            seed = 0
            min_size = 30
            image_idx_path = (
                Path(root_dir)
                / f"image_idx_{num_classes}-classes_{seed}-seed_{min_size}-min-class-size.npy"
            )
            self.image_label_path = (
                Path(root_dir)
                / f"image_label_{num_classes}-classes_{seed}-seed_{min_size}-min-class-size.npy"
            )
            if image_idx_path.is_file():
                self.imgidx = np.load(image_idx_path)
                self.labels = np.load(self.image_label_path)
            else:
                print(f"Listing images of {num_classes} random classes...")
                rng = np.random.default_rng(seed)
                unique_labels, unique_counts = np.unique(
                    self.labels, return_counts=True
                )
                unique_labels_thresh = unique_labels[unique_counts > min_size]
                # selected_classes = rng.choice(
                #     unique_labels_thresh, num_classes, replace=False
                # )
                selected_classes = np.sort(
                    rng.choice(unique_labels_thresh, num_classes, replace=False)
                )
                imgidx_short = []
                labels_short = []
                for selected_class in tqdm(selected_classes):
                    index = self.labels == selected_class
                    imgidx_short.extend(list(self.imgidx[index]))
                    labels_short.extend(list(self.labels[index]))
                self.imgidx = np.array(imgidx_short)
                self.labels = np.array(labels_short)
                np.save(image_idx_path, self.imgidx)
                np.save(self.image_label_path, self.labels)

    def create_identification_meta(
        self, identification_ds_path: Path, gallery_size: int
    ):
        # seed = 0
        # rng = np.random.default_rng(seed)
        identification_ds_path.mkdir(exist_ok=True)
        meta_path = identification_ds_path / "meta"
        meta_path.mkdir(exist_ok=True)
        embeddings_path = identification_ds_path / "embeddings"
        embeddings_path.mkdir(exist_ok=True)

        num_probe_templates = 4
        mids = np.arange(len(self.labels))
        names = np.zeros_like(mids)
        tids = []
        tids_probe = []
        tids_gallery = []
        sids = []
        sids_probe = []
        sids_gallery = []
        i = 0

        (class_ids, poses, counts) = np.unique(
            self.labels, return_index=True, return_counts=True
        )
        for class_id, pos, count in zip(class_ids, poses, counts):
            sids.extend([class_id] * count)
            sids_probe.extend(
                [class_id] * (count // (num_probe_templates + 1)) * num_probe_templates
            )
            if i < gallery_size:
                sids_gallery.extend(
                    [class_id]
                    * (
                        count % (num_probe_templates + 1)
                        + count // (num_probe_templates + 1)
                    )
                )
            for j in range(num_probe_templates):
                probe_templates = [i * (num_probe_templates + 1) + j] * (
                    count // (num_probe_templates + 1)
                )
                tids.extend(probe_templates)
                tids_probe.extend(probe_templates)
            gallery_templates = [
                i * (num_probe_templates + 1) + num_probe_templates
            ] * (count % (num_probe_templates + 1) + count // (num_probe_templates + 1))
            tids.extend(gallery_templates)
            if i < gallery_size:
                tids_gallery.extend(gallery_templates)
            i += 1

        # assert len(tids) == len(tids_probe) + len(tids_gallery)
        # assert len(sids) == len(sids_probe) + len(sids_gallery)
        assert len(np.unique(sids_gallery)) == gallery_size
        out_file_tid_mid = meta_path / Path("ms1m_face_tid_mid.txt")
        with open(out_file_tid_mid, "w") as fd:
            for name, tid, sid, mid in zip(names, tids, sids, mids):
                fd.write(f"{name} {tid} {mid} {sid}\n")

        out_file_probe = meta_path / Path("ms1m_1N_probe_mixed.csv")
        out_file_gallery = meta_path / Path("ms1m_1N_gallery_G1.csv")

        probe = pd.DataFrame(
            {
                "TEMPLATE_ID": tids_probe,
                "SUBJECT_ID": sids_probe,
                "FILENAME": np.zeros_like(tids_probe),
            }
        )
        gallery = pd.DataFrame(
            {
                "TEMPLATE_ID": tids_gallery,
                "SUBJECT_ID": sids_gallery,
                "FILENAME": np.zeros_like(tids_gallery),
            }
        )

        probe.to_csv(out_file_probe, sep=",", index=False)
        gallery.to_csv(out_file_gallery, sep=",", index=False)

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        if self.test:
            return sample
        else:
            return sample, label

    def __len__(self):
        return len(self.imgidx)


class WhaleDataset(Dataset):
    def __init__(self, config_path: str, image_dir: str, test: bool = False):
        super().__init__()
        val_bbox_name = "fullbody"
        cfg = load_config(
            config_path,
            "sandbox/happy_whale/kaggle-happywhale-1st-place/config/default.yaml",
        )
        df = load_df(image_dir, cfg, "train.csv", True)
        self.index = df.index
        self.x_paths = np.array(df.image)
        self.ids = (
            np.array(df.individual_id, dtype=int)
            if hasattr(df, "individual_id")
            else np.full(len(df), -1)
        )
        self.species = (
            np.array(df.species, dtype=int)
            if hasattr(df, "species")
            else np.full(len(df), -1)
        )
        self.cfg = cfg
        self.image_dir = f"{image_dir}/train_images"
        self.df = df
        self.val_bbox_name = val_bbox_name
        self.test = test
        self.data_aug = not test
        augments = []
        if self.data_aug:
            aug = cfg.aug
            augments = [
                A.Affine(
                    rotate=(-aug.rotate, aug.rotate),
                    translate_percent=(0.0, aug.translate),
                    shear=(-aug.shear, aug.shear),
                    p=aug.p_affine,
                ),
                A.RandomResizedCrop(
                    self.cfg.image_size[0],
                    self.cfg.image_size[1],
                    scale=(aug.crop_scale, 1.0),
                    ratio=(aug.crop_l, aug.crop_r),
                ),
                A.ToGray(p=aug.p_gray),
                A.GaussianBlur(blur_limit=(3, 7), p=aug.p_blur),
                A.GaussNoise(p=aug.p_noise),
                A.Downscale(scale_min=0.5, scale_max=0.5, p=aug.p_downscale),
                A.RandomGridShuffle(grid=(2, 2), p=aug.p_shuffle),
                A.Posterize(p=aug.p_posterize),
                A.RandomBrightnessContrast(p=aug.p_bright_contrast),
                A.CoarseDropout(p=aug.p_cutout),
                A.RandomSnow(p=aug.p_snow),
                A.RandomRain(p=aug.p_rain),
                A.HorizontalFlip(p=0.5),
            ]
        augments.append(
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        augments.append(ToTensorV2())  # HWC to CHW
        self.transform = A.Compose(augments)

    def __len__(self):
        return len(self.ids)

    def get_original_image(self, i: int):
        bgr = cv2.imread(f"{self.image_dir}/{self.x_paths[i]}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def __getitem__(self, i: int):
        image = self.get_original_image(i)
        # crop
        if self.data_aug:
            bbox_name = np.random.choice(
                list(self.cfg.bboxes.keys()), p=list(self.cfg.bboxes.values())
            )
        else:
            bbox_name = self.val_bbox_name
        bbox = None if bbox_name == "none" else self.df[bbox_name].iloc[i]
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            image = image[ymin:ymax, xmin:xmax]
        # resize
        image = cv2.resize(image, self.cfg.image_size, interpolation=cv2.INTER_CUBIC)
        # data augmentation
        augmented = self.transform(image=image)["image"]
        if self.test:
            return augmented
        else:
            return augmented, self.ids[i]

    # {
    #         "original_index": self.index[i],
    #         "image": augmented,
    #         "label": self.ids[i],
    #         "label_species": self.species[i],
    #     }


class UncertaintyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset,
        predict_dataset: Dataset,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.predict_dataset = predict_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # Assign train/predict datasets for use in dataloaders
        if stage == "fit":
            pass
            # self.ms1m_dataset = MXFaceDataset(self.data_train_dir)
            # self.ms1m_dataset = torch.utils.data.Subset(self.ms1m_dataset, np.random.choice(len(self.ms1m_dataset), 5000, replace=False))

        if stage == "predict":
            pass
            # self.ijb_dataset = IJB_aligned_images(self.data_predict_dir, self.data_predict_subset)
            # self.predict_dataset = torch.utils.data.Subset(self.predict_dataset, np.random.choice(len(self.predict_dataset), 5000, replace=False))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    num_classes = 3531
    ds = MXFaceDataset("/app/datasets/ms1m/", True, num_classes)
    t = ds[0]
    gallery_size = 1772
    # ds.create_identification_meta(Path("/app/datasets/ms1m_ident"), gallery_size)
