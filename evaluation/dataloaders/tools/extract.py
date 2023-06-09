import os
import numpy as np
import pandas as pd
from joblib import Memory

from ...template_pooling_strategies import AbstractTemplatePooling


def read_IJB_meta_columns_to_int(file_path, columns, sep=" ", skiprows=0, header=None):
    # meta = np.loadtxt(file_path, skiprows=skiprows, delimiter=sep)
    meta = pd.read_csv(file_path, sep=sep, skiprows=skiprows, header=header).values
    return (meta[:, ii].astype("int") for ii in columns)


@Memory("/app/cache/ijb_data/").cache
def extract_IJB_data_11(data_path, subset):
    if subset == "IJBB":
        media_list_path = os.path.join(data_path, "IJBB/meta/ijbb_face_tid_mid.txt")
        pair_list_path = os.path.join(
            data_path, "IJBB/meta/ijbb_template_pair_label.txt"
        )
        img_path = os.path.join(data_path, "IJBB/loose_crop")
        img_list_path = os.path.join(data_path, "IJBB/meta/ijbb_name_5pts_score.txt")
    else:
        media_list_path = os.path.join(data_path, "IJBC/meta/ijbc_face_tid_mid.txt")
        pair_list_path = os.path.join(
            data_path, "IJBC/meta/ijbc_template_pair_label.txt"
        )
        img_path = os.path.join(data_path, "IJBC/loose_crop")
        img_list_path = os.path.join(data_path, "IJBC/meta/ijbc_name_5pts_score.txt")

    print(">>>> Loading templates and medias...")
    templates, medias = read_IJB_meta_columns_to_int(
        media_list_path, columns=[1, 2]
    )  # ['1.jpg', '1', '69544']
    print(
        "templates: %s, medias: %s, unique templates: %s"
        % (templates.shape, medias.shape, np.unique(templates).shape)
    )
    # templates: (227630,), medias: (227630,), unique templates: (12115,)

    print(">>>> Loading pairs...")
    p1, p2, label = read_IJB_meta_columns_to_int(
        pair_list_path, columns=[0, 1, 2]
    )  # ['1', '11065', '1']
    print("p1: %s, unique p1: %s" % (p1.shape, np.unique(p1).shape))
    print("p2: %s, unique p2: %s" % (p2.shape, np.unique(p2).shape))
    print(
        "label: %s, label value counts: %s"
        % (label.shape, dict(zip(*np.unique(label, return_counts=True))))
    )
    # p1: (8010270,), unique p1: (1845,)
    # p2: (8010270,), unique p2: (10270,) # 10270 + 1845 = 12115 --> np.unique(templates).shape
    # label: (8010270,), label value counts: {0: 8000000, 1: 10270}

    print(">>>> Loading images...")
    with open(img_list_path, "r") as ff:
        # 1.jpg 46.060 62.026 87.785 60.323 68.851 77.656 52.162 99.875 86.450 98.648 0.999
        img_records = np.array([ii.strip().split(" ") for ii in ff.readlines()])

    img_names = np.array([os.path.join(img_path, ii) for ii in img_records[:, 0]])
    landmarks = img_records[:, 1:-1].astype("float32").reshape(-1, 5, 2)
    face_scores = img_records[:, -1].astype("float32")
    print(
        "img_names: %s, landmarks: %s, face_scores: %s"
        % (img_names.shape, landmarks.shape, face_scores.shape)
    )
    # img_names: (227630,), landmarks: (227630, 5, 2), face_scores: (227630,)
    print(
        "face_scores value counts:", dict(zip(*np.histogram(face_scores, bins=9)[::-1]))
    )
    # {0.1: 2515, 0.2: 0, 0.3: 62, 0.4: 94, 0.5: 136, 0.6: 197, 0.7: 291, 0.8: 538, 0.9: 223797}

    return templates, medias, face_scores


@Memory("/app/cache/prob_data/").cache
def extract_gallery_prob_data(data_path, subset):
    if subset == "IJBC":
        meta_dir = os.path.join(data_path, "IJBC/meta")
        gallery_s1_record = os.path.join(meta_dir, "ijbc_1N_gallery_G1.csv")
        gallery_s2_record = os.path.join(meta_dir, "ijbc_1N_gallery_G2.csv")
        probe_mixed_record = os.path.join(meta_dir, "ijbc_1N_probe_mixed.csv")
    else:
        meta_dir = os.path.join(data_path, "IJBB/meta")
        gallery_s1_record = os.path.join(meta_dir, "ijbb_1N_gallery_S1.csv")
        gallery_s2_record = os.path.join(meta_dir, "ijbb_1N_gallery_S2.csv")
        probe_mixed_record = os.path.join(meta_dir, "ijbb_1N_probe_mixed.csv")

    print(">>>> Loading gallery feature...")
    s1_templates, s1_subject_ids = read_IJB_meta_columns_to_int(
        gallery_s1_record, columns=[0, 1], skiprows=1, sep=","
    )
    s2_templates, s2_subject_ids = read_IJB_meta_columns_to_int(
        gallery_s2_record, columns=[0, 1], skiprows=1, sep=","
    )
    print(
        "s1 gallery: %s, ids: %s, unique: %s"
        % (s1_templates.shape, s1_subject_ids.shape, np.unique(s1_templates).shape)
    )
    print(
        "s2 gallery: %s, ids: %s, unique: %s"
        % (s2_templates.shape, s2_subject_ids.shape, np.unique(s2_templates).shape)
    )

    print(">>>> Loading probe feature...")
    probe_mixed_templates, probe_mixed_subject_ids = read_IJB_meta_columns_to_int(
        probe_mixed_record, columns=[0, 1], skiprows=1, sep=","
    )
    print(
        "probe_mixed_templates: %s, unique: %s"
        % (probe_mixed_templates.shape, np.unique(probe_mixed_templates).shape)
    )
    print(
        "probe_mixed_subject_ids: %s, unique: %s"
        % (probe_mixed_subject_ids.shape, np.unique(probe_mixed_subject_ids).shape)
    )

    return (
        s1_templates,
        s1_subject_ids,
        s2_templates,
        s2_subject_ids,
        probe_mixed_templates,
        probe_mixed_subject_ids,
    )
    