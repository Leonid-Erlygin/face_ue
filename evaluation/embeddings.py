import cv2
import numpy as np
from tqdm import tqdm

# from skimage import transform
from sklearn.preprocessing import normalize


def process_embeddings(
    embs,
    embs_f=[],
    use_flip_test=True,
    use_norm_score=False,
    use_detector_score=True,
    face_scores=None,
):
    if use_flip_test and len(embs_f) != 0:
        embs = embs + embs_f
    if use_norm_score:
        embs = normalize(embs)
    if use_detector_score and not np.isnan(face_scores).any():
        # print("Using detection score normalization")
        embs = embs * np.expand_dims(face_scores, -1)
    return embs
