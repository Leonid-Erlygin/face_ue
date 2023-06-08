import cv2
import numpy as np
from tqdm import tqdm
from skimage import transform
from sklearn.preprocessing import normalize


def face_align_landmark(img, landmark, image_size=(112, 112), method="similar"):
    tform = (
        transform.AffineTransform()
        if method == "affine"
        else transform.SimilarityTransform()
    )
    src = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.729904, 92.2041],
        ],
        dtype=np.float32,
    )
    tform.estimate(landmark, src)
    # ndimage = transform.warp(img, tform.inverse, output_shape=image_size)
    # ndimage = (ndimage * 255).astype(np.uint8)
    if tform.params is not None:
        M = tform.params[0:2, :]
    else:
        raise ValueError
    
    ndimage = cv2.warpAffine(img, M, image_size, borderValue=0.0)
    if len(ndimage.shape) == 2:
        ndimage = np.stack([ndimage, ndimage, ndimage], -1)
    else:
        ndimage = cv2.cvtColor(ndimage, cv2.COLOR_BGR2RGB)
    return ndimage


def get_embeddings(model_interf, img_names, landmarks, batch_size=64, flip=True):
    steps = int(np.ceil(len(img_names) / batch_size))
    embs, embs_f = [], []
    for batch_id in tqdm(
        range(0, len(img_names), batch_size), "Embedding", total=steps
    ):
        batch_imgs, batch_landmarks = (
            img_names[batch_id : batch_id + batch_size],
            landmarks[batch_id : batch_id + batch_size],
        )
        ndimages = [
            face_align_landmark(cv2.imread(img), landmark)
            for img, landmark in zip(batch_imgs, batch_landmarks)
        ]
        ndimages = np.stack(ndimages)
        embs.extend(model_interf(ndimages))
        if flip:
            embs_f.extend(model_interf(ndimages[:, :, ::-1, :]))
    return np.array(embs), np.array(embs_f)


def process_embeddings(
    embs,
    embs_f=[],
    use_flip_test=True,
    use_norm_score=False,
    use_detector_score=True,
    face_scores=None,
):
    print(
        ">>>> process_embeddings: Norm {}, Detect_score {}, Flip {}".format(
            use_norm_score, use_detector_score, use_flip_test
        )
    )
    if use_flip_test and len(embs_f) != 0:
        embs = embs + embs_f
    if use_norm_score:
        embs = normalize(embs)
    if use_detector_score and face_scores is not None:
        embs = embs * np.expand_dims(face_scores, -1)
    return embs

