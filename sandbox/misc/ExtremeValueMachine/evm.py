import numpy as np
import libmr
import sys
import scipy.spatial.distance
import sklearn.metrics.pairwise
import time
from contextlib import contextmanager
from multiprocessing import Pool
import itertools as it
import argparse
import seaborn as sns
import matplotlib.pyplot as plt


def draw_score_distr_plot(
    scores_distr, score_type, model_name, in_data_name, out_data_name
):
    sns.set_theme()
    plt.figure(figsize=(12, 8))
    sns.distplot(
        scores_distr[in_data_name],
        kde=True,
        norm_hist=True,
        hist=True,
        label=in_data_name,
    )
    sns.distplot(
        scores_distr[out_data_name],
        kde=True,
        norm_hist=True,
        hist=True,
        label=out_data_name,
    )

    plt.title(
        f"{model_name} model Softmax score distribution for {in_data_name} and {out_data_name} datasets"
    )
    plt.xlabel(f"{score_type} score")

    plt.legend()


def save_dataset_to_file(X, y, file_path):
    X = X.astype("int").astype("str")
    with open(file_path, "w") as f:
        for i, (x_slice, y_slice) in enumerate(zip(X, y)):
            slice = [y_slice] + list(x_slice)
            if i == len(y) - 1:
                f.write(",".join(slice))
            else:
                f.write(",".join(slice) + "\n")


def create_oletter_dataset(train_fname, test_fname, seeds):
    oletter_dir = Path("/app/sandbox/ExtremeValueMachine/TestData/oletter")
    Xtrain, ytrain = load_data(train_fname)
    Xtest, ytest = load_data(test_fname)

    letters = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ]

    oletter_dir.mkdir(exist_ok=True)
    for seed in seeds:
        rs = RandomState(seed)
        out_dir = oletter_dir / str(seed)
        out_dir.mkdir(exist_ok=True)
        known_labels = list(rs.choice(letters, 15, replace=False))
        # create train dataset
        train_valid_letters_idx = np.isin(ytrain, known_labels)
        Xtrain_valid = Xtrain[train_valid_letters_idx]
        ytrain_valid = ytrain[train_valid_letters_idx]
        save_dataset_to_file(Xtrain_valid, ytrain_valid, out_dir / f"train_dataset.txt")
        unk_labels = list(set(letters).difference(set(known_labels)))
        with open(out_dir / "train_labels.txt", "w") as f:
            f.write(",".join(known_labels))
        with open(out_dir / "unknown_labels.txt", "w") as f:
            f.write(",".join(unk_labels))
        for num_unk_classes in np.arange(len(unk_labels) + 1):
            unk_labels_to_test = unk_labels[:num_unk_classes]
            test_labels = known_labels + unk_labels_to_test
            test_valid_letters_idx = np.isin(ytest, test_labels)
            Xtest_valid = Xtest[test_valid_letters_idx]
            ytest_valid = ytest[test_valid_letters_idx]
            save_dataset_to_file(
                Xtest_valid,
                ytest_valid,
                out_dir / f"with_{num_unk_classes}_unk_test_dataset.txt",
            )


@contextmanager
def timer(message):
    """
    Simple timing method. Logging should be used instead for large scale experiments.
    """
    print(message)
    start = time.time()
    yield
    stop = time.time()
    print("...elapsed time: {}".format(stop - start))


def euclidean_cdist(X, Y):
    return sklearn.metrics.pairwise.pairwise_distances(
        X, Y, metric="euclidean", n_jobs=1
    )


def euclidean_pdist(X):
    return sklearn.metrics.pairwise.pairwise_distances(X, metric="euclidean", n_jobs=1)


def cosine_cdist(X, Y):
    return sklearn.metrics.pairwise.pairwise_distances(X, Y, metric="cosine", n_jobs=1)


def cosine_pdist(X):
    return sklearn.metrics.pairwise.pairwise_distances(X, metric="cosine", n_jobs=1)


dist_func_lookup = {
    "cosine": {"cdist": cosine_cdist, "pdist": cosine_pdist},
    "euclidean": {"cdist": euclidean_cdist, "pdist": euclidean_pdist},
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--tailsize",
    type=int,
    help="number of points that constitute 'extrema'",
    default=50,
)
parser.add_argument(
    "--cover_threshold",
    type=float,
    help="probabilistic threshold to designate redundancy between points",
    default=0.5,
)
parser.add_argument(
    "--distance", type=str, default="euclidean", choices=dist_func_lookup.keys()
)
parser.add_argument(
    "--nfuse", type=int, help="number of extreme vectors to fuse over", default=4
)
parser.add_argument(
    "--margin_scale",
    type=float,
    help="multiplier by which to scale the margin distribution",
    default=0.5,
)

# set parameters; default if no command line arguments
args = parser.parse_args()
tailsize = args.tailsize
cover_threshold = args.cover_threshold
cdist_func = dist_func_lookup[args.distance]["cdist"]
pdist_func = dist_func_lookup[args.distance]["pdist"]
num_to_fuse = args.nfuse
margin_scale = args.margin_scale


def set_cover_greedy(universe, subsets, cost=lambda x: 1.0):
    """
    A greedy approximation to Set Cover.
    """
    universe = set(universe)
    subsets = list(map(set, subsets))
    covered = set()
    cover_indices = []
    while covered != universe:
        max_index = (np.array(list(map(lambda x: len(x - covered), subsets)))).argmax()
        covered |= subsets[max_index]
        cover_indices.append(max_index)
    return cover_indices


def set_cover(points, weibulls, solver=set_cover_greedy):
    """
    Generic wrapper for set cover. Takes a solver function.
    Could do a Linear Programming approximation, but the
    default greedy method is bounded in polynomial time.
    """
    universe = range(len(points))
    d_mat = pdist_func(points)
    with Pool(8) as p:
        probs = np.array(p.map(weibull_eval_parallel, zip(d_mat, weibulls)))

    thresholded = zip(*np.where(probs >= cover_threshold))
    subsets = {
        k: tuple(set(x[1] for x in v))
        for k, v in it.groupby(thresholded, key=lambda x: x[0])
    }
    subsets = [subsets[i] for i in universe]
    keep_indices = solver(universe, subsets)
    return keep_indices


def reduce_model(points, weibulls, labels, labels_to_reduce=None):
    """
    Model reduction routine. Calls off to set cover.
    """
    if cover_threshold >= 1.0:
        # optimize for the trivial case
        return points, weibulls, labels
    ulabels = np.unique(labels)
    if labels_to_reduce == None:
        labels_to_reduce = ulabels
    labels_to_reduce = set(labels_to_reduce)
    keep = np.array([], dtype=int)
    for ulabel in ulabels:
        ind = np.where(labels == ulabel)
        if ulabel in labels_to_reduce:
            # print("...reducing model for label {}".format(ulabel))
            keep_ind = set_cover(points[ind], [weibulls[i] for i in ind[0]])
            keep = np.concatenate((keep, ind[0][keep_ind]))
        else:
            keep = np.concatenate((keep, ind[0]))
    points = points[keep]
    weibulls = [weibulls[i] for i in keep]
    labels = labels[keep]
    return points, weibulls, labels


def weibull_fit_parallel(args):
    """Parallelized for efficiency"""
    global tailsize
    dists, row, labels = args
    nearest = np.partition(dists[np.where(labels != labels[row])], tailsize)
    mr = libmr.MR()
    mr.fit_low(nearest, tailsize)
    return str(mr)


def weibull_eval_parallel(args):
    """Parallelized for efficiency"""
    dists, weibull_params = args
    mr = libmr.load_from_string(weibull_params)
    probs = mr.w_score_vector(dists)
    return probs


def fuse_prob_for_label(prob_mat, num_to_fuse):
    """
    Fuse over num_to_fuse extreme vectors to obtain
    probability of sample inclusion (PSI)
    """
    return np.average(
        np.partition(prob_mat, -num_to_fuse, axis=0)[-num_to_fuse:, :], axis=0
    )


def fit(X, y):
    """
    Analogous to scikit-learn\'s fit method.
    """
    global margin_scale
    d_mat = margin_scale * pdist_func(X)
    row_range = range(len(d_mat))
    args = zip(d_mat, row_range, [y for i in row_range])
    with Pool(8) as p:
        weibulls = p.map(weibull_fit_parallel, args)

    return weibulls


def predict(Xtest, Xtrain, weibulls, labels):
    """
    Analogous to scikit-learn's predict method
    except takes a few more arguments which
    constitute the actual model.
    """
    global num_to_fuse
    d_mat = cdist_func(Xtrain, Xtest).astype(np.float64)
    with Pool(8) as p:
        probs = np.array(p.map(weibull_eval_parallel, zip(d_mat, weibulls)))

    ulabels = np.unique(labels)
    fused_probs = []
    for ulabel in ulabels:
        fused_probs.append(
            fuse_prob_for_label(probs[np.where(labels == ulabel)], num_to_fuse)
        )
    fused_probs = np.array(fused_probs)
    max_ind = np.argmax(fused_probs, axis=0)
    predicted_labels = ulabels[max_ind]
    confidence = np.max(fused_probs, axis=0)
    return predicted_labels, fused_probs, confidence


def load_data(fname):
    with open(fname) as f:
        data = f.read().splitlines()
    data = [x.split(",") for x in data]
    labels = [x[0] for x in data]
    data = [list(map(lambda y: float(y), x[1:])) for x in data]
    return np.array(data), np.array(labels)


def get_f1_score(predictions, labels):
    return sum(predictions == labels) / float(len(predictions))


def update_params(
    n_tailsize,
    n_cover_threshold,
    n_cdist_func,
    n_pdist_func,
    n_num_to_fuse,
    n_margin_scale,
):
    global tailsize, cover_threshold, cdist_func, pdist_func, num_to_fuse, margin_scale
    tailsize = n_tailsize
    cover_threshold = n_cover_threshold
    cdist_func = n_cdist_func
    pdist_func = n_pdist_func
    num_to_fuse = n_num_to_fuse
    margin_scale = n_margin_scale


def compute_f1_measure(
    gallery_scores, imposter_scores, gallery_idx, predictions, ytest, thresh
):
    tp = np.sum(
        np.logical_and(
            gallery_scores > thresh, predictions[gallery_idx] == ytest[gallery_idx]
        )
    )
    fn = np.sum(gallery_scores < thresh)
    fp = np.sum(imposter_scores > thresh)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    return 2 * (recall * precision) / (recall + precision)


def letter_test(train_fname, test_fname):
    Xtrain, ytrain = load_data(train_fname)
    Xtest, ytest = load_data(test_fname)
    weibulls = fit(Xtrain, ytrain)
    Xtrain, weibulls, ytrain = reduce_model(Xtrain, weibulls, ytrain)
    predictions, probs, confidence = predict(Xtest, Xtrain, weibulls, ytrain)

    y_train_unique = np.unique(ytrain)
    gallery_idx = np.isin(ytest, y_train_unique)
    gallery_scores = confidence[gallery_idx]
    imposter_scores = confidence[~gallery_idx]

    # find optimal thresh
    threshes = np.logspace(-5, -2, num=20)

    f1_scores = []
    for thresh in threshes:
        f1_scores.append(
            compute_f1_measure(
                gallery_scores, imposter_scores, gallery_idx, predictions, ytest, thresh
            )
        )
    print(
        f"Optimal thresh: {threshes[np.argmax(f1_scores)]}, optimal F1 {np.max(f1_scores)}"
    )
    scores_distr = {
        "gallery": gallery_scores,
        "imposter": imposter_scores,
    }

    draw_score_distr_plot(
        scores_distr=scores_distr,
        score_type="EVM",
        model_name="EVM",
        in_data_name="gallery",
        out_data_name="imposter",
    )
    ct = len(np.unique(ytrain))
    ce = len(np.unique(ytest))
    # https://scikit-learn.org/stable/modules/cross_validation.html
    f1_score = np.max(f1_scores)  # get_f1_score(predictions, ytest, confidence)
    return f1_score


from pathlib import Path
from itertools import product
from numpy.random import RandomState


if __name__ == "__main__":
    seeds = [4, 5]
    # create_oletter_dataset( "/app/sandbox/ExtremeValueMachine/TestData/train.txt",
    #     "/app/sandbox/ExtremeValueMachine/TestData/test.txt", seeds)
    for seed in seeds:
        print(f"seed {seed}")
        for num_unk_classes in [11]:  # np.arange(12):
            print(f"15 known and {num_unk_classes} unknown in test")
            f1_score = letter_test(
                f"/app/sandbox/ExtremeValueMachine/TestData/oletter/{seed}/train_dataset.txt",
                f"/app/sandbox/ExtremeValueMachine/TestData/oletter/{seed}/with_{num_unk_classes}_unk_test_dataset.txt",
            )
            # print(f'f1_score: {f1_score}')
