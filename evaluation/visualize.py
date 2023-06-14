import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

def plot_rejection_scores(scores, names=None):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    for id, score in enumerate(scores):
        name = None if names is None else names[id]
        if isinstance(score, str) and score.endswith(".npz"):
            aa = np.load(score)
            score, name = aa.get("scores")[0], aa.get("names")[0]
        fractions, metric_value = score[0], score[1]
        name = name if name is not None else str(id)

        #auc_value = auc(rank, cmc)
        label = "[%s]" % (name)
        plt.plot(fractions, metric_value, lw=1, label=label)

    plt.xlabel("Fractions")
    # plt.xlim([0.0001, 1])
    #plt.xscale("log")
    plt.ylabel("AUC value")
    # plt.ylim([0, 1])

    plt.grid(linestyle="--", linewidth=1)
    plt.legend(fontsize="x-small")
    plt.tight_layout()
    return fig

def plot_roc_and_calculate_tpr(scores, names=None, label=None):
    print(">>>> plot roc and calculate tpr...")
    score_dict = {}
    for id, score in enumerate(scores):
        name = None if names is None else names[id]
        if isinstance(score, str) and score.endswith(".npz"):
            aa = np.load(score)
            score = aa.get("scores", [])
            label = aa["label"] if label is None and "label" in aa else label
            score_name = aa.get("names", [])
            for ss, nn in zip(score, score_name):
                score_dict[nn] = ss
        elif isinstance(score, str) and score.endswith(".npy"):
            name = (
                name
                if name is not None
                else os.path.splitext(os.path.basename(score))[0]
            )
            score_dict[name] = np.load(score)
        elif isinstance(score, str) and score.endswith(".txt"):
            # IJB meta data like ijbb_template_pair_label.txt
            label = pd.read_csv(score, sep=" ", header=None).values[:, 2]
        else:
            name = name if name is not None else str(id)
            score_dict[name] = score
    if label is None:
        print("Error: Label data is not provided")
        return None, None

    x_labels = [10 ** (-ii) for ii in range(1, 7)[::-1]]
    fpr_dict, tpr_dict, roc_auc_dict, tpr_result = {}, {}, {}, {}
    for name, score in score_dict.items():
        fpr, tpr, _ = roc_curve(label, score)
        roc_auc = auc(fpr, tpr)
        fpr, tpr = np.flipud(fpr), np.flipud(tpr)  # select largest tpr at same fpr
        tpr_result[name] = [tpr[np.argmin(abs(fpr - ii))] for ii in x_labels]
        fpr_dict[name], tpr_dict[name], roc_auc_dict[name] = fpr, tpr, roc_auc
    tpr_result_df = pd.DataFrame(tpr_result, index=x_labels).T
    tpr_result_df["AUC"] = pd.Series(roc_auc_dict)
    tpr_result_df.columns.name = "Methods"
    print(tpr_result_df.to_markdown())
    # print(tpr_result_df)

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        for name in score_dict:
            plt.plot(
                fpr_dict[name],
                tpr_dict[name],
                lw=1,
                label="[%s (AUC = %0.4f%%)]" % (name, roc_auc_dict[name] * 100),
            )
        title = (
            "ROC on IJB" + name.split("IJB")[-1][0] if "IJB" in name else "ROC on IJB"
        )

        plt.xlim([10**-6, 0.1])
        plt.xscale("log")
        plt.xticks(x_labels)
        plt.xlabel("False Positive Rate")
        plt.ylim([0.3, 1.0])
        plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
        plt.ylabel("True Positive Rate")

        plt.grid(linestyle="--", linewidth=1)
        plt.title(title)
        plt.legend(loc="lower right", fontsize="x-small")
        plt.tight_layout()
        plt.show()
    except:
        print("matplotlib plot failed")
        fig = None

    return tpr_result_df, fig


def plot_cmc_scores(scores, names=None):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    for id, score in enumerate(scores):
        name = None if names is None else names[id]
        if isinstance(score, str) and score.endswith(".npz"):
            aa = np.load(score)
            score, name = aa.get("scores")[0], aa.get("names")[0]
        rank, cmc = score[0], score[1]
        name = name if name is not None else str(id)

        auc_value = auc(rank, cmc)
        label = "[%s]" % (name)
        plt.plot(rank, cmc, lw=1, label=label)

    plt.xlabel("Rank")
    # plt.xlim([0.0001, 1])
    plt.xscale("log")
    plt.ylabel("Identification rate")
    # plt.ylim([0, 1])

    plt.grid(linestyle="--", linewidth=1)
    plt.legend(fontsize="x-small")
    plt.tight_layout()
    return fig


def plot_dir_far_scores(scores, names=None):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    for id, score in enumerate(scores):
        name = None if names is None else names[id]
        if isinstance(score, str) and score.endswith(".npz"):
            aa = np.load(score)
            score, name = aa.get("scores")[0], aa.get("names")[0]
        fars, tpirs = score[0], score[1]
        name = name if name is not None else str(id)

        auc_value = auc(fars, tpirs)
        label = "[%s (AUC = %0.4f%%)]" % (name, auc_value * 100)
        plt.plot(fars, tpirs, lw=1, label=label)

    plt.xlabel("False Alarm Rate")
    plt.xlim([0.0001, 1])
    plt.xscale("log")
    plt.ylabel("Detection & Identification Rate (%)")
    plt.ylim([0, 1])

    plt.grid(linestyle="--", linewidth=1)
    plt.legend(fontsize="x-small")
    plt.tight_layout()
    # except:
    #     print("matplotlib plot failed")
    #     fig = None

    return fig


def plot_tar_far_scores(scores, names=None):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    for id, score in enumerate(scores):
        name = None if names is None else names[id]
        if isinstance(score, str) and score.endswith(".npz"):
            aa = np.load(score)
            score, name = aa.get("scores")[0], aa.get("names")[0]
        fars, tpirs = score[0], score[1]
        name = name if name is not None else str(id)

        auc_value = auc(fars, tpirs)
        label = "[%s (AUC = %0.4f%%), tar %0.4f%% at far %.1E]" % (
            name,
            auc_value * 100,
            tpirs[0],
            fars[0],
        )
        plt.plot(fars, tpirs, lw=1, label=label)

    plt.xlabel("False Acceptance Rate")
    plt.xlim([0.00001, 1])
    plt.xscale("log")
    plt.ylabel("True Acceptance Rate (%)")
    plt.ylim([0, 1])

    plt.grid(linestyle="--", linewidth=1)
    plt.legend(fontsize="x-small")
    plt.tight_layout()
    # except:
    #     print("matplotlib plot failed")
    #     fig = None

    return fig
