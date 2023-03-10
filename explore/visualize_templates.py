from argparse import ArgumentParser
import sys
import re
import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import auc
sys.path.append('.')
# import face_lib.evaluation.plots as plots


parser = ArgumentParser()
parser.add_argument('--test_folder', default='/gpfs/gpfs0/k.fedyanin/space/figures/test')
parser.add_argument('--last_timestamp', action="store_true")
parser.add_argument('--fusion', action='store_true')
parser.add_argument('--full', action='store_true')
args = parser.parse_args()

FARs = [0.0001, 0.001, 0.05]
rejected_portions = np.arange(0, 0.51, 0.02)



methods = [
    # {
    #     'name': 'scale', 'functions': ('mean', 'cosine', 'similarity'), 'label': 'Cosine similarity'
    # },
    {
        'name': 'emb_norm', 'functions': ('mean', 'cosine', 'mean'), 'label': 'Norm'
    },
    # {
    #     'name': 'head', 'functions': ('mean', 'cosine', 'mean'), 'label': 'PFE'
    # },
    {
        'name': 'head', 'functions': ('PFE', 'MLS', 'mean'), 'label': 'PFE (MLS)'
    },
    {
        'name': 'magface', 'functions': ('mean', 'cosine', 'mean'), 'label': 'MagFace'
    },
    {
        'name': 'scale', 'functions': ('mean', 'cosine', 'mean'), 'label': 'ScaleFace (ours)'
    },
    {
        'name': 'scale', 'functions': ('mean', 'scale-sqrt-harmonic-biased-cosine', 'harmonic-harmonic'), 'label': r'$\mu$-ScaleFace (ours)'
    },
    # {
    #     'name': 'scale', 'functions': ('weighted', 'cosine', 'mean'), 'label': 'ScaleFace-w (ours)'
    # },
]

defaults = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

colors = {
    'Norm': defaults[0],
    'PFE': defaults[1],
    'PFE (MLS)': defaults[1],
    'MagFace': defaults[2],
    'ScaleFace (ours)': defaults[3],
    'ScaleFace-w (ours)': defaults[4]
}

# prop_cycle = plt.rcParams['axes.prop_cycle']
# colors = prop_cycle.by_key()['color']

def plot_TAR_FAR_different_methods(
        results, rejected_portions, AUCs, title=None, save_figs_path=None
):
    def pretty_matplotlib_config(fontsize=15):
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rcParams.update({'font.size': fontsize})

    pretty_matplotlib_config(24)

    plots_indices = {
        FAR: idx for idx, FAR in enumerate(FARs)
    }
    fig, axes = plt.subplots(
        ncols=len(plots_indices), nrows=1, figsize=(9 * len(plots_indices), 10)
    )
    for key, table in results.items():
        for FAR, TARs in table.items():
            if FAR not in FARs:
                continue
            auc = AUCs[key][FAR]

            if type(key) != str:
                label = '\_'.join(key)
            else:
                label = key
            print(label + ", AUC=" + str(round(auc, 5)))

            axes[plots_indices[FAR]].plot(
                rejected_portions, TARs,
                label=label, marker=" ", linewidth=3,  alpha=0.8
            )

            axes[plots_indices[FAR]].set_title(f"TAR@FAR={FAR}")
            axes[plots_indices[FAR]].set_xlabel("Rejected portion")
            axes[plots_indices[FAR]].set_ylabel("TAR")
            axes[plots_indices[FAR]].legend()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if title:
        fig.suptitle(title)
    if save_figs_path:
        fig.savefig(save_figs_path, dpi=150, format='pdf')
    return fig


if __name__ == '__main__':
    folder = Path(args.test_folder)
    all_results = OrderedDict()

    for method in methods:
        name = method['name']

        if args.last_timestamp:
            files = os.listdir(folder)
            pattern = r'table_' + name +r'[\d_-]*'
            if args.full:
                pattern += r'_full\.pt'
            else:
                pattern += r'\_single.pt'
            files = [f for f in files if re.match(pattern, f)]
            file = sorted(files)[-1]
        else:
            file = f'table_{name}.pt'
        print(file)
        local_results = torch.load(folder / file)
        all_results[method['label']] = local_results[method['functions']]

    res_AUCs = OrderedDict()
    for method, table in all_results.items():
        res_AUCs[method] = {
            far: auc(rejected_portions, TARs) for far, TARs in table.items()
        }

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_TAR_FAR_different_methods(
        all_results,
        rejected_portions,
        res_AUCs,
        save_figs_path=os.path.join(folder, f"all_methods_together_last.pdf")
    )