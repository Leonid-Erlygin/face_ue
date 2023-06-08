from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from typing import Any, Dict, List
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from matplotlib import pyplot as plt

from ..dataloader.data1N import Query1N


@dataclass
class Plot:
    xs: np.ndarray
    ys: np.ndarray
    score: float = float("nan")


class BaseMetric:
    plots: Dict[str, List[Plot]] = defaultdict(list)
    call_plt_with_args: Dict[str, Any] = {}

    @property
    def name(self):
        return self.__class__.__name__

    def reduce_ys(self, ys: List[np.ndarray]):
        return np.mean(ys, axis=0)

    def reduce_scores(self, scores: List[float]):
        return np.mean(scores, axis=0)

    def evaluate(
        self,
        probe_ids: np.ndarray,
        gallery_ids: np.ndarray,
        similarity_matrix: np.ndarray,
        confidence: np.ndarray,
    ) -> Plot:
        raise NotImplementedError

    def __call__(
        self,
        query: Query1N,
        similarity_matrix: np.ndarray,
        confidence: np.ndarray,
        name: str,
    ):
        new_plot = self.evaluate(
            query.probe_ids, query.gallery_ids, similarity_matrix, confidence
        )
        self.plots[name].append(new_plot)

    def setup_plt(self, fig: Figure, ax: Axes):
        pass

    def plot(self):
        fig = plt.figure()
        ax = fig.axes[0]
        for name, plot_list in self.plots.items():
            xs_list = [p.xs for p in plot_list]
            assert reduce(np.equal, xs_list).all()
            xs = xs_list[0]
            ys = self.reduce_ys([p.ys for p in plot_list])
            score = self.reduce_scores([p.score for p in plot_list])
            ax.plot(xs, ys, label=f"[{name}: {score}]")

        self.setup_plt(fig, ax)

        return fig
