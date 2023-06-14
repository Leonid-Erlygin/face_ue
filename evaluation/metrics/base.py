from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from operator import attrgetter
from typing import Any, Dict, List
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

from ..dataloaders import Query1N


@dataclass
class Plot:
    xs: np.ndarray
    ys: np.ndarray
    score: float = float("nan")

class BaseMetric:
    def __init__(self) -> None:
        self.plots: Dict[str, List[Plot]] = defaultdict(list)
        self.call_plt_with_args: Dict[str, Any] = {}        

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
        """Evaluate results using your custom metric.

        Args:
            probe_ids (np.ndarray): Ids of probe samples
            gallery_ids (np.ndarray): Ids of gallery samples
            similarity_matrix (np.ndarray): resulting similarity matrix
            confidence (np.ndarray): resulting confidence

        Raises:
            NotImplementedError: please implement this method

        Returns:
            Plot: xs, ys and score to display in label
        """
        raise NotImplementedError

    def __call__(
        self,
        query: Query1N,
        similarity_matrix: np.ndarray,
        confidence: np.ndarray,
        name: str,
    ):
        """Calls evaluate, adds a value to the corresponding list of plots. Most cases, you wouldn't need to override this method

        Args:
            query (Query1N): Incoming query
            similarity_matrix (np.ndarray): result of similarity_function
            confidence (np.ndarray): result of confidence_function
            name (str): name to display on plot
        """
        new_plot = self.evaluate(
            query.probe_ids, query.gallery_ids, similarity_matrix, confidence
        )
        self.plots[name].append(new_plot)

    def setup_plt(self, fig: Figure, ax: Axes):
        pass

    def plot(self):
        fig, ax = plt.subplots()
        for name, plot_list in self.plots.items():
            assert reduce(np.equal, map(attrgetter("xs"), plot_list)).all()
            ys = self.reduce_ys(np.stack([p.ys for p in plot_list]))
            score = self.reduce_scores([p.score for p in plot_list])
            ax.plot(plot_list[0].xs, ys, label=f"[{name}: {score}]")

        ax.legend(fontsize="x-small")
        self.setup_plt(fig, ax)
        return fig
