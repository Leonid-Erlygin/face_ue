from typing import Literal, Optional, List
import numpy as np
from joblib import Parallel, delayed
from sklearnex.decomposition import PCA
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.base import TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import warnings

from evaluation.metrics import EvalMetricsT

# , compute_detection_and_identification_rate
from evaluation.distance_functions.open_set_identification.abc import Abstract1NEval
from evaluation.distance_functions.distance_functions import ScfSim


class SVM(Abstract1NEval):
    def __init__(
        self,
        use_unc: bool,
        C: float,
        loss: Literal["squared_hinge", "hinge"] = "squared_hinge",
        scale: bool = True,
        frac_pca_components: Optional[float] = None,
        shift: int = 0,
    ) -> None:
        self.use_unc = use_unc
        self.C = C
        self.shift = shift
        self.scale = scale
        self.loss: Literal["squared_hinge", "hinge"] = loss
        self.frac_pca_components = frac_pca_components
        self.compute_scf_sim = ScfSim()

    @property
    def __name__(self):
        attrs = [
            "SVM",
            "unc" if self.use_unc else "basic",
            "C" + str(self.C),
            "shift" + str(self.shift),
        ]
        if self.frac_pca_components:
            attrs.append("pca" + str(self.frac_pca_components))

        if self.scale:
            attrs.append("scaled")

        return "_".join(attrs)

    def __call__(
        self,
        probe_feats: np.ndarray,
        probe_unc: np.ndarray,
        gallery_feats: np.ndarray,
        gallery_unc: np.ndarray,
        probe_ids: np.ndarray,
        gallery_ids: np.ndarray,
        fars: np.ndarray,
    ) -> EvalMetricsT:
        if self.use_unc:
            d = probe_feats.shape[1]
            XX = self.compute_scf_sim(
                mu_ij=2 * np.dot(gallery_feats, gallery_feats.T),
                X_unc=gallery_unc + self.shift,
                Y_unc=gallery_unc + self.shift,
                d=d,
            )

            YX = self.compute_scf_sim(
                mu_ij=2 * np.dot(probe_feats, gallery_feats.T),
                X_unc=gallery_unc + self.shift,
                Y_unc=probe_unc + self.shift,
                d=d,
            )  # (test, train)

            transforms: List[TransformerMixin] = [FunctionTransformer()]

            if self.frac_pca_components is not None:
                transforms.append(PCA(n_components=int(d * self.frac_pca_components)))

            if self.scale:
                transforms.append(StandardScaler())

            pipeline = make_pipeline(
                *transforms, OneVsRestClassifier(LinearSVC(C=self.C, loss=self.loss))
            )

            warnings.filterwarnings("error")
            try:
                with Parallel(-1) as backend:
                    pipeline.fit(XX, gallery_ids)
                    decision_scores = np.concatenate(
                        backend(
                            map(
                                delayed(pipeline.decision_function),
                                np.array_split(YX, 16),
                            )
                        )
                    )  # type: ignore
            except ConvergenceWarning:
                print("SVM didn't converge with given parameters, returning nans")
                return (
                    0,
                    0,
                    0,
                    [np.nan] * len(fars),
                    [np.nan] * len(fars),
                    [(np.nan, np.nan)] * len(fars),
                )
            finally:
                warnings.resetwarnings()
        else:
            model = LinearSVC(C=self.C).fit(gallery_feats, gallery_ids)
            decision_scores = model.decision_function(probe_feats)
        return decision_scores, decision_scores.max(1)
        return compute_detection_and_identification_rate(
            fars,
            probe_ids,
            gallery_ids,
            decision_scores,
            decision_scores.max(1),
            labels_sorted=True,
        )
