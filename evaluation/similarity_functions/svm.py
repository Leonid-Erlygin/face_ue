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

from ..dataloaders import Query1N
from .base import BaseSimilarity
from .scf import compute_scf_sim


class SVM(BaseSimilarity):
    def __init__(
        self,
        use_unc: bool,
        C: float,
        loss: Literal['squared_hinge', 'hinge'] = "squared_hinge",
        scale: bool = True,
        frac_pca_components: Optional[float] = None,
        shift: int = 0,
    ) -> None:
        self.use_unc = use_unc
        self.C = C
        self.shift = shift
        self.scale = scale
        self.loss: Literal['squared_hinge', 'hinge'] = loss
        self.frac_pca_components = frac_pca_components
        self.similarity_sorted = True
    
    @property
    def name(self):
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

    def compute(
        self,
        query: Query1N,
    ):
        probe_feats = query.probe_feats
        gallery_feats = query.gallery_feats
        gallery_unc = query.gallery_unc
        probe_unc = query.probe_unc
        gallery_ids = query.gallery_ids

        if self.use_unc:
            d = probe_feats.shape[1]
            XX = compute_scf_sim(
                mu_ij=2 * np.dot(gallery_feats, gallery_feats.T),
                X_unc=gallery_unc + self.shift,
                Y_unc=gallery_unc + self.shift,
                d=d,
            )

            YX = compute_scf_sim(
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
                *transforms,
                OneVsRestClassifier(LinearSVC(C=self.C, loss=self.loss))
            )

            warnings.filterwarnings("error")
            try:
                with Parallel(-1) as backend:
                    pipeline.fit(XX, gallery_ids)
                    decision_scores = np.concatenate(backend(map(
                            delayed(pipeline.decision_function),
                            np.array_split(YX, 16),
                    ))) # type: ignore
            except ConvergenceWarning:
                print("SVM didn't converge with given parameters, returning nans")
                return np.full((probe_feats.shape[0], probe_feats.shape[0]), np.nan)
            finally:
                warnings.resetwarnings()
        else:
            model = LinearSVC(C=self.C).fit(gallery_feats, gallery_ids)
            decision_scores = model.decision_function(probe_feats)

        return decision_scores
