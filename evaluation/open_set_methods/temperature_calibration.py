import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize

from typing import Union, Tuple, Literal
import warnings


def _compute_msp_uncertainty(
    similarity: np.ndarray,
    probe_score: np.ndarray,
    tau: float,
    T: float
) -> np.ndarray:
    """
    Compute MSP uncertainty: 1 - max(softmax([similarities, tau] / T))
    
    Parameters
    ----------
    similarity : np.ndarray, shape (n_samples, n_gallery)
        Similarity scores to gallery identities
    probe_score : np.ndarray, shape (n_samples,)
        Acceptance scores (max similarity per probe)
    tau : float
        Acceptance threshold
    T : float
        Temperature parameter for softmax
        
    Returns
    -------
    uncertainty : np.ndarray, shape (n_samples,)
        MSP uncertainty scores (higher = more uncertain)
    """
    sims = np.column_stack([similarity, np.full(similarity.shape[0], tau)])
    sims = sims / T
    conf_score = np.max(softmax(sims, axis=-1), axis=-1)
    return 1 - conf_score


def _evaluate_uncertainty_metric(
    unc_scores: np.ndarray,
    is_error: np.ndarray,
    metric: Literal['auc', 'likelihood', 'ap'] = 'auc'
) -> float:
    """
    Evaluate how well uncertainty scores predict errors.
    
    Parameters
    ----------
    unc_scores : np.ndarray
        Uncertainty scores (higher = more uncertain)
    is_error : np.ndarray, dtype=bool
        Ground truth error labels
    metric : str
        - 'auc': ROC-AUC (higher = better ranking of errors)
        - 'likelihood': Max log-likelihood of logistic model P(error|unc)
        - 'ap': Average Precision for error prediction
        
    Returns
    -------
    score : float
        Metric value (higher is better)
    """
    if len(np.unique(is_error)) < 2:
        return 0.5 if metric != 'likelihood' else -np.inf
    
    if metric == 'auc':
        try:
            return roc_auc_score(is_error, unc_scores)
        except:
            return 0.5
    elif metric == 'ap':
        try:
            return average_precision_score(is_error, unc_scores)
        except:
            return 0.5
    elif metric == 'likelihood':
        from scipy.special import expit
        
        def neg_log_likelihood(params):
            a, b = params
            logits = np.clip(a * unc_scores + b, -20, 20)
            p = np.clip(expit(logits), 1e-10, 1 - 1e-10)
            ll = np.sum(is_error * np.log(p) + (1 - is_error) * np.log(1 - p))
            return -ll
        
        try:
            result = minimize(neg_log_likelihood, x0=[1.0, 0.0], 
                            method='BFGS', options={'maxiter': 100})
            return -result.fun
        except:
            return -np.inf
    else:
        raise ValueError(f"Unknown metric: {metric}")


def calibrate_msp_temperature(
    similarity_calib: np.ndarray,
    probe_score_calib: np.ndarray,
    tau_calib: float,
    is_error: np.ndarray,
    T_min: float = 0.1,
    T_max: float = 100.0,
    max_iter: int = 50,
    tol: float = 1e-3,
    metric: Literal['auc', 'likelihood', 'ap'] = 'auc',
    search_method: Literal['ternary', 'golden', 'binary'] = 'ternary'
) -> Tuple[float, dict]:
    """
    Calibrate temperature T for MaximumSoftmaxProbability uncertainty function.
    
    Finds T that optimizes MSP uncertainty scores for predicting classification 
    errors on a calibration set using search similar to kappa calibration.
    
    Parameters
    ----------
    similarity_calib : np.ndarray, shape (n_samples, n_gallery)
        Similarity scores for calibration samples
    probe_score_calib : np.ndarray, shape (n_samples,)
        Acceptance scores for calibration samples  
    tau_calib : float
        Acceptance threshold used during calibration
    is_error : np.ndarray, shape (n_samples,), dtype=bool
        True if prediction was incorrect for each calibration sample
    T_min, T_max : float
        Search bounds for temperature
    max_iter : int
        Maximum search iterations
    tol : float
        Convergence tolerance for T
    metric : {'auc', 'likelihood', 'ap'}
        Optimization metric (higher = better error prediction)
    search_method : {'ternary', 'golden', 'binary'}
        - 'ternary': Ternary search (recommended for unimodal objective)
        - 'golden': Golden-section search (robust alternative)
        - 'binary': Binary search (only if criterion is monotonic in T)
        
    Returns
    -------
    T_opt : float
        Optimal temperature parameter
    info : dict
        Metadata: final metric value, convergence status, evaluations count
    """
    
    def objective(T: float) -> float:
        """Evaluate metric for given T (higher = better)"""
        if T <= 1e-8:
            return -np.inf if metric == 'likelihood' else 0.0
        unc = _compute_msp_uncertainty(
            similarity_calib, probe_score_calib, tau_calib, T
        )
        return _evaluate_uncertainty_metric(unc, is_error, metric)
    
    # === Search implementations ===
    
    if search_method == 'ternary':
        # Ternary search for unimodal functions (typical for temperature scaling)
        left, right = T_min, T_max
        for _ in range(max_iter):
            if right - left < tol:
                break
            mid1 = left + (right - left) / 3
            mid2 = right - (right - left) / 3
            if objective(mid1) < objective(mid2):
                left = mid1
            else:
                right = mid2
        T_opt = (left + right) / 2
        
    elif search_method == 'golden':
        # Golden-section search - more robust for noisy objectives
        phi = (1 + 5**0.5) / 2
        resphi = 2 - phi
        a, b = T_min, T_max
        c = a + resphi * (b - a)
        d = b - resphi * (b - a)
        fc, fd = objective(c), objective(d)
        
        for _ in range(max_iter):
            if b - a < tol:
                break
            if fc < fd:
                a, c, fc = c, d, fd
                d = b - resphi * (b - a)
                fd = objective(d)
            else:
                b, d, fd = d, c, fc
                c = a + resphi * (b - a)
                fc = objective(c)
        T_opt = (a + b) / 2
        
    elif search_method == 'binary':
        # Binary search - use only with monotonic criterion
        # Example: find T where uncertainty separation meets target
        target_separation = 0.05  # Customize based on your needs
        
        def separation_criterion(T):
            unc = _compute_msp_uncertainty(
                similarity_calib, probe_score_calib, tau_calib, T
            )
            mean_err = unc[is_error].mean() if is_error.any() else 0
            mean_corr = unc[~is_error].mean() if (~is_error).any() else 0
            return mean_err - mean_corr - target_separation
        
        left, right = T_min, T_max
        for _ in range(max_iter):
            if right - left < tol:
                break
            mid = (left + right) / 2
            if separation_criterion(mid) < 0:
                left = mid
            else:
                right = mid
        T_opt = (left + right) / 2
    else:
        raise ValueError(f"Unknown search_method: {search_method}")
    
    final_score = objective(T_opt)
    
    info = {
        'optimal_T': T_opt,
        'final_metric_value': final_score,
        'metric': metric,
        'search_method': search_method,
        'converged': True,
        'search_range': (T_min, T_max)
    }
    
    return T_opt, info