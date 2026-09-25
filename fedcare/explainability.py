"""
fedcare/explainability.py – Explainable AI (XAI) for FedCare.

Provides SHAP-based model explanations for the heart disease MLP:
    - Global feature importance (mean |SHAP|)
    - Per-prediction force-plot data (waterfall breakdowns)
    - Feature contribution summaries for the Risk Calculator

Uses KernelExplainer (model-agnostic) wrapping the PyTorch Net.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import torch


# Feature names matching the 13-column dataset
FEATURE_NAMES = [
    "Age", "Resting BP", "Cholesterol", "Max Heart Rate",
    "BMI", "Glucose", "Sex", "Smoker",
    "Diabetes History", "Family History",
    "CP: Atypical Angina", "CP: Non-Anginal", "CP: Typical Angina",
]


def _model_predict_proba(X: np.ndarray, model: torch.nn.Module) -> np.ndarray:
    """Wrapper: NumPy array → model → class-1 probability."""
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(X, dtype=torch.float32)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    return probs


def compute_shap_values(
    model: torch.nn.Module,
    background_data: np.ndarray,
    explain_data: np.ndarray,
    n_background: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """
    Compute SHAP values for the given input samples.

    Parameters
    ----------
    model : torch.nn.Module
        Trained FedCare MLP.
    background_data : np.ndarray
        Scaled training data used as the reference distribution.
    explain_data : np.ndarray
        Scaled input samples to explain (shape: [N, 13]).
    n_background : int
        Number of background samples to use (for speed).
    seed : int
        Random seed for reproducible background sampling.

    Returns
    -------
    np.ndarray
        SHAP values array of shape [N, 13].
    """
    try:
        import shap
    except ImportError:
        # Fallback: return a simple gradient-based approximation
        return _gradient_feature_importance(model, explain_data)

    rng = np.random.default_rng(seed)
    n_bg = min(n_background, len(background_data))
    idx = rng.choice(len(background_data), n_bg, replace=False)
    bg = background_data[idx]

    predict_fn = lambda x: _model_predict_proba(x, model)
    explainer = shap.KernelExplainer(predict_fn, bg)

    shap_values = explainer.shap_values(explain_data, nsamples=200, silent=True)
    return np.array(shap_values)


def _gradient_feature_importance(
    model: torch.nn.Module, X: np.ndarray
) -> np.ndarray:
    """Fallback gradient-based feature importance when SHAP is unavailable."""
    model.eval()
    tensor = torch.tensor(X, dtype=torch.float32)
    tensor.requires_grad_(True)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[:, 1]
    probs.sum().backward()

    grads = tensor.grad.numpy()
    return grads * X  # input × gradient approximation


def get_global_feature_importance(
    model: torch.nn.Module,
    background_data: np.ndarray,
    n_samples: int = 200,
    seed: int = 42,
) -> dict[str, float]:
    """
    Compute global feature importance as mean |SHAP| across samples.

    Returns
    -------
    dict[str, float]
        Feature name → mean absolute SHAP value.
    """
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(background_data))
    idx = rng.choice(len(background_data), n, replace=False)
    samples = background_data[idx]

    shap_vals = compute_shap_values(model, background_data, samples, seed=seed)
    mean_abs = np.abs(shap_vals).mean(axis=0)

    importance = {}
    for i, name in enumerate(FEATURE_NAMES):
        importance[name] = float(mean_abs[i])
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def explain_single_prediction(
    model: torch.nn.Module,
    background_data: np.ndarray,
    input_scaled: np.ndarray,
    seed: int = 42,
) -> dict:
    """
    Explain a single prediction with per-feature SHAP contributions.

    Parameters
    ----------
    input_scaled : np.ndarray
        Single scaled input of shape [1, 13].

    Returns
    -------
    dict with keys:
        - 'base_value': float – expected output over background
        - 'prediction': float – model output for this input
        - 'contributions': list[dict] – per-feature {name, value, contribution}
    """
    shap_vals = compute_shap_values(model, background_data, input_scaled, seed=seed)
    single_shap = shap_vals[0]

    base_val = float(_model_predict_proba(background_data[:100], model).mean())
    pred_val = float(_model_predict_proba(input_scaled, model)[0])

    contributions = []
    for i, name in enumerate(FEATURE_NAMES):
        contributions.append({
            "feature": name,
            "shap_value": float(single_shap[i]),
            "abs_shap": abs(float(single_shap[i])),
        })
    contributions.sort(key=lambda x: x["abs_shap"], reverse=True)

    return {
        "base_value": base_val,
        "prediction": pred_val,
        "contributions": contributions,
    }
