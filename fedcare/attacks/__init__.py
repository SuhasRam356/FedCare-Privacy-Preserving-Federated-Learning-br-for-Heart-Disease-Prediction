"""
fedcare/attacks/__init__.py - Adversarial attack modules for Federated Learning.
"""

from fedcare.attacks.label_flip import apply_label_flip
from fedcare.attacks.model_poison import poison_weights

__all__ = ["apply_label_flip", "poison_weights"]
