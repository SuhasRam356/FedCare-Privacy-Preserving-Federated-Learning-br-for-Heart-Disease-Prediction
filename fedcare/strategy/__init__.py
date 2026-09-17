"""
fedcare/strategy/__init__.py - Export all aggregation strategies.
"""

from fedcare.strategy.fedavg_weighted import FedAvgWeighted
from fedcare.strategy.fedprox import FedProx
from fedcare.strategy.krum import MultiKrum, aggregate_krum
from fedcare.strategy.median import CoordinateMedian, aggregate_median
from fedcare.strategy.trimmed_mean import TrimmedMean, aggregate_trimmed_mean

__all__ = [
    "FedAvgWeighted",
    "FedProx",
    "TrimmedMean",
    "aggregate_trimmed_mean",
    "CoordinateMedian",
    "aggregate_median",
    "MultiKrum",
    "aggregate_krum",
]
