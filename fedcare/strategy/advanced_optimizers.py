import csv
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from flwr.common import Parameters, Scalar, parameters_to_ndarrays
from flwr.server.strategy import FedAdam, FedYogi, QFedAvg

from fedcare.strategy.fedavg_weighted import weighted_average_metrics

logger = logging.getLogger("fedcare.strategy.advanced_optimizers")

class MetricsHistoryMixin:
    """Mixin to add checkpointing and metric history to any Flower strategy."""
    def __init__(self, checkpoint_dir: Optional[Union[str, Path]] = None, **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
        self.history: list[dict[str, Union[int, float]]] = []
        self.best_eval_auc: float = -1.0
        self.best_parameters = None

    def evaluate(self, server_round: int, parameters: Parameters):
        eval_res = super().evaluate(server_round, parameters)
        if eval_res is None:
            return None
            
        loss, metrics = eval_res
        acc = float(metrics.get("accuracy", 0.0))
        auc = float(metrics.get("auc", 0.0))
        
        record = {
            "round": server_round,
            "server_loss": float(loss),
            "server_accuracy": acc,
            "server_auc": auc,
        }
        self.history.append(record)
        
        if auc > self.best_eval_auc:
            self.best_eval_auc = auc
            ndarrays = parameters_to_ndarrays(parameters)
            self.best_parameters = ndarrays
            if self.checkpoint_dir:
                ckpt_path = self.checkpoint_dir / f"best_{self.__class__.__name__.lower()}_model.pt"
                np.savez_compressed(
                    str(ckpt_path),
                    **{f"layer_{i}": arr for i, arr in enumerate(ndarrays)},
                )
        return loss, metrics

    def save_history_to_csv(self, file_path: Union[str, Path]) -> None:
        out_path = Path(file_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.history:
            return
        fieldnames = list(self.history[0].keys())
        with open(out_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)


class FedAdamWeighted(MetricsHistoryMixin, FedAdam):
    def __init__(self, checkpoint_dir=None, **kwargs):
        kwargs["fit_metrics_aggregation_fn"] = weighted_average_metrics
        kwargs["evaluate_metrics_aggregation_fn"] = weighted_average_metrics
        super().__init__(checkpoint_dir=checkpoint_dir, **kwargs)


class FedYogiWeighted(MetricsHistoryMixin, FedYogi):
    def __init__(self, checkpoint_dir=None, **kwargs):
        kwargs["fit_metrics_aggregation_fn"] = weighted_average_metrics
        kwargs["evaluate_metrics_aggregation_fn"] = weighted_average_metrics
        super().__init__(checkpoint_dir=checkpoint_dir, **kwargs)


class QFedAvgWeighted(MetricsHistoryMixin, QFedAvg):
    def __init__(self, checkpoint_dir=None, q_param=0.2, qffl_learning_rate=0.1, **kwargs):
        kwargs["q_param"] = q_param
        kwargs["qffl_learning_rate"] = qffl_learning_rate
        kwargs["fit_metrics_aggregation_fn"] = weighted_average_metrics
        kwargs["evaluate_metrics_aggregation_fn"] = weighted_average_metrics
        super().__init__(checkpoint_dir=checkpoint_dir, **kwargs)
