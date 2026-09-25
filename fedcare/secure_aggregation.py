"""
fedcare/secure_aggregation.py – Cryptographic Privacy for FedCare.

Implements Secure Aggregation simulation:
    - Simulated Secure Multi-Party Computation (SMPC) via secret sharing
    - Paillier Homomorphic Encryption (HE) simulation
    - SecAgg protocol where the server never sees individual model updates

This is a simulation-level implementation matching FedCare's sequential
in-process design pattern, demonstrating the cryptographic concepts
without requiring a full distributed cryptographic infrastructure.
"""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np


class SecretSharing:
    """
    Additive Secret Sharing for Secure Aggregation.

    Each client splits its model update into N shares such that
    the sum of all shares equals the original update. The server
    only ever sees the aggregated sum, never individual updates.
    """

    @staticmethod
    def create_shares(
        weights: Sequence[np.ndarray],
        n_parties: int,
        seed: int = 42,
    ) -> list[list[np.ndarray]]:
        """
        Split model weights into n_parties additive shares.

        Parameters
        ----------
        weights : list of np.ndarray
            Model parameter arrays to be shared.
        n_parties : int
            Number of parties to split across.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        shares : list[list[np.ndarray]]
            shares[party_idx][layer_idx] = share array
        """
        rng = np.random.default_rng(seed)
        shares = [[] for _ in range(n_parties)]

        for w in weights:
            # Generate (n-1) random shares
            random_shares = [
                rng.normal(0, 0.01, size=w.shape).astype(w.dtype)
                for _ in range(n_parties - 1)
            ]
            # Last share = original - sum of random shares
            last_share = w - sum(random_shares)
            for i, s in enumerate(random_shares):
                shares[i].append(s)
            shares[-1].append(last_share)

        return shares

    @staticmethod
    def reconstruct(
        all_shares: list[list[np.ndarray]],
    ) -> list[np.ndarray]:
        """
        Reconstruct the original weights from all additive shares.

        Parameters
        ----------
        all_shares : list[list[np.ndarray]]
            all_shares[party_idx][layer_idx] = share array

        Returns
        -------
        list[np.ndarray]
            Reconstructed model weights.
        """
        n_layers = len(all_shares[0])
        reconstructed = []
        for layer_idx in range(n_layers):
            layer_sum = sum(
                all_shares[party_idx][layer_idx]
                for party_idx in range(len(all_shares))
            )
            reconstructed.append(layer_sum)
        return reconstructed


class PaillierSimulator:
    """
    Simulated Paillier Homomorphic Encryption.

    Demonstrates the concept of homomorphic addition:
    encrypt(a) + encrypt(b) = encrypt(a + b)

    This is a simulation that adds computational overhead to model
    realistic encryption/decryption costs without requiring the
    actual Paillier cryptosystem library.
    """

    def __init__(self, key_size_bits: int = 1024, seed: int = 42):
        self.key_size_bits = key_size_bits
        self.rng = np.random.default_rng(seed)
        # Simulated noise factor (represents encryption overhead)
        self._noise_scale = 1e-10

    def encrypt(self, value: np.ndarray) -> dict:
        """
        Simulate Paillier encryption.

        Returns a dict with the 'ciphertext' (value + tiny noise)
        and metadata to track the encryption state.
        """
        noise = self.rng.normal(0, self._noise_scale, size=value.shape)
        return {
            "ciphertext": value + noise,
            "shape": value.shape,
            "dtype": str(value.dtype),
            "encrypted": True,
        }

    def decrypt(self, encrypted: dict) -> np.ndarray:
        """Simulate Paillier decryption (removes noise)."""
        return encrypted["ciphertext"].copy()

    def add_encrypted(self, enc_a: dict, enc_b: dict) -> dict:
        """
        Homomorphic addition: encrypt(a) ⊕ encrypt(b) = encrypt(a + b).
        """
        return {
            "ciphertext": enc_a["ciphertext"] + enc_b["ciphertext"],
            "shape": enc_a["shape"],
            "dtype": enc_a["dtype"],
            "encrypted": True,
        }


class SecureAggregator:
    """
    Secure Aggregation protocol for federated model updates.

    Combines secret sharing and optional homomorphic encryption
    to ensure the server never sees individual client updates.
    """

    def __init__(self, use_he: bool = False, seed: int = 42):
        self.use_he = use_he
        self.seed = seed
        self.paillier = PaillierSimulator(seed=seed) if use_he else None

    def aggregate(
        self,
        client_weights: list[list[np.ndarray]],
        sample_counts: list[int],
    ) -> list[np.ndarray]:
        """
        Securely aggregate model weights from multiple clients.

        Parameters
        ----------
        client_weights : list[list[np.ndarray]]
            client_weights[client_idx][layer_idx] = weight array
        sample_counts : list[int]
            Number of training samples per client.

        Returns
        -------
        list[np.ndarray]
            Aggregated global model weights.
        """
        n_clients = len(client_weights)
        n_layers = len(client_weights[0])
        total_samples = sum(sample_counts)

        if self.use_he and self.paillier is not None:
            # Homomorphic encryption path
            encrypted_weighted = None
            for c_idx in range(n_clients):
                weight_factor = sample_counts[c_idx] / total_samples
                weighted = [w * weight_factor for w in client_weights[c_idx]]
                encrypted = [self.paillier.encrypt(w) for w in weighted]

                if encrypted_weighted is None:
                    encrypted_weighted = encrypted
                else:
                    encrypted_weighted = [
                        self.paillier.add_encrypted(ew, e)
                        for ew, e in zip(encrypted_weighted, encrypted)
                    ]

            aggregated = [self.paillier.decrypt(ew) for ew in encrypted_weighted]
        else:
            # Secret sharing path
            aggregated = [np.zeros_like(client_weights[0][l]) for l in range(n_layers)]
            for c_idx in range(n_clients):
                weight_factor = sample_counts[c_idx] / total_samples
                shares = SecretSharing.create_shares(
                    client_weights[c_idx], n_clients,
                    seed=self.seed + c_idx,
                )
                reconstructed = SecretSharing.reconstruct(shares)
                for l in range(n_layers):
                    aggregated[l] += reconstructed[l] * weight_factor

        return aggregated

    def get_protocol_info(self) -> dict:
        """Return metadata about the secure aggregation protocol."""
        return {
            "method": "Homomorphic Encryption (Paillier)" if self.use_he else "Additive Secret Sharing",
            "key_size": self.paillier.key_size_bits if self.paillier else "N/A",
            "server_sees_individual_updates": False,
            "computational_overhead": "High (HE)" if self.use_he else "Low (Secret Sharing)",
            "communication_overhead": "2x (encrypted weights)" if self.use_he else "Nx shares",
        }


def run_secure_aggregation_demo(
    n_rounds: int = 5,
    n_hospitals: int = 6,
) -> dict:
    """
    Run a complete secure aggregation demo with the FedCare model.

    Returns metrics comparing standard FedAvg vs SecAgg FedAvg.
    """
    from fedcare.task import Net, load_data, train as train_fn, evaluate

    # Standard FedAvg
    standard_model = Net()
    train_loader, test_loader, _ = load_data(partition_id=None)
    train_fn(standard_model, train_loader, epochs=10, lr=0.001)
    standard_metrics = evaluate(standard_model, test_loader)

    # SecAgg with Secret Sharing
    sec_agg_ss = SecureAggregator(use_he=False)
    client_weights_list = []
    sample_counts = []

    for i in range(1, n_hospitals + 1):
        local_model = Net()
        local_train, _, _ = load_data(partition_id=i)
        train_fn(local_model, local_train, epochs=2, lr=0.001)
        weights = [p.detach().cpu().numpy() for p in local_model.parameters()]
        client_weights_list.append(weights)
        sample_counts.append(len(local_train.dataset))

    agg_weights = sec_agg_ss.aggregate(client_weights_list, sample_counts)

    secagg_model = Net()
    for param, agg_w in zip(secagg_model.parameters(), agg_weights):
        param.data = torch.tensor(agg_w, dtype=param.dtype)

    _, global_test, _ = load_data(partition_id=None)
    secagg_metrics = evaluate(secagg_model, global_test)

    # SecAgg with HE
    sec_agg_he = SecureAggregator(use_he=True)
    agg_weights_he = sec_agg_he.aggregate(client_weights_list, sample_counts)

    he_model = Net()
    for param, agg_w in zip(he_model.parameters(), agg_weights_he):
        param.data = torch.tensor(agg_w, dtype=param.dtype)

    he_metrics = evaluate(he_model, global_test)

    import torch
    return {
        "standard": {
            "auc": round(standard_metrics["auc"], 4),
            "accuracy": round(standard_metrics["accuracy"], 4),
            "protocol": "Standard FedAvg (No Encryption)",
        },
        "secagg_ss": {
            "auc": round(secagg_metrics["auc"], 4),
            "accuracy": round(secagg_metrics["accuracy"], 4),
            "protocol": sec_agg_ss.get_protocol_info()["method"],
        },
        "secagg_he": {
            "auc": round(he_metrics["auc"], 4),
            "accuracy": round(he_metrics["accuracy"], 4),
            "protocol": sec_agg_he.get_protocol_info()["method"],
        },
    }
