"""
baseline_centralized.py – Centralized (pooled-data) baseline.

Simulates the scenario where ALL hospital data is pooled in one place
and a single model is trained.  This provides the upper-bound accuracy
that federated learning aims to match.

Usage:
    python baseline_centralized.py
"""

from fedcare.reproducibility import seed_everything
from fedcare.task import Net, load_data, train, evaluate

# ── Configuration ─────────────────────────────────────────────────────
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = "cpu"


def main() -> None:
    seed_everything(42)
    print("=" * 60)
    print("  CENTRALIZED BASELINE  (all hospitals pooled)")
    print("=" * 60)

    # Load combined dataset
    train_loader, test_loader, scaler = load_data(partition_id=None)
    print(f"\nTrain batches : {len(train_loader)}")
    print(f"Test  batches : {len(test_loader)}")

    # Initialize model
    model = Net()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params  : {total_params:,}")

    # Training loop
    print(f"\nTraining for {EPOCHS} epochs ...")
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, train_loader, epochs=1, lr=LEARNING_RATE, device=DEVICE)
        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate(model, test_loader, device=DEVICE)
            print(
                f"  Epoch {epoch:3d} | "
                f"train_loss={loss:.4f}  "
                f"test_loss={metrics['loss']:.4f}  "
                f"acc={metrics['accuracy']:.4f}  "
                f"auc={metrics['auc']:.4f}"
            )

    # Final evaluation
    final = evaluate(model, test_loader, device=DEVICE)
    print("\n" + "-" * 60)
    print("  CENTRALIZED RESULTS")
    print("-" * 60)
    print(f"  Loss     : {final['loss']:.4f}")
    print(f"  Accuracy : {final['accuracy']:.4f}")
    print(f"  AUC      : {final['auc']:.4f}")
    print("-" * 60)


if __name__ == "__main__":
    main()
