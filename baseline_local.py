"""
baseline_local.py – Local-only (lone hospital) baseline.

Simulates the scenario where each hospital trains a model using ONLY its
own data, with no collaboration.  This provides the lower-bound accuracy
that federated learning should improve upon.

Usage:
    python baseline_local.py
"""

from fedcare.task import Net, load_data, train, evaluate

# ── Configuration ─────────────────────────────────────────────────────
NUM_HOSPITALS = 6
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = "cpu"


def main() -> None:
    print("=" * 60)
    print("  LOCAL-ONLY BASELINE  (each hospital trains alone)")
    print("=" * 60)

    results: list[dict[str, float]] = []

    for h_id in range(1, NUM_HOSPITALS + 1):
        print(f"\n{'-' * 60}")
        print(f"  Hospital {h_id}")
        print(f"{'-' * 60}")

        train_loader, test_loader, scaler = load_data(partition_id=h_id)
        print(f"  Train batches: {len(train_loader)}  |  "
              f"Test batches: {len(test_loader)}")

        model = Net()

        # Train
        for epoch in range(1, EPOCHS + 1):
            loss = train(model, train_loader, epochs=1, lr=LEARNING_RATE, device=DEVICE)

        # Evaluate
        metrics = evaluate(model, test_loader, device=DEVICE)
        metrics["hospital"] = h_id
        results.append(metrics)

        print(f"  Loss     : {metrics['loss']:.4f}")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  AUC      : {metrics['auc']:.4f}")

    # Summary table
    print("\n" + "=" * 60)
    print("  SUMMARY  -  Local-Only Results")
    print("=" * 60)
    print(f"  {'Hospital':>10}  {'Accuracy':>10}  {'AUC':>10}  {'Loss':>10}")
    print(f"  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}")

    accs, aucs = [], []
    for r in results:
        print(f"  {int(r['hospital']):>10}  "
              f"{r['accuracy']:>10.4f}  "
              f"{r['auc']:>10.4f}  "
              f"{r['loss']:>10.4f}")
        accs.append(r["accuracy"])
        aucs.append(r["auc"])

    print(f"  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
    avg_acc = sum(accs) / len(accs)
    avg_auc = sum(aucs) / len(aucs)
    print(f"  {'AVG':>10}  {avg_acc:>10.4f}  {avg_auc:>10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
