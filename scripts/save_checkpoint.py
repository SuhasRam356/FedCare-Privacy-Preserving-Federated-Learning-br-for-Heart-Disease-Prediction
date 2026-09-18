"""Train and save a global model checkpoint for the dashboard risk calculator."""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fedcare.task import Net, load_data, train, evaluate


def main():
    model = Net()
    train_loader, test_loader, _ = load_data(partition_id=None)

    for epoch in range(15):
        loss = train(model, train_loader, epochs=1, lr=0.001)
        if (epoch + 1) % 5 == 0:
            metrics = evaluate(model, test_loader)
            acc = metrics["accuracy"]
            auc = metrics["auc"]
            print(f"Epoch {epoch + 1}: Loss={loss:.4f}, Acc={acc:.4f}, AUC={auc:.4f}")

    ckpt_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    # This is the centralized baseline model, NOT the federated model.
    torch.save(model.state_dict(), ckpt_dir / "centralized_model.pt")
    print("Checkpoint saved to checkpoints/centralized_model.pt")

    # Final evaluation
    final = evaluate(model, test_loader)
    print(f"Final - Acc={final['accuracy']:.4f}, AUC={final['auc']:.4f}")


if __name__ == "__main__":
    main()
