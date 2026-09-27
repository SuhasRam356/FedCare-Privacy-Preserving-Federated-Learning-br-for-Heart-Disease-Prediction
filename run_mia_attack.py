"""
run_mia_attack.py - Membership Inference Attack (MIA) Baseline for FedCare.

This script implements a simple loss-threshold Membership Inference Attack 
to provide a real security baseline for the model's privacy leakage.
It tests whether an attacker can determine if a specific patient record 
was used in the training dataset by evaluating the model's confidence/loss.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fedcare.task import load_data, Net
from fedcare.reproducibility import seed_everything


def run_membership_inference(model_path: str = "checkpoints/final_fedavg_model.pt", seed: int = 42):
    """
    Run a threshold-based Membership Inference Attack.
    We assume the attacker has access to:
    1. A trained model (white-box or black-box with probabilities)
    2. A target dataset containing some training members and some non-members.
    """
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("==================================================")
    print("   Membership Inference Attack (MIA) Evaluation   ")
    print("==================================================")
    
    # 1. Load the target global model
    model = Net().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded global model from {model_path}")
    except FileNotFoundError:
        print(f"Model checkpoint not found at {model_path}. Please run `python demo.py` or `python run_federated.py` first.")
        return

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')

    # 2. Get Members (Hospital 1 Training Data) and Non-Members (Global Test Data)
    print("Loading datasets...")
    # Load hospital 1 (Member data)
    train_loader, _, _ = load_data(partition_id=1, batch_size=256)
    
    # Load global test data (Non-member data)
    _, test_loader, _ = load_data(partition_id=None, batch_size=256)

    # 3. Compute losses
    def get_losses(loader):
        losses = []
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device).long()
                outputs = model(data)
                loss = criterion(outputs, target)
                losses.extend(loss.cpu().numpy().flatten())
        return np.array(losses)

    print("Computing inference losses for Members (Hospital 1 Train)...")
    member_losses = get_losses(train_loader)
    
    print("Computing inference losses for Non-Members (Global Test)...")
    non_member_losses = get_losses(test_loader)

    # 4. Perform the threshold attack
    # An attacker predicts "Member" if the loss is below a certain threshold.
    # We will compute the MIA ROC-AUC to measure attack success.
    # A random guess is 0.5. A perfect attack is 1.0.
    
    y_true = np.concatenate([np.ones(len(member_losses)), np.zeros(len(non_member_losses))])
    # The attack score is negative loss (lower loss means higher confidence of being a member)
    y_scores = np.concatenate([-member_losses, -non_member_losses])
    
    mia_auc = roc_auc_score(y_true, y_scores)
    
    # Calculate accuracy at optimal threshold (using median loss as a simple threshold)
    threshold = np.median(y_scores)
    y_pred = (y_scores > threshold).astype(int)
    mia_acc = accuracy_score(y_true, y_pred)
    
    print("\n--- MIA Results ---")
    print(f"MIA ROC-AUC: {mia_auc:.4f}")
    print(f"MIA Accuracy:  {mia_acc:.4f}")
    
    if mia_auc > 0.6:
        print("\n[WARNING]: The model exhibits significant privacy leakage (MIA AUC > 0.6).")
        print("This suggests the model is memorizing patient records.")
    else:
        print("\n[INFO]: The model exhibits minimal privacy leakage (MIA AUC <= 0.6).")
        print("This suggests good generalization with low risk of patient re-identification.")

if __name__ == "__main__":
    run_membership_inference()
