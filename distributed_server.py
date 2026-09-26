"""
distributed_server.py - True RPC Server for FedCare
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import flwr as fl
from fedcare.server_app import create_fedavg_strategy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()

    # Create the FedAvg strategy defined in fedcare
    strategy = create_fedavg_strategy(num_rounds=args.rounds)

    print(f"Starting Flower Server for {args.rounds} rounds...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
