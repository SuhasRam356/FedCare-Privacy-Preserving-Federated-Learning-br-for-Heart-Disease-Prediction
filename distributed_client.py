"""
distributed_client.py - True RPC Client for FedCare
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import flwr as fl
from fedcare.client_app import FlowerClient
from fedcare.reproducibility import seed_everything

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-id", type=int, required=True)
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()

    # Ensure each client has a slightly different seed for reproducibility but variance
    seed_everything(42 + args.node_id)
    
    # Instantiate the client
    client = FlowerClient(partition_id=args.node_id)
    
    print(f"Starting Flower Client {args.node_id} connecting to {args.server_address}...")
    try:
        fl.client.start_client(
            server_address=args.server_address,
            client=client.to_client(),
        )
    except AttributeError:
        # Fallback for older flower versions
        fl.client.start_numpy_client(
            server_address=args.server_address,
            client=client,
        )

if __name__ == "__main__":
    main()
