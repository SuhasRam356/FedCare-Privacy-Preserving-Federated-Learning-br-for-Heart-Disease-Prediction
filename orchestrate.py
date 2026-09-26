"""
orchestrate.py - Single command orchestrator for distributed FedCare
Spawns the Flower Server and 6 Hospital Clients as separate background processes.
"""
import subprocess
import time
import sys
import atexit
import os

processes = []

def cleanup():
    print("\nShutting down all processes...")
    for p in processes:
        if p.poll() is None:
            p.terminate()
            p.wait()
    print("Cleanup complete.")

def main():
    atexit.register(cleanup)

    print("="*64)
    print("  FEDCARE DISTRIBUTED ORCHESTRATOR (No Docker)")
    print("="*64)
    print("Starting server process on port 8080...")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(".")

    # Start Server
    server_process = subprocess.Popen([sys.executable, "distributed_server.py", "--rounds", "5"], env=env)
    processes.append(server_process)
    
    # Wait for server to bind to port
    time.sleep(3)

    print("Starting 6 hospital client processes...")
    for i in range(1, 7):
        p = subprocess.Popen([sys.executable, "distributed_client.py", "--node-id", str(i)], env=env)
        processes.append(p)
        time.sleep(0.5) # stagger startup slightly so logs don't collide as much

    print("\n[SUCCESS] All 7 processes are running simultaneously in the background!")
    print("Their logs will appear here. Press Ctrl+C to stop everything at any time.\n")

    try:
        # Wait for the server to finish
        server_process.wait()
    except KeyboardInterrupt:
        pass # Cleanup will run on exit

if __name__ == "__main__":
    main()
