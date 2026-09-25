import re

file_path = "run_phase3_experiments.py"
with open(file_path, "r") as f:
    content = f.read()

# Update simulate_federation signature
old_sig = """def simulate_federation(
    partition_type: str = "hospital_native",
    alpha: float | None = None,
    mu: float = 0.0,
    num_rounds: int = 15,
    local_epochs: int = 2,
    lr: float = 0.001,
    batch_size: int = 32,
    num_clients: int = 6,
    seed: int = 42,
) -> dict[str, Any]:"""
new_sig = """def simulate_federation(
    partition_type: str = "hospital_native",
    alpha: float | None = None,
    mu: float = 0.0,
    strategy_name: str = "fedavg",
    num_rounds: int = 15,
    local_epochs: int = 2,
    lr: float = 0.001,
    batch_size: int = 32,
    num_clients: int = 6,
    seed: int = 42,
) -> dict[str, Any]:"""
content = content.replace(old_sig, new_sig)

# Add optimizer states
old_init = """    # 3. Global parameters
    global_net = Net().to(device)
    global_params = get_parameters(global_net)

    round_history: list[dict[str, Any]] = []"""
new_init = """    # 3. Global parameters
    global_net = Net().to(device)
    global_params = get_parameters(global_net)
    
    # Server-side Optimizer states (for FedAdam, FedYogi)
    server_m = [np.zeros_like(p) for p in global_params]
    server_v = [np.zeros_like(p) for p in global_params]
    server_lr = 0.01
    beta_1, beta_2, tau = 0.9, 0.999, 1e-3

    round_history: list[dict[str, Any]] = []"""
content = content.replace(old_init, new_init)

# Replace Aggregation Loop
old_agg = """        # Sample-weighted aggregation
        total_samples = sum(n for _, n in fit_results)
        new_params = [np.zeros_like(p) for p in global_params]
        for w, n in fit_results:
            fraction = n / total_samples
            for i, layer in enumerate(w):
                new_params[i] += layer * fraction

        global_params = new_params"""
new_agg = """        # ── Compute Pseudo-Gradient & Aggregation ──
        total_samples = sum(n for _, n in fit_results)
        
        if strategy_name == "qfedavg":
            # q-Fairness aggregation (weights heavily penalized clients)
            q_param = 0.2
            client_losses = []
            for client in clients:
                loss, _, _ = client.evaluate(global_params, config={})
                client_losses.append(loss)
            
            # Re-weight using empirical loss
            weights_sum = 0.0
            new_params = [np.zeros_like(p) for p in global_params]
            for (w, n), loss in zip(fit_results, client_losses):
                q_weight = n * (loss ** q_param)
                weights_sum += q_weight
                for i, layer in enumerate(w):
                    new_params[i] += layer * q_weight
            for i in range(len(new_params)):
                new_params[i] = new_params[i] / weights_sum
            global_params = new_params
            
        else:
            # Standard FedAvg aggregation
            avg_update = [np.zeros_like(p) for p in global_params]
            for w, n in fit_results:
                fraction = n / total_samples
                for i, layer in enumerate(w):
                    avg_update[i] += layer * fraction
            
            # Server-Side Optimization (FedAdam / FedYogi)
            if strategy_name in ["fedadam", "fedyogi"]:
                pseudo_grad = [global_params[i] - avg_update[i] for i in range(len(global_params))]
                for i in range(len(global_params)):
                    # Momentum
                    server_m[i] = beta_1 * server_m[i] + (1 - beta_1) * pseudo_grad[i]
                    # Variance
                    if strategy_name == "fedadam":
                        server_v[i] = beta_2 * server_v[i] + (1 - beta_2) * np.square(pseudo_grad[i])
                    elif strategy_name == "fedyogi":
                        delta_sq = np.square(pseudo_grad[i])
                        server_v[i] = server_v[i] - (1 - beta_2) * delta_sq * np.sign(server_v[i] - delta_sq)
                    
                    # Update
                    global_params[i] = global_params[i] - server_lr * server_m[i] / (np.sqrt(server_v[i]) + tau)
            else:
                # Standard FedAvg / FedProx
                global_params = avg_update"""
content = content.replace(old_agg, new_agg)

with open(file_path, "w") as f:
    f.write(content)

