import cvxpy as cp
import yaml
import argparse

def solve_lora_optimization(n_layers, h_attn, m_mlp, b, s, total_gpus, gpu_memory, base_weights_memory, base_activations_memory, frag_memory, K, max_rank_exponent, fsdp_level=None):
    # Decision variable for FSDP level (if not specified)
    if fsdp_level is None:
        fsdp_level = cp.Variable(integer=True)  # Decision variable to choose FSDP level
        fsdp_constraints = [fsdp_level >= 0, fsdp_level <= 3]
    else:
        fsdp_constraints = []  # No constraints if fsdp_level is fixed

    # Decision variable for the number of GPUs per configuration
    # We are solving for a single number, gpus_per_config, which should be a divisor of total_gpus and a power of two
    gpus_per_config = cp.Variable(integer=True)

    # Constraint: gpus_per_config must be a power of two and divide total_gpus
    gpu_constraints = [gpus_per_config >= 1, gpus_per_config <= total_gpus, cp.log2(gpus_per_config) == cp.floor(cp.log2(gpus_per_config))]
    gpu_constraints += [total_gpus % gpus_per_config == 0]  # Ensure that total_gpus is divisible by gpus_per_config

    # LoRA configurations (example of K configurations)
    e_lora = cp.Variable(K, integer=True)  # Decision variable for LoRA ranks

    # LoRA rank as powers of 2: r_lora_k = 2^e_lora_k
    r_lora = cp.power(2, e_lora)

    # Tensor and pipeline parallelism sizes are set based on the number of GPUs per configuration
    tp_size = gpus_per_config  # All GPUs per configuration are used for Tensor Parallelism
    pp_size = 1  # Assuming no Pipeline Parallelism (pp_size is 1)

    # Compute memory components based on LoRA rank
    # LoRA Parameter Memory (MB)
    lora_param_memory = n_layers * (16 * h_attn * r_lora + 12 * m_mlp * r_lora) / (tp_size * pp_size) / (1024**2)
    
    # LoRA Optimizer Memory (MB) - AdamW optimizer
    optimizer_factor = 2.0  # AdamW requires two buffers (m, v) per parameter
    grad_factor = 1.0  # Gradient is the same size as the parameter memory
    lora_opt_memory = lora_param_memory * optimizer_factor  # Two additional states
    lora_grad_memory = lora_param_memory * grad_factor  # Gradient memory

    # Adjust for sharding based on FSDP/ZeRO level
    lora_grad_memory_sharded = cp.Variable(K)  # Memory for gradients under different sharding
    lora_opt_memory_sharded = cp.Variable(K)   # Memory for optimizer state under different sharding
    lora_param_memory_sharded = cp.Variable(K) # Memory for parameters under different sharding

    # Constraints for different FSDP levels
    fsdp_constraints += [
        # Gradient memory: Only sharded in ZeRO-2 and ZeRO-3 (not in ZeRO-1)
        lora_grad_memory_sharded == cp.where(fsdp_level == 1, lora_grad_memory, cp.where(fsdp_level >= 2, lora_grad_memory / tp_size, lora_grad_memory)),

        # Optimizer state: Sharded starting from ZeRO-1
        lora_opt_memory_sharded == cp.where(fsdp_level >= 1, lora_opt_memory / tp_size, lora_opt_memory),

        # Parameter memory: Only sharded in ZeRO-3
        lora_param_memory_sharded == cp.where(fsdp_level == 3, lora_param_memory / tp_size, lora_param_memory)
    ]

    # LoRA Activation Memory (MB)
    lora_act_memory = (b * s * (h_attn + m_mlp) * r_lora) / (tp_size * pp_size) / (1024**2)

    # Total LoRA Memory for each configuration (MB), scaled by the number of GPUs per config
    lora_total_memory = (lora_param_memory_sharded + lora_grad_memory_sharded + lora_opt_memory_sharded + lora_act_memory) * (total_gpus / gpus_per_config)
    print(f"LoRA total memory is {lora_total_memory}")

    # Free GPU memory (MB) - adjusted by FSDP/ZeRO level
    base_total_memory = base_weights_memory + base_activations_memory
    if fsdp_level is None or cp.max(fsdp_level) > 0:  # If FSDP/ZeRO is enabled, scale the activation memory
        base_total_memory *= cp.where(fsdp_level == 3, 0.5, cp.where(fsdp_level >= 1, 0.75, 1.0))

    free_memory = gpu_memory - (base_total_memory + frag_memory)
    print(f"Free GPU memory is {free_memory}")

    # Objective function: Maximize total memory used by LoRA configurations
    objective = cp.Maximize(cp.sum(lora_total_memory))

    # Constraints:
    # 1. The total memory used by all configurations must be less than or equal to the free GPU memory.
    constraints = [cp.sum(lora_total_memory) <= free_memory] + fsdp_constraints + gpu_constraints

    # 2. Ensure that the exponent is bounded (e.g., between 0 and max_rank_exponent)
    constraints += [e_lora >= 0, e_lora <= max_rank_exponent]

    # Formulate the problem
    prob = cp.Problem(objective, constraints)

    # Solve the problem
    result = prob.solve(solver=cp.GLPK_MI)  # Use a mixed-integer solver (GLPK_MI)

    # Output the results
    print(f"Optimal total memory used: {result / 1024:.2f} MB")
    print(f"Optimal LoRA ranks for configurations: {r_lora.value}")
    print(f"Optimal GPUs per configuration: {gpus_per_config.value}")
    print(f"Optimal FSDP Level: {fsdp_level.value}")
    return r_lora.value, fsdp_level.value, gpus_per_config.value, result


def update_nested_yaml_with_lora_config(yaml_file_path, gpus_per_config, r_lora_values):
    # Open and read the existing YAML file
    with open(yaml_file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    # Navigate to the 'model' section and update the 'lora_rank' field
    if 'model' in yaml_data:
        model_config = yaml_data['model']

        # Update the lora_rank with the corresponding r_lora values
        model_config['lora_rank'] = list(map(int, r_lora_values))  # Update LoRA ranks
        
        # Optionally, if you want to set gpus_per_config here or elsewhere in the YAML
        model_config['gpus_per_config'] = int(gpus_per_config)  # Update gpus_per_config

    # Write the updated YAML data back to the file
    with open(yaml_file_path, 'w') as file:
        yaml.dump(yaml_data, file)

    print(f"Updated YAML file '{yaml_file_path}' with new gpus_per_config and lora_rank values.")


# Command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Solve LoRA optimization problem with varying parameters.")
    
    parser.add_argument('--n_layers', type=int, required=True, help='Number of layers')
    parser.add_argument('--h_attn', type=int, required=True, help='Attention hidden size')
    parser.add_argument('--m_mlp', type=int, required=True, help='MLP intermediate size')
    parser.add_argument('--b', type=int, required=True, help='Batch size during the forward pass')
    parser.add_argument('--s', type=int, required=True, help='Sequence length of input during the forward pass')
    parser.add_argument('--total_gpus', type=int, required=True, help='Total number of GPUs available')
    parser.add_argument('--gpu_memory', type=float, required=True, help='Total GPU memory in MB')
    parser.add_argument('--base_weights_memory', type=float, required=True, help='Base model weights memory in MB')
    parser.add_argument('--base_activations_memory', type=float, required=True, help='Base model activations memory in MB')
    parser.add_argument('--frag_memory', type=float, required=True, help='Fragmentation factor in MB')
    parser.add_argument('--K', type=int, required=True, help='Number of LoRA configurations')
    parser.add_argument('--max_rank_exponent', type=int, required=True, help='Maximum rank exponent')
    parser.add_argument('--fsdp_level', type=int, help='FSDP/ZeRO level (0 = None, 1 = ZeRO-1, 2 = ZeRO-2, 3 = ZeRO-3). If not specified, the best FSDP level will be selected.')

    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Call the function with the parsed arguments
    solve_lora_optimization(
        n_layers=args.n_layers,
        h_attn=args.h_attn,
        m_mlp=args.m_mlp,
        b=args.b,
        s=args.s,
        total_gpus=args.total_gpus,
        gpu_memory=args.gpu_memory,
        base_weights_memory=args.base_weights_memory,
        base_activations_memory=args.base_activations_memory,
        frag_memory=args.frag_memory,
        K=args.K,
        max_rank_exponent=args.max_rank_exponent,
        fsdp_level=args.fsdp_level  # Optional, if not provided, we optimize it
    )
