import cvxpy as cp
import numpy as np
import json
import argparse

MIN_LORA_EXPONENT = 3
MAX_LORA_EXPONENT = 7

def solve_lora_optimization(n_layers, h_attn, m_mlp, b, s, total_gpus, gpu_memory, base_weights_memory, base_activations_memory, frag_memory, lora_rank_limits, fsdp_level=None):
    # Decision variable for FSDP level (if not specified)
    if fsdp_level is None:
        fsdp_level = cp.Variable(integer=True)  # Decision variable to choose FSDP level
        fsdp_constraints = [fsdp_level >= 0, fsdp_level <= 3]
    else:
        fsdp_constraints = []  # No constraints if fsdp_level is fixed

    # Decision variable for the number of GPUs per group (dynamically solve per group)
    gpus_per_group = cp.Variable(total_gpus, integer=True)

    # Binary variable to indicate whether a group is active (uses GPUs)
    is_active = cp.Variable(total_gpus, boolean=True)

    # Constraints: gpus_per_group must be a power of two and sum to total_gpus
    gpu_constraints = [
        gpus_per_group >= 0,  # Allow gpus_per_group to be zero
        cp.log(gpus_per_group + 1e-6) / cp.log(2) == cp.floor(cp.log(gpus_per_group + 1e-6) / cp.log(2)),
        cp.sum(gpus_per_group) <= total_gpus  # Total GPUs used must be <= total_gpus
    ]

    # Link is_active to gpus_per_group using a big-M constraint
    M = total_gpus  # Big M constant (can be set to a large number, e.g., total_gpus)
    for i in range(total_gpus):
        gpu_constraints.append(gpus_per_group[i] <= M * is_active[i])  # If not active, gpus_per_group[i] must be 0
        gpu_constraints.append(gpus_per_group[i] >= 1 - M * (1 - is_active[i]))  # If active, gpus_per_group[i] must be > 0

    # Decision variables for the number of LoRA configurations per group
    num_loras_per_group = cp.Variable(total_gpus, integer=True)

    # Ensure that when gpus_per_group[i] == 0, num_loras_per_group[i] == 0
    for i in range(total_gpus):
        gpu_constraints.append(num_loras_per_group[i] <= M * is_active[i])  # If not active, num_loras_per_group[i] must be 0
        gpu_constraints.append(num_loras_per_group[i] >= 0)  # Non-negative number of LoRAs when active


    MAX_LORAS = 10  # Assuming a fixed maximum number of LoRAs per group
    M = 100

    # Decision variable for LoRA ranks in each group (each group can have up to MAX_LORAS configurations)
    r_lora = []
    for i in range(total_gpus):
        r_lora.append(cp.Variable(MAX_LORAS, integer=True))  # LoRA rank for each config in group i

    # Constraints for LoRA ranks (ensure LoRA ranks are between 2^3 and 2^7, or set to 0 if unused)
    lora_constraints = []
    for i in range(total_gpus):
        for j in range(MAX_LORAS):
            lora_constraints.append(r_lora[i][j] >= MIN_LORA_EXPONENT)
            lora_constraints.append(r_lora[i][j] <= MAX_LORA_EXPONENT)
            lora_constraints.append(r_lora[i][j] <= M * cp.pos(num_loras_per_group[i] - j))  # Set to 0 if unused

    # Tensor and pipeline parallelism sizes for each group
    tp_size = gpus_per_group  # Tensor parallelism is based on the number of GPUs per group
    pp_size = 1  # Assuming no Pipeline Parallelism

    # Compute memory components for each group (only for non-zero LoRA configurations)
    lora_param_memory = []
    for i in range(total_gpus):
        if num_loras_per_group[i].value is not None and num_loras_per_group[i].value > 0:
            lora_param_memory.append(
                cp.sum([n_layers * (16 * h_attn * cp.power(2, r_lora[i][j]) + 12 * m_mlp * cp.power(2, r_lora[i][j])) 
                        for j in range(int(num_loras_per_group[i].value))]
                ) / (tp_size[i] * pp_size) / (1024**2)
            )
        else:
            lora_param_memory.append(0)  # No memory if no LoRAs are assigned

    # LoRA Optimizer Memory (MB) - AdamW optimizer
    lora_opt_memory = []
    lora_grad_memory = []
    for i in range(total_gpus):
        if num_loras_per_group[i].value is not None and num_loras_per_group[i].value > 0:
            optimizer_factor = 2.0  # AdamW requires two buffers (m, v) per parameter
            grad_factor = 1.0  # Gradient is the same size as the parameter memory
            lora_opt_memory.append(lora_param_memory[i] * optimizer_factor)
            lora_grad_memory.append(lora_param_memory[i] * grad_factor)
        else:
            lora_opt_memory.append(0)  # No optimizer memory if no LoRAs are assigned
            lora_grad_memory.append(0)  # No gradient memory if no LoRAs are assigned

    # Adjust for sharding based on FSDP/ZeRO level
    lora_grad_memory_sharded = []
    lora_opt_memory_sharded = []
    lora_param_memory_sharded = []
    for i in range(total_gpus):
        if num_loras_per_group[i].value is not None and num_loras_per_group[i].value > 0:
            lora_grad_memory_sharded.append(
                cp.where(fsdp_level == 1, lora_grad_memory[i], 
                         cp.where(fsdp_level >= 2, lora_grad_memory[i] / tp_size[i], lora_grad_memory[i]))
            )
            lora_opt_memory_sharded.append(
                cp.where(fsdp_level >= 1, lora_opt_memory[i] / tp_size[i], lora_opt_memory[i])
            )
            lora_param_memory_sharded.append(
                cp.where(fsdp_level == 3, lora_param_memory[i] / tp_size[i], lora_param_memory[i])
            )
        else:
            lora_grad_memory_sharded.append(0)
            lora_opt_memory_sharded.append(0)
            lora_param_memory_sharded.append(0)

    # LoRA Activation Memory (MB)
    lora_act_memory = []
    for i in range(total_gpus):
        if num_loras_per_group[i].value is not None and num_loras_per_group[i].value > 0:
            lora_act_memory.append(
                cp.sum([(b * s * (h_attn + m_mlp) * cp.power(2, r_lora[i][j])) / (tp_size[i] * pp_size) / (1024**2)
                        for j in range(int(num_loras_per_group[i].value))])
            )
        else:
            lora_act_memory.append(0)

    # Total LoRA Memory for each group
    lora_total_memory = []
    for i in range(total_gpus):
        if num_loras_per_group[i].value is not None and num_loras_per_group[i].value > 0:
            lora_total_memory.append(
                lora_param_memory_sharded[i] + lora_grad_memory_sharded[i] + lora_opt_memory_sharded[i] + lora_act_memory[i]
            )
        else:
            lora_total_memory.append(0)  # No total memory if no LoRAs are assigned

    # Free GPU memory (MB) - adjusted by FSDP/ZeRO level
    base_total_memory = base_weights_memory + base_activations_memory
    # Binary variables to select the scaling factor based on FSDP level
    is_fsdp_3 = cp.Variable(boolean=True)
    is_fsdp_1_or_2 = cp.Variable(boolean=True)
    is_fsdp_none = cp.Variable(boolean=True)

    # Constraints to ensure only one scale factor is applied
    fsdp_constraints = []
    fsdp_constraints.append(is_fsdp_3 + is_fsdp_1_or_2 + is_fsdp_none == 1)  # Only one can be active

    # Constraints to determine which FSDP level is active
    fsdp_constraints.append(is_fsdp_3 == (fsdp_level == 3))
    fsdp_constraints.append(is_fsdp_1_or_2 == (fsdp_level >= 1) & (fsdp_level <= 2))
    fsdp_constraints.append(is_fsdp_none == (fsdp_level == 0))

    # Apply scaling factors based on FSDP level
    scale_factor = 0.5 * is_fsdp_3 + 0.75 * is_fsdp_1_or_2 + 1.0 * is_fsdp_none

    # Scale the base memory
    base_total_memory *= scale_factor

    free_memory = gpu_memory - (base_total_memory + frag_memory)
    print(f"Free GPU memory is {free_memory}")

    # Objective function: Maximize total memory used by LoRA configurations across all groups
    total_lora_memory = cp.sum(lora_total_memory)
    objective = cp.Maximize(total_lora_memory)

    # Binary variable to indicate if each LoRA configuration has the specified rank
    is_rank = []  # This will be a 2D list where is_rank[i][j] is binary and indicates if r_lora[i][j] == rank
    for i in range(total_gpus):
        is_rank.append([cp.Variable(boolean=True) for j in range(MAX_LORAS)])

    # For each group, for each LoRA configuration, enforce that is_rank[i][j] is 1 if r_lora[i][j] == rank
    rank_constraints = []
    for i in range(total_gpus):
        for j in range(MAX_LORAS):
            for rank, max_count in lora_rank_limits.items():
                # Add constraints to enforce that is_rank[i][j] is 1 if r_lora[i][j] equals the given rank
                rank_constraints.append(is_rank[i][j] <= cp.pos(r_lora[i][j] - rank))  # If not the rank, force to 0
                rank_constraints.append(is_rank[i][j] >= 1 - M * cp.pos(r_lora[i][j] - rank))  # If the rank, force to 1

            # Now, add constraints to ensure the number of LoRAs with each rank in the group does not exceed max_count
            rank_constraints.append(cp.sum(is_rank[i]) <= max_count)

    # Constraints:
    constraints = [total_lora_memory <= free_memory] + fsdp_constraints + gpu_constraints + lora_constraints + rank_constraints

    # Formulate the problem
    prob = cp.Problem(objective, constraints)

    # Solve the problem
    result = prob.solve(solver=cp.GUROBI)  # Use a mixed-integer solver (GUROBI)

    # Output the results
    for i in range(total_gpus):
        if gpus_per_group[i].value is not None and gpus_per_group[i].value > 0:
            print(f"Group {i+1}:")
            print(f"  GPUs per group: {gpus_per_group[i].value}")
            print(f"  LoRA ranks: {[r_lora[i][j].value for j in range(int(num_loras_per_group[i].value))]}")
    print(f"Optimal total memory used: {result / 1024:.2f} MB")
    print(f"Optimal FSDP Level: {fsdp_level.value}")
    return gpus_per_group.value, r_lora.value, fsdp_level.value, result




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
    parser.add_argument('--lora_rank_limits', type=str, required=True, help='List of rank limits')
    parser.add_argument('--fsdp_level', type=int, help='FSDP/ZeRO level (0 = None, 1 = ZeRO-1, 2 = ZeRO-2, 3 = ZeRO-3). If not specified, the best FSDP level will be selected.')

    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    yaml_file_path = "../../recipes/configs/llama3/8B_lora.yaml"

    # Parse the lora_rank_limits argument as a dictionary
    lora_rank_limits = json.loads(args.lora_rank_limits)

    # Call the function with the parsed arguments
    lora_configs, fsdp_level, gpu_config, result = solve_lora_optimization(
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
        lora_rank_limits=lora_rank_limits,  # Pass the converted list
        fsdp_level=args.fsdp_level  # Optional, if not provided, we optimize it
    )

    update_nested_yaml_with_lora_config(yaml_file_path, gpu_config, lora_configs)


