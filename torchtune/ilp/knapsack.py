import cvxpy as cp
import numpy as np
import json
import argparse

# def solve_lora_optimization(n_layers, h_attn, m_mlp, b, s, total_gpus, gpu_memory,
#                             base_weights_memory, base_activations_memory, frag_memory,
#                             lora_rank_limits, fsdp_level=None):
#     MIN_LORA_EXPONENT = 3
#     MAX_LORA_EXPONENT = 7
#     pp_size=1

#     # Possible exponents and corresponding ranks
#     possible_exponents = list(range(MIN_LORA_EXPONENT, MAX_LORA_EXPONENT + 1))
#     possible_ranks = [2 ** exp for exp in possible_exponents]  # [8,16,32,64,128]

#     # Decision variable for FSDP level (if not specified)
#     if fsdp_level is None:
#         fsdp_level_var = cp.Variable(integer=True)  # Decision variable to choose FSDP level
#         fsdp_constraints = [fsdp_level_var >= 0, fsdp_level_var <= 3]
#     else:
#         fsdp_level_var = fsdp_level  # Use the fixed FSDP level
#         fsdp_constraints = []  # No constraints if fsdp_level is fixed

#     # Powers of two for GPUs per group
#     powers_of_two = [1, 2, 4, 8]  # Adjust based on total_gpus
#     num_groups = total_gpus  # Number of groups (adjust as needed)

#     # Binary variable to indicate whether a group is active (uses GPUs)
#     is_active = cp.Variable(num_groups, boolean=True)

#     # Binary variables to select tp_size (gpus_per_group)
#     tp_size_select = [cp.Variable(len(powers_of_two), boolean=True) for _ in range(num_groups)]

#     # Decision variable for number of LoRAs per group
#     num_loras_per_group = cp.Variable(num_groups, integer=True)

#     # Constraints list
#     gpu_constraints = []

#     # Constraint: gpus_per_group[i] == sum of selected powers of two
#     gpus_per_group = []
#     for i in range(num_groups):
#         gpus_per_group_i = cp.sum([tp_size_select[i][k] * powers_of_two[k] for k in range(len(powers_of_two))])
#         gpus_per_group.append(gpus_per_group_i)
#         # Ensure that if group is active, exactly one power of two is selected
#         gpu_constraints.append(cp.sum(tp_size_select[i]) == is_active[i])

#     # Constraint: Total GPUs used must be <= total_gpus
#     gpu_constraints.append(cp.sum(gpus_per_group) <= total_gpus)

#     # Constraints for active/inactive groups
#     M = total_gpus  # Big-M constant
#     MAX_LORAS = max(lora_rank_limits.values())  # Adjust as needed

#     for i in range(num_groups):
#         # If group is not active, num_loras_per_group[i] must be 0
#         gpu_constraints.append(num_loras_per_group[i] <= MAX_LORAS * is_active[i])
#         gpu_constraints.append(num_loras_per_group[i] >= 0)  # Non-negative number of LoRAs

#     # Binary variable indicating if LoRA configuration is used
#     is_used = [[cp.Variable(boolean=True) for _ in range(MAX_LORAS)] for _ in range(num_groups)]

#     lora_constraints = []

#     for i in range(num_groups):
#         for j in range(MAX_LORAS):
#             # Constraints to link is_used[i][j] with num_loras_per_group[i]
#             lora_constraints.append(num_loras_per_group[i] - j - 1 >= -M * (1 - is_used[i][j]))
#             lora_constraints.append(num_loras_per_group[i] - j - 1 <= M * (1 - is_used[i][j]))

#     # Binary variable to select which rank is chosen for each LoRA configuration
#     num_possible_ranks = len(possible_ranks)
#     is_rank = [[[cp.Variable(boolean=True) for _ in range(num_possible_ranks)] for _ in range(MAX_LORAS)] for _ in range(num_groups)]

#     # Constraints to ensure that when a LoRA config is used, exactly one rank is selected
#     rank_constraints = []

#     for i in range(num_groups):
#         for j in range(MAX_LORAS):
#             # For each LoRA configuration, ensure that only one rank is selected when is_used[i][j] == 1
#             rank_constraints.append(cp.sum(is_rank[i][j]) == is_used[i][j])

#     # Precompute reciprocal factors
#     reciprocal_factors = [1.0 / val for val in powers_of_two]

#     # Compute LoRA parameter memory
#     lora_param_memory = []
#     for i in range(num_groups):
#         reciprocal = cp.sum([tp_size_select[i][k] * reciprocal_factors[k] for k in range(len(powers_of_two))]) + (1 - is_active[i])
#         param_memory = (
#             cp.sum([
#                 n_layers * (16 * h_attn * possible_ranks[k2] + 12 * m_mlp * possible_ranks[k2]) * is_rank[i][j][k2]
#                 for j in range(MAX_LORAS) for k2 in range(num_possible_ranks)
#             ]) * reciprocal / pp_size / (1024 ** 2)
#         ) * is_active[i]
#         lora_param_memory.append(param_memory)

#     # LoRA Optimizer and Gradient Memory (MB) - AdamW optimizer
#     lora_opt_memory = []
#     lora_grad_memory = []
#     for i in range(num_groups):
#         optimizer_factor = 2.0  # AdamW requires two buffers (m, v) per parameter
#         grad_factor = 1.0  # Gradient is the same size as the parameter memory
#         opt_memory = lora_param_memory[i] * optimizer_factor
#         grad_memory = lora_param_memory[i] * grad_factor
#         lora_opt_memory.append(opt_memory)
#         lora_grad_memory.append(grad_memory)

#     # Binary variables to select the FSDP level
#     if fsdp_level is None:
#         is_fsdp = [cp.Variable(boolean=True) for _ in range(4)]  # For levels 0 to 3
#         fsdp_constraints = [
#             cp.sum(is_fsdp) == 1,  # Only one FSDP level selected
#             fsdp_level_var == cp.sum([level * is_fsdp[level] for level in range(4)])
#         ]
#     else:
#         # Fixed FSDP level
#         is_fsdp = [1 if fsdp_level_var == level else 0 for level in range(4)]
#         fsdp_constraints = []

#     # Adjust for sharding based on FSDP/ZeRO level
#     lora_grad_memory_sharded = []
#     lora_opt_memory_sharded = []
#     lora_param_memory_sharded = []

#     for i in range(num_groups):
#         reciprocal = cp.sum([tp_size_select[i][k] * reciprocal_factors[k] for k in range(len(powers_of_two))]) + (1 - is_active[i])
#         grad_memory_sharded = (
#             lora_grad_memory[i] * is_fsdp[0] +
#             lora_grad_memory[i] * is_fsdp[1] +
#             (lora_grad_memory[i] * reciprocal) * (is_fsdp[2] + is_fsdp[3])
#         )
#         lora_grad_memory_sharded.append(grad_memory_sharded)

#         opt_memory_sharded = (
#             lora_opt_memory[i] * is_fsdp[0] +
#             (lora_opt_memory[i] * reciprocal) * (is_fsdp[1] + is_fsdp[2] + is_fsdp[3])
#         )
#         lora_opt_memory_sharded.append(opt_memory_sharded)

#         param_memory_sharded = (
#             lora_param_memory[i] * (is_fsdp[0] + is_fsdp[1] + is_fsdp[2]) +
#             (lora_param_memory[i] * reciprocal) * is_fsdp[3]
#         )
#         lora_param_memory_sharded.append(param_memory_sharded)

#     # LoRA Activation Memory (MB)
#     lora_act_memory = []
#     for i in range(num_groups):
#         reciprocal = cp.sum([tp_size_select[i][k] * reciprocal_factors[k] for k in range(len(powers_of_two))]) + (1 - is_active[i])
#         act_memory = (
#             cp.sum([
#                 (b * s * (h_attn + m_mlp) * possible_ranks[k2] * is_rank[i][j][k2])
#                 for j in range(MAX_LORAS) for k2 in range(num_possible_ranks)
#             ]) * reciprocal / pp_size / (1024 ** 2)
#         ) * is_active[i]
#         lora_act_memory.append(act_memory)

#     # Total LoRA Memory for each group
#     lora_total_memory = []
#     for i in range(num_groups):
#         total_memory = (
#             lora_param_memory_sharded[i] + lora_grad_memory_sharded[i] +
#             lora_opt_memory_sharded[i] + lora_act_memory[i]
#         ) * is_active[i]
#         lora_total_memory.append(total_memory)

#     # Free GPU memory (MB) - adjusted by FSDP/ZeRO level
#     base_total_memory = base_weights_memory + base_activations_memory

#     # Apply scaling factors based on FSDP level
#     scale_factor = (
#         1.0 * is_fsdp[0] + 0.75 * (is_fsdp[1] + is_fsdp[2]) + 0.5 * is_fsdp[3]
#     )
#     base_total_memory_scaled = base_total_memory * scale_factor

#     free_memory = gpu_memory - (base_total_memory_scaled + frag_memory)
#     print(f"Free GPU memory is {free_memory}")

#     # Objective function: Maximize total memory used by LoRA configurations across all groups
#     total_lora_memory = cp.sum(lora_total_memory)
#     objective = cp.Maximize(total_lora_memory)

#     # Constraints to ensure the total number of LoRAs with each rank does not exceed max_count
#     rank_count_constraints = []
#     for k2, rank in enumerate(possible_ranks):
#         max_count = lora_rank_limits.get(rank, MAX_LORAS * num_groups)  # If no limit specified, use a large number
#         total_rank_count = cp.sum([is_rank[i][j][k2] for i in range(num_groups) for j in range(MAX_LORAS)])
#         rank_count_constraints.append(total_rank_count <= max_count)

#     # Combine all constraints
#     constraints = [total_lora_memory <= free_memory] + fsdp_constraints + gpu_constraints + lora_constraints + rank_constraints + rank_count_constraints

#     # Formulate the problem
#     prob = cp.Problem(objective, constraints)

#     # Solve the problem
#     result = prob.solve(solver=cp.GUROBI)  # Use a mixed-integer solver (e.g., GUROBI)

#     # Output the results
#     for i in range(num_groups):
#         if is_active[i].value > 0.5:
#             print(f"Group {i+1}:")
#             tp_size = int(sum([powers_of_two[k] * tp_size_select[i][k].value for k in range(len(powers_of_two))]))
#             print(f"  GPUs per group: {tp_size}")
#             num_loras = int(round(num_loras_per_group[i].value))
#             lora_ranks = []
#             for j in range(num_loras):
#                 for k2 in range(num_possible_ranks):
#                     if is_rank[i][j][k2].value > 0.5:
#                         lora_ranks.append(possible_ranks[k2])
#                         break
#             print(f"  LoRA ranks: {lora_ranks}")
#     print(f"Optimal total memory used: {result / 1024:.2f} GB")
#     if fsdp_level is None:
#         fsdp_level_value = int(round(fsdp_level_var.value))
#         print(f"Optimal FSDP Level: {fsdp_level_value}")
#     else:
#         fsdp_level_value = fsdp_level_var
#         print(f"Fixed FSDP Level: {fsdp_level_value}")
#     return gpus_per_group, is_rank, fsdp_level_value, result

# import cvxpy as cp
# import numpy as np

def solve_lora_optimization(n_layers, h_attn, m_mlp, b, s, total_gpus, gpu_memory,
                            base_weights_memory, base_activations_memory, frag_memory,
                            lora_rank_limits, fsdp_level=None):
    MIN_LORA_EXPONENT = 3
    MAX_LORA_EXPONENT = 7
    pp_size = 1 

    # Possible exponents and corresponding ranks
    possible_exponents = list(range(MIN_LORA_EXPONENT, MAX_LORA_EXPONENT + 1))
    possible_ranks = [2 ** exp for exp in possible_exponents]  # [8,16,32,64,128]

    # Possible tensor parallel sizes (powers of two up to total_gpus)
    tp_sizes = [2 ** i for i in range(int(np.log2(total_gpus)) + 1) if 2 ** i <= total_gpus]

    num_groups = len(tp_sizes)  # Each group corresponds to a tp_size

    # Decision variables: Whether to use a specific tp_size
    use_tp_size = [cp.Variable(boolean=True) for _ in range(num_groups)]

    # Decision variables: Number of LoRA configurations for each (tp_size, rank)
    num_possible_ranks = len(possible_ranks)
    num_loras = [[cp.Variable(integer=True) for _ in range(num_possible_ranks)] for _ in range(num_groups)]

    # Constraints
    constraints = []

    # Constraint: Total GPUs used cannot exceed total_gpus
    total_gpus_used = cp.sum([tp_sizes[i] * use_tp_size[i] for i in range(num_groups)])
    constraints.append(total_gpus_used <= total_gpus)

    # Constraint: If tp_size is not used, num_loras must be zero
    for i in range(num_groups):
        for k in range(num_possible_ranks):
            constraints.append(num_loras[i][k] >= 0)
            constraints.append(num_loras[i][k] <= lora_rank_limits.get(possible_ranks[k], total_gpus))  # Apply rank limits
            constraints.append(num_loras[i][k] <= total_gpus * use_tp_size[i])

    # Precompute memory requirements for each configuration
    param_memory = np.zeros((num_groups, num_possible_ranks))
    act_memory = np.zeros((num_groups, num_possible_ranks))

    for i in range(num_groups):
        tp = tp_sizes[i]
        reciprocal = 1.0 / tp
        for k, rank in enumerate(possible_ranks):
            # Parameter Memory (MB)
            mem = n_layers * (16 * h_attn * rank + 12 * m_mlp * rank) * reciprocal / (pp_size * 1024 ** 2)
            param_memory[i][k] = mem

            # Activation Memory (MB)
            mem_act = (b * s * (h_attn + m_mlp) * rank) * reciprocal / (pp_size * 1024 ** 2)
            act_memory[i][k] = mem_act

    # LoRA Optimizer and Gradient Memory (MB) - AdamW optimizer
    optimizer_factor = 2.0  # AdamW requires two buffers (m, v) per parameter
    grad_factor = 1.0  # Gradient is the same size as the parameter memory
    opt_memory = param_memory * optimizer_factor
    grad_memory = param_memory * grad_factor

    # Adjust for sharding based on FSDP/ZeRO level
    if fsdp_level is None:
        fsdp_level_var = cp.Variable(integer=True)
        fsdp_constraints = [fsdp_level_var >= 0, fsdp_level_var <= 3]
        constraints += fsdp_constraints
    else:
        fsdp_level_var = fsdp_level

    # Scaling factors based on FSDP level
    scale_factor = cp.Variable()
    constraints += [
        scale_factor >= 0,
        scale_factor <= 1,
        scale_factor == cp.multiply(1.0, fsdp_level_var == 0) +
                        cp.multiply(0.75, (fsdp_level_var == 1) + (fsdp_level_var == 2)) +
                        cp.multiply(0.5, fsdp_level_var == 3)
    ]

    # Convert precomputed memory into CVXPY expressions using hstack for broadcasting
    param_memory_expr = [cp.hstack(param_memory[i]) for i in range(num_groups)]
    act_memory_expr = [cp.hstack(act_memory[i]) for i in range(num_groups)]
    grad_memory_expr = [cp.hstack(grad_memory[i]) for i in range(num_groups)]
    opt_memory_expr = [cp.hstack(opt_memory[i]) for i in range(num_groups)]
    num_loras_expr = [cp.hstack(num_loras[i]) for i in range(num_groups)]

    # Total memory per group
    total_memory_per_group = []
    for i in range(num_groups):
        # Sum over all ranks
        total_param_mem = cp.sum(cp.multiply(param_memory_expr[i], num_loras_expr[i]))
        total_act_mem = cp.sum(cp.multiply(act_memory_expr[i], num_loras_expr[i]))
        total_grad_mem = cp.sum(cp.multiply(grad_memory_expr[i], num_loras_expr[i]))
        total_opt_mem = cp.sum(cp.multiply(opt_memory_expr[i], num_loras_expr[i]))

        # Adjust for sharding based on FSDP level (simplified here)
        total_group_mem = total_param_mem + total_act_mem + total_grad_mem + total_opt_mem
        total_memory_per_group.append(total_group_mem)

    # Total LoRA memory
    total_lora_memory = cp.sum(total_memory_per_group)

    # Total base model memory (scaled)
    base_total_memory = (base_weights_memory + base_activations_memory) * scale_factor

    # Free GPU memory
    free_memory = gpu_memory - (base_total_memory + frag_memory)
    print(f"Free GPU memory is {free_memory}")

    # Memory constraint
    constraints.append(total_lora_memory <= free_memory)

    # Objective function: Maximize total memory used by LoRA configurations across all groups
    objective = cp.Maximize(total_lora_memory)

    # Combine all constraints
    constraints = [total_lora_memory <= free_memory] + constraints

    # Formulate the problem
    prob = cp.Problem(objective, constraints)

    # Solve the problem
    result = prob.solve(solver=cp.GUROBI)  # Use a mixed-integer solver

    # Output the results
    for i in range(num_groups):
        if use_tp_size[i].value > 0.5:
            print(f"Group with tp_size {tp_sizes[i]}:")
            lora_ranks = []
            lora_counts = []
            for k in range(num_possible_ranks):
                count = int(num_loras[i][k].value)
                if count > 0:
                    lora_ranks.append(possible_ranks[k])
                    lora_counts.append(count)
            print(f"  LoRA ranks and counts: {list(zip(lora_ranks, lora_counts))}")
    print(f"Optimal total memory used: {result / 1024:.2f} GB")
    if fsdp_level is None:
        fsdp_level_value = int(fsdp_level_var.value)
        print(f"Optimal FSDP Level: {fsdp_level_value}")
    else:
        fsdp_level_value = fsdp_level_var
        print(f"Fixed FSDP Level: {fsdp_level_value}")
    return None, None, fsdp_level_value, result



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


