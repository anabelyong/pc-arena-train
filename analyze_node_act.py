import torch
import torch.nn.functional as F
import argparse
import os
import pyjuice as juice
import numpy as np
from omegaconf import OmegaConf
import sys
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

sys.path.append("../../")
from src.utils import instantiate_from_config, collect_data_from_dsets, ProgressBar

def calculate_total_entropy(root_ns, k=1.0, verbose=True):
    all_node_entropies = []
    num_sum_groups = 0
    
    for ns in root_ns:
        if not ns.is_sum():
            continue
                
        source_ns = ns.get_source_ns()
        num_sum_groups += 1
        params = source_ns.get_params()
        
        if params is None:
            continue

        if source_ns == root_ns:
            probs = params.view(1, -1)
        else:
            block_size = source_ns.block_size
            num_nodes = source_ns.num_nodes
            num_blocks = num_nodes // block_size
            probs = params.reshape(num_blocks, num_blocks, block_size, block_size).permute(0, 2, 1, 3).reshape(num_nodes, num_nodes)

        # power scaling to sharpen the distribution
        # p_new = p^k / sum(p^k)
        if k != 1.0:
            probs_k = torch.pow(probs, k)
            perturbed_probs = probs_k / (torch.sum(probs_k, dim=1, keepdim=True) + 1e-12)
        else:
            perturbed_probs = probs

        entropies_per_node = -torch.sum(perturbed_probs * torch.log(perturbed_probs + 1e-9), dim=1)
        avg_entropy_for_ns = torch.mean(entropies_per_node)
            
        if verbose:
            if source_ns == root_ns:
                print(f"  - Root SumNode: Avg. Edge Entropy = {avg_entropy_for_ns.item():.6f}")
            else:
                print(f"  - Intermediate SumNode Group #{num_sum_groups}: Avg. Entropy = {avg_entropy_for_ns.item():.6f}")
            
        all_node_entropies.append(avg_entropy_for_ns.item())

    if not all_node_entropies:
        return 0.0

    total_average_entropy = sum(all_node_entropies) / len(all_node_entropies)
    
    if verbose:
        print(f"\nFound and processed {num_sum_groups} sum node group(s).")
        
    return total_average_entropy

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Prune dead nodes from a trained Probabilistic Circuit based on value or flow")

    parser.add_argument("--model-config", type=str, required=True, help="Path to the trained PC checkpoint (.jpc file)")
    parser.add_argument("--optim-config", type=str, required=True, help="Path to the trained PC checkpoint (.jpc file)")
    parser.add_argument("--data-config", type=str, required=True, help="Name of the data configuration file (e.g., 'mnist')")

    parser.add_argument("--dead-node-factor", type=float, default=10, help="Factor to determine the dead node threshold. threshold = (1/num_children) / factor.")
    parser.add_argument("--device-id", type=int, default=0, help="GPU device ID to use")
    parser.add_argument("--gpu-batch-size", type=int, default=128, help="Batch size for GPU processing")

    return parser.parse_args()

def evaluate_log_likelihood(pc, data_loader, device):
    pc.eval()
    log_likelihoods = []
    with torch.no_grad():
        for x_batch in tqdm(data_loader, desc="Evaluating LL"):
            x_batch = x_batch.to(device)
            lls = pc(x_batch)
            log_likelihoods.append(lls.cpu())
    
    return torch.cat(log_likelihoods).mean().item()

def main():
    args = parse_arguments()
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device_id}')
        print(f"Using GPU: {device}")
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU.")

    print(f"Loading dataset from config: '{args.data_config}'...")
    data_config_path = os.path.join("../../configs/data/", args.data_config + ".yaml")
    data_config = OmegaConf.load(data_config_path)
    # data_config["params"]["batch_size"] = args.gpu_batch_size
    data_config["params"]["batch_size"] = 32

    dsets = instantiate_from_config(data_config)
    dsets.setup()
    
    train_loader = dsets._train_dataloader()
    val_loader = dsets._val_dataloader()
    print("Dataset and dataloaders are ready.")
    
    # Load and compile the original PC
    model_config_path = os.path.join("../../configs/model/", args.model_config + ".yaml")
    model_config = OmegaConf.load(model_config_path)
    model_kwargs = {}
    for k, v in model_config["params"].items():
        if isinstance(v, str) and v.startswith("__train_data__:"):
            num_samples = int(v.split(":")[1])
            data = collect_data_from_dsets(dsets, num_samples = num_samples, split = "train")
            model_config["params"].pop(k, None)
            model_kwargs[k] = data.cuda()
            
    folder_name = f"/nfs-shared-2/anji/{args.data_config}/[{args.model_config}]-[{args.optim_config}]/"
    base_folder = os.path.join("logs/", folder_name)
    pc_filename = f"best.jpc"
    pc_filepath = os.path.join(base_folder, pc_filename)
    if os.path.exists(pc_filepath):
        print(f"PC checkpoint found at: {pc_filepath}")
        root_ns = juice.load(pc_filepath)
    else:
        print(f"Constructing PC...")
        print(f"Not found at {pc_filepath}")
        root_ns = instantiate_from_config(model_config, recursive = True, **model_kwargs)
        print(f"PC constructed...")
    
    pc = juice.compile(root_ns)
    pc.to(device)
    pc.print_statistics()
    
    # original_params = [p.clone() for p in pc.params]
    # original_ll = evaluate_log_likelihood(pc, val_loader, device)
    # print(f"\nOriginal PC Validation LL: {original_ll:.4f}")
    # original_entropy = calculate_total_entropy(root_ns, k=1.0, verbose=False)
    # print(f"\nOriginal Overall Average Entropy: {original_entropy:.4f}")
    
    print("\nCalculating average activations over the training set...")
    print("\nGathering all sum nodes from the compiled circuit...")
    all_sum_nodes = []
    seen_sum_nodes = set()
    for ns in pc.root_ns: 
        if ns.is_sum():
            if ns not in seen_sum_nodes:
                all_sum_nodes.append(ns)
                seen_sum_nodes.add(ns)
    print(f"Found {len(all_sum_nodes)} unique sum node groups.")

    cache_dir = "./prob_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_filename = os.path.basename(model_config_path).replace('.yaml', f'_{args.optim_config}_{args.data_config}_avg_probs_bs_{data_config["params"]["batch_size"]}.pt')
    cache_filepath = os.path.join(cache_dir, cache_filename)
    
    if os.path.exists(cache_filepath):
        print(f"\nLoading cached average probabilities from '{cache_filepath}'...")
        cached_data = torch.load(cache_filepath)
        final_avg_val_probs = cached_data['val']
        final_avg_flow_probs = cached_data['flow']
        print("Loading complete.")
    else:
        print("\nCache not found. Calculating average probabilities。..")
    
        sum_val_probs = {ns: 0.0 for ns in all_sum_nodes}
        sum_flow_probs = {ns: 0.0 for ns in all_sum_nodes}
        total_samples = 0

        with torch.no_grad():
            for x_batch in tqdm(train_loader, desc="Forward/Backward Pass"):
                x_batch = x_batch.to(device)
                
                lls = pc(x_batch)
                pc.backward(x_batch, allow_modify_flows=False, logspace_flows=True)

                first_node = next(iter(all_sum_nodes))
                num_batch_samples = pc.get_node_mars(first_node).shape[1]
                total_samples += num_batch_samples

                all_val_entropies = []
                all_flow_entropies = []

                for ns in all_sum_nodes:
                    val_acts = pc.get_node_mars(ns).detach()
                    flow_acts = pc.get_node_flows(ns).detach()
                    
                    val_probs = F.softmax(val_acts, dim=0)
                    flow_probs = F.softmax(flow_acts, dim=0)

                    val_entropy = -torch.sum(val_probs * torch.log(val_probs + 1e-9), dim=0)
                    flow_entropy = -torch.sum(flow_probs * torch.log(flow_probs + 1e-9), dim=0)
                    
                    all_val_entropies.append(val_entropy)
                    all_flow_entropies.append(flow_entropy)

                stacked_val_entropy = torch.stack(all_val_entropies) 
                sample_wise_val_mean = torch.mean(stacked_val_entropy, dim=0)
                sample_wise_val_var  = torch.var(stacked_val_entropy, dim=0)

                for i in range(val_acts.shape[1]):
                    print(f"Sample {i}: Mean Entropy across nodes: {sample_wise_val_mean[i]:.4f}, Var: {sample_wise_val_var[i]:.4f}")
                    
                num_samples = stacked_val_entropy.shape[1]  # 32
                rows, cols = 4, 8
                fig, axes = plt.subplots(rows, cols, figsize=(20, 10), constrained_layout=True)
                axes = axes.flatten()

                for i in range(num_samples):
                    # Data for the current sample across all node sets
                    # sample_entropies = stacked_val_entropy[:, i].cpu().numpy()
                    
                    # node_indices = np.arange(len(sample_entropies)) # 0 to 127

                    # axes[i].bar(node_indices, sample_entropies, color='skyblue', width=1.0)
                    # axes[i].set_title(f"Sample {i}", fontsize=9)
                    # axes[i].set_xlabel("Node Set Index", fontsize=7)
                    # axes[i].set_ylabel("Entropy", fontsize=7)
                    
                    sample_entropies = stacked_val_entropy[:, i].cpu().numpy()
    
                    axes[i].hist(sample_entropies, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
                    
                    # Adding titles and labels
                    axes[i].set_title(f"Sample {i}", fontsize=9)
                    axes[i].tick_params(axis='both', which='major', labelsize=8)
                    axes[i].set_ylim(0, 150)

                fig.suptitle("Entropy Distribution across Node Sets (Per Sample)", fontsize=16)
                plt.savefig(f"entropy_distribution_per_sample_{args.model_config}_{args.optim_config}.png")
                import pdb; pdb.set_trace()

        print("\nCalculating final average probabilities...")
        final_avg_val_probs = dict()
        final_avg_flow_probs = dict()

        for ns in tqdm(all_sum_nodes, desc="Calculating final averages"):
            final_avg_val_probs[ns] = sum_val_probs[ns] / total_samples
            final_avg_flow_probs[ns] = sum_flow_probs[ns] / total_samples

        print("Average probabilities calculated. Caching results...")

        torch.save({
            'val': list(final_avg_val_probs.values()), 
            'flow': list(final_avg_flow_probs.values())
        }, cache_filepath)
        print(f"Saved cache to '{cache_filepath}'")

    # value
    print(f"\n---------- Pruning by value (factor = {args.dead_node_factor}) ----------")
    total_dead_val_nodes = 0
    num_latents = 512

    for i, ns in enumerate(tqdm(all_sum_nodes, desc="Pruning dead value nodes")):
        block_size = ns.block_size
        num_blocks = num_prod_blocks = num_latents // ns.block_size

        if torch.any(dead_indices_mask):
            total_dead_val_nodes += torch.sum(dead_indices_mask).item()
            parameter_matrix = ns.get_source_ns().get_params().reshape(num_blocks, num_blocks, block_size, block_size).permute(0, 2, 1, 3).reshape(num_latents, num_latents)
            parameter_matrix[:, dead_indices_mask] = 0
            ns.get_source_ns().set_params(parameter_matrix)

    pc = juice.compile(root_ns)
    pc.to(device)
    print(f"Pruning complete. Avg dead value nodes: {total_dead_val_nodes / len(all_sum_nodes)}")
    
    val_pruned_ll = evaluate_log_likelihood(pc, val_loader, device)
    print(f"\nValue-Pruned PC Validation LL: {val_pruned_ll:.4f}")
    print(f"   - Change in LL: {val_pruned_ll - original_ll:+.4f}")
    val_pruned_entropy = calculate_total_entropy(root_ns, k=1.0, verbose=False)
    print(f"\nValue pruned Overall Average Entropy: {val_pruned_entropy:.4f}")
    juice.save(pc_val_pruning_filename, root_ns)
    print(f"Pruned PC saved to '{pc_val_pruning_filename}'")

    with torch.no_grad():
        for i, p_orig in enumerate(original_params):
            pc.params[i].data.copy_(p_orig.data)

    # flow
    print(f"\n---------- Pruning by flow (factor = {args.dead_node_factor}) ----------")
    total_dead_flow_nodes = 0
    
    for i, ns in enumerate(tqdm(all_sum_nodes, desc="Pruning dead flow nodes")):
        avg_probs = final_avg_flow_probs[i]
        flow_threshold = (1.0 / num_latents) / args.dead_node_factor
        dead_indices_mask = avg_probs.to(device) < flow_threshold
        block_size = ns.block_size
        num_blocks = num_prod_blocks = num_latents // ns.block_size
        
        if torch.any(dead_indices_mask):
            total_dead_flow_nodes += torch.sum(dead_indices_mask).item()
            parameter_matrix = ns.get_source_ns().get_params().reshape(num_blocks, num_blocks, block_size, block_size).permute(0, 2, 1, 3).reshape(num_latents, num_latents)
            parameter_matrix[:, dead_indices_mask] = 0

            ns.get_source_ns().set_params(parameter_matrix)

    pc = juice.compile(root_ns)
    pc.to(device)
    print(f"Pruning complete. Avg dead flow nodes: {total_dead_flow_nodes  / len(all_sum_nodes)}")

    # Evaluate the modified PC
    flow_pruned_ll = evaluate_log_likelihood(pc, val_loader, device)
    print(f"\nFlow-Pruned PC Validation LL: {flow_pruned_ll:.4f}")
    print(f"   - Change in LL: {flow_pruned_ll - original_ll:+.4f}")
    flow_pruned_entropy = calculate_total_entropy(root_ns, k=1.0, verbose=False)
    print(f"Flow pruned Overall Average Entropy: {flow_pruned_entropy:.4f}")
    juice.save(pc_flow_pruning_filename, root_ns)
    print(f"Pruned PC saved to '{pc_flow_pruning_filename}'")

if __name__ == "__main__":
    main()