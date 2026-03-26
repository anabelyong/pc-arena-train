import math
import torch
import torch.multiprocessing as mp
import argparse
import os
import torch.distributed as dist
import pyjuice as juice
import socket
import shutil
import re
import time

from omegaconf import OmegaConf
import sys

sys.path.append("../")

from src.utils import instantiate_from_config, collect_data_from_dsets, ProgressBar
from src.data.subsampler import DistributedSubsetSampler
from src.sgd import SGDWrapper

from src.layers.monarchlayer import create_monarch_layers, create_dense_layer
from src.structures.HCLTMonarch import HCLTGeneral
sys.setrecursionlimit(15000)

import wandb

def ddp_setup(rank: int, world_size: int, port: int):
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = str(port)

    backend = 'gloo' if os.name == 'nt' else 'nccl'
    torch.distributed.init_process_group(backend = backend, rank = rank, world_size = world_size)


def get_free_port():
    sock = socket.socket()
    sock.bind(('', 0))
    return sock.getsockname()[1]


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)


def copy_configs(path, args):
    mkdir_p(path)
    shutil.copy(os.path.join("../configs/data/", args.data_config + ".yaml"), os.path.join(path, "data_config.yaml"))
    shutil.copy(os.path.join("../configs/model/", args.model_config + ".yaml"), os.path.join(path, "model_config.yaml"))
    shutil.copy(os.path.join("../configs/optim/", args.optim_config + ".yaml"), os.path.join(path, "optim_config.yaml"))


def find_largest_epoch(file_path):
    # Regular expression to match epoch numbers
    epoch_pattern = re.compile(r'\[Epoch (\d+)\]')
    
    # Read the file content
    with open(file_path, 'r') as file:
        file_content = file.read()
    
    # Find all epoch numbers
    epochs = epoch_pattern.findall(file_content)
    
    # Convert epoch numbers to integers and find the maximum
    if epochs:
        epochs = list(map(int, epochs))
        largest_epoch = max(epochs)
        return largest_epoch
    else:
        return None


def resolve_tuple(*args):
    return tuple(args)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train PC with DDP")

    parser.add_argument("--data-config", type = str, default = "imagenet32")
    parser.add_argument("--model-config", type = str, default = "hclt_256")
    parser.add_argument("--optim-config", type = str, default = "full_em")

    parser.add_argument("--gpu-batch-size", type = int, default = 256)
    
    parser.add_argument("--layer-type", type=str, default="dense", choices=["dense", "monarch"], help="Type of layer to use for the model.")

    parser.add_argument("--resume", default = False, action = "store_true")

    parser.add_argument("--port", type = int, default = 0)

    # Weights & Biases (wandb) minimal options
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging (rank 0 only)")
    parser.add_argument("--wandb-project", type=str, default="pc-arena", help="Wandb project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="Wandb run name")

    args = parser.parse_args()

    return args


def main(rank, world_size, args):
    torch.set_num_threads(8)
    OmegaConf.register_new_resolver('as_tuple', resolve_tuple)

    ddp_setup(rank, world_size, args.port)

    device = torch.device(f'cuda:{rank}')

    epoch_start = 1

    # Logging
    base_folder = os.path.join("logs/", f"{args.data_config}/[{args.model_config}]-[{args.optim_config}]-[pseudocount-1e-6]/")
    mkdir_p(base_folder)
    config_folder = os.path.join(base_folder, "configs/")
    copy_configs(config_folder, args)
    logfile = os.path.join(base_folder, "logs.txt")
    pcfile_last = os.path.join(base_folder, "last.jpc")
    pcfile_best = os.path.join(base_folder, "best.jpc")

    # Dataset
    data_config = OmegaConf.load(os.path.join("../configs/data/", args.data_config + ".yaml"))
    if args.gpu_batch_size > 0:
        data_config["params"]["batch_size"] = args.gpu_batch_size
    dsets = instantiate_from_config(data_config)
    dsets.setup()
    print(f"[rank {rank}] Dataset prepared")

    # Model
    if rank == 0:
        if args.resume:
            assert os.path.exists(pcfile_last) and os.path.exists(logfile)
            root_ns = juice.load(pcfile_last)
            epoch_start = find_largest_epoch(logfile) + 1
            print(f"[rank {rank}] PC loaded...")
        else:
            model_config = OmegaConf.load(os.path.join("../configs/model/", args.model_config + ".yaml"))
            model_kwargs = {}
            for k, v in model_config["params"].items():
                if isinstance(v, str) and v.startswith("__train_data__:"):
                    num_samples = int(v.split(":")[1])
                    data = collect_data_from_dsets(dsets, num_samples = num_samples, split = "train")
                    model_config["params"].pop(k, None)
                    model_kwargs[k] = data.cuda()
            
            if "monarch" in args.model_config:
                layer_fn = create_monarch_layers
                if 'num_latents' in model_config['params']:
                    model_config['params']['block_size'] = int(model_config['params']['num_latents'] ** 0.5)

                model_config['params']['homogeneous_inputs'] = True
                model_kwargs['layer_fn'] = layer_fn
            
            print(f"[rank {rank}] Constructing PC...")
            root_ns = instantiate_from_config(model_config, recursive = True, **model_kwargs)
            juice.save(pcfile_last, root_ns)
            print(f"[rank {rank}] PC constructed and saved...")

    dist.barrier()

    if rank != 0:
        time.sleep(10)
        root_ns = juice.load(pcfile_last)
        print(f"[rank {rank}] PC loaded...")
        if args.resume:
            epoch_start = find_largest_epoch(logfile) + 1

    dist.barrier()

    pc = juice.compile(root_ns)
    #if rank == 0:
        #pc.print_statistics()
    pc.to(device)

    # Data loader
    SUBSET_SIZE = 1000000
    train_sampler = DistributedSubsetSampler(
        dsets.datasets["train"], 
        subset_size=SUBSET_SIZE, 
        shuffle=True
    )
    tr_loader = dsets._train_dataloader(sampler = train_sampler)
    vl_loader = dsets._val_dataloader()
    print(f"[rank {rank}] Dataloaders constructed")

    # Optimizer
    optim_config = OmegaConf.load(os.path.join("../configs/optim/", args.optim_config + ".yaml"))
    optim_mode = optim_config["mode"]
    num_epochs = optim_config["num_epochs"]
    if optim_mode == "full_em":
        momentum = 0.0
        step_size = 1.0
        niters_per_update = len(tr_loader)
    elif optim_mode == "mini_em" or optim_mode == "mini_em_scaled":
        step_size = optim_config["step_size"]
        niters_per_update = optim_config["batch_size"] // world_size // args.gpu_batch_size
        assert niters_per_update > 0, f"Please make sure that [gpu_batch_size ({args.gpu_batch_size})] x [world_size ({world_size})] <= [batch_size ({optim_config['batch_size']})]"
        momentum = 0.0
        if "momentum" in optim_config:
            momentum = optim_config["momentum"]
    elif optim_mode == "adam":
        momentum = 0.0
        step_size = optim_config["lr"]
        niters_per_update = optim_config["batch_size"] // world_size // args.gpu_batch_size
        cum_batch_size = optim_config["batch_size"]
        assert niters_per_update > 0, f"Please make sure that [gpu_batch_size ({args.gpu_batch_size})] x [world_size ({world_size})] <= [batch_size ({optim_config['batch_size']})]"

        pcopt = SGDWrapper(pc)

        adam_kwargs = dict()
        if "beta1" in optim_config:
            adam_kwargs["beta1"] = optim_config["beta1"]
        if "beta2" in optim_config:
            adam_kwargs["beta2"] = optim_config["beta2"]
    else:
        raise NotImplementedError()
    print(f"[rank {rank}] Optimizer set")

    # Scheduler (optional)
    if "scheduler" in optim_config:
        lr_scheduler = instantiate_from_config(optim_config["scheduler"])
    else:
        lr_scheduler = None

    # Initialize Weights & Biases on rank 0 if requested (minimal)
    use_wandb = (rank == 0) and bool(getattr(args, "wandb", False))
    wandb_run = None
    if use_wandb:
        if wandb is None:
            print("[rank 0] wandb is not installed; proceeding without logging.")
            use_wandb = False
        else:
            run_name = args.wandb_name or f"{args.data_config}-{args.model_config}-{args.optim_config}-{time.strftime('%Y%m%d_%H%M%S')}"
            try:
                wandb_run = wandb.init(project=args.wandb_project, name=run_name, config={
                    "data_config": args.data_config,
                    "model_config": args.model_config,
                    "optim_config": args.optim_config,
                    "gpu_batch_size": args.gpu_batch_size,
                    "layer_type": args.layer_type,
                    "world_size": world_size,
                    "optim_mode": optim_mode,
                    "niters_per_update": niters_per_update,
                })
                print(f"[rank 0] wandb initialized: project={args.wandb_project}, name={run_name}")
            except Exception as e:
                print(f"[rank 0] Failed to initialize wandb: {e}. Proceeding without wandb.")
                use_wandb = False

    dist.barrier()

    # Sanity check
    for x in tr_loader:
        assert x.dim() == 2 and x.size(1) == pc.num_vars
        break
    
    # CUDA Graph
    for batch in tr_loader:
        x = batch.to(device)

        with torch.cuda.device(f'cuda:{rank}'):
            lls = pc(x, propagation_alg = "LL", record_cudagraph = True)
            pc.backward(x, flows_memory = 1.0, allow_modify_flows = False,
                        propagation_alg = "LL", logspace_flows = True)

        break

    # Main training loop
    if rank == 0:
        progress_bar = ProgressBar(num_epochs, len(tr_loader), ["LL"], cumulate_statistics = True)
        progress_bar.set_epoch_id(epoch_start - 1)
        best_val_ll = -torch.inf

    pc.init_param_flows(flows_memory = 0.0)

    # Momentum
    if momentum > 0.0:
        momentum_flows = dict()
        momentum_flows["sum_flows"] = torch.zeros(pc.param_flows.size(), dtype = torch.float32, device = device)
        for i, layer in enumerate(pc.input_layer_group):
            momentum_flows[f"input_flows_{i}"] = torch.zeros(layer.param_flows.size(), dtype = torch.float32, device = device)

    step_count = 0
    global_step_count = 0
    wandb_step = 0

    for epoch in range(epoch_start, num_epochs + 1):
        train_sampler.set_epoch(epoch)
        if rank == 0:
            progress_bar.new_epoch_begin()

        for x in tr_loader:

            x = x.to(device)

            with torch.cuda.device(f'cuda:{rank}'):
                lls = pc(x, propagation_alg = "LL")
                pc.backward(x, flows_memory = 1.0, allow_modify_flows = False,
                            propagation_alg = "LL", logspace_flows = True)

                if optim_mode == "adam":
                    pcopt.partition_eval(negate_pflows = True)
                    lls = lls - pc.node_mars[-1,0]

            curr_ll = lls.mean().detach().cpu().numpy().item()

            stats = torch.tensor([curr_ll]).to(device)
            dist.all_reduce(stats, op = dist.ReduceOp.SUM)
            if rank == 0:
                curr_ll = stats[0].item() / world_size

                progress_bar.new_batch_done([curr_ll])
                # per-batch logging
                if use_wandb:
                    wandb.log({
                        "train/ll": curr_ll,
                        "trainer/epoch": epoch,
                        "trainer/step_in_epoch": step_count,
                        "trainer/global_step": wandb_step,
                    }, step=wandb_step)
                    wandb_step += 1

            step_count += 1
            if step_count >= niters_per_update:
                step_count = 0

                if lr_scheduler is not None:
                    step_size = lr_scheduler.step()

                dist.barrier()
                dist.all_reduce(pc.param_flows, op = dist.ReduceOp.SUM)
                for layer in pc.input_layer_group:
                    dist.all_reduce(layer.param_flows, op = dist.ReduceOp.SUM)
                dist.barrier()

                if optim_mode != "adam":
                    if optim_mode == "mini_em_scaled":
                        pc._cum_flow *= world_size

                        if momentum > 0.0:
                            with torch.no_grad():
                                pc.param_flows.mul_(1.0 - momentum)
                                momentum_flows["sum_flows"].mul_(momentum)
                                momentum_flows["sum_flows"].add_(pc.param_flows)
                                pc.param_flows[:] = momentum_flows["sum_flows"]
                                pc.param_flows.div_(1.0 - math.pow(momentum, global_step_count + 1))
                                for i, layer in enumerate(pc.input_layer_group):
                                    layer.param_flows.mul_(1.0 - momentum)
                                    momentum_flows[f"input_flows_{i}"].mul_(momentum)
                                    momentum_flows[f"input_flows_{i}"].add_(layer.param_flows)
                                    layer.param_flows[:] = momentum_flows[f"input_flows_{i}"]
                                    layer.param_flows.div_(1.0 - math.pow(momentum, global_step_count + 1))

                    with torch.cuda.device(f'cuda:{rank}'):
                        pc.mini_batch_em(step_size = step_size, pseudocount = 1e-6, step_size_rescaling = (optim_mode == "mini_em_scaled"))
                else:
                    pcopt.apply_update(cum_batch_size, step_size, **adam_kwargs)
                    pcopt.normalize_by_flows()

                pc.init_param_flows(flows_memory = 0.0)

                global_step_count += 1
                if rank == 0 and use_wandb:
                    wandb.log({
                        "optim/step_size": float(step_size),
                        "trainer/global_update": global_step_count,
                    }, step=wandb_step)

        if rank == 0:
            aveg_train_ll = progress_bar.epoch_ends()[0]
            with open(logfile, "a+") as f:
                f.write(f"[Epoch {epoch:05d}] - Aveg train LL: {aveg_train_ll:.4f}; Step size: {step_size:.4f}\n")
            if use_wandb:
                wandb.log({
                    "train/ll_epoch": aveg_train_ll,
                    "optim/step_size": float(step_size),
                    "trainer/epoch": epoch,
                }, step=wandb_step)

        if epoch % 5 == 0:
            local_ll_sum = 0.0
            for x in vl_loader:

                x = x.to(device)

                with torch.cuda.device(f'cuda:{rank}'):
                    lls = pc(x, propagation_alg = "LL")

                    if optim_mode == "adam":
                        pcopt.partition_eval(negate_pflows = True)
                        lls = lls - pc.node_mars[-1,0]
                
                local_ll_sum += lls.mean().item()

            stats = torch.tensor([local_ll_sum]).to(device)
            dist.all_reduce(stats, op = dist.ReduceOp.SUM)
            if rank == 0:
                aveg_valid_ll = stats[0].item() / world_size / len(vl_loader)
                print(f"[Epoch {epoch:05d}] - Aveg validation LL: {aveg_valid_ll:.4f}")

                with open(logfile, "a+") as f:
                    f.write(f"[Epoch {epoch:05d}] - Aveg validation LL: {aveg_valid_ll:.4f}\n")

                if use_wandb:
                    wandb.log({
                        "valid/ll": aveg_valid_ll,
                        "trainer/epoch": epoch,
                    }, step=wandb_step)

                juice.save(pcfile_last, pc)

                if aveg_valid_ll > best_val_ll:
                    best_val_ll = aveg_valid_ll
                    shutil.copy(pcfile_last, pcfile_best)

                print("> PC saved.")

            dist.barrier()

    # Finish wandb run gracefully
    if rank == 0 and use_wandb and wandb is not None:
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    world_size = torch.cuda.device_count()

    args = parse_arguments()

    if args.port == 0:
        port = get_free_port()
        args.port = port

    if world_size == 1:
        main(0, world_size, args)
    else:
        mp.spawn(
            main,
            args = (world_size, args),
            nprocs = world_size,
        )  # nprocs - total number of processes - # gpus