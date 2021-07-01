from typing import Dict, List, Tuple
from types import FunctionType
import os
import time
import argparse
import wandb
from rich.progress import track
from rich import print
import yaml
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as ddp

from view_synthesis.cfgnode import CfgNode
import view_synthesis.datasets as datasets
import view_synthesis.models as network_arch
from view_synthesis.utils import prepare_device, is_main_process, prepare_logging, mse2psnr, get_minibatches
from view_synthesis.nerf import RaySampler, PointSampler, volume_render_radiance_field, PositionalEmbedder


def prepare_dataloader(rank: int, cfg: CfgNode) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """ Prepare the dataloader considering DataDistributedParallel

    :function:
        rank: Process rank. 0 == main process
    :returns: TODO

    """
    dataset = getattr(datasets, cfg.dataset.type)(
        cfg.dataset.basedir, cfg.dataset.resolution_level)
    train_size = int(len(dataset) * 0.75)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_dataloader, val_dataloader = None, None
    if cfg.setup_ddp == True:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=cfg.gpus,
            rank=rank
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=cfg.gpus,
            rank=rank
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(
            train_dataset,
            replacement=True,
            num_samples=cfg.experiment.train_iters
        )
        val_sampler = torch.utils.data.RandomSampler(
            val_dataset,
            replacement=True,
            num_samples=cfg.experiment.train_iters
        )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.dataset.train_batch_size, shuffle=False, num_workers=0, sampler=train_sampler, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.dataset.val_batch_size, shuffle=False, num_workers=0, sampler=val_sampler, pin_memory=True)
    return train_dataloader, val_dataloader


def prepare_models(rank: int, cfg: CfgNode) -> OrderedDict:
    """ Prepare the torch models

    :function:
        rank: Process rank. 0 == main process
    :returns: TODO

    """
    models = OrderedDict()
    models['coarse'] = getattr(network_arch, cfg.models.coarse.type)(
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
    )
    models["coarse"].to(rank)
    if hasattr(cfg.models, "fine"):
        models['fine'] = getattr(network_arch, cfg.models.fine.type)(
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
        )
        models["fine"].to(rank)

    if cfg.setup_ddp == True:
        models["coarse"] = ddp(models['coarse'], device_ids=[rank],
                               output_device=rank)
        if hasattr(cfg.models, "fine"):
            models["fine"] = ddp(models['coarse'], device_ids=[rank],
                                 output_device=rank)

    return models


def prepare_optimizer(rank: int, cfg: CfgNode, models: "OrderedDict[str, torch.nn.Module]") -> Tuple[OrderedDict, OrderedDict]:
    """ Load the optimizer and learning schedule according to the configuration

    :function: TODO
    :returns: TODO

    """
    optimizers, schedulers = OrderedDict(), OrderedDict()
    for model_name, model in models.items():
        optimizers[model_name] = getattr(torch.optim, cfg.optimizer.type)(
            model.parameters(), lr=cfg.optimizer.lr)
        # TODO: Define custom scheduler which handles this from config
        schedulers[model_name] = torch.optim.lr_scheduler.LambdaLR(
            optimizers[model_name], lr_lambda=lambda epoch: 0.1 ** (epoch / cfg.experiment.train_iters))

    return optimizers, schedulers


def load_checkpoint(rank: int, cfg: CfgNode, models: "OrderedDict[str, torch.nn.Module]", optimizers: "OrderedDict[str, torch.optim.Optimizer]") -> int:
    """TODO: Docstring for load_pretrained.

    :function: TODO
    :returns: TODO

    """
    start_iter = 0
    print(Path(cfg.load_checkpoint))
    checkpoint_file = Path(cfg.load_checkpoint)
    if checkpoint_file.exists() and checkpoint_file.is_file() and checkpoint_file.suffix == ".ckpt":
        if cfg.setup_ddp == True:
            map_location = {"cuda:0": f"cuda:{rank}"}
            checkpoint = torch.load(
                cfg.load_checkpoint, map_location=map_location)
        else:
            checkpoint = torch.load(cfg.load_checkpoint)

        for model_name, model in models.items():
            model.load_state_dict(
                checkpoint[f"model_{model_name}_state_dict"])
            optimizers[model_name].load_state_dict(
                checkpoint["optimizer_{model_name}_state_dict"])
        start_iter = checkpoint["start_iter"]

    # Ensure that all loading by all processes is done before any process has started saving models
    torch.distributed.barrier()
    return start_iter


def prepare_embedding_fns(cfg):
    """Load the embedding functions for points and viewdirs from config file

    :function: TODO
    :returns: TODO

    """
    encode_position_fn = get_embedding_function(
        num_encoding_functions=cfg.num_encoding_fn_xyz,
        include_input=cfg.include_input_xyz,
        log_sampling=cfg.log_sampling_xyz,
    )

    encode_direction_fn = None
    if cfg.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.num_encoding_fn_dir,
            include_input=cfg.include_input_dir,
            log_sampling=cfg.log_sampling_dir,
        )

    return encode_position_fn, encode_direction_fn


def train(rank: int, cfg: CfgNode) -> None:
    """
    Main training loop for the model

    :function:
        rank: Process rank. 0 == main process
    :returns: TODO

    """
    # Seed experiment for repeatability (Each process should sample different rays)
    seed = cfg.experiment.randomseed + rank
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set device and logdir_path
    device, logdir_path = None, None
    if is_main_process(rank) or cfg.setup_ddp == False:
        logdir_path = prepare_logging(cfg)
        device = f"cuda:0"
    if cfg.setup_ddp == True:
        device = f"cuda:{rank}"
        torch.distributed.barrier()
        torch.cuda.set_device(rank)

    # Load Data
    train_dataloader, val_dataloader = prepare_dataloader(rank, cfg)

    # Prepare Model, Optimizer, and load checkpoint
    models = prepare_models(rank, cfg)
    if is_main_process(rank):
        for _, model in models.items():
            wandb.watch(model)
    optimizers, schedulers = prepare_optimizer(rank, cfg, models)
    start_iter = load_checkpoint(rank, cfg, models, optimizers)

    # Prepare RaySampler
    first_data_sample = next(iter(train_dataloader))
    (height, width), intrinsic, datatype = first_data_sample["color"][0].shape[
        :2], first_data_sample["intrinsic"][0], first_data_sample["intrinsic"][0].dtype

    ray_sampler = RaySampler(height,
                             width,
                             intrinsic,
                             sample_size=cfg.nerf.train.num_random_rays,
                             device=device)

    point_sampler = PointSampler(cfg.nerf.train.num_coarse,
                                 cfg.dataset.near,
                                 cfg.dataset.far,
                                 spacing_mode=cfg.nerf.spacing_mode,
                                 perturb=cfg.nerf.train.perturb,
                                 dtype=datatype,
                                 device=device)

    # Prepare Positional Embedding functions
    embedder_xyz = PositionalEmbedder(num_freq=cfg.models.coarse.num_encoding_fn_xyz,
                                      log_sampling=cfg.models.coarse.log_sampling_xyz,
                                      include_input=cfg.models.coarse.include_input_xyz,
                                      dtype=datatype,
                                      device=device)

    embedder_dir = None
    if cfg.nerf.use_viewdirs:
        embedder_dir = PositionalEmbedder(num_freq=cfg.models.coarse.num_encoding_fn_dir,
                                          log_sampling=cfg.models.coarse.log_sampling_dir,
                                          include_input=cfg.models.coarse.include_input_dir,
                                          dtype=datatype,
                                          device=device)

    for i in range(start_iter, cfg.experiment.train_iters):

        for model_name, model in models.items():
            models[model_name].train()

        train_data = next(iter(train_dataloader))
        for key, val in train_data.items():
            if torch.is_tensor(val):
                train_data[key] = train_data[key].to(device, non_blocking=True)

        ro, rd, select_inds = ray_sampler.sample(
            tform_cam2world=train_data["pose"])

        target_pixels = train_data["color"].view(-1, 4)[select_inds, :]
        weights, viewdirs, pts, z_vals = None, None, None, None
        coarse_loss, fine_loss = None, None

        then = time.time()
        logs = str()
        for model_name, model in models.items():
            optimizer, scheduler = optimizers[model_name], schedulers[model_name]
            if model_name == "coarse":
                pts, z_vals = point_sampler.sample_uniform(ro, rd)
            else:
                assert weights is not None, "Weights need to be updated by the coarse network"
                pts, z_vals = point_sampler.sample_pdf(
                    ro, rd, weights[..., 1:-1])

            pts_flat = pts.reshape(-1, 3)

            embedded = embedder_xyz.embed(pts_flat)
            if cfg.nerf.use_viewdirs:
                viewdirs = rd
                viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
                input_dirs = viewdirs.repeat([1, z_vals.shape[-1], 1])
                input_dirs_flat = input_dirs.reshape(-1, input_dirs.shape[-1])
                embedded_dirs = embedder_dir.embed(input_dirs_flat)
                embedded = torch.cat((embedded, embedded_dirs), dim=-1)

            radiance_field = model(embedded)
            radiance_field = radiance_field.reshape(
                list(z_vals.shape) + [radiance_field.shape[-1]])

            (rgb, _, _, weights, _) = volume_render_radiance_field(
                radiance_field,
                z_vals,
                rd,
                white_background=getattr(cfg.nerf, "train").white_background)

            loss = torch.nn.functional.mse_loss(
                rgb[..., :3], target_pixels[..., :3])

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            psnr = mse2psnr(loss.item())
            if is_main_process(rank):
                wandb.log({f"train/{model_name}_loss": loss.item(),
                          f"train/{model_name}_psnr": psnr})
            logs += f"{model_name} Loss: {loss.item():>4.4f}, {model_name} PSNR: {psnr:>4.4f} "

        if is_main_process(rank) and i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
            print(
                f"[TRAIN] Iter: {i:>8} Time taken: {time.time() - then:>4.4f} {logs} ")

        if is_main_process(rank) and i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1:
            checkpoint_dict = {
                "iter": i,
                "model_coarse_state_dict": models["coarse"].state_dict(),
                "model_fine_state_dict": models["fine"].state_dict(),
                "optimizer_coarse_state_dict": optimizers["coarse"].state_dict(),
                "optimizer_fine_state_dict": optimizers["fine"].state_dict(),
            }
            torch.save(checkpoint_dict, logdir_path / f"checkpoint{i:5d}.ckpt")
            print("================== Saved Checkpoint =================")

        # TODO: Parallel validation and rendering of images
        if (i % cfg.experiment.validate_every == 0 or i == cfg.experiment.train_iters - 1):
            val_data = next(iter(val_dataloader))
            for key, val in val_data.items():
                if torch.is_tensor(val):
                    val_data[key] = val_data[key].to(device)

            val_then = time.time()
            rgb_coarse, rgb_fine = validation_render_image(rank,
                                                           val_data["pose"],
                                                           cfg,
                                                           models,
                                                           ray_sampler,
                                                           point_sampler,
                                                           [embedder_xyz, embedder_dir],
                                                           device)
            if is_main_process(rank) and rgb_coarse is not None and rgb_fine is not None:
                target_pixels = val_data["color"].view(-1, 4).cpu()
                coarse_loss = torch.nn.functional.mse_loss(
                    rgb_coarse[..., :3], target_pixels[..., :3])
                fine_loss = torch.nn.functional.mse_loss(
                    rgb_fine[..., :3], target_pixels[..., :3])
                loss = 0.0
                loss = coarse_loss + (fine_loss if fine_loss is not None else 0.0)
                psnr = mse2psnr(loss.item())

                rgb_coarse = rgb_coarse.reshape(
                    list(val_data["color"].shape[:-1]) + [rgb_coarse.shape[-1]])
                rgb_fine = rgb_fine.reshape(
                    list(val_data["color"].shape[:-1]) + [rgb_fine.shape[-1]])
                target_rgb = target_pixels.reshape(
                    list(val_data["color"].shape[:-1]) + [4])

                wandb.log({"validation/loss": loss.item(),
                           "validation/psnr": psnr,
                           "validation/rgb_coarse": [wandb.Image(rgb_coarse[i, ...].permute(2, 0, 1)) for i in range(rgb_coarse.shape[0])],
                           "validation/rgb_fine": [wandb.Image(rgb_fine[i, ...].permute(2, 0, 1)) for i in range(rgb_fine.shape[0])],
                           "validation/target": [wandb.Image(target_rgb[i, ..., :3].permute(2, 0, 1)) for i in range(val_data["color"].shape[0])]
                           })
                print(f"[bold magenta][VAL  ][/bold magenta] Iter: {i:>8} Time taken: {time.time() - val_then:>4.4f} Loss: {loss.item():>4.4f}, PSNR: {psnr:>4.4f}")


def validation_render_image(rank: int,
                            pose: torch.Tensor,
                            cfg: CfgNode,
                            models: "OrderedDict[torch.nn.Module, torch.nn.Module]",
                            ray_sampler: RaySampler,
                            point_sampler: PointSampler,
                            embedders: List[PositionalEmbedder],
                            device: torch.cuda.Device):
    """Parallely render images on multiple GPUs for validation
    """
    for model_name, model in models.items():
        models[model_name].to(device)
        models[model_name].eval()

    embedder_xyz, embedder_dir = embedders[0], embedders[1]
    with torch.no_grad():

        ray_origins, ray_directions = ray_sampler.get_bundle(
            tform_cam2world=pose)
        ro, rd = ray_origins.reshape(-1, 3), ray_directions.reshape(-1, 3)
        num_rays = ro.shape[0]
        # msg = f"Number of pixels {ro.shape[0]} in image is not divisible by number of GPUs {cfg.gpus}"
        # assert (num_rays // cfg.gpus) * cfg.gpus == num_rays, msg

        batch_per_process = [num_rays // cfg.gpus, ] * cfg.gpus
        padding = num_rays - sum(batch_per_process)
        batch_per_process[-1] = num_rays - sum(batch_per_process[:-1])

        padded_batch_per_process = batch_per_process[0] + padding

        # Only use the split of the rays for the current process
        ro = torch.split(ro, batch_per_process)[rank].to(device)
        rd = torch.split(rd, batch_per_process)[rank].to(device)

        padded_ro = torch.empty((padded_batch_per_process, ro.shape[-1]), dtype=ro.dtype, device=ro.device)
        padded_rd = torch.empty((padded_batch_per_process, rd.shape[-1]), dtype=rd.dtype, device=rd.device)
        padded_ro[:ro.shape[0], ...] = ro
        padded_rd[:rd.shape[0], ...] = rd

        # Batch the rays allocated to current process
        ro_batches = get_minibatches(padded_ro, cfg.nerf.validation.chunksize)
        rd_batches = get_minibatches(padded_rd, cfg.nerf.validation.chunksize)

        rgb_coarse_batches, rgb_fine_batches = [], []
        for ro, rd in zip(ro_batches, rd_batches):
            weights, viewdirs, pts, z_vals = None, None, None, None
            for model_name, model in models.items():
                if model_name == "coarse":
                    pts, z_vals = point_sampler.sample_uniform(ro, rd)
                else:
                    assert weights is not None, "Weights need to be updated by the coarse network"
                    pts, z_vals = point_sampler.sample_pdf(
                        ro, rd, weights[..., 1:-1])

                pts_flat = pts.reshape(-1, 3)

                embedded = embedder_xyz.embed(pts_flat)
                if cfg.nerf.use_viewdirs:
                    viewdirs = rd
                    viewdirs = viewdirs / \
                        viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
                    input_dirs = viewdirs.repeat([1, z_vals.shape[-1], 1])
                    input_dirs_flat = input_dirs.reshape(
                        -1, input_dirs.shape[-1])
                    embedded_dirs = embedder_dir.embed(input_dirs_flat)
                    embedded = torch.cat((embedded, embedded_dirs), dim=-1)

                radiance_field = model(embedded)
                radiance_field = radiance_field.reshape(
                    list(z_vals.shape) + [radiance_field.shape[-1]])

                (rgb, _, _, weights, _) = volume_render_radiance_field(
                    radiance_field,
                    z_vals,
                    rd,
                    white_background=getattr(cfg.nerf, "train").white_background)

                if model_name == "coarse":
                    rgb_coarse_batches.append(rgb.cpu())
                else:
                    rgb_fine_batches.append(rgb.cpu())
            torch.cuda.empty_cache()

        rgb_coarse_batches = torch.cat(rgb_coarse_batches, dim=0)
        rgb_fine_batches = torch.cat(rgb_fine_batches, dim=0)

        if is_main_process(rank):
            all_rgb_coarse_batches = [torch.zeros(padded_ro.shape,
                                       dtype=padded_ro.dtype)] * len(batch_per_process)
            all_rgb_fine_batches = [torch.zeros(padded_ro.shape,
                                       dtype=padded_rd.dtype)] * len(batch_per_process)
            torch.distributed.gather(
                rgb_coarse_batches, all_rgb_coarse_batches)
            torch.distributed.gather(
                rgb_fine_batches, all_rgb_fine_batches)
            for i, size in enumerate(batch_per_process):
                all_rgb_coarse_batches[i] = all_rgb_coarse_batches[i][:size, ...]
                all_rgb_fine_batches[i] = all_rgb_fine_batches[i][:size, ...]

            all_rgb_coarse_batches = torch.cat(all_rgb_coarse_batches, dim=0)
            all_rgb_fine_batches = torch.cat(all_rgb_fine_batches, dim=0)
            return all_rgb_coarse_batches, all_rgb_fine_batches

        torch.distributed.gather(rgb_coarse_batches)
        torch.distributed.gather(rgb_fine_batches)
        return None, None


def init_process(rank: int, fn: FunctionType, cfg: CfgNode, backend: str = "gloo"):
    """TODO: Docstring for init_process.

    :function: TODO
    :returns: TODO

    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank,  world_size=cfg.gpus)
    fn(rank, cfg)


def main(cfg: CfgNode):
    """ Main function setting up the training loop

    :function: TODO
    :returns: TODO

    """
    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)

    _, device_ids = prepare_device(cfg.gpus, cfg.setup_ddp)

    if len(device_ids) > 1 and configargs.setup_ddp == True:
        # TODO: Setup DataDistributedParallel
        print(f"Using {len(device_ids)} GPUs for training")
        mp.spawn(init_process, args=(train, cfg), nprocs=cfg.gpus, join=True)
    else:
        train(0, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='Number of gpus per node')
    parser.add_argument("--setup-ddp", type=bool, default=False,
                        help="Run the models in DataDistributedParallel")
    configargs = parser.parse_args()
    # Read config file.
    cfg = CfgNode(vars(configargs), new_allowed=True)
    cfg.merge_from_file(configargs.config)

    main(cfg)
