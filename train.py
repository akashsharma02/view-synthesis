from typing import Dict, List, Tuple, Union
from types import FunctionType
import os
import time
import argparse
import wandb
import yaml
from pathlib import Path
from collections import OrderedDict
import cv2

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from apex.parallel import DistributedDataParallel as ddp

from view_synthesis.cfgnode import CfgNode
import view_synthesis.datasets as datasets
import view_synthesis.models as network_arch
from view_synthesis.utils import prepare_device, is_main_process, prepare_logging, mse2psnr, get_minibatches
from view_synthesis.nerf import RaySampler, PointSampler, volume_render_radiance_field, PositionalEmbedder


def prepare_dataloader(cfg: CfgNode) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
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

    train_sampler = torch.utils.data.RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=cfg.experiment.load_iters
    )

    val_sampler = torch.utils.data.RandomSampler(
        val_dataset,
        replacement=True,
        num_samples=cfg.experiment.load_iters
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.dataset.train_batch_size, shuffle=False, num_workers=0, sampler=train_sampler, pin_memory=True)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.dataset.val_batch_size, shuffle=False, num_workers=0, sampler=val_sampler, pin_memory=True)

    return train_dataloader, val_dataloader


def prepare_models(cfg: CfgNode) -> OrderedDict:
    """ Prepare the torch models

    :function:
        rank: Process rank. 0 == main process
    :returns: TODO

    """
    rank = dist.get_rank()
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

    if cfg.is_distributed == True:
        models["coarse"] = ddp(models['coarse'])
        if hasattr(cfg.models, "fine"):
            models["fine"] = ddp(models['coarse'])

    return models


def prepare_optimizer(cfg: CfgNode, models: "OrderedDict[str, torch.nn.Module]") -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """ Load the optimizer and learning schedule according to the configuration

    :function: TODO
    :returns: TODO

    """
    trainable_params = []
    for model_name, model in models.items():
        trainable_params += list(model.parameters())
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_params, lr=cfg.optimizer.lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: cfg.optimizer.scheduler_gamma ** (epoch / cfg.optimizer.scheduler_step_size))

    return optimizer, scheduler


def load_checkpoint(rank: int, cfg: CfgNode, models: "OrderedDict[str, torch.nn.Module]", optimizer: torch.optim.Optimizer) -> int:
    """TODO: Docstring for load_pretrained.

    :function: TODO
    :returns: TODO

    """
    start_iter = 0
    print(Path(cfg.load_checkpoint))
    checkpoint_file = Path(cfg.load_checkpoint)
    if checkpoint_file.exists() and checkpoint_file.is_file() and checkpoint_file.suffix == ".ckpt":
        if cfg.is_distributed == True:
            map_location = {"cuda:0": f"cuda:{rank}"}
            checkpoint = torch.load(
                cfg.load_checkpoint, map_location=map_location)
        else:
            checkpoint = torch.load(cfg.load_checkpoint)

        for model_name, model in models.items():
            model.load_state_dict(
                checkpoint[f"model_{model_name}_state_dict"])

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["start_iter"]

    # Ensure that all loading by all processes is done before any process has started saving models
    if cfg.is_distributed == True:
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

    # Set device and logdir_path
    device, logdir_path = None, None
    if is_main_process(rank) or cfg.is_distributed == False:
        logdir_path = prepare_logging(cfg)
        device = f"cuda:0"
    if cfg.is_distributed == True:
        device = f"cuda:{rank}"
        torch.cuda.set_device(rank)

    # Load Data
    train_dataloader, val_dataloader = prepare_dataloader(cfg)

    # Prepare Model, Optimizer, and load checkpoint
    models = prepare_models(cfg)
    if is_main_process(rank):
        for _, model in models.items():
            wandb.watch(model)

    optimizer, scheduler = prepare_optimizer(cfg, models)
    start_iter = load_checkpoint(rank, cfg, models, optimizer)

    # Prepare RaySampler
    first_data_sample = next(iter(train_dataloader))
    (height, width), intrinsic, datatype = first_data_sample["color"][0].shape[
        :2], first_data_sample["intrinsic"][0], first_data_sample["intrinsic"][0].dtype

    ray_sampler = RaySampler(seed,
                             height,
                             width,
                             intrinsic,
                             sample_size=cfg.nerf.train.num_random_rays,
                             device=device)

    point_sampler = PointSampler(seed,
                                 cfg.nerf.train.num_coarse,
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
    i = 0
    for load_iter in range(start_iter, cfg.experiment.load_iters):

        for model_name, model in models.items():
            models[model_name].train()

        train_data = next(iter(train_dataloader))
        for key, value in train_data.items():
            if torch.is_tensor(value):
                train_data[key] = train_data[key].to(device, non_blocking=True)

        ro_batch, rd_batch, select_inds = ray_sampler.sample(
            tform_cam2world=train_data["pose"])

        tgt_pixel_batch = train_data["color"].view(-1, 4)[select_inds, :]

        # Batching to reduce loading time
        ro_minibatches          = get_minibatches(ro_batch, cfg.nerf.train.chunksize)
        rd_minibatches          = get_minibatches(rd_batch, cfg.nerf.train.chunksize)
        tgt_pixel_minibatches   = get_minibatches(tgt_pixel_batch, cfg.nerf.train.chunksize)
        num_batches = len(ro_minibatches)

        msg = "Mismatch in batch length of ray origins, ray directions and target pixels"
        assert num_batches == len(rd_minibatches) == len(
            tgt_pixel_minibatches), msg

        weights, viewdirs, pts, z_vals = None, None, None, None
        coarse_loss, fine_loss = None, None
        for j, (ro, rd, target_pixels) in enumerate(zip(ro_minibatches, rd_minibatches, tgt_pixel_minibatches)):
            weights, viewdirs, pts, z_vals = None, None, None, None
            then = time.time()
            loss = None
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
                    viewdirs = rd / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
                    input_dirs = viewdirs.repeat([1, z_vals.shape[-1], 1])
                    input_dirs_flat = input_dirs.reshape(-1, input_dirs.shape[-1])
                    embedded_dirs = embedder_dir.embed(input_dirs_flat)
                    embedded = torch.cat((embedded, embedded_dirs), dim=-1)

                radiance_field = model(embedded)
                radiance_field = radiance_field.reshape(list(z_vals.shape) + [radiance_field.shape[-1]])

                (rgb, _, _, weights, _) = volume_render_radiance_field(
                    radiance_field,
                    z_vals,
                    rd,
                    white_background=getattr(cfg.nerf, "train").white_background)

                if loss is None:
                    loss = torch.nn.functional.mse_loss(
                        rgb[..., :3], target_pixels[..., :3])
                else:
                    loss += torch.nn.functional.mse_loss(
                        rgb[..., :3], target_pixels[..., :3])

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            scheduler.step()

            psnr = mse2psnr(loss.item())

            i = load_iter * num_batches + j

            if is_main_process(rank) and i % cfg.experiment.print_every == 0 or load_iter == cfg.experiment.load_iters - 1:
                wandb.log({f"train/loss": loss.item(), f"train/psnr": psnr})
                print(f"[TRAIN] Iter: {i:>8} Time taken: {time.time() - then:>4.4f} Learning rate: {scheduler.get_lr():4.4f} Loss: {loss.item():>4.4f}, PSNR: {psnr:>4.4f}")

        if is_main_process(rank) and load_iter % cfg.experiment.save_every == 0 or load_iter == cfg.experiment.load_iters - 1:
            checkpoint_dict = {
                "iter": i,
                "model_coarse_state_dict": models["coarse"].state_dict(),
                "model_fine_state_dict": models["fine"].state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint_dict, logdir_path / f"checkpoint{i:5d}.ckpt")
            print("================== Saved Checkpoint =================")

        # Parallel rendering of image for Validation
        if load_iter % cfg.experiment.validate_every == 0 or load_iter == cfg.experiment.load_iters - 1:

            # Load data independently in all processes as a list of tuples
            val_data = next(iter(val_dataloader))

            # Broadcast validation data in rank 0 to all the processes
            if cfg.is_distributed == True:
                val_data = list(val_data.items())
                torch.distributed.broadcast_object_list(val_data, 0)
                val_data = dict(val_data)

            for key, val in val_data.items():
                if torch.is_tensor(val):
                    val_data[key] = val_data[key].to(device)

            val_then = time.time()
            rgb_coarse, rgb_fine = parallel_image_render(rank,
                                                         val_data["pose"],
                                                         cfg,
                                                         models,
                                                         ray_sampler,
                                                         point_sampler,
                                                         [embedder_xyz, embedder_dir],
                                                         device)

            if is_main_process(rank) or cfg.is_distributed == False:
                assert rgb_coarse is not None, "Main process must contain rgb_coarse"
                assert rgb_fine is not None, "Main process must contain rgb_fine"

                target_pixels = val_data["color"].view(-1, 4)

                coarse_loss = torch.nn.functional.mse_loss(
                    rgb_coarse[..., :3], target_pixels[..., :3])
                fine_loss = torch.nn.functional.mse_loss(
                    rgb_fine[..., :3], target_pixels[..., :3])

                loss = coarse_loss + fine_loss
                psnr = mse2psnr(loss)

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
                print(
                    f"[VAL  ] Iter: {i:>8} Time taken: {time.time() - val_then:>4.4f} Loss: {loss.item():>4.4f}, PSNR: {psnr:>4.4f}")


def parallel_image_render(rank: int,
                          pose: torch.Tensor,
                          cfg: CfgNode,
                          models: "OrderedDict[torch.nn.Module, torch.nn.Module]",
                          ray_sampler: RaySampler,
                          point_sampler: PointSampler,
                          embedders: List[Union[PositionalEmbedder, None]],
                          device: torch.cuda.Device):
    """Parallely render images on multiple GPUs for validation
    """
    embedder_xyz, embedder_dir = embedders[0], embedders[1]
    assert embedder_xyz is not None, "XYZ PositionalEmbedder is None "
    assert embedder_dir is not None, "Direction PositionalEmbedder is None "

    for model_name, model in models.items():
        models[model_name].to(device)
        models[model_name].eval()

    with torch.no_grad():

        ray_origins, ray_directions = ray_sampler.get_bundle(
            tform_cam2world=pose)
        ro, rd = ray_origins.reshape(-1, 3), ray_directions.reshape(-1, 3)
        num_rays = ro.shape[0]

        batchsize_per_process = torch.full(
            [cfg.gpus], (num_rays / cfg.gpus), dtype=int)
        padding = num_rays - torch.sum(batchsize_per_process)
        batchsize_per_process[-1] = num_rays - \
            torch.sum(batchsize_per_process[:-1])
        assert torch.sum(
            batchsize_per_process) == num_rays, "Mismatch in batchsize per process and total number of rays"

        padding_per_process = torch.zeros([cfg.gpus], dtype=int)
        if padding > 0:
            padding_per_process[:-1] = padding
        assert padding + \
            batchsize_per_process[0] == batchsize_per_process[-1], "Incorrect calculation of padding"

        # Only use the split of the rays for the current process
        ro_batch = torch.split(ro, batchsize_per_process.tolist())[
            rank].to(device)
        rd_batch = torch.split(rd, batchsize_per_process.tolist())[
            rank].to(device)

        # Minibatch the rays allocated to current process
        ro_minibatches = get_minibatches(
            ro_batch, cfg.nerf.validation.chunksize)
        rd_minibatches = get_minibatches(
            rd_batch, cfg.nerf.validation.chunksize)

        rgb_coarse_batches, rgb_fine_batches = [], []
        for ro, rd in zip(ro_minibatches, rd_minibatches):
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
                    white_background=getattr(cfg.nerf, "validation").white_background)

                if model_name == "coarse":
                    rgb_coarse_batches.append(rgb)
                else:
                    rgb_fine_batches.append(rgb)

        rgb_coarse_batches = torch.cat(rgb_coarse_batches, dim=0)
        rgb_fine_batches = torch.cat(rgb_fine_batches, dim=0)

        if cfg.is_distributed == False:
            return rgb_coarse_batches, rgb_fine_batches

        # Pad image chunks to get equal chunksize for all_gather/gather
        padded_rgb_coarse = torch.zeros((padding_per_process[rank], rgb_coarse_batches.shape[-1]),
                                        dtype=rgb_coarse_batches.dtype,
                                        device=rgb_coarse_batches.device)
        padded_rgb_fine = torch.zeros((padding_per_process[rank], rgb_fine_batches.shape[-1]),
                                      dtype=rgb_fine_batches.dtype,
                                      device=rgb_fine_batches.device)
        rgb_coarse_batches = torch.cat([rgb_coarse_batches, padded_rgb_coarse], dim=0)
        rgb_fine_batches = torch.cat([rgb_fine_batches, padded_rgb_fine], dim=0)

        all_rgb_coarse_batches = [torch.zeros_like(rgb_coarse_batches)
                                  for _ in range(cfg.gpus)]
        all_rgb_fine_batches = [torch.zeros_like(rgb_fine_batches)
                                for _ in range(cfg.gpus)]
        torch.distributed.all_gather(
            all_rgb_coarse_batches, rgb_coarse_batches)
        torch.distributed.all_gather(all_rgb_fine_batches, rgb_fine_batches)

        # Return only from main_process
        if is_main_process(rank):
            for i, size in enumerate(batchsize_per_process):
                all_rgb_coarse_batches[i] = all_rgb_coarse_batches[i][:size, ...]
                all_rgb_fine_batches[i] = all_rgb_fine_batches[i][:size, ...]

            all_rgb_coarse_batches = torch.cat(all_rgb_coarse_batches, dim=0)
            all_rgb_fine_batches = torch.cat(all_rgb_fine_batches, dim=0)
            return all_rgb_coarse_batches, all_rgb_fine_batches
        else:
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
    torch.distributed.destroy_process_group()


def main(cfg: CfgNode):
    """ Main function setting up the training loop

    :function: TODO
    :returns: TODO

    """
    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)

    _, device_ids = prepare_device(cfg.gpus, cfg.is_distributed)

    if len(device_ids) > 1 and configargs.is_distributed == True:
        # TODO: Setup DataDistributedParallel
        print(f"Using {len(device_ids)} GPUs for training")
        mp.spawn(init_process, args=(train, cfg, "nccl"),
                 nprocs=cfg.gpus, join=True)
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
    parser.add_argument("--distributed", action='store_true', dest="is_distributed",
                        help="Run the models in DataDistributedParallel")
    configargs = parser.parse_args()

    # Read config file.
    cfg = CfgNode(vars(configargs), new_allowed=True)
    cfg.merge_from_file(configargs.config)

    main(cfg)
