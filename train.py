from typing import Dict, List, Tuple
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
import torch.distributed
import torch.multiprocessing
from torch.nn.parallel import DistributedDataParallel as ddp

from view_synthesis.cfgnode import CfgNode
import view_synthesis.datasets as datasets
import view_synthesis.models as network_arch
from view_synthesis.utils import prepare_device, is_main_process, prepare_logging, mse2psnr, get_minibatches
from view_synthesis.nerf import RaySampler, PointSampler, get_embedding_function, volume_render_radiance_field, PositionalEmbedder


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
        train_dataset, batch_size=cfg.dataset.train_batch_size, shuffle=False, num_workers=0, sampler=train_sampler)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.dataset.val_batch_size, shuffle=False, num_workers=0, sampler=val_sampler)
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
    if hasattr(cfg.models, "fine"):
        models['fine'] = getattr(network_arch, cfg.models.fine.type)(
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
        )
    if cfg.setup_ddp == True:
        torch.cuda.set_device(rank)
        ddp(models['coarse'], device_ids=[rank],
            output_device=rank, find_unused_parameters=True)

    return models


def prepare_optimizer(rank: int, cfg: CfgNode, models: "OrderedDict[str, torch.nn.Module]") -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """ Load the optimizer and learning schedule according to the configuration

    :function: TODO
    :returns: TODO

    """
    trainable_params = []
    for model_name, model in models.items():
        trainable_params += models[model_name].parameters()
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_params, lr=cfg.optimizer.lr)
    # TODO: Define custom scheduler which handles this from config
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1 ** (epoch / cfg.experiment.train_iters))

    # scheduler = getattr(torch.optim.lr_scheduler, cfg.optimizer.scheduler_type)(
    #     optimizer=optimizer, **cfg.optimizer.scheduler_args)
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
        if cfg.setup_ddp == True:
            # map_location = {"cuda:0": f"cuda:{rank}"}
            # checkpoint = torch.load(cfg.load_checkpoint, map_location=map_location)
            # TODO
            pass
        else:
            checkpoint = torch.load(cfg.load_checkpoint)
            for model_name, model in models.items():
                model.load_state_dict(
                    checkpoint[f"model_{model_name}_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_iter = checkpoint["start_iter"]
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
    device = None
    if is_main_process(rank) or cfg.setup_ddp == False:
        prepare_logging(cfg)
        device = f"cuda:0"
    if cfg.setup_ddp == True:
        device = f"cuda:{rank}"
        torch.distributed.barrier()

    # Load Data
    train_dataloader, val_dataloader = prepare_dataloader(rank, cfg)

    # Prepare Model, Optimizer, and load checkpoint
    models = prepare_models(rank, cfg)
    for _, model in models.items():
        wandb.watch(model)
    optimizer, scheduler = prepare_optimizer(rank, cfg, models)
    start_iter = load_checkpoint(rank, cfg, models, optimizer)

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

    for i in track(range(start_iter, cfg.experiment.train_iters), description="Training..."):

        for model_name, model in models.items():
            models[model_name].to(device)
            models[model_name].train()

        train_data = next(iter(train_dataloader))
        for key, val in train_data.items():
            if torch.is_tensor(val):
                train_data[key] = train_data[key].to(device)

        ro, rd, select_inds = ray_sampler.sample(
            tform_cam2world=train_data["pose"])

        target_pixels = train_data["color"].view(-1, 4)[select_inds, :]
        weights, viewdirs, pts, z_vals = None, None, None, None
        coarse_loss, fine_loss = None, None

        then = time.time()
        for model_name, model in models.items():

            if model_name == "coarse":
                pts, z_vals = point_sampler.sample_uniform(ro, rd)
            else:
                assert weights is not None, "Weights need to be updated by the coarse network"
                pts, z_vals = point_sampler.sample_pdf(ro, rd, weights[..., 1:-1])


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

            if model_name == "coarse":
                coarse_loss = torch.nn.functional.mse_loss(
                                rgb[..., :3], target_pixels[..., :3])
            else:
                fine_loss = torch.nn.functional.mse_loss(
                                rgb[..., :3], target_pixels[..., :3])

        loss = coarse_loss + (fine_loss if fine_loss is not None else 0.0)

        loss.backward()
        optimizer.step()
        scheduler.step()


        optimizer.zero_grad()

        psnr = mse2psnr(loss.item())
        if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
            wandb.log({"train/loss": loss.item(), "train/psnr": psnr})
            print(f"[TRAIN] Iter: {i:>8} Time taken: {time.time() - then:>4.4f} Loss: {loss.item():>4.4f}, PSNR: {psnr:>4.4f}")
            # tqdm.tqdm.write(f"Ray: {ray_sample_time}, Point: {point_sample_time}, embed: {embedding_time}, model: {model_time}, render: {render_time}, backward: {backward_time}")

        if (i % cfg.experiment.validate_every == 0 or i == cfg.experiment.train_iters - 1):
            for model_name, model in models.items():
                models[model_name].to(device)
                models[model_name].eval()
            with torch.no_grad():
                val_data = next(iter(val_dataloader))
                for key, val in val_data.items():
                    if torch.is_tensor(val):
                        val_data[key] = val_data[key].to(device)

                ray_origins, ray_directions = ray_sampler.get_bundle(tform_cam2world=val_data["pose"])
                ro, rd = ray_origins.reshape(-1, 3), ray_directions.reshape(-1, 3)
                target_pixels = val_data["color"].view(-1, 4)

                ro_batches = get_minibatches(ro, cfg.nerf.validation.chunksize)
                rd_batches = get_minibatches(rd, cfg.nerf.validation.chunksize)

                rgb_coarse, rgb_fine = [], []
                coarse_loss, fine_loss = None, None
                val_then = time.time()
                # TODO: Clean this shit up
                for ro, rd in zip(ro_batches, rd_batches):
                    weights, viewdirs, pts, z_vals = None, None, None, None
                    for model_name, model in models.items():
                        if model_name == "coarse":
                            pts, z_vals = point_sampler.sample_uniform(ro, rd)
                        else:
                            pts, z_vals = point_sampler.sample_pdf(ro, rd, weights[..., 1:-1])
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
                        if model_name == "coarse":
                            rgb_coarse.append(rgb)
                        else:
                            rgb_fine.append(rgb)

                rgb_coarse = torch.cat(rgb_coarse, dim=0)
                rgb_fine = torch.cat(rgb_fine, dim=0)

                coarse_loss = torch.nn.functional.mse_loss(
                                rgb_coarse[..., :3], target_pixels[..., :3])
                fine_loss = torch.nn.functional.mse_loss(
                                rgb_fine[..., :3], target_pixels[..., :3])
                loss = 0.0
                loss = coarse_loss + (fine_loss if fine_loss is not None else 0.0)
                psnr = mse2psnr(loss.item())

                rgb_coarse = rgb_coarse.reshape(list(val_data["color"].shape[:-1]) + [rgb_coarse.shape[-1]])
                rgb_fine = rgb_fine.reshape(list(val_data["color"].shape[:-1]) + [rgb_fine.shape[-1]])
                target_rgb = target_pixels.reshape(list(val_data["color"].shape[:-1]) + [4])

                wandb.log({"validation/loss": loss.item(),
                           "validation/psnr": psnr,
                           "validation/rgb_coarse": [wandb.Image(rgb_coarse[i, ...].permute(2, 0, 1)) for i in range(rgb_coarse.shape[0])],
                           "validation/rgb_fine": [wandb.Image(rgb_fine[i, ...].permute(2, 0, 1)) for i in range(rgb_fine.shape[0])],
                           "validation/target": [wandb.Image(target_rgb[i, ..., :3].permute(2, 0, 1)) for i in range(val_data["color"].shape[0])]
                           })
                print(f"[bold magenta][VAL  ][/bold magenta] Iter: {i:>8} Time taken: {time.time() - val_then:>4.4f} Loss: {loss.item():>4.4f}, PSNR: {psnr:>4.4f}")

def main(cfg: CfgNode):
    """ Main function setting up the training loop

    :function: TODO
    :returns: TODO

    """
    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)

    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    device, device_ids = prepare_device(cfg.gpus, cfg.setup_ddp)

    if len(device_ids) > 1 and configargs.setup_ddp == True:
        # TODO: Setup DataDistributedParallel
        pass

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
