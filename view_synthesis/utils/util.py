from typing import Tuple, List, Optional
import math
import torch
import numpy as np
from pathlib import Path
import yaml
import wandb

from view_synthesis.cfgnode import CfgNode



def prepare_device(n_gpus_to_use: int, setup_ddp: bool) -> Tuple[torch.device, List[int]]:
    """ Prepare GPU device if available, and get GPU indices for DataDistributedParallel

    :function: TODO
    :returns: TODO

    """
    n_gpu = torch.cuda.device_count()
    if n_gpus_to_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpus_to_use = 0
    if n_gpus_to_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpus_to_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpus_to_use = n_gpu
    if n_gpus_to_use < 2 and setup_ddp == True:
        print(f"Warning: Setting up DataDistributedParallel is forbidden with 1 GPU.")
        setup_ddp = False

    main_device = torch.device('cuda:0' if n_gpus_to_use > 0 else 'cpu')
    list_ids = list(range(n_gpus_to_use))

    return main_device, list_ids


def is_main_process(rank: int) -> bool:
    """ Test whether process is the main worker process
    """
    return rank == 0


def prepare_logging(cfg: CfgNode):
    """TODO: Docstring for prepare_logging.

    :function: TODO
    :returns: TODO

    """
    wandb.init(project="ibr-view-synthesis", config=cfg)

    logdir_path = Path(cfg.experiment.logdir) / cfg.experiment.id
    logdir_path.mkdir(parents=True, exist_ok=True)
    with open(Path(logdir_path) / "config.yml", "w") as f:
        f.write(cfg.dump())

    return logdir_path


def mse2psnr(mse_val: float) -> float:
    """
    Calculate PSNR from MSE

    :function: TODO
    :returns: TODO

    """
    # For numerical stability, avoid a zero mse loss.
    if mse_val == 0:
        mse_val = 1e-5
    return -10.0 * math.log10(mse_val)


def get_minibatches(inputs: torch.Tensor, chunksize: Optional[int] = 1024 * 8):
    r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    return [inputs[i: i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


def meshgrid_xy(
    tensor1: torch.Tensor, tensor2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mimick np.meshgrid(..., indexing="xy") in pytorch. torch.meshgrid only allows "ij" indexing.
    :Function:
      tensor1 (torch.Tensor): Tensor whose elements define the first dimension of the returned meshgrid.
      tensor2 (torch.Tensor): Tensor whose elements define the second dimension of the returned meshgrid.
    """
    # TESTED
    ii, jj = torch.meshgrid(tensor1, tensor2)
    return ii.transpose(-1, -2), jj.transpose(-1, -2)

