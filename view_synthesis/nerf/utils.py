from typing import Tuple, Union, Literal
from numpy.typing import DTypeLike
import numpy as np
import torch


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


class RaySampler(object):

    """RaySampler samples rays for a given image size and intrinsics """

    def __init__(self, height: int, width: int, intrinsics: Union[torch.Tensor, np.ndarray], sample_size: int, device: torch.cuda.Device):
        """ Prepares a ray bundle for a given image size and intrinsics

        :Function: intrinsics: torch.Tensor 4x4

        """
        assert height > 0 and width > 0, "Height and width must be positive integers"
        assert sample_size > 0 and sample_size < height * width, "Sample size must be a positive number less than height * width"

        self.height = height
        self.width = width
        self.sample_size = sample_size
        self.device = device


        if isinstance(intrinsics, np.ndarray):
            intrinsics = torch.from_numpy(intrinsics)
        assert intrinsics.shape == torch.Size([4, 4]), "Incorrect intrinsics shape"
        self.intrinsics = intrinsics
        self.intrinsics = self.intrinsics.to(device)
        self.focal_length = self.intrinsics[..., 0, 0]

        ii, jj = meshgrid_xy(
            torch.arange(
                width, dtype=self.intrinsics.dtype, device=self.device
            ),
            torch.arange(
                height, dtype=self.intrinsics.dtype, device=self.device
            ),
        )
        self.directions = torch.stack(
            [
                (ii - width * 0.5) / self.focal_length,
                -(jj - height * 0.5) / self.focal_length,
                -torch.ones_like(ii),
            ],
            dim=-1,
        )

    def sample(self, tform_cam2world: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Rotate the bundle of rays given the camera pose and return a random subset of rays

        :function:
            tform_cam2world: [batch, 4, 4] torch.Tensor camera pose (SE3)
        :returns:
            ray origins: torch.Tensor [3, batch_size*sample_size]
            ray directions: torch.Tensor [3, batch_size*sample_size]
            select_inds: np.ndarray [batch_size*sample_size]

        """
        # TODO: Returns same sample across batch
        batch_size = tform_cam2world.shape[0]

        ray_origins, ray_directions = self.get_bundle(tform_cam2world)
        ro, rd = ray_origins.reshape(-1, 3), ray_directions.reshape(-1, 3)

        rng = np.random.default_rng()
        select_inds = rng.choice(
            ro.shape[-2], size=(
                self.sample_size*batch_size), replace=False
        )

        ray_origins = ro[select_inds, :]
        ray_directions = rd[select_inds, :]

        return ray_origins, ray_directions, select_inds

    def get_bundle(self, tform_cam2world: torch.Tensor):
        """
            Rotate the bundle of rays given the camera pose

        :function:
            tform_cam2world: 4x4 torch.Tensor camera pose (SE3)
        :returns:
            ray origins: torch.Tensor [batch, 3, H, W]
            ray directions: torch.Tensor [batch, 3, H, W]

        """
        ray_directions = torch.tensordot(
            tform_cam2world[..., :3, :3],
            self.directions.T,
            dims=([2], [0])
        ).transpose(1, -1).contiguous()

        ray_origins = tform_cam2world[..., :3, -1].expand(ray_directions.shape)
        return ray_origins, ray_directions


class PointSampler(object):

    """Sample 3D points along the given rays"""

    def __init__(self, num_samples: int, near: float, far: float, spacing_mode: Literal["lindisp", "lindepth"], perturb: bool, dtype: DTypeLike, device: torch.cuda.Device):
        assert num_samples > 0, "Number of samples along a ray should be positive integer"
        assert near >= 0 and far > near, "Near and far ranges should be positive values, and far > near"
        self.num_samples = num_samples
        self.near = near
        self.far = far
        self.spacing_mode = spacing_mode
        self.perturb = perturb
        self.dtype = dtype
        self.device = device

        self.t_vals = torch.linspace(
            0.0,
            1.0,
            self.num_samples,
            dtype=self.dtype,
            device=self.device
        )
        if self.spacing_mode == "lindisp":
            self.z_vals = near * (1.0 - self.t_vals) + far * self.t_vals
        else:
            self.z_vals = 1.0 / (1.0 / near * (1.0 - self.t_vals) + 1.0 / far * self.t_vals)


    def sample_uniform(self, ro: torch.Tensor, rd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Uniform sample points according to spacing mode along the ray

        :function:
            ro: [num_random_rays, 3] ray_origins
            rd: [num_random_rays, 3] ray_directions
        :returns:
            pts: [num_random_rays*num_samples, 3] pts alongs the ray
            z_vals: [num_random_rays, num_samples, 3] z_vals along the ray

        """
        num_random_rays = ro.shape[-2]
        z_vals = self.z_vals.expand(num_random_rays, self.num_samples)

        if self.perturb:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
            lower = torch.cat((z_vals[..., :1], mids), dim=-1)
            t_rand = torch.rand(z_vals.shape, dtype=ro.dtype, device=ro.device)
            z_vals = lower + (upper - lower) * t_rand

        assert z_vals.shape == torch.Size([num_random_rays, self.num_samples]), "Incorrect shape of depth samples z_vals"
        pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]
        return pts, z_vals

    def sample_pdf(self, ro: torch.Tensor, rd: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Sample points according to spacing mode along ray given a probability distribution
        :function:
            ro: [num_random_rays, 3] ray_origins
            rd: [num_random_rays, 3] ray_directions
        :returns:
            z_vals: [num_random_rays, num_samples, 3] z_vals along the ray

        """
        num_random_rays = ro.shape[-2]
        z_vals = self.z_vals.expand(num_random_rays, self.num_samples)

        bins = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])

        weights = weights + 1e-5
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat(
            [torch.zeros_like(cdf[..., :1]), cdf], dim=-1
        )  # (batchsize, len(bins))

        # Take uniform samples
        if self.perturb:
            u = torch.rand(
                list(cdf.shape[:-1]) + [self.num_samples],
                dtype=weights.dtype,
                device=weights.device,
            )
        else:
            u = torch.linspace(
                0.0, 1.0, steps=self.num_samples, dtype=weights.dtype, device=weights.device
            )
            u = u.expand(list(cdf.shape[:-1]) + [self.num_samples])

        # Invert CDF
        u = u.contiguous()
        cdf = cdf.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack((below, above), dim=-1)  # (batchsize, num_samples, 2)

        matched_shape = (inds_g.shape[0], inds_g.shape[1], cdf.shape[-1])
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        z_samples = samples.detach()
        z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
        pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

        return pts, z_vals



def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True, progress=1.0
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    alpha = int(progress * num_encoding_functions)
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    def get_weight(alpha, k):
        if alpha < k:
            weight = 0.0
        elif alpha - k >= 0 and alpha - k < 1:
            weight = (1 - torch.cos(torch.tensor((alpha - k) * math.pi))) / 2
        else:
            weight = 1.0
        return weight

    for i, freq in enumerate(frequency_bands):
        weight = get_weight(alpha, i)
        for func in [torch.sin, torch.cos]:
            encoding.append(weight * func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


def get_embedding_function(
    num_encoding_functions=6, include_input=True, log_sampling=True, alpha=6
):
    r"""Returns a lambda function that internally calls positional_encoding.
    """
    return lambda x, progress=1.0: positional_encoding(
        x, num_encoding_functions, include_input, log_sampling, progress
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Test RaySampler")
    parser.add_argument("--dataset-dir", type=str, help="Directory for the blender dataset root", required=True)
    parser.add_argument("--num-random-rays", type=int, help="Number of random rays to sample", required=True)
    args = parser.parse_args()
    from view_synthesis.datasets.dataset import BlenderNeRFDataset
    dataset = BlenderNeRFDataset(args.dataset_dir, resolution_level=16, mode="val")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    first_data_sample = next(iter(dataloader))

    (height, width), intrinsic = first_data_sample["color"].shape[1:-1], first_data_sample["intrinsic"]

    print(f" Height: {height}, Width: {width}, Intrinsics:\n {intrinsic}")

    device=None
    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"

    ray_sampler = RaySampler(height, width, intrinsic[0], sample_size=args.num_random_rays, device=device)
    pose = first_data_sample["pose"].to(device)
    ray_origins, ray_directions, select_inds = ray_sampler.sample(tform_cam2world=pose)

    print(f"Ray bundle shape: {ray_origins.shape}, {ray_directions.shape}, {select_inds.shape}")
    assert torch.equal(ray_origins[..., 0], pose[..., :3, 3]), "ray_origin is not equal to the pose origin"

    print(f"Ray directions: \n{ray_directions}")

