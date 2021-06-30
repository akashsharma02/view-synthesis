from typing import Tuple, Union, Literal
from numpy.typing import DTypeLike
import numpy as np
import torch


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
            self.z_vals = 1.0 / \
                (1.0 / near * (1.0 - self.t_vals) + 1.0 / far * self.t_vals)

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

        assert z_vals.shape == torch.Size(
            [num_random_rays, self.num_samples]), "Incorrect shape of depth samples z_vals"
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
        # (batchsize, num_samples, 2)
        inds_g = torch.stack((below, above), dim=-1)

        matched_shape = (inds_g.shape[0], inds_g.shape[1], cdf.shape[-1])
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(
            1).expand(matched_shape), 2, inds_g)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        z_samples = samples.detach()

        z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
        pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

        return pts, z_vals


if __name__ == "__main__":
    # Manual testing of RaySampler and PointSampler
    import argparse
    parser = argparse.ArgumentParser("Test PointSampler")
    parser.add_argument("--dataset-dir", type=str,
                        help="Directory for the Car Interior dataset root", required=True)
    parser.add_argument("--num-random-rays", type=int,
                        help="Number of random rays to sample", required=True)
    parser.add_argument("--num-samples", type=int,
                        help="Number of samples along the ray", required=True)

    args = parser.parse_args()
    np.random.seed(42)
    torch.manual_seed(42)

    from view_synthesis.datasets.dataset import CarInteriorDataset
    from view_synthesis.nerf.ray_sampler import RaySampler

    dataset = CarInteriorDataset(args.dataset_dir, resolution_level=32)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0)

    first_data_sample = next(iter(dataloader))

    (height,
     width), intrinsic = first_data_sample["color"].shape[1:-1], first_data_sample["intrinsic"]

    print(f" Height: {height}, Width: {width}, Intrinsics:\n {intrinsic}")

    device = None
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    ray_sampler = RaySampler(
        height, width, intrinsic[0], sample_size=args.num_random_rays, device=device)
    pose = first_data_sample["pose"].to(device)
    ray_origins, ray_directions, select_inds = ray_sampler.sample(
        tform_cam2world=pose)

    print(
        f"Ray bundle shape: {ray_origins.shape}, {ray_directions.shape}, {select_inds.shape}")

    point_sampler = PointSampler(num_samples=args.num_samples, near=0, far=5,
                                 spacing_mode="lindisp", perturb=True, dtype=intrinsic[0].dtype, device=device)
    pts, z_vals = point_sampler.sample_uniform(ray_origins, ray_directions)

    # Take the z value from the depth image of the first left-top pixel
    depth_img = first_data_sample["depth"]
    weights = point_sampler.z_vals
    z_mean = depth_img[..., :].reshape(-1, 1)[select_inds].to(weights)
    z_sigma = torch.Tensor([0.25]).expand(z_mean.shape).to(
        weights)  # 0.25meters std deviation

    # Get linearly spaced points between near and far

    def gaussian(weights: torch.Tensor, mean: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """TODO: Docstring for gaussian.

        :function: TODO
        :returns: TODO

        """
        print(weights.shape, mean.shape)
        mean, sigma = mean[:, None], sigma[:, None]
        weights = weights[None, :].repeat(mean.shape)
        print(weights.device, mean.device, sigma.device)
        weights = -0.5 * ((weights - mean) / (2 * sigma ** 2)) ** 2
        return torch.exp(weights)

    weights = gaussian(weights=weights, mean=z_mean, sigma=z_sigma)
    print(weights.shape, ray_origins.shape, ray_directions.shape)
    pts, z_vals = point_sampler.sample_pdf(
        ray_origins, ray_directions, weights.squeeze()[..., 1:-1])
    print(pts.shape, z_vals.shape)
    print(f"Sampled points:\n {z_vals[0]}, {z_vals[0].shape}")
    print(f"Mean: {z_mean[0]}")
