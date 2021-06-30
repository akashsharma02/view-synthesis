from typing import List, Union, Dict, Literal

from pathlib import Path
import matplotlib.pyplot as plt
import json

import numpy as np
import imageio
import cv2
import torch


class CarInteriorDataset(torch.utils.data.Dataset):

    """Docstring for CarInteriorDataset. """

    def __init__(self, root_dir: str, resolution_level: int = 1, max_depth: float = 5):
        """
        Args:
            root_dir: Path to the root directory containing the data
            resolution_level: resolution level to resize the images

        """
        self.root_dir = Path(root_dir)
        self.color_img_dir = self.root_dir / "color"
        self.depth_img_dir = self.root_dir / "depth"
        self.normal_img_dir = self.root_dir / "normals"
        self.pose_dir = self.root_dir / "poses"
        self.max_depth = max_depth

        self.intrinsic = np.loadtxt(
            self.root_dir / "intrinsics.txt").reshape(4, 4)

        self.color_img_fnames = np.array([f for f in self.color_img_dir.iterdir(
        ) if f.is_file() and f.suffix in [".png", ".jpg"]])
        self.depth_img_fnames = np.array([f for f in self.depth_img_dir.iterdir(
        ) if f.is_file() and f.suffix in [".png", ".jpg"]])
        self.normal_img_fnames = np.array([f for f in self.normal_img_dir.iterdir(
        ) if f.is_file() and f.suffix in [".png", ".jpg"]])
        self.pose_fnames = np.array(
            [f for f in self.pose_dir.iterdir() if f.is_file() and f.suffix == ".txt"])

        msg = "Number of images between different folders are inconsistent"
        assert len(self.color_img_fnames) == len(self.depth_img_fnames) == len(
            self.normal_img_fnames) == len(self.pose_fnames), msg

        self.length = len(self.color_img_fnames)

        assert resolution_level > 0, "Resolution level needs to be a positive integer"
        self.resolution_level = resolution_level

    def __len__(self):
        """ Returns length of the dataset

        :function: TODO
        :returns: int

        """
        return self.length

    def __getitem__(self, idx: Union[int, List[int], torch.Tensor, List[torch.Tensor]]) -> Dict:
        """TODO: Docstring for __get_item__.

        :function: TODO
        :returns: TODO

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        color_img_fname = self.color_img_fnames[idx]
        depth_img_fname = self.depth_img_fnames[idx]
        normal_img_fname = self.normal_img_fnames[idx]
        pose_fname = self.pose_fnames[idx]

        color_img = imageio.imread(color_img_fname)
        depth_img = cv2.imread(str(depth_img_fname), cv2.IMREAD_UNCHANGED)
        normal_img = imageio.imread(normal_img_fname)
        pose = np.loadtxt(pose_fname).reshape(4, 4)

        color_img = (np.array(color_img) / 255.0)
        depth_img = (np.array(depth_img) * self.max_depth / 65535.0)
        depth_img = depth_img[..., 0][..., None]
        normal_img = (np.array(normal_img) / 255.0)

        sample = {'color': color_img.astype(np.float32),
                  'depth': depth_img.astype(np.float32),
                  'normal': normal_img.astype(np.float32),
                  'pose': pose.astype(np.float32),
                  'intrinsic': np.copy(self.intrinsic).astype(np.float32)}

        if self.resolution_level != 1:
            H, W = color_img.shape[:2]
            H, W = H // self.resolution_level, W // self.resolution_level

            sample['intrinsic'][:2, :3] = sample['intrinsic'][:2,
                                                              :3] // self.resolution_level
            sample['color'] = cv2.resize(
                sample['color'], (W, H), interpolation=cv2.INTER_AREA)
            sample['depth'] = cv2.resize(
                sample['depth'], (W, H), interpolation=cv2.INTER_NEAREST)
            sample['normal'] = cv2.resize(
                sample['normal'], (W, H), interpolation=cv2.INTER_NEAREST)

        return sample


class BlenderNeRFDataset(torch.utils.data.Dataset):

    """Docstring for BlenderNeRFDataset. """

    def __init__(self, root_dir: str, resolution_level: int = 1, mode: Literal["train", "test", "val"] = "train"):
        """
        Args:
            root_dir: Path to the root directory containing the data
            resolution_level: resolution level to resize the images

        """
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / mode
        self.pose_fname = self.root_dir / f"transforms_{mode}.json"
        assert self.img_dir.exists() and self.img_dir.is_dir(
        ), "Incorrect mode, should be either train, test, or val"
        assert self.pose_fname.exists() and self.pose_fname.is_file(
        ), "Pose file name transforms_{mode}.json does not exist"

        metadata = None
        with open(self.pose_fname, "r") as fp:
            metadata = json.load(fp)

        self.img_fnames, self.poses = [], []
        for frame in metadata["frames"]:
            self.img_fnames.append(
                self.root_dir / (frame["file_path"] + ".png"))
            self.poses.append(np.array(frame["transform_matrix"]))

        self.img_fnames = np.array(self.img_fnames)
        self.poses = np.array(self.poses)

        height, width = imageio.imread(self.img_fnames[0]).shape[:2]
        camera_angle_x = metadata["camera_angle_x"]
        focal = 0.5 * width / np.tan(0.5 * camera_angle_x)
        self.intrinsic = np.array([[focal, 0, width/2.0, 0],
                                   [0, focal, height/2.0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])

        msg = "Inconsistent number of images or poses in the dataset folder"
        assert len(self.img_fnames) == len(self.poses), msg

        self.length = len(self.img_fnames)

        assert resolution_level > 0, "Resolution level needs to be a positive integer"
        self.resolution_level = resolution_level

    def __len__(self):
        """ Returns length of the dataset

        :function: TODO
        :returns: int

        """
        return self.length

    def __getitem__(self, idx: Union[int, List[int], torch.Tensor, List[torch.Tensor]]) -> Dict:
        """TODO:
        Get a sample from the dataset given an index or index list

        :function: TODO
        :returns: TODO

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        color_img_fname = self.img_fnames[idx]
        pose = self.poses[idx].reshape(4, 4)

        assert np.isclose(np.linalg.det(
            pose[:3, :3]), 1), "Incorrect rotation does not determinant = 1"

        color_img = imageio.imread(color_img_fname)
        color_img = (np.array(color_img) / 255.0)

        sample = {'color': color_img.astype(np.float32), 'pose': pose.astype(
            np.float32), 'intrinsic': np.copy(self.intrinsic).astype(np.float32)}

        if self.resolution_level != 1:
            H, W = color_img.shape[:2]
            H, W = H // self.resolution_level, W // self.resolution_level

            sample['intrinsic'][:2, :3] = sample['intrinsic'][:2,
                                                              :3] // self.resolution_level
            sample['color'] = cv2.resize(
                sample['color'], (W, H), interpolation=cv2.INTER_AREA)

        return sample


if __name__ == "__main__":
    # Test Dataset
    car_interior_dataset = CarInteriorDataset(
        root_dir="/home/fyusion/Documents/datasets/bmw-simulated", resolution_level=2)

    print(f"Length of the dataset: {len(car_interior_dataset)}")

    train_size = int(len(car_interior_dataset) * 0.75)
    test_size = len(car_interior_dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        car_interior_dataset, [train_size, test_size])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_dataloader):
        sample = car_interior_dataset[i]

        Path("./dataset_test").mkdir(parents=True, exist_ok=True)
        plt.imsave(f"./dataset_test/sample_color_{i}.png", sample['color'])
        plt.imsave(f"./dataset_test/sample_depth_{i}.png", sample['depth'])
        plt.imsave(f"./dataset_test/sample_normal_{i}.png", sample['normal'])

        print(f"Pose of sample {i}:\n {sample['pose']}")
        print(f"Intrinsic of sample {i}:\n {sample['intrinsic']}")

        if i > 10:
            break

    for i, sample in enumerate(test_dataloader):
        sample = car_interior_dataset[i]

        Path("./dataset_test").mkdir(parents=True, exist_ok=True)
        plt.imsave(
            f"./dataset_test/test_sample_color_{i}.png", sample['color'])
        plt.imsave(
            f"./dataset_test/test_sample_depth_{i}.png", sample['depth'])
        plt.imsave(
            f"./dataset_test/test_sample_normal_{i}.png", sample['normal'])

        print(f"Pose of sample {i}:\n {sample['pose']}")
        print(f"Intrinsic of sample {i}:\n {sample['intrinsic']}")

        if i > 10:
            break
