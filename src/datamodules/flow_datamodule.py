from functools import partial
import os.path as osp

from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, Compose, Lambda  # Normalize

from src.datamodules.datasets.flow_dataset import TripletFlowDataset
from src.utils.utils import xy_to_polar, polar_to_xy, load_csv


class TripletFlowDataModule(LightningDataModule):
    """Initialize train, val and test base data loader.

    :param split_dir: path to the directory with train/val/test splits.
    :param unity_dir: path to the directory with Unity flows.
    :param prcpt_dir: path to the directory with precomputed flows.
    :param n_frames: number of frames in a sample (fixed by the model).
    :param frame_size: frame size to input (resizing)
    :param unity_mod_max: maximum value of the unity module flow (norm).
    :param prcpt_mod_max: maximum value of the precomputed module flow (norm).
    :param batch_size: size of batches.
    :param num_workers: number of workers.
    """

    def __init__(
        self,
        split_dir: str,
        unity_dir: str,
        prcpt_dir: str,
        n_frames: int,
        frame_size: int,
        unity_mod_max: float,
        prcpt_mod_max: float,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()

        train_split_path = osp.join(split_dir, "train.csv")
        self._train_clip_dirnames = load_csv(train_split_path)[0].tolist()
        val_split_path = osp.join(split_dir, "val.csv")
        self._val_clip_dirnames = load_csv(val_split_path)[0].tolist()
        test_split_path = osp.join(split_dir, "test.csv")
        self._test_clip_dirnames = load_csv(test_split_path)[0].tolist()

        self._unity_dir = unity_dir
        self._prcpt_dir = prcpt_dir

        self._n_frames = n_frames
        self._frame_size = frame_size
        self._unity_mod_max = unity_mod_max
        self._prcpt_mod_max = prcpt_mod_max

        self.batch_size = batch_size
        self.num_workers = num_workers

    @staticmethod
    def _scale_module(
        flows: torch.Tensor,
        mod_max: float,
    ) -> torch.Tensor:
        """Scale module in polar coordinates between -1 adn 1."""
        scaled_flows = torch.zeros_like(flows)
        # scaled_flows[:, 0] = (2 * (flows[:, 0] - mod_max) / mod_max) + 1
        scaled_flows[:, 0] = torch.ones_like(scaled_flows[:, 0])
        scaled_flows[:, 1] = flows[:, 1]
        return scaled_flows

    def _transform(self, mod_max: float) -> Compose:
        """Flow transformations to apply (T, C, H, W)."""
        transform_list = [
            CenterCrop(self._frame_size),
            xy_to_polar,
            Lambda(partial(self._scale_module, mod_max=mod_max)),
            polar_to_xy,
            # Normalize(mean=norm_mean, std=norm_std),
        ]
        transform = Compose(transform_list)

        return transform

    def train_dataloader(self) -> DataLoader:
        """Load train set loader."""
        self.train_set = TripletFlowDataset(
            clip_dirnames=self._train_clip_dirnames,
            unity_dir=self._unity_dir,
            prcpt_dir=self._prcpt_dir,
            n_frames=self._n_frames,
            unity_transforms=self._transform(self._unity_mod_max),
            prcpt_transforms=self._transform(self._prcpt_mod_max),
        )

        return DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Load val set loader."""
        self.val_set = TripletFlowDataset(
            clip_dirnames=self._val_clip_dirnames,
            unity_dir=self._unity_dir,
            prcpt_dir=self._prcpt_dir,
            n_frames=self._n_frames,
            unity_transforms=self._transform(self._unity_mod_max),
            prcpt_transforms=self._transform(self._prcpt_mod_max),
        )

        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Load test set loader."""
        self.test_set = TripletFlowDataset(
            clip_dirnames=self._test_clip_dirnames,
            unity_dir=self._unity_dir,
            prcpt_dir=self._prcpt_dir,
            n_frames=self._n_frames,
            unity_transforms=self._transform(self._unity_mod_max),
            prcpt_transforms=self._transform(self._prcpt_mod_max),
        )

        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
