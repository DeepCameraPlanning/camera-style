import os.path as osp

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from character_encoder.src.datamodules.datasets.latent_bbox_dataset import (
    LatentBboxDataset,
)
from utils.file_utils import load_csv


class LatentBboxDataModule(LightningDataModule):
    """Initialize train, val and test base data loader.

    :param split_dir: path to the directory with train/val/test splits.
    :param bbox_dir: directory containing pre-extracted detections.
    :param feature_dir: directory containing pre-extracted flow features.
    :param flow_dir: directory containing pre-compute flow frames.
    :param batch_size: size of batches.
    :param num_workers: number of workers.
    """

    def __init__(
        self,
        split_dir: str,
        bbox_dir: str,
        feature_dir: str,
        flow_dir: str,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()

        train_split_path = osp.join(split_dir, "train.csv")
        self._train_clip_dirnames = load_csv(train_split_path)[0].tolist()
        val_split_path = osp.join(split_dir, "val.csv")
        self._val_clip_dirnames = load_csv(val_split_path)[0].tolist()
        test_split_path = osp.join(split_dir, "val.csv")
        self._test_clip_dirnames = load_csv(test_split_path)[0].tolist()

        self._bbox_dir = bbox_dir
        self._feature_dir = feature_dir
        self._flow_dir = flow_dir

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        """Load train set loader."""
        self.train_set = LatentBboxDataset(
            clip_dirnames=self._train_clip_dirnames,
            bbox_dir=self._bbox_dir,
            feature_dir=self._feature_dir,
            flow_dir=self._flow_dir,
        )

        return DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Load val set loader."""
        self.val_set = LatentBboxDataset(
            clip_dirnames=self._val_clip_dirnames,
            bbox_dir=self._bbox_dir,
            feature_dir=self._feature_dir,
            flow_dir=self._flow_dir,
        )

        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Load test set loader."""
        self.test_set = LatentBboxDataset(
            clip_dirnames=self._test_clip_dirnames,
            bbox_dir=self._bbox_dir,
            feature_dir=self._feature_dir,
            flow_dir=self._flow_dir,
        )

        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
