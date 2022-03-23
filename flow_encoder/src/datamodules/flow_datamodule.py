import os.path as osp

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from flow_encoder.src.datamodules.datasets.triplet_dataset import (
    TripletFlowDataset,
)
from utils.file_utils import load_csv


class TripletFlowDataModule(LightningDataModule):
    """Initialize train, val and test base data loader.

    :param split_dir: path to the directory with train/val/test splits.
    :param unity_dir: path to the directory with Unity flows.
    :param prcpt_dir: path to the directory with precomputed flows.
    :param n_frames: number of frames in a sample (fixed by the model).
    :param stride: number of frames between 2 consecutive samples.
    :param batch_size: size of batches.
    :param num_workers: number of workers.
    """

    def __init__(
        self,
        split_dir: str,
        unity_dir: str,
        prcpt_dir: str,
        n_frames: int,
        stride: int,
        frame_size: int,
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
        # self._test_clip_dirnames = load_csv(test_split_path)
        # self._test_clip_dirnames = (
        #     self._test_clip_dirnames[0].tolist()
        #     if self._test_clip_dirnames.size
        #     else []
        # )

        self._unity_dir = unity_dir
        self._prcpt_dir = prcpt_dir

        self._n_frames = n_frames
        self._stride = stride

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        """Load train set loader."""
        self.train_set = TripletFlowDataset(
            clip_dirnames=self._train_clip_dirnames,
            unity_dir=self._unity_dir,
            prcpt_dir=self._prcpt_dir,
            n_frames=self._n_frames,
            stride=self._stride,
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
            stride=self._stride,
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
            stride=self._stride,
        )

        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
