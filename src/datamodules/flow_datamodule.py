# from typing import Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from src.datamodules.datasets.flow_dataset import TripletFlowDataset


class TripletFlowDataModule(LightningDataModule):
    """Initialize train, val and test base data loader.

    :param unity_dir: path to the directory with precomputed Unity flows.
    :param raft_dir: path to the directory with precomputed RAFT flows.
    :param n_frames: number of frames in a sample (fixed by the model).
    # :param norm_mean: pixel RGB mean.
    # :param norm_std: pixel RGB standard deviation.
    :param batch_size: size of batches.
    :param num_workers: number of workers.
    """

    def __init__(
        self,
        unity_dir: str,
        raft_dir: str,
        n_frames: int,
        # norm_mean: Tuple[float, float, float],
        # norm_std: Tuple[float, float, float],
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()

        self._unity_dir = unity_dir
        self._raft_dir = raft_dir

        self._n_frames = n_frames
        # self._norm_mean = norm_mean
        # self._norm_std = norm_std

        self.batch_size = batch_size
        self.num_workers = num_workers

    def _transform(self) -> Compose:
        """Video transformations to apply (SoundNet)."""
        transform_list = [
            Resize((self._frame_size, self._frame_size)),
            ToTensor(),
            Normalize(mean=self._norm_mean, std=self._norm_std),
        ]
        transform = Compose(transform_list)

        return transform

    def train_dataloader(self) -> DataLoader:
        """Load train set loader."""
        self.train_set = TripletFlowDataset(
            unity_dir=self._unity_dir,
            raft_dir=self._raft_dir,
            n_frames=self._n_frames,
            transform=None,
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
            unity_dir=self._unity_dir,
            raft_dir=self._raft_dir,
            n_frames=self._n_frames,
            transform=None,
        )

        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Load test set loader."""
        self.test_set = TripletFlowDataset(
            unity_dir=self._unity_dir,
            raft_dir=self._raft_dir,
            n_frames=self._n_frames,
            transform=None,
        )

        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
