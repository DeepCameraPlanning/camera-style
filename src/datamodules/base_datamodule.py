from typing import Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from src.datamodules.datasets.base_dataset import BaseDataset


class BaseDataModule(LightningDataModule):
    """Initialize train, val and test base data loader.

    :param dataset_dir: path to the dataset directory.
    :param frame_size: resizing target height and width.
    :param norm_mean: pixel RGB mean.
    :param norm_std: pixel RGB standard deviation.
    :param n_frames: number of frames in a sample (fixed by the model).
    :param batch_size: size of batches.
    :param num_workers: number of workers.
    """

    def __init__(
        self,
        dataset_dir: str,
        n_frames: int,
        frame_size: int,
        norm_mean: Tuple[float, float, float],
        norm_std: Tuple[float, float, float],
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()

        self._dataset_dir = dataset_dir

        self._frame_size = frame_size
        self._norm_mean = norm_mean
        self._norm_std = norm_std
        self._n_frames = n_frames

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
        self.train_set = BaseDataset(
            dataset_dir=self._dataset_dir,
            set_mode="train",
            n_frames=self._n_frames,
        )

        return DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Load val set loader."""
        self.val_set = BaseDataset(
            dataset_dir=self._dataset_dir,
            set_mode="val",
            n_frames=self._n_frames,
        )

        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Load test set loader."""
        self.test_set = BaseDataset(
            dataset_dir=self._dataset_dir,
            set_mode="test",
            n_frames=self._n_frames,
        )

        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
