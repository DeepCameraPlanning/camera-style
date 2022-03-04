import argparse
import os
import os.path as osp
from typing import List, Tuple

from sklearn.model_selection import train_test_split

from src.utils.utils import create_dir, save_csv


def parse_arguments() -> str:
    """Parse input arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "raw_dir",
        type=str,
        help="Path to the directory containing the raw frame directories",
    )
    parser.add_argument(
        "split_dir",
        type=str,
        help="Path to the directory to save dataset splits",
    )

    args = parser.parse_args()

    return args.raw_dir, args.split_dir


def split_dataset(raw_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """Split clips into train/val/test (0.6/0.2/0.2)."""
    raw_files = sorted(os.listdir(raw_dir))
    trainval_files, test_files = train_test_split(
        raw_files, test_size=0.2, random_state=1
    )
    train_files, val_files = train_test_split(
        trainval_files, test_size=0.25, random_state=1
    )
    return train_files, val_files, test_files


if __name__ == "__main__":
    raw_dir, split_dir = parse_arguments()
    create_dir(split_dir)

    train_files, val_files, test_files = split_dataset(raw_dir)
    val_files.append("pan")

    train_split_path = osp.join(split_dir, "train.csv")
    val_split_path = osp.join(split_dir, "val.csv")
    test_split_path = osp.join(split_dir, "test.csv")

    save_csv(train_files, train_split_path)
    save_csv(val_files, val_split_path)
    save_csv(test_files, test_split_path)
