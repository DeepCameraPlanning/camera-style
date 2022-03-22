import argparse
import os
import os.path as osp
from typing import List, Tuple

from sklearn.model_selection import train_test_split

from src.utils.file_utils import create_dir, save_csv


def parse_arguments() -> Tuple[str, str, str, bool]:
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
    parser.add_argument(
        "--augment-dir",
        "-a",
        type=str,
        default=None,
        help="Path to the directory containing augmented frame directories",
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Wether to add a test set",
    )
    args = parser.parse_args()

    return args.raw_dir, args.split_dir, args.augment_dir, args.test


def split_dataset(
    raw_dir: str, augment_dir: str, test_set: bool
) -> Tuple[List[str], List[str], List[str]]:
    """Split clips into train/val (0.8/0.2) or train/val/test (0.6/0.2/0.2)."""
    raw_files = sorted(os.listdir(raw_dir))

    if test_set:
        trainval_files, test_files = train_test_split(
            raw_files, test_size=0.2, random_state=32
        )
        train_files, val_files = train_test_split(
            trainval_files, test_size=0.25, random_state=32
        )

    else:
        train_files, val_files = train_test_split(
            raw_files, test_size=0.2, random_state=32
        )
        test_files = val_files

    if augment_dir is not None:
        for filename in train_files:
            reversed_filename = "reversed_" + filename
            if osp.exists(osp.join(augment_dir, reversed_filename)):
                train_files.append(reversed_filename)
            speedup_filename = "speedup_" + filename
            if osp.exists(osp.join(augment_dir, speedup_filename)):
                train_files.append(speedup_filename)

    return train_files, val_files, test_files


if __name__ == "__main__":
    raw_dir, split_dir, augment_dir, test = parse_arguments()
    create_dir(split_dir)

    train_files, val_files, test_files = split_dataset(
        raw_dir, augment_dir, test
    )

    train_split_path = osp.join(split_dir, "train.csv")
    val_split_path = osp.join(split_dir, "val.csv")
    test_split_path = osp.join(split_dir, "test.csv")

    save_csv(test_files, test_split_path)
    save_csv(train_files, train_split_path)
    save_csv(val_files, val_split_path)
