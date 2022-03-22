import argparse
from collections import Counter
from itertools import product
import os
import os.path as osp
from typing import List, Tuple
import warnings

import cv2
from kneed import KneeLocator
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from movie_style.tools.utils import get_patches, progressbar

# Ignore "no elbow is found" warnings (cf `KneeLocator`/`_get_optimalcluster`)
warnings.filterwarnings("ignore")


class LayerDetector:
    @staticmethod
    def _get_optimalcluster(depth_maps: np.array, n_features: int = 1) -> int:
        """
        Find the optimal cluster number from a given depth map with the elbow
        method applied on the iniertia curve (`None` if no elbow is found).
        """
        X = np.array(
            [cv2.resize(_map, (640, 340)) for _map in depth_maps]
        ).T.reshape(-1, n_features)
        # X = depth_map.reshape(-1, 1)

        # Compute cluster map for clusters between 2 and 7 and store inertia
        clusters = range(2, 7)
        criterion = []
        for k_clusters in clusters:
            kmeans = MiniBatchKMeans(n_clusters=k_clusters, batch_size=2048)
            kmeans.fit(X)
            criterion.append(kmeans.inertia_)
        # Find the optimal cluster number with the elbow method
        optimal_k = KneeLocator(
            clusters, criterion, S=1, curve="convex", direction="decreasing"
        ).knee

        return optimal_k

    def _estimate_clusternumber(
        self, depth_maps: List[np.array], verbose: bool = False
    ) -> List[int]:
        """Find the optimal cluster number for each frame."""
        n_frames = len(depth_maps)
        optimal_clusters = []
        if verbose:
            print("[Optimal Cluster Estimation]")
        for k in progressbar(range(n_frames), n_frames, verbose=verbose):
            selected_indices = slice(k, k + 1)
            current_depthmaps = depth_maps[selected_indices]
            optimal_k = self._get_optimalcluster(current_depthmaps)
            # Handle no elbow case (ie: `optimal_k` is `None`)
            if not optimal_k:
                # If first frame `optimal_k` is set to 3 by default
                if not optimal_clusters:
                    optimal_k = 3
                # Else take the same value as the previous one
                else:
                    optimal_k = optimal_clusters[-1]
            optimal_clusters.append(optimal_k)

        # Add the `window_size` missing starting and ending cluster numbers
        optimal_clusters = (
            [optimal_clusters[0]] + optimal_clusters + [optimal_clusters[-1]]
        )

        return optimal_clusters

    @staticmethod
    def _get_clustermap(depth_map: np.array, k_clusters: int) -> np.array:
        """Compute the cluster map from a depth map given a cluster number."""
        height, width = depth_map.shape
        X = depth_map.reshape(-1, 1)
        kmeans = MiniBatchKMeans(n_clusters=k_clusters, batch_size=2048)
        cluster_map = kmeans.fit_predict(X).reshape(height, width)

        return cluster_map

    @staticmethod
    def _get_clusterconsitency(
        prev_clustermap: np.array, curr_clustermap: np.array
    ) -> Tuple[np.array, np.array]:
        """Map index of the previous cluster map with the current one."""
        # If this is the first frame there is no consistency to ensure
        if prev_clustermap is None:
            return curr_clustermap

        prev_k_clusters = np.unique(prev_clustermap).size
        curr_k_clusters = np.unique(curr_clustermap).size
        # Compute IoU between previous and current clusters
        cross_iou = np.zeros((prev_k_clusters, curr_k_clusters))
        for prev_k, curr_k in product(
            range(prev_k_clusters), range(curr_k_clusters)
        ):
            prev_binarymap = prev_clustermap == prev_k
            curr_binarymap = curr_clustermap == curr_k
            map_intersection = (prev_binarymap & curr_binarymap).sum()
            map_union = (prev_binarymap | curr_binarymap).sum()
            cross_iou[prev_k, curr_k] = map_intersection / map_union

        # Map previous and current clusters based on their IoU
        min_cluster = min(prev_k_clusters, curr_k_clusters)
        cluster_correspondences = -np.ones(curr_k_clusters)
        for _ in range(min_cluster):
            prev_k, curr_k = np.unravel_index(
                cross_iou.argmax(), cross_iou.shape
            )
            cluster_correspondences[curr_k] = prev_k
            cross_iou[prev_k] = -1
            cross_iou[:, curr_k] = -1
        # Add unmapped clusters (if there are more clusters in current map)
        cluster_correspondences[cluster_correspondences == -1] = np.arange(
            min_cluster, curr_k_clusters
        )
        # Map current cluster numbers to previous ones
        cluster_map = np.array(cluster_correspondences)[curr_clustermap]

        return cluster_map

    @staticmethod
    def _smooth_choice(
        optimal_clusters: List[int], current_index: int, window_size: int
    ) -> int:
        """Max pooling in the window around the `current_index`."""
        n_frames = len(optimal_clusters)
        slice_indices = slice(
            max(0, current_index - window_size),
            min(n_frames, current_index + window_size),
        )
        selected_clusters = optimal_clusters[slice_indices]
        smoothed_choice = max(selected_clusters, key=selected_clusters.count)

        return smoothed_choice

    def detect_layer(
        self, depth_maps: List[np.array], verbose: bool = False
    ) -> List[np.array]:
        """Compute cluster maps and ensure cluster consistency."""
        optimal_clusters = self._estimate_clusternumber(
            depth_maps, verbose=verbose
        )

        n_frames = len(depth_maps)
        prev_clustermap = None
        cluster_maps = []
        if verbose:
            print("[Layer Detection]")
        for k in progressbar(range(n_frames), n_frames, verbose=verbose):
            # Temporal smoothing of the optimal cluster number
            optimal_k = self._smooth_choice(
                optimal_clusters, k, window_size=10
            )
            cluster_map = self._get_clustermap(depth_maps[k], optimal_k)
            # Map the previous and current maps to ensure color consistency
            cluster_map = self._get_clusterconsitency(
                prev_clustermap, cluster_map
            )
            cluster_maps.append(cluster_map)
            prev_clustermap = cluster_map

        return cluster_maps

    @staticmethod
    def get_layerfeatures(
        layer_maps: List[np.array],
        depth_maps: List[np.array],
        featuremap_dims: Tuple[int, int],
    ) -> np.array:
        """
        Compute layer features: get the average depth value over each detected
        layer and qunatize it spatilly over a grid.

        :param layer_maps: layer maps.
        :param depth_maps: depth maps.
        :param featuremap_dims: dimensions of the output layer grid.
        :return: binarized layer grids.
        """
        n_row, n_col = featuremap_dims
        layer_features = []
        for depth_map, layer_map in zip(depth_maps, layer_maps):
            depth_layer_map = -np.ones(depth_map.shape)
            for cluster_index in np.unique(layer_map):
                cluster_mask = layer_map == cluster_index
                depth_layer_map[cluster_mask] = depth_map[cluster_mask].mean()

            layer_patches = get_patches(depth_layer_map, (n_row, n_col))
            layer_grid = -np.ones((n_row, n_col))
            for i, j in product(np.arange(n_row), np.arange(n_col)):
                patch = layer_patches[i][j].reshape(-1)
                # Find the most frequent depth level in the patch and store it
                layer_grid[i][j] = Counter(patch).most_common(1)[0][0]

            layer_features.append(layer_grid)

        # Normalize feature within the sequence
        layer_features = (layer_features - np.min(layer_features)) / (
            np.max(layer_features) - np.min(layer_features)
        )

        return layer_features


def parse_arguments():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "save_dir",
        type=str,
        help="Path to the custom feature directory",
    )
    args = parser.parse_args()

    return args


def compute_depthmaps(save_dir: str):
    depth_dir = osp.join(save_dir, "depth_maps")
    layer_dir = osp.join(save_dir, "layer_maps")
    if not osp.exists(layer_dir):
        os.makedirs(layer_dir)

    layer_detector = LayerDetector()
    for clip_year in sorted(os.listdir(depth_dir)):
        clip_year_dir = osp.join(depth_dir, clip_year)
        layer_year_dir = osp.join(layer_dir, clip_year)
        if not osp.exists(layer_year_dir):
            os.makedirs(layer_year_dir)

        for clip_id in sorted(os.listdir(clip_year_dir)):
            clip_id_dir = osp.join(clip_year_dir, clip_id)
            layer_id_dir = osp.join(layer_year_dir, clip_id)
            if not osp.exists(layer_id_dir):
                os.makedirs(layer_id_dir)

            for clip_shot in sorted(os.listdir(clip_id_dir)):
                clip_shot_dir = osp.join(clip_id_dir, clip_shot)
                layer_shot_dir = osp.join(layer_id_dir, clip_shot)
                if not osp.exists(layer_shot_dir):
                    os.makedirs(layer_shot_dir)

                shot_depths = []
                for depth_filename in sorted(os.listdir(clip_shot_dir)):
                    depth_path = osp.join(clip_shot_dir, depth_filename)
                    map_filename = osp.join(
                        layer_shot_dir, depth_filename[:-4] + ".npy"
                    )
                    if not osp.exists(map_filename):
                        with open(depth_path, "rb") as f:
                            depth_map = np.load(f)
                        shot_depths.append(depth_map)

                if not shot_depths:
                    continue

                layer_maps = layer_detector.detect_layer(shot_depths)
                for layer_filename, layer_map in zip(
                    sorted(os.listdir(clip_shot_dir)), layer_maps
                ):
                    map_filename = osp.join(
                        layer_shot_dir, layer_filename[:-4] + ".npy"
                    )
                    with open(map_filename, "wb") as f:
                        np.save(f, layer_map)
                    print(map_filename)
                print()


if __name__ == "__main__":
    args = parse_arguments()
    compute_depthmaps(args.save_dir)
