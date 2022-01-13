import os
import os.path as osp
import sys
from typing import List, Tuple

import numpy as np
import torch

# Avoid local import issues
root_dir = [os.sep] + osp.dirname(osp.abspath(__file__)).split(os.sep)[:-2]
import_dir = ["lib", "camera_control"]
sys.path.append(osp.join(*root_dir + import_dir))
from movie_processing import extract_movie_feature, load_ckpt
from Net import combined_CNN


class ToricEstimator:
    def __init__(self, sequence_length: int = 4):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_name = "toric_estimation"
        self._checkpoint_filename = osp.join(
            *root_dir, "models", model_name + ".tar"
        )
        model = combined_CNN(seq_length=2 * sequence_length, channels=28)
        load_ckpt(model, self._checkpoint_filename)
        self.model = model.to(self.device)

    def estimate_toric(
        self, poselets: List[np.array], frame_dim: Tuple[int, int]
    ) -> np.array:
        """Estimate toric angles.
        Code adapted from: https://github.com/jianghd1996/Camera-control.

        :param poselets: list of detected poselets for each farmes
            (n_detections, n_frames, 2*n_joints).
        :param frame_dim: height and width of the source frame.
        :return: (n_frames, 12):
            - pB, pA, pY, theta, phi.
            - relative head, relative shoulder, shoulder1-line, shoudler2-line,
              head1-line, head2-line.
        """
        # TODO: explain
        raw = extract_movie_feature(poselets, "raw", size=frame_dim)
        raw = torch.tensor(raw).to(self.device)
        # TODO: explain
        no_pX = extract_movie_feature(poselets, "no_pX", size=frame_dim)
        no_pX = torch.tensor(no_pX).to(self.device)
        # TODO: explain
        no_pY = extract_movie_feature(poselets, "no_pY", size=frame_dim)
        no_pY = torch.tensor(no_pY).to(self.device)
        # TODO: explain
        no_all = extract_movie_feature(poselets, "no_all", size=frame_dim)
        no_all = torch.tensor(no_all).to(self.device)

        # Infer the model
        with torch.no_grad():
            output = self.model([raw, no_pX, no_pY, no_all]).cpu().numpy()

        n_frames = len(raw)
        output_features = np.zeros((n_frames, 12), dtype="float32")
        for j in range(n_frames):
            output_features[j][0] = raw[j][0][4] * 2  # x0
            output_features[j][1] = raw[j][14][4] * 2  # x1
            output_features[j][2] = -raw[j][1][4] + -raw[j][15][4]  # y0
            output_features[j][3:] = output[j][:]

        return output_features
