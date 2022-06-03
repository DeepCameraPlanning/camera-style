"""This code is adapted from: https://github.com/piergiaj/pytorch-i3d."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_encoder.src.models.modules.vector_quantizer import VectorQuantizer


class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()

        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):
    def __init__(
        self,
        in_channels,
        output_channels,
        kernel_shape=(1, 1, 1),
        stride=(1, 1, 1),
        padding=0,
        activation_fn=F.relu,
        use_batch_norm=True,
        use_bias=False,
        name="unit_3d",
    ):

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self._output_channels,
            kernel_size=self._kernel_shape,
            stride=self._stride,
            padding=0,
            bias=self._use_bias,
        )

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(
                self._output_channels, eps=0.001, momentum=0.01
            )

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()

        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_0/Conv3d_0a_1x1",
        )
        self.b1a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_1/Conv3d_0a_1x1",
        )
        self.b1b = Unit3D(
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_shape=[3, 3, 3],
            name=name + "/Branch_1/Conv3d_0b_3x3",
        )
        self.b2a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_2/Conv3d_0a_1x1",
        )
        self.b2b = Unit3D(
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_shape=[3, 3, 3],
            name=name + "/Branch_2/Conv3d_0b_3x3",
        )
        self.b3a = MaxPool3dSamePadding(
            kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0
        )
        self.b3b = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_3/Conv3d_0b_1x1",
        )
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class ReducedInceptionI3d(nn.Module):
    """
    Smaller I3D architecture, with output dimension: 490 / 980 / 2058.
    """

    VALID_ENDPOINTS = (
        "Conv3d_1a_7x7",
        "MaxPool3d_2a_3x3",
        "Conv3d_2b_1x1",
        "Conv3d_2c_3x3",
        "MaxPool3d_3a_3x3",
        "Mixed_3b",
        "Mixed_3c",
        "MaxPool3d_4a_3x3",
        "Mixed_4b",
        "Mixed_4c",
        "Mixed_4d",
        "Mixed_4e",
        "Mixed_4f",
        "MaxPool3d_5a_2x2",
        "Mixed_5b",
        "Mixed_5c",
    )

    def __init__(
        self,
        in_channels=2,
        out_channels=2058,
        final_endpoint="Mixed_5c",
        name="inception_i3d",
    ):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError("Unknown final endpoint %s" % final_endpoint)

        super(ReducedInceptionI3d, self).__init__()
        self._final_endpoint = final_endpoint

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError(
                "Unknown final endpoint %s" % self._final_endpoint
            )

        self.end_points = {}
        end_point = "Conv3d_1a_7x7"
        self.end_points[end_point] = Unit3D(
            in_channels=in_channels,
            output_channels=3,
            kernel_shape=[7, 7, 7],
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = "MaxPool3d_2a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0
        )
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = "Conv3d_2b_1x1"
        self.end_points[end_point] = Unit3D(
            in_channels=3,
            output_channels=3,
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = "Conv3d_2c_3x3"
        self.end_points[end_point] = Unit3D(
            in_channels=3,
            output_channels=4,
            kernel_shape=[3, 3, 3],
            padding=1,
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = "MaxPool3d_3a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0
        )
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = "Mixed_3b"
        self.end_points[end_point] = InceptionModule(
            4, [2, 3, 4, 1, 1, 1], name + end_point
        )
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = "Mixed_3c"
        self.end_points[end_point] = InceptionModule(
            8, [4, 4, 6, 2, 4, 2], name + end_point
        )
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = "MaxPool3d_4a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0
        )
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = "Mixed_4b"
        self.end_points[end_point] = InceptionModule(
            16, [6, 3, 10, 1, 2, 4], name + end_point
        )
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = "Mixed_4c"
        self.end_points[end_point] = InceptionModule(
            22, [4, 4, 10, 1, 4, 4], name + end_point
        )
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = "Mixed_4d"
        self.end_points[end_point] = InceptionModule(
            22, [2, 2, 12, 4, 4, 4], name + end_point
        )
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = "Mixed_4e"
        self.end_points[end_point] = InceptionModule(
            22, [2, 2, 14, 6, 4, 4], name + end_point
        )
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = "Mixed_4f"
        out = (
            [4, 2, 16, 6, 4, 4]
            if out_channels in [980, 2058]
            else [4, 2, 10, 6, 4, 4]  # 490
        )
        self.end_points[end_point] = InceptionModule(24, out, name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = "MaxPool3d_5a_2x2"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0
        )
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = "Mixed_5b"
        i = 28 if out_channels in [980, 2058] else 22
        out = (
            [4, 2, 16, 6, 4, 4]
            if out_channels in [980, 2058]
            else [2, 2, 8, 6, 3, 3]  # 490
        )
        self.end_points[end_point] = InceptionModule(i, out, name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = "Mixed_5c"
        i = 28 if out_channels in [980, 2058] else 16
        out = (
            [4, 4, 11, 6, 3, 3]
            if out_channels == 2058
            else [3, 3, 3, 3, 2, 2]
            if out_channels == 980
            else [1, 2, 2, 2, 1, 1]  # 490
        )
        self.end_points[end_point] = InceptionModule(i, out, name + end_point)
        if self._final_endpoint == end_point:
            self.build()
            return

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)

        return x

    def extract_features(self, x, avg_out=True):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)

        return x


class DecoderConv3D(nn.Module):
    """3D CNN flow decoder."""

    VALID_ENDPOINTS = (
        "Conv3d_1_1x1",
        "Upconv3d_2a",
        # "Conv3d_2b_3x3",
        "Conv3d_2c_3x3",
        "Upconv3d_3a",
        # "Conv3d_3b_3x3",
        "Conv3d_3c_3x3",
        "Upconv3d_4a",
        # "Conv3d_4b_3x3",
        "Conv3d_4c_3x3",
        "Upconv3d_5a",
        # "Conv3d_5b_3x3",
        "Upconv3d_6",
    )

    def __init__(self, in_channels: int):
        super(DecoderConv3D, self).__init__()

        self.end_points = {}

        end_point = "Conv3d_1_1x1"
        self.end_points[end_point] = Unit3D(
            in_channels=in_channels,
            output_channels=28,
            kernel_shape=[1, 1, 1],
            padding=0,
            name=end_point,
        )

        end_point = "Upconv3d_2a"
        block_dim = 28
        self.end_points[end_point] = nn.Sequential(
            nn.ConvTranspose3d(
                block_dim, block_dim, (4, 4, 4), (2, 2, 2), 1, bias=False
            ),
            nn.BatchNorm3d(block_dim),
            nn.ReLU(inplace=True),
        )
        end_point = "Conv3d_2b_3x3"
        self.end_points[end_point] = Unit3D(
            in_channels=block_dim,
            output_channels=block_dim,
            kernel_shape=[3, 3, 3],
            padding=0,
            name=end_point,
        )
        end_point = "Conv3d_2c_3x3"
        self.end_points[end_point] = Unit3D(
            in_channels=block_dim,
            output_channels=16,
            kernel_shape=[3, 3, 3],
            padding=0,
            name=end_point,
        )

        end_point = "Upconv3d_3a"
        block_dim = 16
        self.end_points[end_point] = nn.Sequential(
            nn.ConvTranspose3d(block_dim, block_dim, 4, 2, 1, bias=False),
            nn.BatchNorm3d(block_dim),
            nn.ReLU(inplace=True),
        )
        end_point = "Conv3d_3b_3x3"
        self.end_points[end_point] = Unit3D(
            in_channels=block_dim,
            output_channels=block_dim,
            kernel_shape=[3, 3, 3],
            padding=0,
            name=end_point,
        )
        end_point = "Conv3d_3c_3x3"
        self.end_points[end_point] = Unit3D(
            in_channels=block_dim,
            output_channels=4,
            kernel_shape=[3, 3, 3],
            padding=0,
            name=end_point,
        )

        end_point = "Upconv3d_4a"
        block_dim = 4
        self.end_points[end_point] = nn.Sequential(
            nn.ConvTranspose3d(
                block_dim, block_dim, (3, 4, 4), (1, 2, 2), 1, bias=False
            ),
            nn.BatchNorm3d(block_dim),
            nn.ReLU(inplace=True),
        )
        end_point = "Conv3d_4b_3x3"
        self.end_points[end_point] = Unit3D(
            in_channels=block_dim,
            output_channels=block_dim,
            kernel_shape=[3, 3, 3],
            padding=0,
            name=end_point,
        )
        end_point = "Conv3d_4c_3x3"
        self.end_points[end_point] = Unit3D(
            in_channels=block_dim,
            output_channels=3,
            kernel_shape=[3, 3, 3],
            padding=0,
            name=end_point,
        )

        end_point = "Upconv3d_5a"
        block_dim = 3
        self.end_points[end_point] = nn.Sequential(
            nn.ConvTranspose3d(
                block_dim, block_dim, (3, 4, 4), (1, 2, 2), 1, bias=False
            ),
            nn.BatchNorm3d(block_dim),
            nn.ReLU(inplace=True),
        )
        end_point = "Conv3d_5b_3x3"
        self.end_points[end_point] = Unit3D(
            in_channels=block_dim,
            output_channels=block_dim,
            kernel_shape=[3, 3, 3],
            padding=0,
            name=end_point,
        )

        end_point = "Upconv3d_6"
        self.end_points[end_point] = nn.Sequential(
            nn.ConvTranspose3d(3, 2, (4, 4, 4), (2, 2, 2), 1, bias=False),
            nn.BatchNorm3d(2),
        )

        self.tanh = nn.Tanh()
        self.build()

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x, residual_features) -> torch.Tensor:
        up_conv_count = 0
        for end_point in self.decoder.VALID_ENDPOINTS:
            x = self.decoder._modules[end_point](x)
            print(end_point, x.shape)
            if (
                isinstance(self.decoder._modules[end_point], nn.Sequential)
                and end_point != "Upconv3d_6"
            ):
                up_conv_count += 1
                x = torch.hstack([x, residual_features[-up_conv_count]])
        out = self.decoder.tanh(x)
        return out


class AutoencoderI3D(nn.Module):
    """Flow autoencoder with an I3D encoder. Latent dim: 490 / 980 / 2058."""

    def __init__(self, in_channels: int, latent_dim: int):
        super(AutoencoderI3D, self).__init__()
        self.encoder = ReducedInceptionI3d(
            in_channels=in_channels, out_channels=latent_dim
        )
        # Dimension of the latent code = [latent_dim, 2, 7, 7]
        latent_in_channels = latent_dim // (7 * 7 * 2)
        self.decoder = DecoderConv3D(in_channels=latent_in_channels)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_features, residual_features = self._encode(x)
        out = self._decode(latent_features, residual_features)
        return latent_features, out

    def extract_features(self, x) -> torch.Tensor:
        for end_point in self.encoder.VALID_ENDPOINTS:
            if end_point in self.encoder.end_points:
                x = self.encoder._modules[end_point](x)
        return x

    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual_features = []
        for end_point in self.encoder.VALID_ENDPOINTS:
            if end_point in self.encoder.end_points:
                if isinstance(self.encoder._modules[end_point], nn.MaxPool3d):
                    residual_features.append(x)
                x = self.encoder._modules[end_point](x)
                # print(end_point, x.shape)
        return x, residual_features

    def _decode(
        self, x: torch.Tensor, residual_features: torch.Tensor
    ) -> torch.Tensor:
        # up_conv_count = 0
        for end_point in self.decoder.VALID_ENDPOINTS:
            x = self.decoder._modules[end_point](x)
            # if (
            #     isinstance(self.decoder._modules[end_point], nn.Sequential)
            #     and end_point != "Upconv3d_6"
            # ):
            #     up_conv_count += 1
            #     x = torch.hstack([x, residual_features[-up_conv_count]])
            # print(end_point, x.shape)
        out = self.decoder.tanh(x)

        return out

    def _get_shared_layer(self) -> torch.Tensor:
        """Return last encoder layer parameters."""
        return self.encoder.Mixed_5c.b3b.parameters()


class VQVAEI3D(nn.Module):
    """Flow VQ-VAE with an I3D encoder. Latent dim: 490 / 980 / 2058."""

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        n_embeddings: int,
        commitment_cost: float,
    ):
        super(VQVAEI3D, self).__init__()
        self.encoder = ReducedInceptionI3d(
            in_channels=in_channels, out_channels=latent_dim
        )
        # Dimension of the latent code = [latent_dim, 2, 7, 7]
        latent_in_channels = latent_dim // (7 * 7 * 2)
        self.decoder = DecoderConv3D(in_channels=latent_in_channels)
        self.quantizer = VectorQuantizer(
            n_embeddings=n_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
        )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_features, residual_features = self._encode(x)
        vq_loss, quantized_features, encodings = self._quantize(
            latent_features
        )
        out = self._decode(quantized_features, residual_features)
        return quantized_features, vq_loss, out

    def extract_features(self, x) -> torch.Tensor:
        for end_point in self.encoder.VALID_ENDPOINTS:
            if end_point in self.encoder.end_points:
                x = self.encoder._modules[end_point](x)
        return x

    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual_features = []
        for end_point in self.encoder.VALID_ENDPOINTS:
            if end_point in self.encoder.end_points:
                if isinstance(self.encoder._modules[end_point], nn.MaxPool3d):
                    residual_features.append(x)
                x = self.encoder._modules[end_point](x)
        return x, residual_features

    def _quantize(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss, quantized, perplexity, _ = self.quantizer(x)
        return loss, quantized, perplexity

    def _decode(
        self, x: torch.Tensor, residual_features: torch.Tensor
    ) -> torch.Tensor:
        # up_conv_count = 0
        for end_point in self.decoder.VALID_ENDPOINTS:
            x = self.decoder._modules[end_point](x)
            # if (
            #     isinstance(self.decoder._modules[end_point], nn.Sequential)
            #     and end_point != "Upconv3d_6"
            # ):
            #     up_conv_count += 1
            #     x = torch.hstack([x, residual_features[-up_conv_count]])
            # print(end_point, x.shape)
        out = self.decoder.tanh(x)

        return out

    def _get_shared_layer(self) -> torch.Tensor:
        """Return last encoder layer parameters."""
        return self.encoder.Mixed_5c.b3b.parameters()


def make_flow_encoder(pretrained_path: str, size: str = "large") -> nn.Module:
    """Load a standard flow I3D (2 channels) architecture.

    :param pretrained_path: if provided load the model stored at this location.
    :param size:
        - "small" -> 490 dimensional output feature.
        - "large" -> 960 dimensional output feature.
        - "huge" -> 2058 dimensional output feature.
    :return: configured flow I3D.
    """
    out_channels = 490 if size == "small" else 980 if size == "large" else 2058
    model = ReducedInceptionI3d(in_channels=2, out_channels=out_channels)

    if pretrained_path:
        pretrained_params = torch.load(pretrained_path)
        state_dict = {
            ".".join(k.split(".")[1:]): v
            for k, v in pretrained_params["state_dict"].items()
        }
        model.load_state_dict(state_dict)

    return model


def make_flow_autoencoder(
    pretrained_path: str, size: str = "large"
) -> nn.Module:
    """Load a flow (2 channels) autoencoder (I3D encoder) architecture.

    :param pretrained_path: if provided load the model stored at this location.
    :param size:
        - "small" -> 490 dimensional latent code.
        - "large" -> 960 dimensional latent code.
        - "huge" -> 2058 dimensional latent code.
    :return: configured flow autoencoder.
    """
    latent_dim = 490 if size == "small" else 980 if size == "large" else 2058
    model = AutoencoderI3D(in_channels=2, latent_dim=latent_dim)

    if pretrained_path:
        pretrained_params = torch.load(pretrained_path)
        state_dict = {
            ".".join(k.split(".")[1:]): v
            for k, v in pretrained_params["state_dict"].items()
        }
        model.load_state_dict(state_dict)

    return model


def make_flow_vqvae(pretrained_path: str, size: str = "large") -> nn.Module:
    """Load a flow (2 channels) vqvae (I3D encoder) architecture.

    :param pretrained_path: if provided load the model stored at this location.
    :param size:
        - "small" -> 490 dimensional latent code.
        - "large" -> 960 dimensional latent code.
        - "huge" -> 2058 dimensional latent code.
    :return: configured flow vq-vae.
    """
    latent_dim = 490 if size == "small" else 980 if size == "large" else 2058
    model = VQVAEI3D(
        in_channels=2,
        latent_dim=latent_dim,
        n_embeddings=64,
        commitment_cost=0.25,
    )

    if pretrained_path:
        pretrained_params = torch.load(pretrained_path)
        state_dict = {
            ".".join(k.split(".")[1:]): v
            for k, v in pretrained_params["state_dict"].items()
        }
        model.load_state_dict(state_dict)

    return model
