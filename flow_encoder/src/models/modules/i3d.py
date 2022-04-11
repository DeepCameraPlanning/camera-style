"""This code is adapted from: https://github.com/piergiaj/pytorch-i3d."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints
    # up to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
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
        "Logits",
        "Predictions",
    )

    def __init__(
        self,
        num_classes=400,
        spatial_squeeze=True,
        final_endpoint="Logits",
        name="inception_i3d",
        in_channels=3,
        dropout_keep_prob=0.5,
    ):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400,
            which matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the
            logits before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be
              built up to. In addition to the output at `final_endpoint`, all
              the outputs at endpoints up to `final_endpoint` will also be
              returned, in a dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError("Unknown final endpoint %s" % final_endpoint)

        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError(
                "Unknown final endpoint %s" % self._final_endpoint
            )

        self.end_points = {}
        end_point = "Conv3d_1a_7x7"
        self.end_points[end_point] = Unit3D(
            in_channels=in_channels,
            output_channels=64,
            kernel_shape=[7, 7, 7],
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_2a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Conv3d_2b_1x1"
        self.end_points[end_point] = Unit3D(
            in_channels=64,
            output_channels=64,
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Conv3d_2c_3x3"
        self.end_points[end_point] = Unit3D(
            in_channels=64,
            output_channels=192,
            kernel_shape=[3, 3, 3],
            padding=1,
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_3a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_3b"
        self.end_points[end_point] = InceptionModule(
            192, [64, 96, 128, 16, 32, 32], name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_3c"
        self.end_points[end_point] = InceptionModule(
            256, [128, 128, 192, 32, 96, 64], name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_4a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4b"
        self.end_points[end_point] = InceptionModule(
            128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4c"
        self.end_points[end_point] = InceptionModule(
            192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4d"
        self.end_points[end_point] = InceptionModule(
            160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4e"
        self.end_points[end_point] = InceptionModule(
            128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4f"
        self.end_points[end_point] = InceptionModule(
            112 + 288 + 64 + 64,
            [256, 160, 320, 32, 128, 128],
            name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_5a_2x2"
        self.end_points[end_point] = MaxPool3dSamePadding(
            kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_5b"
        self.end_points[end_point] = InceptionModule(
            256 + 320 + 128 + 128,
            [256, 160, 320, 32, 128, 128],
            name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_5c"
        self.end_points[end_point] = InceptionModule(
            256 + 320 + 128 + 128,
            [384, 192, 384, 48, 128, 128],
            name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Logits"
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(
            in_channels=384 + 384 + 128 + 128,
            output_channels=self._num_classes,
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name="logits",
        )

        self.build()

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(
            in_channels=384 + 384 + 128 + 128,
            output_channels=self._num_classes,
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name="logits",
        )

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

        if avg_out:
            x = self.avg_pool(x)

        return x


class ReducedInceptionI3d(nn.Module):
    """ """

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
        final_endpoint="Mixed_5c",
        name="inception_i3d",
        in_channels=2,
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
            # output_channels=4,
            output_channels=3,
            kernel_shape=[7, 7, 7],
            stride=(2, 4, 4),
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
            # in_channels=4,
            # output_channels=4,
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
            # in_channels=4,
            # output_channels=8,
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
        # 8, [4, 6, 8, 1, 2, 2], name + end_point
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = "Mixed_3c"
        self.end_points[end_point] = InceptionModule(
            8, [4, 4, 6, 2, 4, 2], name + end_point
        )
        # 16, [8, 8, 12, 2, 8, 4], name + end_point
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
        # 48, [18, 8, 20, 1, 4, 6], name + end_point
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = "Mixed_4d"
        self.end_points[end_point] = InceptionModule(
            22, [2, 2, 12, 4, 4, 4], name + end_point
        )
        # 48, [12, 12, 26, 4, 6, 6], name + end_point
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = "Mixed_4e"
        self.end_points[end_point] = InceptionModule(
            22, [2, 2, 14, 6, 4, 4], name + end_point
        )
        # 50, [12, 14, 30, 6, 8, 8], name + end_point
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = "Mixed_4f"
        self.end_points[end_point] = InceptionModule(
            24, [4, 2, 16, 6, 4, 4], name + end_point
        )
        # 58, [12, 14, 30, 6, 8, 8], name + end_point
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
        self.end_points[end_point] = InceptionModule(
            28, [4, 2, 16, 6, 4, 4], name + end_point
        )
        # 58, [12, 14, 30, 6, 8, 8], name + end_point
        if self._final_endpoint == end_point:
            self.build()
            return

        end_point = "Mixed_5c"
        self.end_points[end_point] = InceptionModule(
            28, [6, 4, 18, 6, 4, 4], name + end_point
        )
        # 58, [16, 14, 32, 6, 8, 8], name + end_point
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


class SimpleDecoderConv3D(nn.Module):
    """3D CNN flow decoder."""

    def __init__(self, in_channels: int, latent_dim: int):
        super(SimpleDecoderConv3D, self).__init__()

        ndf = latent_dim // 8
        # dconv0 out: 512, 4, 14, 14
        self.deconv0 = nn.Sequential(
            nn.ConvTranspose3d((ndf * 8), ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.ReLU(inplace=True),
        )
        # dconv1 out: 256, 8, 28, 28
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(ndf * 4, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.ReLU(inplace=True),
        )
        # dconv2 out: 128, 16, 56, 56
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(ndf * 2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf),
            nn.ReLU(inplace=True),
        )
        # dconv3 out: 2, 16, 224, 224
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(
                ndf, in_channels, (3, 9, 9), (1, 7, 7), 1, bias=False
            ),
            # nn.Sigmoid(),
            nn.Tanh(),
        )

    def forward(self, x) -> torch.Tensor:
        x = self.deconv0(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x


class DecoderConv3D(nn.Module):
    """3D CNN flow decoder."""

    VALID_ENDPOINTS = (
        "Conv3d_1_1x1",
        "Upconv3d_2a",
        "Conv3d_2b_3x3",
        "Conv3d_2c_3x3",
        "Upconv3d_3a",
        "Conv3d_3b_3x3",
        "Conv3d_3c_3x3",
        "Upconv3d_4a",
        "Conv3d_4b_3x3",
        "Conv3d_4c_3x3",
        "Upconv3d_5a",
        "Conv3d_5b_3x3",
        "Upconv3d_6",
    )

    def __init__(self, in_channels: int, latent_dim: int):
        super(DecoderConv3D, self).__init__()

        self.end_points = {}

        end_point = "Conv3d_1_1x1"
        self.end_points[end_point] = Unit3D(
            in_channels=latent_dim,
            # output_channels=58,
            output_channels=28,
            kernel_shape=[1, 1, 1],
            padding=0,
            name=end_point,
        )

        end_point = "Upconv3d_2a"
        # block_dim = 58
        block_dim = 28
        self.end_points[end_point] = nn.Sequential(
            nn.ConvTranspose3d(
                block_dim, block_dim, (4, 3, 3), (2, 2, 2), 1, bias=False
            ),
            nn.BatchNorm3d(block_dim),
            nn.ReLU(inplace=True),
        )
        end_point = "Conv3d_2b_3x3"
        self.end_points[end_point] = Unit3D(
            in_channels=2 * block_dim,
            output_channels=block_dim,
            kernel_shape=[3, 3, 3],
            padding=0,
            name=end_point,
        )
        end_point = "Conv3d_2c_3x3"
        self.end_points[end_point] = Unit3D(
            in_channels=block_dim,
            # output_channels=32,
            output_channels=16,
            kernel_shape=[3, 3, 3],
            padding=0,
            name=end_point,
        )

        end_point = "Upconv3d_3a"
        # block_dim = 32
        block_dim = 16
        self.end_points[end_point] = nn.Sequential(
            nn.ConvTranspose3d(block_dim, block_dim, 4, 2, 1, bias=False),
            nn.BatchNorm3d(block_dim),
            nn.ReLU(inplace=True),
        )
        end_point = "Conv3d_3b_3x3"
        self.end_points[end_point] = Unit3D(
            in_channels=2 * block_dim,
            output_channels=block_dim,
            kernel_shape=[3, 3, 3],
            padding=0,
            name=end_point,
        )
        end_point = "Conv3d_3c_3x3"
        self.end_points[end_point] = Unit3D(
            in_channels=block_dim,
            # output_channels=8,
            output_channels=4,
            kernel_shape=[3, 3, 3],
            padding=0,
            name=end_point,
        )

        end_point = "Upconv3d_4a"
        # block_dim = 8
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
            in_channels=2 * block_dim,
            output_channels=block_dim,
            kernel_shape=[3, 3, 3],
            padding=0,
            name=end_point,
        )
        end_point = "Conv3d_4c_3x3"
        self.end_points[end_point] = Unit3D(
            in_channels=block_dim,
            # output_channels=4,
            output_channels=3,
            kernel_shape=[3, 3, 3],
            padding=0,
            name=end_point,
        )

        end_point = "Upconv3d_5a"
        # block_dim = 4
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
            in_channels=2 * block_dim,
            output_channels=block_dim,
            kernel_shape=[3, 3, 3],
            padding=0,
            name=end_point,
        )

        end_point = "Upconv3d_6"
        self.end_points[end_point] = nn.Sequential(
            # nn.ConvTranspose3d(4, 2, (4, 6, 6), (2, 4, 4), 1, bias=False),
            nn.ConvTranspose3d(3, 2, (4, 6, 6), (2, 4, 4), 1, bias=False),
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


class SimpleAutoencoderI3D(nn.Module):
    """Flow autoencoder with an I3D encoder."""

    def __init__(self, in_channels: int, latent_dim: int):
        super(SimpleAutoencoderI3D, self).__init__()
        self.encoder = InceptionI3d(in_channels=in_channels)
        self.decoder = SimpleDecoderConv3D(in_channels, latent_dim)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_features = self.encoder.extract_features(x, avg_out=False)
        out = self.decoder(latent_features)

        return latent_features, out

    def extract_features(self, x) -> torch.Tensor:
        latent_features = self.encoder.extract_features(x, avg_out=False)
        return latent_features


class AutoencoderI3D(nn.Module):
    """Flow autoencoder with an I3D encoder."""

    def __init__(self, in_channels: int, latent_dim: int):
        super(AutoencoderI3D, self).__init__()
        self.encoder = ReducedInceptionI3d(in_channels=in_channels)
        self.decoder = DecoderConv3D(in_channels, latent_dim)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode
        residual_features = []
        for end_point in self.encoder.VALID_ENDPOINTS:
            if end_point in self.encoder.end_points:
                if isinstance(self.encoder._modules[end_point], nn.MaxPool3d):
                    residual_features.append(x)
                x = self.encoder._modules[end_point](x)
            print(end_point, x.shape)
        latent_features = x

        # Decode
        up_conv_count = 0
        for end_point in self.decoder.VALID_ENDPOINTS:
            x = self.decoder._modules[end_point](x)
            if (
                isinstance(self.decoder._modules[end_point], nn.Sequential)
                and end_point != "Upconv3d_6"
            ):
                up_conv_count += 1
                x = torch.hstack([x, residual_features[-up_conv_count]])
            print(end_point, x.shape)
        out = self.decoder.tanh(x)

        return latent_features, out

    def extract_features(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        for end_point in self.encoder.VALID_ENDPOINTS:
            if end_point in self.encoder.end_points:
                x = self.encoder._modules[end_point](x)
        return x


def make_flow_encoder(pretrained_path: str) -> nn.Module:
    """Load a standard flow I3D (2 channels) architecture.

    :param pretrained_path: if provided load the model stored at this location.
    :return: configured flow I3D.
    """
    model = InceptionI3d(in_channels=2)

    if pretrained_path:
        pretrained_params = torch.load(pretrained_path)
        model.load_state_dict(pretrained_params)

    return model


def make_flow_autoencoder(pretrained_path: str) -> nn.Module:
    """Load a flow (2 channels) autoencoder (I3D encoder) architecture.

    :param pretrained_path: if provided load the model stored at this location.
    :return: configured flow autoencoder.
    """
    model = SimpleAutoencoderI3D(in_channels=2, latent_dim=1024)

    if pretrained_path:
        pretrained_params = torch.load(pretrained_path)
        model.load_state_dict(pretrained_params)

    return model


if __name__ == "__main__":
    model = AutoencoderI3D(in_channels=2, latent_dim=32)

    x = torch.rand((1, 2, 16, 224, 224))
    model(x)
