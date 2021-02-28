from typing import Any, Dict
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


CONV_DIM = 64
FC_DIM = 128
IMAGE_SIZE = 28
N_BLOCKS = 2
BATCH_NORM = True

class ResBlock(nn.Module):

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, batch_norm: bool) -> None:
        super().__init__()

        if batch_norm:
            self.convs = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=2, padding=kernel_size//2),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
                nn.BatchNorm2d(output_channels)
            )
        
        else: 
            self.convs = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=2, padding=kernel_size//2),
                nn.ReLU(),
                nn.Conv2d(output_channels, output_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            )

        self.pool = nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            of dimensions (B, C, H, W)

        Returns
        -------
        torch.Tensor
            of dimensions (B, C, H, W)
        """
        return F.relu(self.pool(x) + self.convs(x))


class CNN(nn.Module):
    """Simple CNN for recognizing characters in a square image."""

    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        input_dims = data_config["input_dims"]
        num_classes = len(data_config["mapping"])

        conv_dim = self.args.get("conv_dim", CONV_DIM)
        fc_dim = self.args.get("fc_dim", FC_DIM)
        n_blocks = self.args.get("n_blocks", N_BLOCKS)
        batch_norm = self.args.get("batch_norm", BATCH_NORM)

        for i in range(n_blocks):
            if i == 0:
                layers = [ResBlock(input_dims[0], conv_dim, 3, batch_norm)] 
            else:
                layers.append(ResBlock(conv_dim, conv_dim, 3, batch_norm))

        self.res_blocks = nn.Sequential(*layers)

      #  self.conv1 = ResBlock(input_dims[0], conv_dim, 3)
      #  self.conv2 = ResBlock(conv_dim, conv_dim, 3)
        self.dropout = nn.Dropout(0.25)
      #  self.max_pool = nn.MaxPool2d(2)

        # Because our 3x3 convs have padding size 1, they leave the input size unchanged.
        # The 2x2 max-pool divides the input size by 2. Flattening squares it.
        conv_output_size = IMAGE_SIZE // (2*n_blocks)
        fc_input_dim = int(conv_output_size * conv_output_size * conv_dim)
        self.fc1 = nn.Linear(fc_input_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x
            (B, C, H, W) tensor, where H and W must equal IMAGE_SIZE

        Returns
        -------
        torch.Tensor
            (B, C) tensor
        """
        B_, C_, H, W = x.shape
        assert H == W == IMAGE_SIZE
        x = self.res_blocks(x)
     #   x = self.conv1(x)
     #   x = self.conv2(x)
     #   x = self.max_pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--conv_dim", type=int, default=CONV_DIM)
        parser.add_argument("--fc_dim", type=int, default=FC_DIM)
        parser.add_argument("--n_blocks", type=int, default=N_BLOCKS)
        parser.add_argument("--batch_norm", type=bool, default=BATCH_NORM)
        return parser
