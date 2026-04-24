from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def _norm_layer(channels: int, batchnorm: bool) -> nn.Module:
    if batchnorm:
        return nn.BatchNorm2d(channels)
    return nn.Identity()


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batchnorm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        bias = not batchnorm
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.norm1 = _norm_layer(out_channels, batchnorm)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.norm2 = _norm_layer(out_channels, batchnorm)

        if in_channels == out_channels:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
                _norm_layer(out_channels, batchnorm),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.projection(inputs)

        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)

        return self.activation(x + residual)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batchnorm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.block = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            batchnorm=batchnorm,
            dropout=dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(inputs))


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        batchnorm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ResidualBlock(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            batchnorm=batchnorm,
            dropout=dropout,
        )

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(inputs)
        if x.shape[-2:] != skip.shape[-2:]:
            diff_y = skip.size(-2) - x.size(-2)
            diff_x = skip.size(-1) - x.size(-1)
            x = F.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - (diff_x // 2),
                    diff_y // 2,
                    diff_y - (diff_y // 2),
                ],
            )
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class ResUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 48,
        depth: int = 4,
        growth_factor: int = 2,
        dropout: float = 0.2,
        batchnorm: bool = True,
    ) -> None:
        super().__init__()

        encoder_channels = [base_channels]
        for _ in range(depth):
            encoder_channels.append(encoder_channels[-1] * growth_factor)

        self.stem = ResidualBlock(
            in_channels=in_channels,
            out_channels=encoder_channels[0],
            batchnorm=batchnorm,
            dropout=dropout,
        )
        self.down_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    in_channels=encoder_channels[index],
                    out_channels=encoder_channels[index + 1],
                    batchnorm=batchnorm,
                    dropout=dropout,
                )
                for index in range(depth)
            ]
        )
        self.bottleneck = ResidualBlock(
            in_channels=encoder_channels[-1],
            out_channels=encoder_channels[-1],
            batchnorm=batchnorm,
            dropout=dropout,
        )

        decoder_in_channels = list(reversed(encoder_channels[1:]))
        decoder_skip_channels = list(reversed(encoder_channels[:-1]))
        self.up_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    in_channels=decoder_in_channels[index],
                    skip_channels=decoder_skip_channels[index],
                    out_channels=decoder_skip_channels[index],
                    batchnorm=batchnorm,
                    dropout=dropout,
                )
                for index in range(depth)
            ]
        )
        self.head = nn.Conv2d(encoder_channels[0], out_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []

        x = self.stem(inputs)
        skips.append(x)
        for down_block in self.down_blocks:
            x = down_block(x)
            skips.append(x)

        x = self.bottleneck(x)
        for up_block, skip in zip(self.up_blocks, reversed(skips[:-1])):
            x = up_block(x, skip)

        return self.head(x)


def _coerce_model_config(model_config: ModelConfig | Mapping[str, object]) -> ModelConfig:
    if isinstance(model_config, ModelConfig):
        config = model_config
    else:
        raw_config = dict(model_config)
        if "architecture" not in raw_config:
            raise ValueError(
                "Checkpoint model_config is missing 'architecture'. "
                "This checkpoint predates the ResUNet upgrade and is not compatible with the current model code."
            )
        config = ModelConfig(**raw_config)
    config.validate()
    return config


def build_model(model_config: ModelConfig | Mapping[str, object]) -> nn.Module:
    config = _coerce_model_config(model_config)
    if config.architecture != "resunet":
        raise ValueError(f"Unsupported model architecture: {config.architecture}")
    return ResUNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_channels=config.base_channels,
        depth=config.depth,
        growth_factor=config.growth_factor,
        dropout=config.dropout,
        batchnorm=config.batchnorm,
    )
