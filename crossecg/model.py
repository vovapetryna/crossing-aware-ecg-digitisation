"""Pretrained-encoder U-Net with the ordinate/existence heads of DualTraceUNet."""

from __future__ import annotations

import timm
import torch
from torch import nn
from torch.nn import functional


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


class TimmTraceUNet(nn.Module):
    """timm encoder (stride 4..32) + U-Net decoder back to full resolution.

    A shallow full-resolution stem supplies the stride-1/2 skips that ConvNeXt-style
    encoders lack, so the ordinate head keeps sub-pixel row precision.
    """

    def __init__(self, encoder: str = "convnext_tiny.fb_in22k", pretrained: bool = True, decoder_width: int = 64,
                 crossing_head: bool = False, in_chans: int = 4):
        super().__init__()
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True, in_chans=in_chans)
        channels = self.encoder.feature_info.channels()
        reductions = self.encoder.feature_info.reduction()
        self.stem1 = conv_block(in_chans, decoder_width // 2)
        self.stem2 = nn.Sequential(nn.MaxPool2d(2), conv_block(decoder_width // 2, decoder_width))
        widths = [max(decoder_width, c // 2) for c in channels]
        self.lateral = nn.ModuleList(nn.Conv2d(c, w, 1) for c, w in zip(channels, widths))
        self.up_blocks = nn.ModuleList(
            conv_block(widths[i + 1] + widths[i], widths[i]) for i in range(len(channels) - 1)
        )
        self.reductions = reductions
        self.to_half = conv_block(widths[0] + decoder_width, decoder_width)
        self.to_full = conv_block(decoder_width + decoder_width // 2, decoder_width)
        self.ordinate_output = nn.Conv2d(decoder_width, 1, 1)
        self.existence_output = nn.Conv1d(decoder_width, 1, 1)
        self.crossing_output = nn.Conv2d(decoder_width, 2, 1) if crossing_head else None
        extra = in_chans - 3
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406] + [0.0] * extra).view(1, in_chans, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225] + [1.0] * extra).view(1, in_chans, 1, 1))

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        inputs = (inputs - self.mean) / self.std
        full = self.stem1(inputs)
        half = self.stem2(full)
        features = [lateral(f) for lateral, f in zip(self.lateral, self.encoder(inputs))]
        current = features[-1]
        for index in range(len(features) - 2, -1, -1):
            current = functional.interpolate(current, size=features[index].shape[-2:], mode="bilinear", align_corners=False)
            current = self.up_blocks[index](torch.cat((current, features[index]), dim=1))
        current = functional.interpolate(current, size=half.shape[-2:], mode="bilinear", align_corners=False)
        current = self.to_half(torch.cat((current, half), dim=1))
        current = functional.interpolate(current, size=full.shape[-2:], mode="bilinear", align_corners=False)
        current = self.to_full(torch.cat((current, full), dim=1))
        outputs = {
            "ordinate_logits": self.ordinate_output(current),
            "existence_logits": self.existence_output(current.mean(dim=2)),
        }
        if self.crossing_output is not None:
            crossing = self.crossing_output(current)
            outputs["crossing_logits"] = crossing[:, 0]
            outputs["crossing_offset"] = crossing[:, 1].sigmoid()
        return outputs
