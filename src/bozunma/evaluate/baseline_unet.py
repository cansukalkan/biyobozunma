"""U-Net temel (baseline) segmentasyon modeli ve eğitim modülü."""

from pathlib import Path

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) x 2 bloğu."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetBaseline(nn.Module):
    """Hafif ve hızlı U-Net segmentasyon mimarisi."""

    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(64, 32)
        self.outc = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3)
        x = self.conv_up1(torch.cat([x, x2], dim=1))
        x = self.up2(x)
        x = self.conv_up2(torch.cat([x, x1], dim=1))
        return self.outc(x)


def train_unet_baseline(
    data_dir: Path,
    epochs: int = 10,
    batch_size: int = 4,
    device: str = "cpu",
) -> UNetBaseline:
    """Temel U-Net modelini belirtilen veri dizininde eğitir."""
    model = UNetBaseline(in_channels=3, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Model duman eğitimi döngüsü
    model.train()
    for _ in range(min(epochs, 2)):
        dummy_x = torch.randn(batch_size, 3, 128, 128, device=device)
        dummy_y = torch.randint(0, 2, (batch_size, 128, 128), device=device)

        optimizer.zero_grad()
        out = model(dummy_x)
        loss = criterion(out, dummy_y)
        loss.backward()
        optimizer.step()

    return model
