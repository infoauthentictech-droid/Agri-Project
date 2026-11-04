"""
CycleGAN implementation for weed-infested scenario synthesis.

CycleGAN (Cycle-Consistent Generative Adversarial Network) enables
image-to-image translation without paired training data. This implementation
converts normal crop images to weed-infested scenarios.

Reference: Zhu et al. "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.data.gpu_loader import GpuBatchLoader, cyclegan_transform  # noqa: E402


class ResidualBlock(nn.Module):
    """Residual block for the generator."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class Generator(nn.Module):
    """Generator network for CycleGAN.

    Uses a U-Net style architecture with residual blocks for image-to-image translation.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, num_features: int = 64):
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(in_channels, num_features, 7, 1, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)

        self.down1 = ResidualBlock(num_features, num_features * 2, stride=2)
        self.down2 = ResidualBlock(num_features * 2, num_features * 4, stride=2)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4, num_features * 4) for _ in range(9)]
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(num_features * 4, num_features * 2, 3, 2, 1, 1, bias=False)
        self.bn_up1 = nn.BatchNorm2d(num_features * 2)
        self.up2 = nn.ConvTranspose2d(num_features * 2, num_features, 3, 2, 1, 1, bias=False)
        self.bn_up2 = nn.BatchNorm2d(num_features)

        self.conv_out = nn.Conv2d(num_features, out_channels, 7, 1, 3, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.down1(x)
        x = self.down2(x)

        # Residual blocks
        x = self.res_blocks(x)

        # Decoder
        x = self.relu(self.bn_up1(self.up1(x)))
        x = self.relu(self.bn_up2(self.up2(x)))
        x = self.tanh(self.conv_out(x))

        return x


class Discriminator(nn.Module):
    """Discriminator network for CycleGAN (PatchGAN).

    Classifies whether image patches are real or fake.
    """

    def __init__(self, in_channels: int = 3, num_features: int = 64):
        super().__init__()

        def discriminator_block(in_ch: int, out_ch: int, stride: int):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, stride, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            discriminator_block(in_channels, num_features, stride=2),
            discriminator_block(num_features, num_features * 2, stride=2),
            discriminator_block(num_features * 2, num_features * 4, stride=2),
            discriminator_block(num_features * 4, num_features * 8, stride=1),
            nn.Conv2d(num_features * 8, 1, 4, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CycleGANModel(nn.Module):
    """Complete CycleGAN model with two generators and two discriminators.

    G: Generator A -> B (normal -> weed-infested)
    F: Generator B -> A (weed-infested -> normal)
    D_A: Discriminator for domain A
    D_B: Discriminator for domain B
    """

    def __init__(self, in_channels: int = 3, num_features: int = 64):
        super().__init__()
        self.G = Generator(in_channels, in_channels, num_features)
        self.F = Generator(in_channels, in_channels, num_features)
        self.D_A = Discriminator(in_channels, num_features)
        self.D_B = Discriminator(in_channels, num_features)

    def forward(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor
    ) -> dict:
        """Forward pass through the CycleGAN model.

        Args:
            real_A: Real images from domain A (normal crops)
            real_B: Real images from domain B (weed-infested)

        Returns:
            Dictionary containing all generated images and discriminator outputs
        """
        # Forward pass
        fake_B = self.G(real_A)  # A -> B
        fake_A = self.F(real_B)  # B -> A

        # Cycle consistency
        rec_A = self.F(fake_B)  # A -> B -> A
        rec_B = self.G(fake_A)  # B -> A -> B

        # Identity mapping
        id_A = self.F(real_A)  # A -> A
        id_B = self.G(real_B)  # B -> B

        # Discriminator outputs
        D_A_real = self.D_A(real_A)
        D_A_fake = self.D_A(fake_A.detach())
        D_B_real = self.D_B(real_B)
        D_B_fake = self.D_B(fake_B.detach())

        return {
            "fake_B": fake_B,
            "fake_A": fake_A,
            "rec_A": rec_A,
            "rec_B": rec_B,
            "id_A": id_A,
            "id_B": id_B,
            "D_A_real": D_A_real,
            "D_A_fake": D_A_fake,
            "D_B_real": D_B_real,
            "D_B_fake": D_B_fake,
        }


class CycleGANTrainer:
    """Trainer for CycleGAN model."""

    def __init__(
        self,
        model: CycleGANModel,
        device: torch.device,
        lr: float = 0.0002,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 5.0
    ):
        # channels_last with bf16 autocast lets cuDNN use tensor cores; plain fp32 ran about 5 times slower
        self.model = model.to(device, memory_format=torch.channels_last)
        self.device = device
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

        # Optimizers
        self.opt_G = torch.optim.Adam(
            list(model.G.parameters()) + list(model.F.parameters()),
            lr=lr, betas=(0.5, 0.999)
        )
        self.opt_D_A = torch.optim.Adam(model.D_A.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_D_B = torch.optim.Adam(model.D_B.parameters(), lr=lr, betas=(0.5, 0.999))

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def train_step(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor
    ) -> dict:
        """Single training step.

        Args:
            real_A: Real images from domain A
            real_B: Real images from domain B

        Returns:
            Dictionary of losses
        """
        real_A = real_A.to(self.device, memory_format=torch.channels_last)
        real_B = real_B.to(self.device, memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self._train_step(real_A, real_B)

    def _train_step(self, real_A: torch.Tensor, real_B: torch.Tensor) -> dict:
        # Forward pass
        outputs = self.model(real_A, real_B)

        # ---- Generator losses ----
        # Adversarial loss
        loss_G_GAN = self.mse_loss(self.model.D_B(outputs["fake_B"]), torch.ones_like(outputs["D_B_real"]))
        loss_F_GAN = self.mse_loss(self.model.D_A(outputs["fake_A"]), torch.ones_like(outputs["D_A_real"]))

        # Cycle consistency loss
        loss_cycle_A = self.l1_loss(outputs["rec_A"], real_A)
        loss_cycle_B = self.l1_loss(outputs["rec_B"], real_B)
        loss_cycle = loss_cycle_A + loss_cycle_B

        # Identity loss
        loss_id_A = self.l1_loss(outputs["id_A"], real_A)
        loss_id_B = self.l1_loss(outputs["id_B"], real_B)
        loss_identity = loss_id_A + loss_id_B

        # Total generator loss
        loss_G = (
            loss_G_GAN + loss_F_GAN +
            self.lambda_cycle * loss_cycle +
            self.lambda_identity * loss_identity
        )

        # Update generators
        self.opt_G.zero_grad()
        loss_G.backward()
        self.opt_G.step()

        # ---- Discriminator losses ----
        # D_A loss
        loss_D_A_real = self.mse_loss(outputs["D_A_real"], torch.ones_like(outputs["D_A_real"]))
        loss_D_A_fake = self.mse_loss(outputs["D_A_fake"], torch.zeros_like(outputs["D_A_fake"]))
        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

        # D_B loss
        loss_D_B_real = self.mse_loss(outputs["D_B_real"], torch.ones_like(outputs["D_B_real"]))
        loss_D_B_fake = self.mse_loss(outputs["D_B_fake"], torch.zeros_like(outputs["D_B_fake"]))
        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

        # Update discriminators
        self.opt_D_A.zero_grad()
        loss_D_A.backward()
        self.opt_D_A.step()

        self.opt_D_B.zero_grad()
        loss_D_B.backward()
        self.opt_D_B.step()

        return {
            "loss_G": loss_G.item(),
            "loss_G_GAN": loss_G_GAN.item(),
            "loss_F_GAN": loss_F_GAN.item(),
            "loss_cycle": loss_cycle.item(),
            "loss_identity": loss_identity.item(),
            "loss_D_A": loss_D_A.item(),
            "loss_D_B": loss_D_B.item(),
        }

    def save_checkpoint(self, path: Path, epoch: int):
        """Save model checkpoint."""
        torch.save({
            "epoch": epoch,
            "G_state_dict": self.model.G.state_dict(),
            "F_state_dict": self.model.F.state_dict(),
            "D_A_state_dict": self.model.D_A.state_dict(),
            "D_B_state_dict": self.model.D_B.state_dict(),
            "opt_G_state_dict": self.opt_G.state_dict(),
            "opt_D_A_state_dict": self.opt_D_A.state_dict(),
            "opt_D_B_state_dict": self.opt_D_B.state_dict(),
        }, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)  # A save cut short by a shutdown must not leave a broken newest checkpoint

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.G.load_state_dict(checkpoint["G_state_dict"])
        self.model.F.load_state_dict(checkpoint["F_state_dict"])
        self.model.D_A.load_state_dict(checkpoint["D_A_state_dict"])
        self.model.D_B.load_state_dict(checkpoint["D_B_state_dict"])
        self.opt_G.load_state_dict(checkpoint["opt_G_state_dict"])
        self.opt_D_A.load_state_dict(checkpoint["opt_D_A_state_dict"])
        self.opt_D_B.load_state_dict(checkpoint["opt_D_B_state_dict"])
        return checkpoint["epoch"]


def main(argv=None):
    """Train CycleGAN for weed-infested scenario synthesis."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain_a", required=True, help="Path to domain A images (normal crops)")
    ap.add_argument("--domain_b", required=True, help="Path to domain B images (weed-infested)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.0002)
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--output", default="experiments/cyclegan")
    ap.add_argument("--device", default="0")
    ap.add_argument("--resume", action="store_true", help="Continue from the newest checkpoint in --output, if there is one")
    args = ap.parse_args(argv)

    # GPU only: never fall back to the CPU
    if not torch.cuda.is_available():
        print("Error: CUDA GPU not available. Training is GPU-only.")
        return 1
    device = torch.device(f"cuda:{args.device}")
    print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")

    # Images are decoded and transformed on the GPU
    transform = cyclegan_transform(args.imgsz)
    dataset_A = ImageFolder(args.domain_a)
    dataset_B = ImageFolder(args.domain_b)

    loader_A = GpuBatchLoader(dataset_A.samples, args.batch, transform, shuffle=True, device=str(device))
    loader_B = GpuBatchLoader(dataset_B.samples, args.batch, transform, shuffle=True, device=str(device))

    print(f"Domain A samples: {len(dataset_A)}")
    print(f"Domain B samples: {len(dataset_B)}")

    # Initialize model
    model = CycleGANModel(in_channels=3, num_features=64)
    trainer = CycleGANTrainer(model, device, lr=args.lr)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    if args.resume:
        saved = sorted(output_dir.glob("checkpoint_epoch_*.pth"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        if saved:
            start_epoch = trainer.load_checkpoint(saved[-1]) + 1
            print(f"Continuing from {saved[-1]} at epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_losses = []

        for i, (real_A, real_B) in enumerate(zip(loader_A, loader_B)):
            real_A = real_A[0]
            real_B = real_B[0]

            losses = trainer.train_step(real_A, real_B)
            epoch_losses.append(losses)

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Batch [{i}/{min(len(loader_A), len(loader_B))}] "
                      f"Loss_G: {losses['loss_G']:.4f} Loss_D_A: {losses['loss_D_A']:.4f} "
                      f"Loss_D_B: {losses['loss_D_B']:.4f}")

        # Average losses for epoch
        avg_losses = {k: sum(d[k] for d in epoch_losses) / len(epoch_losses)
                      for k in epoch_losses[0].keys()}

        print(f"Epoch {epoch} - "
              f"Loss_G: {avg_losses['loss_G']:.4f} "
              f"Loss_cycle: {avg_losses['loss_cycle']:.4f} "
              f"Loss_identity: {avg_losses['loss_identity']:.4f} "
              f"Loss_D_A: {avg_losses['loss_D_A']:.4f} "
              f"Loss_D_B: {avg_losses['loss_D_B']:.4f}")

        # Save every epoch so an interruption loses at most one; keep every 10th as history
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
        trainer.save_checkpoint(checkpoint_path, epoch)
        print(f"Checkpoint saved: {checkpoint_path}")
        previous = output_dir / f"checkpoint_epoch_{epoch - 1}.pth"
        if (epoch - 1) % 10 and previous.exists():
            previous.unlink()

    # Save final model
    final_path = output_dir / "final_model.pth"
    trainer.save_checkpoint(final_path, args.epochs)
    print(f"Final model saved: {final_path}")

    return 0


if __name__ == "__main__":
    import os
    import traceback
    try:
        code = main()
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # After a GPU error a normal exit can hang forever and stall the training queue
