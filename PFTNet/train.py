import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from models.pftnet import build_pftnet_s, build_pftnet_l


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        def gradient(x):
            dx = x[:, :, :, 1:] - x[:, :, :, :-1]
            dy = x[:, :, 1:, :] - x[:, :, :-1, :]
            return dx, dy

        pred_dx, pred_dy = gradient(pred)
        gt_dx, gt_dy = gradient(target)

        loss = (
            torch.abs(pred_dx - gt_dx).mean()
            + torch.abs(pred_dy - gt_dy).mean()
        )
        return loss


class DehazeDataset(torch.utils.data.Dataset):
    def __init__(self, hazy_dir, clear_dir, transform=None):
        self.hazy_paths = sorted(Path(hazy_dir).glob("*.png"))
        self.clear_paths = sorted(Path(clear_dir).glob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.hazy_paths)

    def __getitem__(self, idx):
        hazy = transforms.functional.to_tensor(
            transforms.functional.pil_to_tensor(
                transforms.functional.pil_loader(self.hazy_paths[idx])
            )
        )
        clear = transforms.functional.to_tensor(
            transforms.functional.pil_to_tensor(
                transforms.functional.pil_loader(self.clear_paths[idx])
            )
        )
        return hazy, clear


def build_model(name: str):
    if name.lower() == "pftnet-s":
        return build_pftnet_s()
    elif name.lower() == "pftnet-l":
        return build_pftnet_l()
    else:
        raise ValueError("Unknown model name")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="pftnet-l")
    parser.add_argument("--train_hazy", type=str, required=True)
    parser.add_argument("--train_clear", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda_grad", type=float, default=0.2)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.model).to(device)

    dataset = DehazeDataset(
        args.train_hazy,
        args.train_clear,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    pixel_loss = CharbonnierLoss().to(device)
    grad_loss = GradientLoss().to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for hazy, clear in loader:
            hazy = hazy.to(device)
            clear = clear.to(device)

            pred = model(hazy)

            loss_char = pixel_loss(pred, clear)
            loss_grad = grad_loss(pred, clear)
            loss = loss_char + args.lambda_grad * loss_grad

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Loss: {total_loss / len(loader):.6f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(args.save_dir, f"{args.model}_epoch_{epoch+1}.pth"),
        )


if __name__ == "__main__":
    main()
