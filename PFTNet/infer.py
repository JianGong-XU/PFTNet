import argparse
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from models.pftnet import build_pftnet_s, build_pftnet_l


def build_model(name: str):
    if name.lower() == "pftnet-s":
        return build_pftnet_s()
    elif name.lower() == "pftnet-l":
        return build_pftnet_l()
    else:
        raise ValueError("Unknown model name")


def load_image(path: Path):
    img = Image.open(path).convert("RGB")
    return transforms.ToTensor()(img).unsqueeze(0)


def save_image(tensor, path: Path):
    tensor = tensor.clamp(0.0, 1.0)
    img = transforms.ToPILImage()(tensor.squeeze(0).cpu())
    img.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="pftnet-l")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.model).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for img_path in input_dir.glob("*.*"):
            img = load_image(img_path).to(device)
            pred = model(img)
            save_image(pred, output_dir / img_path.name)


if __name__ == "__main__":
    main()
