import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dataset import CLASS_NAMES, HAM10000Dataset, load_metadata
from grad_cam import GradCAM, overlay_heatmap
from model import build_model
from transforms import get_eval_transforms, IMAGENET_MEAN, IMAGENET_STD


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def main(data_dir, image_dirs, model_path, output_dir, num_samples_per_class=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=len(CLASS_NAMES), freeze_backbone=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    target_layer = model.features[-1]
    cam_generator = GradCAM(model, target_layer)

    test_df = load_metadata(os.path.join(data_dir, "test.csv"))
    transform = get_eval_transforms()
    dataset = HAM10000Dataset(test_df, image_dirs, transform=transform)

    os.makedirs(output_dir, exist_ok=True)

    samples_per_class = {}
    for idx in range(len(dataset)):
        label = test_df.iloc[idx]["dx"]
        samples_per_class.setdefault(label, []).append(idx)

    for class_name, indices in samples_per_class.items():
        for i, idx in enumerate(indices[:num_samples_per_class]):
            image_tensor, label = dataset[idx]
            input_tensor = image_tensor.unsqueeze(0).to(device)

            cam, predicted_idx = cam_generator.generate(input_tensor)
            predicted_class = CLASS_NAMES[predicted_idx]

            original_img = denormalize(image_tensor.cpu())
            overlay = overlay_heatmap(original_img, cam)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(original_img)
            axes[0].set_title(f"Gercek: {class_name}")
            axes[0].axis("off")

            axes[1].imshow(overlay)
            axes[1].set_title(f"Tahmin: {predicted_class}")
            axes[1].axis("off")

            plt.tight_layout()
            out_path = os.path.join(output_dir, f"gradcam_{class_name}_{i}.png")
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--image_dirs", nargs="+", default=[
        "../data/HAM10000_images_part_1",
        "../data/HAM10000_images_part_2",
    ])
    parser.add_argument("--model_path", default="../outputs/best_model.pth")
    parser.add_argument("--output_dir", default="../outputs/grad_cam")
    parser.add_argument("--num_samples_per_class", type=int, default=2)
    args = parser.parse_args()

    main(args.data_dir, args.image_dirs, args.model_path, args.output_dir, args.num_samples_per_class)
