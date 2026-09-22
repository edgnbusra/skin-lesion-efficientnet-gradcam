import argparse
import os

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, HAM10000Dataset, load_metadata
from model import build_model
from transforms import get_eval_transforms


def evaluate(data_dir, image_dirs, model_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=len(CLASS_NAMES), freeze_backbone=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    test_df = load_metadata(os.path.join(data_dir, "test.csv"))
    dataset = HAM10000Dataset(test_df, image_dirs, transform=get_eval_transforms())
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=3)
    print(report)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    print(f"Saved report and confusion matrix to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--image_dirs", nargs="+", default=[
        "../data/HAM10000_images_part_1",
        "../data/HAM10000_images_part_2",
    ])
    parser.add_argument("--model_path", default="../outputs/best_model.pth")
    parser.add_argument("--output_dir", default="../outputs")
    args = parser.parse_args()

    evaluate(args.data_dir, args.image_dirs, args.model_path, args.output_dir)
