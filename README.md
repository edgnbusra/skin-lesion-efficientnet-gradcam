# Skin Lesion Classification - EfficientNet-B3 + Grad-CAM

## Goal

This project had two main objectives:
1. **EfficientNet-B3 fine-tuning**: adapt a pretrained EfficientNet-B3 model to classify skin lesion images via transfer learning, writing the training loop and early stopping logic from scratch.
2. **Explainability with Grad-CAM**: visualize which region of an image the model focuses on when making a classification decision.

Dataset: [HAM10000 (Skin Cancer MNIST)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) - 10,015 dermatoscopic skin lesion images across 7 classes:

| Code | Meaning |
|---|---|
| akiec | Actinic keratosis |
| bcc | Basal cell carcinoma |
| bkl | Benign keratosis |
| df | Dermatofibroma |
| mel | Melanoma (malignant, most dangerous) |
| nv | Melanocytic nevus (normal mole) |
| vasc | Vascular lesion |

## Hardware decision

The local machine (Intel i5-13420H, 8GB RAM, no dedicated GPU - integrated Intel UHD Graphics only) was found insufficient for EfficientNet-B3 fine-tuning. So:
- **Code writing and Grad-CAM/evaluation** (CPU-friendly work) was done locally.
- **Training** (the GPU-heavy part) was run on Google Colab's free T4 GPU.

## What was done

1. **Data preparation** (`src/dataset.py`, `src/transforms.py`, `src/split_data.py`)
   - HAM10000 metadata loaded into a PyTorch `Dataset` class.
   - Train/val/test split done at the `lesion_id` level (using `GroupShuffleSplit`) to prevent data leakage from different photos of the same lesion ending up in different splits.
   - Result: 7002 train, 1519 val, 1494 test images.
   - Augmentation (random crop, flip, rotation, color jitter) applied to the training set.

2. **Model** (`src/model.py`)
   - `torchvision.models.efficientnet_b3` loaded with ImageNet weights.
   - Backbone frozen, only the final classification layer redefined and trained for the 7 classes.

3. **Class imbalance detection and mitigation**
   - Found a ~53x gap between the `nv` class (4718 samples) and `df` class (89 samples).
   - Added class weights via `compute_class_weight(class_weight="balanced")` to `CrossEntropyLoss`.

4. **Training loop + Early Stopping** (`src/train.py`)
   - Manual training loop: forward pass -> loss -> backward pass -> optimizer.step().
   - Early stopping: training halts if validation loss doesn't improve for 5 consecutive epochs; the best model (`best_model.pth`) is saved separately.
   - Run on Colab (`notebooks/train_colab.ipynb`) with a T4 GPU - **early stopping kicked in at epoch 15**.

5. **Grad-CAM** (`src/grad_cam.py`, `src/run_grad_cam.py`)
   - Implemented from scratch (no third-party library) using PyTorch forward/backward hooks.
   - Activations and gradients from the last convolutional layer are combined via Global Average Pooling to compute channel importance and produce a heatmap.
   - Heatmaps for sample images from every class saved to `outputs/grad_cam/`.

6. **Evaluation** (`src/evaluate.py`)
   - Confusion matrix and per-class precision/recall/F1 report generated on the test set (1494 images).

## Results

**Overall test accuracy: 67.9%**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| akiec | 0.433 | 0.619 | 0.510 |
| bcc | 0.464 | 0.574 | 0.513 |
| bkl | 0.473 | 0.645 | 0.546 |
| df | 0.091 | 0.571 | 0.157 |
| mel | 0.367 | 0.471 | 0.412 |
| nv | 0.937 | 0.734 | 0.823 |
| vasc | 0.306 | 0.714 | 0.429 |

Details: `outputs/classification_report.txt`, `outputs/confusion_matrix.png`

**Most critical finding:** the `mel` (melanoma, cancer) class recall is only 47.1% - more than half of the actual melanoma cases are missed, 37 of them misclassified as "normal mole" (`nv`). In a medical application this is one of the most dangerous error types, and shows the model in its current form (only the final layer trained, a single fine-tuning pass) is not production-ready.

**Grad-CAM observation:** on images with a clear lesion boundary, the heatmap focuses sharply on the lesion itself (the model looks at the right region) - this holds even on misclassified examples, meaning the error isn't "random" but comes from looking at the right place and misjudging it.

## Possible improvements (not done, notes for later)

- Unfreeze the last backbone blocks and run a second fine-tuning pass with a low learning rate.
- Try more epochs / different learning rates.
- Oversampling or additional data collection for minority classes (df, vasc).

## Project structure

```
src/
  dataset.py       - PyTorch Dataset class
  transforms.py    - Augmentation and normalization
  split_data.py    - Train/val/test split (lesion_id-based)
  model.py         - EfficientNet-B3 model definition
  train.py         - Training loop + early stopping (for local testing/Colab)
  grad_cam.py      - Grad-CAM implementation (hook-based)
  run_grad_cam.py  - Script to generate Grad-CAM visualizations
  evaluate.py      - Test set evaluation (confusion matrix, classification report)
notebooks/
  train_colab.ipynb - Notebook for GPU training on Colab
outputs/
  best_model.pth           - Trained model weights
  grad_cam/                - Grad-CAM visualizations
  confusion_matrix.png
  classification_report.txt
```

## Note

The HAM10000 dataset (~5GB) in the `data/` folder is not included in this repo (excluded via `.gitignore`). To download it again:
```
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
```
