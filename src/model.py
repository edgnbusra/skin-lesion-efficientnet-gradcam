import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


def build_model(num_classes, freeze_backbone=True):
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1
    model = efficientnet_b3(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
