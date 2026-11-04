from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def build_model(architecture, num_classes, pretrained):
    if architecture != 'efficientnet_b0':
        raise ValueError(f'Unsupported backbone: {architecture}')
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def set_stage(model, stage):
    for parameter in model.features.parameters():
        parameter.requires_grad = stage == 'finetune'
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def set_train_mode(model, stage):
    model.train()
    if stage == 'head':
        model.features.eval()  # Keep pretrained batch-normalization statistics fixed.
