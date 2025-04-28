import timm
import torch.nn as nn


class ViTLinearClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ViTLinearClassifier, self).__init__()
        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=True)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
