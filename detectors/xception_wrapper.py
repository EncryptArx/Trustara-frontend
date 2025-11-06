from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import timm


class XceptionDetector(nn.Module):
    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        # xception41t has good trade-off; fall back to xception if needed
        self.backbone = timm.create_model("xception41", pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features if hasattr(self.backbone, "num_features") else 2048
        self.head = nn.Linear(in_features, num_classes)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.head(feats)
        probs = self.activation(logits)
        return probs.squeeze(-1)


def load_xception(weights_path: Optional[str] = None) -> XceptionDetector:
    model = XceptionDetector(pretrained=True)
    if weights_path and Path(weights_path).exists():
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Strict=False to allow head shape mismatch if weights differ
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


