import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.squeezenet import squeezenet1_0

class Squeeze(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(Squeeze, self).__init__()

        # Livelli Convoluzionali non si fa uso del classificatore
        features = list(squeezenet1_0(pretrained=True).features)
        
        if device == "cuda":
            self.features = nn.ModuleList(features).cuda().eval()
        else:
            self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        feature_maps = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            print(idx)
            if idx == 1:
                feature_maps.append(x)
        return feature_maps