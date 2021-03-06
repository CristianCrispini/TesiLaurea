import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg19

class VGG19(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(VGG19, self).__init__()

        # Livelli Convoluzionali non si fa uso del classificatore
        features = list(vgg19(pretrained=True).features)
        
        if device == "cuda":
            self.features = nn.ModuleList(features).cuda().eval()
        else:
            self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        feature_maps = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            #print('indice del livello: ' + str(idx) )
            if idx == 1:
                #Sceglie la mappa estratta al livello idx-esimo
                feature_maps.append(x)
        return feature_maps
