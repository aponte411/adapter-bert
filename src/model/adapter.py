from torch import nn
import torch.nn.functional as F

from config import *

class Adapter(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.ff_down_proj = nn.Linear(in_features, ADAPTER_BOTTLENECK)
        self.ff_up_proj = nn.Linear(ADAPTER_BOTTLENECK, in_features)

    def forward(self, X):
        X = X + self.ff_up_proj(F.relu(self.ff_down_proj(X)))
        return X
