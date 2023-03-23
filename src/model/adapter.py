from torch import nn
import torch.nn.functional as F
# TODO: switch to yaml config
from config import *

class AdapterModule(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.ff_down_proj = nn.Linear(in_features, ADAPTER_BOTTLENECK)
        self.ff_up_proj = nn.Linear(ADAPTER_BOTTLENECK, in_features)

    def forward(self, X):
        # Skip connection
        X = X + self.ff_up_proj(F.relu(self.ff_down_proj(X)))
        return X
