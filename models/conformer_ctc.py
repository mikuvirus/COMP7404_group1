import torch.nn as nn
from models.conformer import Conformer

class ConformerCTCModel(nn.Module):
    def __init__(self, input_dim, num_classes, encoder_dim=144, depth=12, **kwargs):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, encoder_dim)
        self.encoder = Conformer(dim=encoder_dim, depth=depth, **kwargs)
        self.output_linear = nn.Linear(encoder_dim, num_classes)

    def forward(self, x):
        # x: [batch, T, input_dim]
        x = self.input_linear(x)
        x = self.encoder(x)
        x = self.output_linear(x)
        return x
