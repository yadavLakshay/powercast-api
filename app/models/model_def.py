import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=128, horizon=6, target_dim=3,
                 num_layers=2, dropout=0.2, bidirectional=True):
        super().__init__()
        self.horizon = horizon
        self.target_dim = target_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
            bidirectional=bidirectional
        )
        gru_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, horizon * target_dim),
        )

    def forward(self, x):
        _, hn = self.gru(x)
        if self.bidirectional:
            out = torch.cat((hn[-2], hn[-1]), dim=-1)
        else:
            out = hn[-1]
        out = self.head(out)
        return out.view(-1, self.horizon, self.target_dim)