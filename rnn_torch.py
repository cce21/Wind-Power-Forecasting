import torch
import torch.nn.functional as F
import torch.nn as nn

class BaselineRNNModel(nn.Module):
    """
    Desc:
        RNN Model
    """
    def __init__(self, settings):
        # type: (dict) -> None
        """
        Desc:
            __init__
        Args:
            settings: a dict of parameters
        """
        super(BaselineRNNModel, self).__init__()
        
        self.input_size = 4
        self.hidden_size = 12
        self.num_layers = 4

        self.in_features = int(settings["input_len"] * self.hidden_size)
        self.out_features = int(settings["output_len"] * settings["out_var"])

        self.rnn = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.projection = nn.Linear(in_features=self.in_features, out_features=self.out_features)

    def forward(self, X):
        # type: (torch.tensor) -> torch.tensor
        """
        Desc:
            Forward pass for RNN
        Args:
            X encoding:
        Returns:
            A tensor
        """
        device = X.device
        hidden = torch.normal(mean=0, std=1, size=(self.num_layers, X.size(0), self.hidden_size)).to(device)
        X, _ = self.rnn(X, hidden)
        X = torch.flatten(X, start_dim=1, end_dim=2)
        output = self.projection(X)

        return output 