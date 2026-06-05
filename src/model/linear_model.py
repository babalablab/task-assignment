import torch.nn as nn
import torch
import numpy as np


class LinearModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        layer_num: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        model = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(layer_num - 2):
            model.append(nn.Linear(hidden_dim, hidden_dim))
            model.append(nn.ReLU())
        model.append(nn.Linear(hidden_dim, out_dim))

        self.model = nn.Sequential(*model)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, **kwargs):
        x = kwargs["embedding"]
        logit = self.model(x)
        prob = self.softmax(logit)
        return {
            "confusion_out": None,
            "cls_prob": prob,
            "weight": None,
            "cls_logit": logit,
        }

    def predicted_accuracy(self, **kwargs) -> np.ndarray:
        out = self(**kwargs)
        prob = out["cls_prob"]
        return prob.detach().cpu().numpy()

    def init_device(self, *args, **kwargs):
        pass
