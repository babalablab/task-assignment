import numpy as np
import torch
import torch.nn as nn
from model.linear_model import LinearModel
from copy import deepcopy


class ConfusionLayer(nn.Module):
    def __identity_init(self, shape):
        out = np.ones(shape) * 0
        if len(shape) == 3:
            for r in range(shape[0]):
                for i in range(shape[1]):
                    out[r, i, i] = 2
        elif len(shape) == 2:
            for i in range(shape[1]):
                out[i, i] = 2
        return torch.Tensor(out)

    def __init__(self, annotator_num: int, out_dim: int) -> None:
        super().__init__()
        # システム毎の混同を推定するモデル
        self.confusion_matrices = nn.Parameter(
            self.__identity_init((annotator_num, out_dim, out_dim)),
            requires_grad=True,
        )

    def forward(self, x):
        annotator_confusion = torch.einsum("ik,jkl->ijl", (x, self.confusion_matrices))
        return annotator_confusion


class ConfusionModel(nn.Module):
    @torch.no_grad()
    def init_weights_for_test(self, m):
        if type(m) == nn.Linear:
            m.weight.fill_(1)
            m.bias.fill_(0)

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        hidden_dim: int,
        annotator_num: int,
        layer_num: int,
        mode: str = "train",
    ):
        super(ConfusionModel, self).__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        # 正解を予測するモデル
        self.classifier = LinearModel(input_dim, hidden_dim, out_dim, layer_num)
        self.confusion_matrices = ConfusionLayer(annotator_num, out_dim)

        self.softmax = nn.Softmax(dim=2)
        if mode == "test":
            self.classifier.apply(self.init_weights_for_test)

    def init_device(self, device: torch.device):
        pass

    def forward(self, **kwargs) -> torch.Tensor:
        if "cls_out" not in kwargs:
            cls_out = self.classifier(**kwargs)
            cls_logit = cls_out["cls_logit"]
            cls_prob = cls_out["cls_prob"]
        else:
            cls_prob = kwargs["cls_out"]
            cls_logit = kwargs["cls_out"]

        annotator_confusion = self.softmax(self.confusion_matrices(cls_prob))
        return {
            "confusion_out": annotator_confusion,
            "cls_prob": cls_prob,
            "cls_logit": cls_logit,
            "weight": 0,
        }

    def predicted_accuracy(self, **kwargs) -> np.ndarray:
        if "cls_out" in kwargs:
            cls_out = kwargs["cls_out"]
        else:
            cls_out = self.classifier(**kwargs)["cls_prob"]
        confusion_matrices = deepcopy(self.confusion_matrices.confusion_matrices)
        annotator_correct = torch.diagonal(confusion_matrices, dim1=1, dim2=2).repeat(
            cls_out.shape[0], 1, 1
        )
        annotator_correct_pred = torch.einsum(
            "ijk,ik->ij", (annotator_correct, cls_out)
        )

        return annotator_correct_pred.cpu().detach().numpy()
