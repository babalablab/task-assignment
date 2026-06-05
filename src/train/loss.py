import torch
import torch.nn as nn


# class LossCommonConfusion:
#     def __init__(self, alpha: float, beta: float) -> None:
#         self.criterion = nn.CrossEntropyLoss()
#         self.alpha = alpha
#         self.beta = beta
#         self.name = "common_confusion"
#
#     def loss_common_confusion(
#         self,
#         confusion_pred: torch.Tensor,
#         annotations: torch.Tensor,
#         common_confusion: torch.Tensor,
#         agent_confusion: torch.Tensor,
#     ) -> torch.Tensor:
#         loss = confusion_pred.gather(2, annotations.unsqueeze(-1))
#         loss = -torch.log(loss + 1e-4)
#         loss = loss.mean()
#         # common_confusion = common_confusion.expand(agent_confusion.shape[0], -1, -1)
#         constraint = 0
#         for i in range(agent_confusion.shape[0]):
#             constraint -= torch.norm(common_confusion - agent_confusion[i])
#         # constraint = -torch.norm(common_confusion - agent_confusion, dim=1).sum()
#         return loss + self.alpha * constraint
#
#     def loss_classification(
#         self, cls_out: torch.Tensor, ground_truth: torch.Tensor
#     ) -> torch.Tensor:
#         return self.criterion(cls_out, ground_truth.squeeze())
#
#     def __call__(
#         self,
#         cls_out: torch.Tensor,
#         ground_truth: torch.Tensor,
#         confusion_out: torch.Tensor,
#         annotations: torch.Tensor,
#         **kwargs,
#     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         model = kwargs["model"]
#         common_confusion = model.global_confusion_matrix
#         agent_confusion = model.agent_confusion_matrix
#         classification_loss = torch.tensor(0)
#
#         if cls_out is not None:
#             classification_loss = self.loss_classification(cls_out, ground_truth)
#         loss = 0
#         confusion_loss = self.loss_common_confusion(
#             confusion_out, annotations, common_confusion, agent_confusion
#         )
#         loss = confusion_loss + self.beta * classification_loss
#         return {
#             "loss": loss,
#             "confusion_loss": confusion_loss,
#             "cls_loss": classification_loss,
#         }


# class LossConfusionWithoutCL:
#     def __init__(self, alpha: float, beta: float) -> None:
#         self.beta = beta
#         self.alpha = alpha
#         self.name = "confusion_ablation"
#
#     def loss_confusion(
#         self, confusion_pred: torch.Tensor, annotations: torch.Tensor
#     ) -> torch.Tensor:
#         # annotations = annotations[annotations != -1]
#         i, j = torch.where(annotations != -1)
#         # annotations = annotations[idx]
#         confusion_pred = confusion_pred[i]
#         annotations = annotations[i]
#         loss = torch.tensor([])
#         for i in range(annotations.shape[0]):
#             annot_dist = confusion_pred[i]
#             annotation = annotations[i]
#
#             for j, ai in enumerate(annotation):
#                 if ai == -1:
#                     continue
#
#                 value = annot_dist[j][ai].unsqueeze(-1)
#                 loss = (
#                     torch.cat((loss, value)) if loss.shape != torch.Size([0]) else value
#                 )
#         # loss = torch.gather(confusion_pred, 2, annotations.unsqueeze(-1))
#         loss = -torch.log(loss + 1e-4).mean()
#         return loss
#
#     def __call__(
#         self,
#         cls_out: torch.Tensor,
#         ground_truth: torch.Tensor,
#         confusion_out: torch.Tensor,
#         annotations: torch.Tensor,
#         **kwargs,
#     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         loss = 0
#         confusion_loss = self.loss_confusion(confusion_out, annotations)
#
#         return {
#             "loss": confusion_loss,
#             "confusion_loss": confusion_loss,
#             "cls_loss": torch.tensor([0.0]),
#         }


class LossConfusion:
    def __init__(self, alpha: float, beta: float) -> None:
        self.criterion = nn.CrossEntropyLoss()
        self.beta = beta
        self.alpha = alpha
        self.name = "confusion"

    def loss_confusion(
        self, confusion_pred: torch.Tensor, annotations: torch.Tensor
    ) -> torch.Tensor:
        # annotations = annotations[annotations != -1]
        i, j = torch.where(annotations != -1)
        # annotations = annotations[idx]
        confusion_pred = confusion_pred[i]
        annotations = annotations[i]
        loss = torch.tensor([])
        for i in range(annotations.shape[0]):
            annot_dist = confusion_pred[i]
            annotation = annotations[i]

            for j, ai in enumerate(annotation):
                if ai == -1:
                    continue

                value = annot_dist[j][ai].unsqueeze(-1)
                loss = (
                    torch.cat((loss, value)) if loss.shape != torch.Size([0]) else value
                )
        # loss = torch.gather(confusion_pred, 2, annotations.unsqueeze(-1))
        loss = -torch.log(loss + 1e-4).mean()
        return loss

    def loss_classification(self, cls_out: torch.Tensor, ground_truth: torch.Tensor):
        return self.criterion(cls_out, ground_truth)

    def __call__(
        self,
        cls_out: torch.Tensor,
        ground_truth: torch.Tensor,
        confusion_out: torch.Tensor,
        annotations: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        classification_loss = torch.tensor(0)
        if cls_out is not None:
            classification_loss = self.loss_classification(cls_out, ground_truth)
        loss = 0

        confusion_loss = self.loss_confusion(confusion_out, annotations)

        loss = confusion_loss + self.beta * classification_loss
        return {
            "loss": loss,
            "confusion_loss": confusion_loss,
            "cls_loss": classification_loss,
        }


# class LossConfusionDiag:
#     def __init__(self, alpha: float, beta: float) -> None:
#         self.criterion = nn.CrossEntropyLoss()
#         self.alpha = alpha
#         self.beta = beta
#         self.name = "confusion_diag_const"
#
#     def loss_confusion(
#         self,
#         confusion_out: torch.Tensor,
#         annotations: torch.Tensor,
#         confusion_matrices: torch.Tensor,
#     ) -> torch.Tensor:
#         loss = torch.gather(confusion_out, 2, annotations.unsqueeze(-1))
#         const = 0
#         loss = -torch.log(loss).mean()
#
#         for confusion in confusion_matrices:
#             const += confusion.diag().sum()
#
#         return loss + self.alpha * const
#
#     def loss_classification(self, cls_out: torch.Tensor, ground_truth: torch.Tensor):
#         return self.criterion(cls_out, ground_truth.squeeze())
#
#     def __call__(
#         self,
#         cls_out: torch.Tensor,
#         ground_truth: torch.Tensor,
#         confusion_out: torch.Tensor,
#         annotations: torch.Tensor,
#         **kwargs,
#     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         model = kwargs["model"]
#         confusion_matrices = model.confusion_matrices.confusion_matrices
#         classification_loss = torch.tensor(0)
#
#         if cls_out is not None:
#             classification_loss = self.loss_classification(cls_out, ground_truth)
#         loss = 0
#         confusion_loss = self.loss_confusion(
#             confusion_out, annotations, confusion_matrices
#         )
#         loss = confusion_loss + self.beta * classification_loss
#         return {
#             "loss": loss,
#             "confusion_loss": confusion_loss,
#             "cls_loss": classification_loss,
#         }


def _device():
    device = ""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


class LossLearningToDefer:
    def __init__(self, *args, **kwargs) -> None:
        self.name = "learning_to_defer"
        self.alpha = kwargs["alpha"]
        self.beta = kwargs["beta"]
        self.weight = torch.tensor(kwargs["weight"]).to(_device())
        self.softmax = nn.Softmax(dim=1)

    def __call__(
        self,
        cls_out: torch.Tensor,
        ground_truth: torch.Tensor,
        annotations: torch.Tensor,
        **kwargs,
    ):
        if len(ground_truth.shape) != 2:
            ground_truth = ground_truth.unsqueeze(-1).expand(-1, annotations.shape[1])
        target = self.weight * (ground_truth == annotations).to(torch.long)

        loss = -target * torch.log(self.softmax(cls_out))
        loss = loss.sum() / torch.sum(cls_out * (annotations != -1).to(torch.long))

        return {
            "loss": loss,
            "confusion_loss": None,
            "cls_loss": None,
        }
