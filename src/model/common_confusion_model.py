# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from copy import deepcopy
# from model.linear_model import LinearModel
# 
# 
# class CommonConfusionModel(nn.Module):
#     def __identity_init(self, shape):
#         out = np.ones(shape) * 0
#         if len(shape) == 3:
#             for r in range(shape[0]):
#                 for i in range(shape[1]):
#                     out[r, i, i] = 2
#         elif len(shape) == 2:
#             for i in range(shape[1]):
#                 out[i, i] = 2
#         return torch.Tensor(out)
# 
#     @torch.no_grad()
#     def init_weights_for_test(self, m):
#         if type(m) == nn.Linear:
#             m.weight.fill_(1)
#             m.bias.fill_(0)
# 
#     def __init__(
#         self,
#         input_dim: int,
#         out_dim: int,
#         hidden_dim: int,
#         annotator_num: int,
#         layer_num: int,
#         mode: str = "train",
#     ):
#         super(CommonConfusionModel, self).__init__()
#         self.mode = mode
#         self.hidden_dim = hidden_dim
#         # 正解を予測するモデル
#         self.classifier = LinearModel(input_dim, hidden_dim, out_dim, layer_num)
# 
#         # Globalな混同を予測するモデル
#         self.global_confusion_matrix = nn.Parameter(
#             self.__identity_init((out_dim, out_dim)),
#             requires_grad=True,
#         )
#         # システム毎の混同を推定するモデル
#         self.agent_confusion_matrix = nn.Parameter(
#             self.__identity_init((annotator_num, out_dim, out_dim)),
#             requires_grad=True,
#         )
# 
#         self.auxiliary_model = AuxiliaryModel(input_dim, out_dim, annotator_num)
#         self.softmax = nn.Softmax(dim=2)
# 
#         if mode == "test":
#             self.classifier.apply(self.init_weights_for_test)
#             self.global_confusion_matrix.requires_grad = False
#             self.agent_confusion_matrix.requires_grad = False
# 
#     def init_device(self, device):
#         self.auxiliary_model = self.auxiliary_model.to(device)
#         self.auxiliary_model.annotator_feature = (
#             self.auxiliary_model.annotator_feature.to(device)
#         )
# 
#     def calc_confusion(
#         self, cls_out: torch.Tensor
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         annotator_confusion = torch.einsum(
#             "ik,jkl->ijl", (cls_out, self.agent_confusion_matrix)
#         )
#         global_confusion = torch.einsum(
#             "ij,jk->ik", (cls_out, self.global_confusion_matrix)
#         )
#         global_confusion = global_confusion.unsqueeze(1).expand(
#             -1, annotator_confusion.shape[1], -1
#         )
# 
#         return global_confusion, annotator_confusion
# 
#     def get_confusion_matrix(self) -> tuple[torch.Tensor, torch.Tensor]:
#         return self.global_confusion_matrix, self.agent_confusion_matrix
# 
#     def forward(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         if "cls_out" in kwargs:
#             cls_logit = kwargs["cls_out"]
#             cls_prob = kwargs["cls_out"]
#         else:
#             out = self.classifier(**kwargs)
#             cls_logit = out["cls_logit"]
#             cls_prob = out["cls_prob"]
# 
#         global_confusion, annotator_confusion = self.calc_confusion(cls_prob)
# 
#         text_embedding = kwargs["embedding"]
#         weight = self.auxiliary_model(text_embedding)
# 
#         expanded_weight = weight.unsqueeze(-1).expand(-1, -1, global_confusion.shape[2])
# 
#         pred_annotaor_dist = (
#             1 - expanded_weight
#         ) * annotator_confusion + expanded_weight * global_confusion
#         pred_annotaor_dist = self.softmax(pred_annotaor_dist)
# 
#         return {
#             "confusion_out": pred_annotaor_dist,
#             "cls_prob": cls_prob,
#             "cls_logit": cls_logit,
#             "weight": weight,
#         }
# 
#     def predicted_accuracy(self, **kwargs) -> np.ndarray:
#         # P(y|x)の計算
#         text_embedding = kwargs["embedding"]
#         if "cls_out" in kwargs:
#             cls_out = kwargs["cls_out"]
#         else:
#             cls_out = self.classifier(**kwargs)
#             cls_out = cls_out["cls_prob"]
# 
#         # 混同行列から対角成分を抽出する(P(\hat{y}=y|y, x)に相当する)
#         global_confusion_matrix = deepcopy(self.global_confusion_matrix)
#         annotator_confusion_matrix = deepcopy(self.agent_confusion_matrix)
#         global_correct = global_confusion_matrix.diag().repeat(cls_out.shape[0], 1)
#         annotator_correct = torch.diagonal(
#             annotator_confusion_matrix, dim1=1, dim2=2
#         ).repeat(cls_out.shape[0], 1, 1)
#         # 周辺化の計算
#         # P(\hat{y}=0|y=0,x)P(y=0|x)+P(\hat{y}=1|y=1,x)P(y=1|x)
#         global_correct_pred = (cls_out * global_correct).sum(dim=1)
#         annotator_correct_pred = torch.einsum(
#             "ijk,ik->ij", (annotator_correct, cls_out)
#         )
# 
#         global_correct_pred = global_correct_pred.unsqueeze(1).expand(
#             -1, annotator_correct_pred.shape[1]
#         )
# 
#         weight = self.auxiliary_model(text_embedding)
#         # 重み付け和の計算
#         acc_tensor = (
#             weight * global_correct_pred + (1 - weight) * annotator_correct_pred
#         )
# 
#         acc_numpy = acc_tensor.cpu().detach().numpy()
#         return acc_numpy
# 
# 
# class AuxiliaryModel(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         out_dim: int,
#         annotator_num: int,
#         mode: str = "train",
#     ):
#         super(AuxiliaryModel, self).__init__()
#         self.feature_embedding = nn.Linear(input_dim, out_dim)
#         self.annotator_embedding = nn.Linear(annotator_num, out_dim)
#         self.annotator_feature = F.one_hot(torch.arange(0, annotator_num)).to(
#             torch.float32
#         )
# 
#         self.sigmoid = nn.Sigmoid()
#         if mode == "test":
#             self.feature_embedding.apply(self.init_weights_for_test)
#             self.annotator_embedding.apply(self.init_weights_for_test)
#             self.annotator_feature = self.annotator_feature.to(torch.float32)
# 
#     @torch.no_grad()
#     def init_weights_for_test(self, m):
#         if type(m) == nn.Linear:
#             m.weight.fill_(1)
#             m.bias.fill_(0)
# 
#     def forward(self, text_embedding: torch.Tensor):
#         vi = F.normalize(self.feature_embedding(text_embedding))
#         ur = F.normalize(self.annotator_embedding(self.annotator_feature))
# 
#         w = torch.einsum("ij,kj->ik", (vi, ur))
#         w = self.sigmoid(w)
# 
#         return w
