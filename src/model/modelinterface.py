# from typing import Any
# 
# import torch.nn as nn
# 
# 
# class ModelInterface(nn.Module):
#     def __init__(
#         self,
#         token_len=None,
#         hidden_dim=None,
#         out_dim=None,
#         dropout_rate=None,
#         kernel_size=None,
#         stride=None,
#         load_bert=False,
#     ) -> None:
#         super().__init__()
# 
#     # def model(self, input):
#     #     raise NotImplementedError
# 
#     def forward(self, batch):
#         raise NotImplementedError
# 
#     @classmethod
#     def predict(self, target_ans: list[Any], target_count: list[int]):
#         raise NotImplementedError
# 
#     def get_params(self):
#         raise NotImplementedError
