# from pathlib import Path
# 
# import torch
# import torch.nn as nn
# 
# from model.modelinterface import ModelInterface
# 
# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# from config import Config
# 
# 
# class ConvolutionModel(ModelInterface):
#     def __init__(
#         self,
#         token_len: int,
#         hidden_dim: int,
#         out_dim: int,
#         dropout_rate: float,
#         config: Config,
#     ) -> None:
#         super().__init__()
#         self.token_len = token_len
#         self.hidden_dim = hidden_dim
#         self.out_dim = out_dim
#         self.dropout_rate = dropout_rate
#         self.config = config
#         self.model = nn.Sequential(
#             nn.Linear(768, self.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.hidden_dim, out_dim),
#             nn.Softmax(dim=1),
#         )
# 
#     def forward(self, emb: torch.Tensor) -> torch.Tensor:
#         out = self.model(emb)
#         return out
# 
#     def predict(self, out, system_dicision, crowd_dicision, annotator):
#         model_ans = []
#         system_crowd = []
#         s_count, c_count, a_count = 0, 0, 0
#         index = torch.argmax(out, dim=1)
#         for i, idx in enumerate(index):
#             if idx == 0:
#                 model_ans.append(system_dicision[i])
#                 s_count += 1
#                 system_crowd.append("system")
#             elif idx == 1:
#                 model_ans.append(crowd_dicision[i])
#                 c_count += 1
#                 system_crowd.append("crowd")
#             else:
#                 model_ans.append(annotator[i])
#                 a_count += 1
#                 system_crowd.append("annotator")
#         model_ans = torch.Tensor(model_ans)
#         return model_ans, s_count, c_count, a_count, system_crowd
# 
#     def get_params(self):
#         return self.model.parameters()
# 
# 
# class AgentAccuracyModel(ModelInterface):
#     def __init__(
#         self,
#         token_len: int,
#         hidden_dim: int,
#         out_dim: int,
#         dropout_rate: float,
#         config: Config,
#         system_num: int,
#     ):
#         super().__init__()
# 
#         assert system_num is not None, "system_num is required"
#         assert system_num > 0, f"system_num is greater than 0, but got {system_num}"
#         self.models: list[nn.Module] = []
#         for _ in range(system_num):
#             self.models.append(
#                 ConvolutionModel(token_len, hidden_dim, out_dim, dropout_rate, config)
#             )
# 
#     def init_device(self, device: torch.device):
#         for i in range(len(self.models)):
#             self.models[i] = self.models[i].to(device)
# 
#     def save_model(self, path: Path):
#         for i, model in enumerate(self.models):
#             torch.save(model.state_dict(), path / f"model_{i}.pth")
# 
#     def forward(self, batch):
#         return [model.forward(batch) for model in self.models]
# 
#     def get_params(self):
#         params = []
#         for model in self.models:
#             params.extend(model.get_params())
#         return params
