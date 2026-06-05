# import torch.nn as nn
# from pathlib import Path
# from sklearn.metrics import accuracy_score
# from collections import defaultdict
# 
# import json
# import numpy as np
# import torch
# from model import BaseMatchingModel
# from train.train import ModelTrainer
# 
# 
# # class ShinraTrainer:
# #     def __init__(
# #         self,
# #         output_dir: str,
# #         weight_model: nn.Module,
# #         matching_model: BaseMatchingModel,
# #         seed: int,
# #         learning_rate: float,
# #         weight_decay: float,
# #         assign_interval: int,
# #         loss,
# #         mode: str = "train",
# #     ):
# #         self.mode = mode
# #         self.seed = seed
# #         super().__init__()
# #         seed_everything(seed)
# #         self.matching_model: BaseMatchingModel = matching_model
# #         self.output_dir = Path(output_dir)
# #         self.output_dir.mkdir(parents=True, exist_ok=True)
# #         if weight_model is not None:
# #             self.weight_model: nn.Module = weight_model.to(self.device)
# #         if mode == "test":
# #             return
# #
# #         self.train_df = pd.read_csv("./data/all_system_train.csv", index_col=0)
# #
# #         self.min_valid_loss: float = 100000000
# #         self.learning_rate = learning_rate
# #         self.weight_decay = weight_decay
# #         self.max_valid_acc: float = 0
# #         Path(self.output_dir).mkdir(parents=True, exist_ok=True)
# #
# #         self.loss = loss
# #         self.weight: np.ndarray = np.array([])
# #         self.assignments: torch.Tensor = torch.empty(0, 0)
# #         self.model_name = f"weight_model_{seed}.pth"
# #         self.model_path = self.output_dir / self.model_name
# #         self.assign_interval = assign_interval
# #         self.save_model()
# #
# #     def forward(self, emb: torch.Tensor):
# #         return self.weight_model(emb)
# #
# #     def on_fit_start(self):
# #         self.weight_model.init_device(self.device)
# #         self.save_model()
# #         self.logger.log_hyperparams({"seed": self.seed})
# #         self.logger.log_hyperparams({"beta": self.loss.beta})
# #         self.logger.log_hyperparams({"alpha": self.loss.alpha})
# #         self.logger.log_hyperparams({"learning_rate": self.learning_rate})
# #         self.logger.log_hyperparams({"weight_decay": self.weight_decay})
# #         self.logger.log_hyperparams({"output_dir": self.output_dir})
# #         self.logger.log_hyperparams({"hidden_dim": self.weight_model.hidden_dim})
# #         self.logger.log_hyperparams({"loss": self.loss.name})
# #         self.logger.log_hyperparams({"dataset": "shinra"})
# #
# #     def training_step(self, batch: dict, _):
# #         emb = batch["bert_embedding"].to(self.device)
# #         annotation = batch["system_pred"].to(torch.long)
# #         ground_truth = batch["annotator_decision"].to(torch.long)
# #         out = self.forward(emb)
# #         cls_out = out["cls_logit"]
# #         confusion_out = out["confusion_out"]
# #
# #         all_loss = self.loss(
# #             cls_out=cls_out,
# #             ground_truth=ground_truth,
# #             confusion_out=confusion_out,
# #             annotations=annotation,
# #             model=self.weight_model,
# #         )
# #
# #         loss = all_loss["loss"]
# #
# #         confusion_loss = (
# #             all_loss["confusion_loss"].item()
# #             if all_loss["confusion_loss"] is not None
# #             else None
# #         )
# #         cls_loss = (
# #             all_loss["cls_loss"].item() if all_loss["cls_loss"] is not None else None
# #         )
# #         self.log_data(
# #             {
# #                 "train loss": loss.item(),
# #                 "train confusion loss": confusion_loss
# #                 if confusion_loss is not None
# #                 else 0,
# #                 "train cls loss": cls_loss if cls_loss is not None else 0,
# #             }
# #         )
# #         self.annotations = torch.cat([self.annotations, annotation])
# #         self.gold_label = torch.cat([self.gold_label, ground_truth])
# #         if cls_out is not None:
# #             self.cls_out = torch.cat([self.cls_out, cls_out])
# #         cost = self.weight_model.predicted_accuracy(emb)
# #         self.cost = (
# #             cost if self.cost.shape == (0,) else np.concatenate([self.cost, cost])
# #         )
# #
# #         return loss
# #
# #     def on_train_epoch_end(self):
# #         if self.current_epoch % self.assign_interval == 0:
# #             annotator_num = self.annotations.shape[1]
# #             if self.conf_out.shape != torch.Size([0]):
# #                 for i in range(annotator_num):
# #                     confusion_i = self.conf_out[:, i].cpu().detach()
# #                     confusion_i = confusion_i.argmax(dim=1).numpy()
# #                     annotation_i = self.annotations[:, i].cpu().numpy()
# #                     annotator_i_acc = accuracy_score(annotation_i, confusion_i)
# #                     self.log_data({f"train_annotator_{i+1}_accuracy": annotator_i_acc})
# #
# #             pred = self.cls_out.argmax(dim=1).cpu().numpy()
# #             acc = accuracy_score(self.gold_label.cpu().numpy(), pred)
# #
# #             assignment, objective = self.matching_model(self.cost)
# #
# #             pred = self.gather_assignment(self.annotations, assignment)
# #
# #             metrics = self.calc_metrics(self.gold_label, pred)
# #             self.log_data(
# #                 {
# #                     "train objective value": (
# #                         objective if objective is not None else 0
# #                     ),
# #                     "train_accuracy": metrics[0],
# #                     "train_precision": metrics[1],
# #                     "train_recall": metrics[2],
# #                     "train_f1": metrics[3],
# #                     "train_cls_accuracy": acc,
# #                 }
# #             )
# #
# #     def validation_step(self, batch: dict, _) -> None:
# #         emb = batch["bert_embedding"].to(self.device)
# #         annotation = batch["system_pred"].to(torch.long)
# #         ground_truth = batch["annotator_decision"].to(torch.long)
# #         out = self.forward(emb)
# #
# #         cls_out = out["cls_logit"]
# #         confusion_out = out["confusion_out"]
# #
# #         all_loss = self.loss(
# #             cls_out=cls_out,
# #             ground_truth=ground_truth,
# #             confusion_out=confusion_out,
# #             annotations=annotation,
# #             model=self.weight_model,
# #         )
# #         loss = all_loss["loss"].item()
# #         confusion_loss = (
# #             all_loss["confusion_loss"].item()
# #             if all_loss["confusion_loss"] is not None
# #             else None
# #         )
# #
# #         cls_loss = (
# #             all_loss["cls_loss"].item() if all_loss["cls_loss"] is not None else None
# #         )
# #
# #         self.log_data(
# #             {
# #                 "val_loss": loss,
# #                 "val_confusion_loss": confusion_loss
# #                 if confusion_loss is not None
# #                 else 0,
# #                 "val_cls_loss": cls_loss if cls_loss is not None else 0,
# #             }
# #         )
# #         self.losses.append(loss)
# #         self.annotations = torch.cat([self.annotations, annotation])
# #         self.gold_label = torch.cat([self.gold_label, ground_truth])
# #
# #         if confusion_out is not None:
# #             self.conf_out = torch.cat([self.conf_out, confusion_out])
# #         if cls_out is not None:
# #             self.cls_out = torch.cat([self.cls_out, cls_out])
# #         cost = self.weight_model.predicted_accuracy(emb)
# #         self.cost = (
# #             cost if self.cost.shape == (0,) else np.concatenate([self.cost, cost])
# #         )
# #
# #     def on_validation_epoch_end(self):
# #         if self.min_valid_loss > np.mean(self.losses):
# #             self.save_model()
# #             self.min_valid_loss = np.mean(self.losses)
# #         annotator_num = self.annotations.shape[1]
# #
# #         if self.conf_out.shape != torch.Size([0]):
# #             for i in range(annotator_num):
# #                 confusion_i = self.conf_out[:, i].cpu().detach()
# #                 confusion_i = confusion_i.argmax(dim=1).numpy()
# #                 annotation_i = self.annotations[:, i].cpu().numpy()
# #                 annotator_i_acc = accuracy_score(annotation_i, confusion_i)
# #                 self.log_data({f"val_annotator_{i+1}_accuracy": annotator_i_acc})
# #
# #         if self.current_epoch % self.assign_interval == 0:
# #             pred = self.cls_out.argmax(dim=1).cpu().numpy()
# #             cls_acc = accuracy_score(self.gold_label.cpu().numpy(), pred)
# #             assignment, objective = self.matching_model(self.cost)
# #             pred = self.gather_assignment(self.annotations, assignment)
# #             metrics = self.calc_metrics(self.gold_label, pred)
# #             assert self.logger is not None, f"logger is None in {self.__class__}"
# #             self.logger.log_metrics(
# #                 {
# #                     "valid objective value": (
# #                         objective if objective is not None else 0
# #                     ),
# #                     "valid_accuracy": metrics[0],
# #                     "valid_train_precision": metrics[1],
# #                     "valid_recall": metrics[2],
# #                     "valid_f1": metrics[3],
# #                     "valid_cls_accuracy": cls_acc,
# #                 }
# #             )
# #
# #     def on_predict_epoch_start(self):
# #         self.annotations = torch.tensor([]).to(self.device)
# #         self.conf_out = torch.tensor([]).to(self.device)
# #         self.cls_out = torch.tensor([]).to(self.device)
# #         self.gold_label = torch.tensor([]).to(self.device)
# #         self.cost = np.array([])
# #         self.importance = torch.tensor([]).to(self.device)
# #         self.batch = {}
# #         self.load_model()
# #
# #     def predict_step(self, batch, _):
# #         self.load_model()
# #         emb = batch["bert_embedding"].to(self.device)
# #         out = self.forward(emb)
# #
# #         cls_out = out["cls_logit"]
# #         confusion_out = out["confusion_out"]
# #         importance = out["weight"]
# #         cost = self.weight_model.predicted_accuracy(emb)
# #
# #         self.annotations = torch.cat([self.annotations, batch["system_pred"]])
# #         self.gold_label = torch.cat([self.gold_label, batch["annotator_decision"]])
# #         if confusion_out is not None:
# #             self.conf_out = torch.cat([self.conf_out, confusion_out])
# #         if cls_out is not None:
# #             self.cls_out = torch.cat([self.cls_out, cls_out])
# #         if isinstance(importance, torch.Tensor):
# #             self.importance = torch.cat([self.importance, importance])
# #         self.cost = (
# #             cost if self.cost.shape == (0,) else np.concatenate([self.cost, cost])
# #         )
# #
# #         for k, v in batch.items():
# #             if k not in self.batch.keys():
# #                 self.batch[k] = v
# #                 continue
# #             if isinstance(v, torch.Tensor):
# #                 self.batch[k] = torch.cat([self.batch[k], v])
# #             elif isinstance(v, list):
# #                 self.batch[k].extend(v)
# #
# #     def on_predict_epoch_end(self):
# #         batch = self.batch
# #         importance = None
# #         cost = self.cost
# #
# #         self.assign_by_acc(batch)
# #         if hasattr(self, "importance"):
# #             importance = self.importance
# #
# #         assignment, _ = self.matching_model(cost)
# #
# #         df, batch = self.create_result_dataframe_acc(
# #             batch, cost, assignment, importance
# #         )
# #         df.to_csv(f"{self.output_dir}/result_{self.mode}.csv", index=False)
# #         if self.conf_out.shape != torch.Size([0]):
# #             self.save_model_weights(batch, self.conf_out)
# #         print(f"dataframe was saved to {self.output_dir}/result_{self.mode}.csv")
# #
# #         random_metrics, random_metrics_std = self.random_model_predict(
# #             batch["system_pred"], batch["annotator_decision"]
# #         )
# #         pred = df.assignment_ans.to_list()
# #         prediction_metrics = self.calc_metrics(pred, batch["annotator_decision"])
# #         data = {}
# #
# #         idx = {
# #             0: "accuracy",
# #             1: "precision",
# #             2: "recall",
# #             3: "f1",
# #         }
# #         for i, v in enumerate(random_metrics):
# #             data[f"random_{idx[i]}"] = v
# #
# #         # for i, v in enumerate(random_metrics_std):
# #         #     data[f"random_{idx[i]}_std"] = v
# #
# #         for i, v in enumerate(prediction_metrics):
# #             data[f"{self.mode}_{idx[i]}"] = v
# #
# #         with open(f"{self.output_dir}/score_{self.mode}.json", "w") as f:
# #             json.dump(data, f, indent=4, ensure_ascii=False)
# #
# #         ground_truth = batch["annotator_decision"].to(torch.long)
# #         cls_pred = self.cls_out.argmax(dim=1)
# #         acc = accuracy_score(ground_truth.cpu().numpy(), cls_pred.cpu().numpy())
# #         data[f"cls_{self.mode}_accuracy"] = acc
# #
# #         assert self.logger is not None, f"logger is None in {self.__class__}"
# #         self.logger.log_metrics(data)
# #         self.scores = data
# #
# #     def save_model_weights(self, batch: dict, weight: torch.Tensor):
# #         system_pred = batch["system_pred"]
# #         correct_pred = batch["correct"]
# #         for i in range(weight.shape[1]):
# #             weight_data: dict[str, list[float]] = {
# #                 f"system_{i}_wrong_prob": [],
# #                 f"system_{i}_correct_prob": [],
# #                 f"system_{i}_pred": [],
# #                 "correct": [],
# #             }
# #             w_i = weight[i]
# #
# #             system_pred_i = system_pred[:, i]
# #             data_num, _ = w_i.shape
# #             for d_num in range(data_num):
# #                 weight_data[f"system_{i}_wrong_prob"].append(w_i[d_num, 0].item())
# #                 weight_data[f"system_{i}_correct_prob"].append(w_i[d_num, 1].item())
# #                 weight_data[f"system_{i}_pred"].append(system_pred_i[d_num].item())
# #                 weight_data["correct"].append(correct_pred[d_num].item())
# #             df = pd.DataFrame(weight_data)
# #             df.to_csv(f"{self.output_dir}/weight_{i}_{self.mode}.csv", index=False)
# #
# #     def create_result_dataframe_acc(
# #         self,
# #         batch: dict,
# #         weight: np.ndarray,
# #         assignment: torch.Tensor,
# #         importance: torch.Tensor | None,
# #     ):
# #         tmp_batch = copy(batch)
# #         b = ["bert_embedding", "tokens", "attention_mask", "n", "decision", "correct"]
# #         for bi in b:
# #             if bi not in batch.keys():
# #                 continue
# #             del batch[bi]
# #
# #         batch["assignment_ans"] = self.gather_assignment(
# #             batch["system_pred"], assignment
# #         )
# #         assignments = []
# #
# #         batch["assignment_idx"] = assignment.argmax(dim=1).tolist()
# #         for a in assignment.argmax(dim=1).tolist():
# #             assignments.append(f"system_{a}")
# #         batch["assignments"] = assignments
# #
# #         # 重みの集計
# #         systems = [f"system_{i}" for i in range(len(batch["system_pred"][0]))]
# #         weights: dict[str, list[float]] = {"weight_" + s: [] for s in systems}
# #
# #         # 推定された正解率の集計
# #         weight_list = weight.tolist()
# #         for wi in weight_list:
# #             for i, w in enumerate(wi):
# #                 weights[f"weight_system_{i}"].append(w)
# #
# #         for i in range(len(batch["system_pred"][0])):
# #             batch[f"pred_system_{i}"] = batch["system_pred"][:, i].tolist()
# #         # if importance is not None:
# #         #     for ii in range(importance.shape[1]):
# #         #         batch[f"importance_system_{ii}"] = importance[:, ii].tolist()
# #         d = batch
# #         del d["system_pred"]
# #         del d["system_pred_results"]
# #         data = {**d, **weights}
# #         for k in data.keys():
# #             if isinstance(data[k], torch.Tensor):
# #                 if len(data[k].shape) > 1:
# #                     data[k] = data[k].squeeze()
# #
# #                 if data[k].device != "cpu":
# #                     data[k] = data[k].cpu()
# #                 data[k] = data[k].tolist()
# #
# #         df = pd.DataFrame(data)
# #         return df, tmp_batch
# #
# #     def configure_optimizers(self):
# #         if self.weight_model is None:
# #             return None
# #         optimizer = torch.optim.Adam(
# #             self.weight_model.parameters(),
# #             lr=self.learning_rate if self.mode == "train" else 0.0,
# #             weight_decay=self.weight_decay,
# #         )
# #         return optimizer
# #
# #     def gather_assignment(self, preds, assignment):
# #         assignment_idx = assignment.argmax(dim=1).to(self.device)
# #         ans = torch.gather(preds, 1, assignment_idx.unsqueeze(-1))
# #         return ans
# 
# 
# # class ShinraTrainerLearningToDefer(ModelTrainer):
# #     def __init__(
# #         self,
# #         weight_model: nn.Module,
# #         matching_model: BaseMatchingModel,
# #         learning_rate: float,
# #         weight_decay: float,
# #         mode: str,
# #         seed: int,
# #         output_dir: Path,
# #         assign_interval: int,
# #         dataset_name: str,
# #         loss,
# #     ) -> None:
# #         super().__init__(
# #             weight_model=weight_model,
# #             matching_model=matching_model,
# #             learning_rate=learning_rate,
# #             weight_decay=weight_decay,
# #             mode=mode,
# #             seed=seed,
# #             output_dir=output_dir,
# #             assign_interval=assign_interval,
# #             dataset_name=dataset_name,
# #             loss=loss,
# #         )
# #         self.out_dim = self.weight_model.out_dim
# #         self.loss_function = loss
# #
# #     def on_fit_start(self):
# #         self.save_model()
# #         self.logger.log_hyperparams({"seed": self.seed})
# #         self.logger.log_hyperparams({"alpha": self.loss.alpha})
# #         self.logger.log_hyperparams({"beta": self.loss.beta})
# #         self.logger.log_hyperparams({"learning_rate": self.learning_rate})
# #         self.logger.log_hyperparams({"weight_decay": self.weight_decay})
# #         self.logger.log_hyperparams({"output_dir": self.output_dir})
# #         self.logger.log_hyperparams({"hidden_dim": self.weight_model.hidden_dim})
# #         self.logger.log_hyperparams({"loss": "learning to defer"})
# #         self.logger.log_hyperparams({"dataset": "shinra"})
# #
# #     def training_step(self, batch: dict, _):
# #         emb = batch["embedding"].to(self.device)
# #         annotation = batch["annotations"].to(torch.long)
# #         ground_truth = batch["label"].to(torch.long)
# #         out = self.forward(emb)
# #         prob = out["cls_prob"]
# #         logit = out["cls_logit"]
# #
# #         loss = self.loss_function(prob, ground_truth, annotation)
# #         loss = loss["loss"]
# #
# #         self.log_data(
# #             {
# #                 "train loss": loss.item(),
# #             }
# #         )
# #         self.annotations = torch.cat([self.annotations, annotation])
# #         self.gold_label = torch.cat([self.gold_label, ground_truth])
# #         self.cls_out = torch.cat([self.cls_out, prob])
# #         self.assignments = torch.cat([self.assignments, logit.argmax(dim=1)])
# #         return loss
# #
# #     def on_train_epoch_end(self):
# #         if self.current_epoch % self.assign_interval == 0:
# #             pred = self.cls_out.argmax(dim=1).cpu().numpy()
# #             cls_acc = accuracy_score(self.gold_label.cpu().numpy(), pred)
# #
# #             pred = torch.gather(
# #                 self.annotations, 1, self.assignments.unsqueeze(-1).to(torch.long)
# #             )
# #             metrics = self.calc_metrics(self.gold_label, pred)
# #             assert self.logger is not None, f"logger is None in {self.__class__}"
# #             self.logger.log_metrics(
# #                 {
# #                     "valid_accuracy": metrics[0],
# #                     "valid_train_precision": metrics[1],
# #                     "valid_recall": metrics[2],
# #                     "valid_f1": metrics[3],
# #                     "valid_cls_accuracy": cls_acc,
# #                 }
# #             )
# #
# #     def validation_step(self, batch: dict, _) -> None:
# #         emb = batch["embedding"].to(self.device)
# #         annotation = batch["annotations"].to(torch.long)
# #         ground_truth = batch["label"].to(torch.long)
# #
# #         out = self.forward(emb)
# #         prob = out["cls_prob"]
# #         logit = out["cls_logit"]
# #
# #         loss = self.loss_function(prob, ground_truth, annotation)
# #         loss = loss["loss"]
# #
# #         self.log_data(
# #             {
# #                 "val loss": loss.item(),
# #             }
# #         )
# #         self.losses.append(loss.item())
# #         self.annotations = torch.cat([self.annotations, annotation])
# #         self.gold_label = torch.cat([self.gold_label, ground_truth])
# #         self.cls_out = torch.cat([self.cls_out, prob])
# #         self.assignments = torch.cat([self.assignments, logit.argmax(dim=1)])
# #
# #     def on_validation_epoch_end(self):
# #         if self.min_valid_loss > np.mean(self.losses):
# #             self.save_model()
# #             self.min_valid_loss = np.mean(self.losses)
# #
# #         if self.current_epoch % self.assign_interval == 0:
# #             pred = self.cls_out.argmax(dim=1).cpu().numpy()
# #             cls_acc = accuracy_score(self.gold_label.cpu().numpy(), pred)
# #
# #             pred = torch.gather(
# #                 self.annotations, 1, self.assignments.unsqueeze(-1).to(torch.long)
# #             )
# #             metrics = self.calc_metrics(self.gold_label, pred)
# #             assert self.logger is not None, f"logger is None in {self.__class__}"
# #             self.logger.log_metrics(
# #                 {
# #                     "valid_accuracy": metrics[0],
# #                     "valid_train_precision": metrics[1],
# #                     "valid_recall": metrics[2],
# #                     "valid_f1": metrics[3],
# #                     "valid_cls_accuracy": cls_acc,
# #                 }
# #             )
# #
# #     def on_predict_epoch_start(self):
# #         self.annotations = torch.tensor([]).to(self.device)
# #         self.conf_out = torch.tensor([]).to(self.device)
# #         self.cls_out = torch.tensor([]).to(self.device)
# #         self.gold_label = torch.tensor([]).to(self.device)
# #         self.importance = torch.tensor([]).to(self.device)
# #         self.assignments = torch.tensor([]).to(self.device)
# #         self.batch = {}
# #         self.load_model()
# #
# #     def predict_step(self, batch, _):
# #         self.load_model()
# #         emb = batch["embedding"].to(self.device)
# #         out = self.forward(emb)
# #         prob = out["cls_prob"]
# #         logit = out["cls_logit"]
# #
# #         self.annotations = torch.cat([self.annotations, batch["annotations"]])
# #         self.gold_label = torch.cat([self.gold_label, batch["label"]])
# #         self.cls_out = torch.cat([self.cls_out, logit])
# #         self.assignments = torch.cat([self.assignments, logit.argmax(dim=1)])
# #
# #         for k, v in batch.items():
# #             if k not in self.batch.keys():
# #                 self.batch[k] = v
# #                 continue
# #             if isinstance(v, torch.Tensor):
# #                 self.batch[k] = torch.cat([self.batch[k], v])
# #             elif isinstance(v, list):
# #                 self.batch[k].extend(v)
# #
# #     def on_predict_epoch_end(self):
# #         # predの計算をする
# #         pred = torch.gather(
# #             self.annotations, 1, self.assignments.unsqueeze(-1).to(torch.long)
# #         )
# #         prediction_metrics = self.calc_metrics(pred, self.gold_label)
# #         data = {}
# #
# #         idx = {
# #             0: "accuracy",
# #             1: "precision",
# #             2: "recall",
# #             3: "f1",
# #         }
# #         for i, v in enumerate(prediction_metrics):
# #             data[f"{self.mode}_{idx[i]}"] = v
# #
# #         with open(f"{self.output_dir}/score_{self.mode}.json", "w") as f:
# #             json.dump(data, f, indent=4, ensure_ascii=False)
# #
# #         assert self.logger is not None, f"logger is None in {self.__class__}"
# #         self.logger.log_metrics(data)
# #         self.scores = data
# #         # 制約違反の評価
# #         task_num = self.assignments.shape[0]
# #         ratio = [1 / self.out_dim for _ in range(self.out_dim)]
# #         maximum_assign_num = [int(task_num * ratio) for ratio in ratio]
# #
# #         i = 0
# #         while sum(maximum_assign_num) < task_num:
# #             maximum_assign_num[i % len(maximum_assign_num)] += 1
# #             i += 1
# #
# #         assigned_num = defaultdict(int)
# #         assignments = self.assignments.tolist()
# #         for assignmtnt in assignments:
# #             assigned_num[assignmtnt] += 1
# #
# #         violations = []
# #         for i, maximum in enumerate(maximum_assign_num):
# #             violations.append(max(assigned_num[i] - maximum, 0))
# #
# #         violations_data = {}
# #         for i, vi in enumerate(violations):
# #             violations_data[f"violation_annotator_{i}"] = vi
# #
# #         violations_data["total_constraint_violation"] = sum(violations)
# #         self.logger.log_metrics(violations_data)
