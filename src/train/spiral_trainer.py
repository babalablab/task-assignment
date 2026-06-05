# import torch
# import torch.nn as nn
# import lightning as L
# from sklearn.metrics import accuracy_score
# from pathlib import Path
# import polars as pl
# import pandas as pd
# import numpy as np
# from lightning.pytorch import seed_everything
# from collections import defaultdict
# 
# from train.icrowd import iCrowdTrainer
# from model import BaseMatchingModel
# from train.train import ModelTrainer
# 
# # class SpiralTrainer:
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
# #         dataset: str,
# #         loss,
# #     ) -> None:
# #         super().__init__()
# #         seed_everything(seed)
# #         self.seed = seed
# #         self.mode = mode
# #         self.output_dir = Path(output_dir)
# #         self.output_dir.mkdir(parents=True, exist_ok=True)
# #         self.weight_model = weight_model
# #         self.matching_model = matching_model
# #         self.learning_rate = learning_rate
# #         self.weight_decay = weight_decay
# #         self.assign_interval = assign_interval
# #         self.model_name = f"weight_model_{seed}.pth"
# #         self.dataset = dataset
# #         self.loss = loss
# #         self.model_path = self.output_dir / self.model_name
# #         self.min_loss = 10000000000000
# #         self.save_model()
# #
# #     def on_fit_start(self):
# #         self.weight_model.init_device(self.device)
# #         self.logger.log_hyperparams({"seed": self.seed})
# #         self.logger.log_hyperparams({"alpha": self.loss.alpha})
# #         self.logger.log_hyperparams({"beta": self.loss.beta})
# #         self.logger.log_hyperparams({"learning_rate": self.learning_rate})
# #         self.logger.log_hyperparams({"weight_decay": self.weight_decay})
# #         self.logger.log_hyperparams({"output_dir": self.output_dir})
# #         self.logger.log_hyperparams({"hidden_dim": self.weight_model.hidden_dim})
# #         self.logger.log_hyperparams({"loss": self.loss.name})
# #         self.logger.log_hyperparams({"dataset": self.dataset})
# #
# #     def training_step(self, x):
# #         feature = x["feature"]
# #         label = x["label"]
# #         annotation = x["annotations"]
# #
# #         out = self.weight_model(feature)
# #         cls_out = out["cls_prob"]
# #         cls_logit = out["cls_logit"]
# #         conf_out = out["confusion_out"]
# #
# #         all_loss = self.loss(
# #             cls_out=cls_logit,
# #             ground_truth=label,
# #             confusion_out=conf_out,
# #             annotations=annotation,
# #             model=self.weight_model,
# #         )
# #
# #         loss = all_loss["loss"]
# #         confusion_loss = (
# #             all_loss["confusion_loss"].item()
# #             if all_loss["confusion_loss"] is not None
# #             else None
# #         )
# #         cls_loss = (
# #             all_loss["cls_loss"].item() if all_loss["cls_loss"] is not None else None
# #         )
# #
# #         self.losses.append(loss.item())
# #         self.log_data(
# #             {
# #                 "train_loss": loss.item(),
# #                 "train_confusion_loss": confusion_loss
# #                 if confusion_loss is not None
# #                 else 0,
# #                 "train_cls_loss": cls_loss if cls_loss is not None else 0,
# #             }
# #         )
# #         self.annotations = torch.cat([self.annotations, annotation])
# #         self.gold_label = torch.cat([self.gold_label, label])
# #         if conf_out is not None:
# #             self.conf_out = torch.cat([self.conf_out, conf_out])
# #         if cls_out is not None:
# #             self.cls_out = torch.cat([self.cls_out, cls_out])
# #         cost = self.weight_model.predicted_accuracy(feature)
# #         self.cost = (
# #             cost if self.cost.shape == (0,) else np.concatenate([self.cost, cost])
# #         )
# #         return loss
# #
# #     def on_train_epoch_end(self):
# #         for i in range(self.annotations.shape[1]):
# #             pred = self.annotations[:, i]
# #             acc = accuracy_score(self.gold_label.cpu(), pred.cpu())
# #             self.training_accuracies.append(acc)
# #
# #         if self.current_epoch % self.assign_interval == 0:
# #             if self.conf_out.shape != torch.Size([0]):
# #                 annotator_num = self.annotations.shape[1]
# #                 for i in range(annotator_num):
# #                     confusion_i = self.conf_out[:, i].cpu().detach()
# #                     confusion_i = confusion_i.argmax(dim=1).numpy()
# #                     annotation_i = self.annotations[:, i].cpu().numpy()
# #                     annotator_i_acc = accuracy_score(annotation_i, confusion_i)
# #                     self.log_data({f"train_annotator_{i+1}_accuracy": annotator_i_acc})
# #
# #             cls_pred = self.cls_out.argmax(dim=1)
# #
# #             cls_acc = accuracy_score(
# #                 self.gold_label.cpu().numpy(), cls_pred.cpu().numpy()
# #             )
# #
# #             assignment, objective = self.matching_model(self.cost)
# #             pred = self.gather_assignment(self.annotations, assignment)
# #             metrics = self.calc_metrics(self.gold_label, pred)
# #             assert self.logger is not None, f"logger is None in {self.__class__}"
# #             self.logger.log_metrics(
# #                 {
# #                     "train_objective_value": (
# #                         objective if objective is not None else 0
# #                     ),
# #                     "train_accuracy": metrics[0],
# #                     "train_precision": metrics[1],
# #                     "train_recall": metrics[2],
# #                     "train_f1": metrics[3],
# #                     "train_cls_accuracy": cls_acc,
# #                 }
# #             )
# #
# #     def validation_step(self, x):
# #         feature = x["feature"]
# #         label = x["label"]
# #         annotation = x["annotations"]
# #
# #         out = self.weight_model(feature)
# #
# #         cls_out = out["cls_prob"]
# #         conf_out = out["confusion_out"]
# #         cls_logit = out["cls_logit"]
# #
# #         all_loss = self.loss(
# #             cls_out=cls_logit,
# #             ground_truth=label,
# #             confusion_out=conf_out,
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
# #
# #         self.losses.append(loss.item())
# #         self.log_data(
# #             {
# #                 "val_loss": loss.item(),
# #                 "val_confusion_loss": confusion_loss
# #                 if confusion_loss is not None
# #                 else 0,
# #                 "val_cls_loss": cls_loss if cls_loss is not None else 0,
# #             }
# #         )
# #         self.annotations = torch.cat([self.annotations, annotation])
# #         self.gold_label = torch.cat([self.gold_label, label])
# #         if conf_out is not None:
# #             self.conf_out = torch.cat([self.conf_out, conf_out])
# #         if cls_out is not None:
# #             self.cls_out = torch.cat([self.cls_out, cls_out])
# #         cost = self.weight_model.predicted_accuracy(feature)
# #         self.cost = (
# #             cost if self.cost.shape == (0,) else np.concatenate([self.cost, cost])
# #         )
# #         self.losses.append(loss.item())
# #         return loss
# #
# #     def on_validation_epoch_end(self):
# #         losses = np.mean(self.losses)
# #         if losses < self.min_loss:
# #             self.min_loss = losses
# #             self.save_model()
# #
# #         if self.current_epoch % self.assign_interval == 0:
# #             if self.conf_out.shape != torch.Size([0]):
# #                 annotator_num = self.annotations.shape[1]
# #                 for i in range(annotator_num):
# #                     confusion_i = self.conf_out[:, i].cpu().detach()
# #                     confusion_i = confusion_i.argmax(dim=1).numpy()
# #                     annotation_i = self.annotations[:, i].cpu().numpy()
# #                     annotator_i_acc = accuracy_score(annotation_i, confusion_i)
# #                     self.log_data({f"train_annotator_{i+1}_accuracy": annotator_i_acc})
# #
# #             cls_pred = self.cls_out.argmax(dim=1)
# #
# #             cls_acc = accuracy_score(
# #                 self.gold_label.cpu().numpy(), cls_pred.cpu().numpy()
# #             )
# #
# #             assignment, objective = self.matching_model(self.cost)
# #             pred = self.gather_assignment(self.annotations, assignment)
# #             metrics = self.calc_metrics(self.gold_label, pred)
# #             assert self.logger is not None, f"logger is None in {self.__class__}"
# #             self.logger.log_metrics(
# #                 {
# #                     "valid_objective_value": (
# #                         objective if objective is not None else 0
# #                     ),
# #                     "valid_accuracy": metrics[0],
# #                     "valid_precision": metrics[1],
# #                     "valid_recall": metrics[2],
# #                     "valid_f1": metrics[3],
# #                     "valid_cls_accuracy": cls_acc,
# #                 }
# #             )
# #
# #     def predict_step(self, batch, _):
# #         feature = batch["feature"]
# #         label = batch["label"]
# #         annotations = batch["annotations"]
# #
# #         out = self.weight_model(feature)
# #         cls_out = out["cls_prob"]
# #         conf_out = out["confusion_out"]
# #
# #         cost = self.weight_model.predicted_accuracy(feature)
# #         self.annotations = torch.cat([self.annotations, annotations])
# #         self.gold_label = torch.cat([self.gold_label, label])
# #         if conf_out is not None:
# #             self.conf_out = torch.cat([self.conf_out, conf_out])
# #         if cls_out is not None:
# #             self.cls_out = torch.cat([self.cls_out, cls_out])
# #         self.cost = (
# #             cost if self.cost.shape == (0,) else np.concatenate([self.cost, cost])
# #         )
# #
# #         for k, v in batch.items():
# #             self.batch[k].extend(v.tolist() if isinstance(v, torch.Tensor) else list(v))
# #
# #     def on_predict_epoch_end(self):
# #         cls_pred = self.cls_out.argmax(dim=1)
# #         cls_acc = accuracy_score(self.gold_label.cpu().numpy(), cls_pred.cpu().numpy())
# #
# #         assignment, objective = self.matching_model(self.cost)
# #         pred = self.gather_assignment(self.annotations, assignment)
# #         metrics = self.calc_metrics(self.gold_label, pred)
# #         assert self.logger is not None, f"logger is None in {self.__class__}"
# #         data = {
# #             f"result_{self.mode}_objective value": (
# #                 objective if objective is not None else 0
# #             ),
# #             f"result_{self.mode}_accuracy": metrics[0],
# #             f"result_{self.mode}_precision": metrics[1],
# #             f"result_{self.mode}_recall": metrics[2],
# #             f"result_{self.mode}_f1": metrics[3],
# #             f"result_{self.mode}_cls_accuracy": cls_acc,
# #         }
# #
# #         random_metrics, random_metrics_std = self.random_model_predict(
# #             self.annotations, self.gold_label
# #         )
# #         idx = {
# #             0: "accuracy",
# #             1: "precision",
# #             2: "recall",
# #             3: "f1",
# #         }
# #         for i, v in enumerate(random_metrics):
# #             data[f"random_{idx[i]}"] = v
# #
# #         accs = np.array([self.training_accuracies])
# #         accs = np.repeat(accs, self.annotations.shape[0], axis=0)
# #
# #         h_assignment, obj = self.matching_model(accs)
# #         h_pred = self.gather_assignment(self.annotations, h_assignment)
# #         h_metrics = self.calc_metrics(self.gold_label, h_pred)
# #         for i, v in enumerate(h_metrics):
# #             data[f"{self.mode}_heuristic_{idx[i]}"] = v
# #
# #         self.logger.log_metrics(data)
# #
# #         pred = pred.squeeze().detach().cpu().tolist()
# #         h_pred = h_pred.squeeze().detach().cpu().tolist()
# #         assignments = assignment.argmax(dim=1)
# #         h_assignments = h_assignment.argmax(dim=1).tolist()
# #
# #         n = self.cost.shape[1]
# #         for i in range(n):
# #             self.batch[f"weight_valid_{i}"] = self.cost[:, i]
# #
# #         self.batch["cls_pred"] = cls_pred.tolist()
# #         self.batch["pred_idx"] = pred
# #         self.batch["assignments"] = assignments
# #         self.batch["h_assignments"] = h_assignments
# #         self.batch["h_pred_idx"] = h_pred
# #
# #         df = pd.DataFrame(self.batch)
# #         df.to_csv(self.output_dir / f"result_{self.mode}.csv")
# #         if self.conf_out.shape != torch.Size([0]):
# #             annotator_num = self.annotations.shape[1]
# #             for i in range(annotator_num):
# #                 confusion_i = self.conf_out[:, i].cpu().detach()
# #                 confusion_i = confusion_i.argmax(dim=1).numpy()
# #                 annotation_i = self.annotations[:, i].cpu().numpy()
# #                 annotator_i_acc = accuracy_score(annotation_i, confusion_i)
# #                 self.logger.log_metrics(
# #                     {f"val_annotator_{i+1}_accuracy": annotator_i_acc}
# #                 )
# #
# #     def gather_assignment(self, preds, assignment):
# #         assignment_idx = assignment.argmax(dim=1).to(self.device)
# #         ans = torch.gather(preds, 1, assignment_idx.unsqueeze(-1))
# #         return ans
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
# 
# 
# # class SpiralTrainerWithDifferentOptimization():
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
# #         dataset: str,
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
# #             dataset=dataset,
# #             loss=loss,
# #         )
# #         self.automatic_optimization = False
# #
# #     def on_fit_start(self):
# #         self.weight_model.init_device(self.device)
# #         self.logger.log_hyperparams({"seed": self.seed})
# #         self.logger.log_hyperparams({"alpha": self.loss.alpha})
# #         self.logger.log_hyperparams({"beta": self.loss.beta})
# #         self.logger.log_hyperparams({"learning_rate": self.learning_rate})
# #         self.logger.log_hyperparams({"weight_decay": self.weight_decay})
# #         self.logger.log_hyperparams({"output_dir": self.output_dir})
# #         self.logger.log_hyperparams({"hidden_dim": self.weight_model.hidden_dim})
# #         self.logger.log_hyperparams({"loss": self.loss.name})
# #         self.logger.log_hyperparams({"dataset": self.dataset})
# #         self.logger.log_hyperparams({"optim": "dual"})
# #
# #     def training_step(self, x):
# #         feature = x["feature"]
# #         label = x["label"]
# #         annotation = x["annotations"]
# #
# #         out = self.weight_model(feature)
# #         cls_out = out["cls_prob"]
# #         cls_logit = out["cls_logit"]
# #         conf_out = out["confusion_out"]
# #
# #         all_loss = self.loss(
# #             cls_out=cls_logit,
# #             ground_truth=label,
# #             confusion_out=conf_out,
# #             annotations=annotation,
# #             model=self.weight_model,
# #         )
# #
# #         loss = all_loss["loss"]
# #         confusion_loss = all_loss["confusion_loss"]
# #         cls_loss = all_loss["cls_loss"]
# #
# #         cls_optim, conf_optim = self.configure_optimizers()
# #
# #         if self.current_epoch % 2 == 0:
# #             cls_optim.zero_grad()
# #             self.manual_backward(cls_loss)
# #             cls_optim.step()
# #         else:
# #             conf_optim.zero_grad()
# #             self.manual_backward(confusion_loss)
# #             conf_optim.step()
# #
# #         self.log_data(
# #             {
# #                 "train_loss": loss.item(),
# #                 "train_confusion_loss": confusion_loss.item()
# #                 if confusion_loss is not None
# #                 else 0,
# #                 "train_cls_loss": cls_loss.item() if cls_loss is not None else 0,
# #             }
# #         )
# #         self.annotations = torch.cat([self.annotations, annotation])
# #         self.gold_label = torch.cat([self.gold_label, label])
# #         if conf_out is not None:
# #             self.conf_out = torch.cat([self.conf_out, conf_out])
# #         if cls_out is not None:
# #             self.cls_out = torch.cat([self.cls_out, cls_out])
# #         cost = self.weight_model.predicted_accuracy(feature)
# #         self.cost = (
# #             cost if self.cost.shape == (0,) else np.concatenate([self.cost, cost])
# #         )
# #
# #     def configure_optimizers(self):
# #         cls_optim = torch.optim.Adam(
# #             self.weight_model.classifier.parameters(),
# #             lr=self.learning_rate if self.mode == "train" else 0.0,
# #             weight_decay=self.weight_decay,
# #         )
# #
# #         conf_optim = torch.optim.Adam(
# #             self.weight_model.confusion_matrices.parameters(),
# #             lr=self.learning_rate if self.mode == "train" else 0.0,
# #             weight_decay=self.weight_decay,
# #         )
# #
# #         return cls_optim, conf_optim
# 
# 
# # class SpiralTrainerLearningToDefer(ModelTrainer):
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
# #         self.loss = loss
# #         self.out_dim = self.weight_model.out_dim
# #         self.val_losses = []
# #         self.min_val_loss = 10**1000
# #
# #     def training_step(self, x):
# #         feature = x["feature"]
# #         label = x["label"]
# #         annotations = x["annotations"]
# #         out = self.weight_model(feature)
# #         logit = out["cls_logit"]
# #         prob = out["cls_prob"]
# #
# #         loss = self.loss(prob, label, annotations)
# #         loss = loss["loss"]
# #         self.log_data(
# #             {
# #                 "train_loss": loss.item(),
# #             }
# #         )
# #         self.annotations = torch.cat([self.annotations, annotations])
# #         self.gold_label = torch.cat([self.gold_label, label])
# #         self.cls_out = torch.cat([self.cls_out, prob])
# #         self.assignments = torch.cat([self.assignments, logit.argmax(dim=1)])
# #
# #         return loss
# #
# #     def on_train_epoch_end(self):
# #         if self.current_epoch % self.assign_interval == 0:
# #             pred = torch.gather(
# #                 self.annotations, 1, self.assignments.unsqueeze(-1).to(torch.long)
# #             )
# #
# #             metrics = self.calc_metrics(self.gold_label, pred)
# #             assert self.logger is not None, f"logger is None in {self.__class__}"
# #             self.logger.log_metrics(
# #                 {
# #                     "train_accuracy": metrics[0],
# #                     "train_precision": metrics[1],
# #                     "train_recall": metrics[2],
# #                     "train_f1": metrics[3],
# #                 }
# #             )
# #
# #     def validation_step(self, x):
# #         feature = x["feature"]
# #         label = x["label"]
# #         annotations = x["annotations"]
# #
# #         out = self.weight_model(feature)
# #         logit = out["cls_logit"]
# #         prob = out["cls_prob"]
# #
# #         loss = self.loss(prob, label, annotations)
# #         loss = loss["loss"]
# #         self.log_data(
# #             {
# #                 "valid_loss": loss.item(),
# #             }
# #         )
# #         self.annotations = torch.cat([self.annotations, annotations])
# #         self.gold_label = torch.cat([self.gold_label, label])
# #         self.cls_out = torch.cat([self.cls_out, prob])
# #
# #         self.assignments = torch.cat([self.assignments, logit.argmax(dim=1)])
# #         self.val_losses.append(loss.item())
# #
# #         return loss
# #
# #     def on_validation_epoch_end(self):
# #         val_loss = np.mean(self.val_losses)
# #         if val_loss < self.min_val_loss:
# #             self.min_loss = val_loss
# #             self.save_model()
# #
# #         if self.current_epoch % self.assign_interval == 0:
# #             pred = torch.gather(
# #                 self.annotations, 1, self.assignments.unsqueeze(-1).to(torch.long)
# #             )
# #             metrics = self.calc_metrics(self.gold_label, pred)
# #             assert self.logger is not None, f"logger is None in {self.__class__}"
# #             self.logger.log_metrics(
# #                 {
# #                     "valid_accuracy": metrics[0],
# #                     "valid_precision": metrics[1],
# #                     "valid_recall": metrics[2],
# #                     "valid_f1": metrics[3],
# #                 }
# #             )
# #
# #     def predict_step(self, batch, _):
# #         feature = batch["feature"]
# #         label = batch["label"]
# #         annotations = batch["annotations"]
# #         out = self.weight_model(feature)
# #         logit = out["cls_logit"]
# #         prob = out["cls_prob"]
# #
# #         self.annotations = torch.cat([self.annotations, annotations])
# #         self.gold_label = torch.cat([self.gold_label, label])
# #         self.cls_out = torch.cat([self.cls_out, prob])
# #         self.assignments = torch.cat([self.assignments, logit.argmax(dim=1)])
# #
# #         for k, v in batch.items():
# #             self.batch[k].extend(v.tolist() if isinstance(v, torch.Tensor) else list(v))
# #
# #     def on_predict_epoch_end(self):
# #         pred = torch.gather(
# #             self.annotations, 1, self.assignments.unsqueeze(-1).to(torch.long)
# #         )
# #         metrics = self.calc_metrics(self.gold_label, pred)
# #         assert self.logger is not None, f"logger is None in {self.__class__}"
# #         data = {
# #             f"result_{self.mode}_accuracy": metrics[0],
# #             f"result_{self.mode}_precision": metrics[1],
# #             f"result_{self.mode}_recall": metrics[2],
# #             f"result_{self.mode}_f1": metrics[3],
# #         }
# #
# #         self.logger.log_metrics(data)
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
# #
# #
# # class SpiralICrowdTaskAssignment:
# #     def __init__(self) -> None:
# #         pass
# #
# #     def __call__(self):
# #         pass
# #
# #
# class SpiraliCrowd(iCrowdTrainer):
#     def get_labels(self, df):
#         return df.select(pl.col("label")).to_numpy()
# 
#     def get_predictions(self, df):
#         return df.select(pl.col("^pred_.*$")).to_numpy()
