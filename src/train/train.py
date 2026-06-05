import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import wandb
from config import Config
from hydra.utils import instantiate
from lightning.pytorch import Trainer, seed_everything
from model import MatchingBatchModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def train_common_confusions(config: Config):
    if config.debug:
        config.train.epoch = 3
        config.name = "debug"

    train_dataset = instantiate(config.dataset.train_dataset)
    valid_dataset = instantiate(config.dataset.valid_dataset)
    test_dataset = instantiate(config.dataset.test_dataset)

    train_dataloader = train_dataset.get_dataloader()
    valid_dataloader = valid_dataset.get_dataloader()
    test_dataloader = test_dataset.get_dataloader()

    if config.abci and "abci" not in config.name:
        config.name = config.name + "_abci"

    logger = instantiate(config.logger) if config.wandb_enabled else False
    modelTrainer = instantiate(config.trainer)

    if config.wandb_enabled:
        wandb.config["loss"] = modelTrainer.loss.name
    trainer = Trainer(max_epochs=config.train.epoch, logger=logger)

    trainer.fit(
        modelTrainer,
        train_dataloader,
        valid_dataloader,
    )
    # modelTrainer.mode = "train"
    # trainer.predict(modelTrainer, train_dataloader)
    modelTrainer.mode = "test"
    _ = trainer.predict(modelTrainer, test_dataloader)


class ModelTrainer(L.LightningModule):
    def __init__(
        self,
        output_dir: str,
        weight_model: nn.Module,
        matching_model: MatchingBatchModel,
        seed: int,
        learning_rate: float,
        weight_decay: float,
        loss,
        dataset_name: str,
        mode: str = "train",
        random_assignment: bool = True,
    ):
        self.mode = mode
        self.seed = seed
        self.random_assignment = random_assignment
        super().__init__()
        seed_everything(seed)
        if matching_model is not None:
            self.matching_model: MatchingBatchModel = matching_model
        if loss.name == "learning_to_defer":
            self.output_dir = Path(
                output_dir.replace("$loss", loss.name + "_assignment")
            )
        else:
            self.output_dir = Path(output_dir.replace("$loss", loss.name))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if weight_model is not None:
            self.weight_model: nn.Module = weight_model.to(self.device)
        if mode == "test":
            return

        self.min_valid_loss: float = 100000000
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_valid_acc: float = 0
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.loss = loss
        self.weight: np.ndarray = np.array([])
        self.assignments: torch.Tensor = torch.empty(0, 0)
        self.model_name = f"weight_model_{seed}.pth"
        self.model_path = self.output_dir / self.model_name

        self.dataset_name = dataset_name
        self.training_accuracies = []
        self.save_model()

    def forward(self, **kwargs):
        return self.weight_model(**kwargs)

    def on_fit_start(self):
        self.weight_model.init_device(self.device)
        self.save_model()

    def _init_variables(self):
        self.annotations = torch.tensor([]).to(self.device)
        self.conf_out = torch.tensor([]).to(self.device)
        self.cls_out = torch.tensor([]).to(self.device)
        self.gold_label = torch.tensor([]).to(self.device)
        self.assignments = torch.tensor([]).to(self.device)
        self.cost = np.array([])
        self.losses = []

    def on_train_epoch_start(self):
        self._init_variables()

    def training_step(self, batch: dict, _):
        emb = batch["embedding"].to(self.device)
        annotation = batch["annotations"].to(torch.long)
        ground_truth = batch["label"].to(torch.long)
        out = self.forward(**batch)

        cls_out = out["cls_logit"]
        confusion_out = out["confusion_out"]

        all_loss = self.loss(
            cls_out=cls_out,
            ground_truth=ground_truth,
            confusion_out=confusion_out,
            annotations=annotation,
            model=self.weight_model,
        )
        loss = all_loss["loss"]
        confusion_loss = (
            all_loss["confusion_loss"].item()
            if all_loss["confusion_loss"] is not None
            else None
        )
        cls_loss = (
            all_loss["cls_loss"].item() if all_loss["cls_loss"] is not None else None
        )
        self.log_data(
            {
                "train/loss": loss.item(),
                "train/confusion_loss": (
                    confusion_loss if confusion_loss is not None else 0
                ),
                "train/cls_loss": cls_loss if cls_loss is not None else 0,
            }
        )
        self.annotations = torch.cat([self.annotations, annotation])
        self.gold_label = torch.cat([self.gold_label, ground_truth.to(torch.long)])
        if cls_out is not None:
            self.cls_out = torch.cat([self.cls_out, cls_out])
        cost = self.weight_model.predicted_accuracy(**batch)
        self.cost = (
            cost if self.cost.shape == (0,) else np.concatenate([self.cost, cost])
        )

        return loss

    def on_train_epoch_end(self):
        if self.current_epoch == 0:
            for i in range(self.annotations.shape[1]):
                pred = self.annotations[:, i]

                acc = accuracy_score(self.gold_label.cpu(), pred.cpu())
                self.training_accuracies.append(acc)

    def on_validation_epoch_start(self):
        self._init_variables()

    def validation_step(self, batch: dict, _) -> None:
        annotation = batch["annotations"].to(torch.long)
        ground_truth = batch["label"].to(torch.long)

        out = self.forward(**batch)

        cls_out = out["cls_logit"]
        confusion_out = out["confusion_out"]
        all_loss = self.loss(
            cls_out=cls_out,
            ground_truth=ground_truth,
            confusion_out=confusion_out,
            annotations=annotation,
            model=self.weight_model,
        )

        loss = all_loss["loss"].item()
        confusion_loss = (
            all_loss["confusion_loss"].item()
            if all_loss["confusion_loss"] is not None
            else None
        )

        cls_loss = (
            all_loss["cls_loss"].item() if all_loss["cls_loss"] is not None else None
        )

        self.log_data(
            {
                "valid/loss": loss,
                "valid/confusion_loss": (
                    confusion_loss if confusion_loss is not None else 0
                ),
                "valid/cls_loss": cls_loss if cls_loss is not None else 0,
            }
        )
        self.losses.append(loss)
        self.annotations = torch.cat([self.annotations, annotation])
        self.gold_label = torch.cat([self.gold_label, ground_truth])
        if confusion_out is not None:
            self.conf_out = torch.cat([self.conf_out, confusion_out])
        if cls_out is not None:
            self.cls_out = torch.cat([self.cls_out, cls_out])
        cost = self.weight_model.predicted_accuracy(**batch)
        self.cost = (
            cost if self.cost.shape == (0,) else np.concatenate([self.cost, cost])
        )

    def on_validation_epoch_end(self):
        if self.min_valid_loss > np.mean(self.losses):
            self.save_model()
            self.min_valid_loss = np.mean(self.losses)

    def on_predict_epoch_start(self):
        self.annotations = torch.tensor([]).to(self.device)
        self.conf_out = torch.tensor([]).to(self.device)
        self.cls_out = torch.tensor([]).to(self.device)
        self.gold_label = torch.tensor([]).to(self.device)
        self.assignments = torch.tensor([]).to(self.device)
        self.cost = np.array([])
        self.importance = torch.tensor([]).to(self.device)
        self.batch = defaultdict(lambda: [])
        self.load_model()

    def predict_step(self, batch, _):
        self.load_model()
        out = self.forward(**batch)

        cls_out = out["cls_logit"]
        confusion_out = out["confusion_out"]
        importance = out["weight"]
        cost = self.weight_model.predicted_accuracy(**batch)

        self.annotations = torch.cat([self.annotations, batch["annotations"]])
        self.gold_label = torch.cat([self.gold_label, batch["label"]])
        if confusion_out is not None:
            self.conf_out = torch.cat([self.conf_out, confusion_out])
        if cls_out is not None:
            self.cls_out = torch.cat([self.cls_out, cls_out])
        if isinstance(importance, torch.Tensor):
            self.importance = torch.cat([self.importance, importance])
        self.cost = (
            cost if self.cost.shape == (0,) else np.concatenate([self.cost, cost])
        )

        for k, v in batch.items():
            if k not in self.batch.keys():
                self.batch[k] = v
                continue
            if isinstance(v, torch.Tensor):
                self.batch[k] = torch.cat([self.batch[k], v])
            elif isinstance(v, list):
                self.batch[k].extend(v)

    def on_predict_epoch_end(self):
        assert self.cost.shape != (0,), "cost is empty"

        print("assignment by proposed method")

        # ここから実行時間を計測する
        start_time = time.process_time()
        assignment, _ = self.matching_model(self.cost)
        end_time = time.process_time()
        process_time = end_time - start_time
        print(f"matching model execution time: {end_time - start_time} seconds")
        pred = self.gather_assignment(self.annotations, assignment)
        prediction_metrics = self.calc_metrics(pred, self.gold_label)
        self.batch["assigned_idx"] = assignment.argmax(dim=1).tolist()
        self.batch["assigned_pred"] = pred.squeeze().tolist()
        if isinstance(self.batch["idx"], torch.Tensor):
            self.batch["idx"] = self.batch["idx"].tolist()
        self.batch["label"] = self.batch["label"].tolist()
        del self.batch["annotations"]
        del self.batch["embedding"]
        gc.collect()

        accs = np.array([self.training_accuracies])
        accs = np.repeat(accs, self.annotations.shape[0], axis=0)

        print("assignment by AW")
        h_assignment, _ = self.matching_model(accs)
        h_pred = self.gather_assignment(self.annotations, h_assignment)
        h_metrics = self.calc_metrics(self.gold_label, h_pred)

        data = {}
        data["run_time(s)"] = process_time
        idx = {
            0: "accuracy",
            1: "precision",
            2: "recall",
            3: "f1",
        }
        # random_metrics, _ = self.random_model_predict(self.annotations, self.gold_label)
        random_metrics = None
        for i, v in enumerate(assignment.sum(dim=0).tolist()):
            data[f"{self.mode}/assigned_annotator_{i}"] = v
        if random_metrics is not None:
            for i, v in enumerate(random_metrics):
                data[f"{self.mode}/random_{idx[i]}"] = v

        for i, v in enumerate(prediction_metrics):
            data[f"{self.mode}/{idx[i]}"] = v

        for i, v in enumerate(h_metrics):
            data[f"{self.mode}/heuristic_{idx[i]}"] = v
        data["index"] = 0
        print(data)
        self.log_table(
            key=f"{self.mode}/score", dataframe=pd.DataFrame(data, index=["index"])
        )
        with open(f"{self.output_dir}/score_{self.mode}.json", "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        if self.cls_out.shape != torch.Size([0]):
            cls_pred = self.cls_out.argmax(dim=1)
            acc = accuracy_score(self.gold_label.cpu().numpy(), cls_pred.cpu().numpy())
            data[f"{self.mode}/cls_accuracy"] = acc

        for k in self.batch.keys():
            if not isinstance(self.batch[k], torch.Tensor):
                continue
            self.batch[k] = self.batch[k].cpu().numpy()

        if "cls_out" in self.batch.keys():
            del self.batch["cls_out"]

        df = pd.DataFrame(self.batch)
        print(df.head())
        self.log_table(
            key=f"{self.mode}/result", columns=df.columns, dataframe=df
        )

        self.log_metrics(data)
        self.scores = data

    def gather_assignment(self, preds, assignment):
        assignment_idx = assignment.argmax(dim=1).to(self.device)
        ans = torch.gather(preds, 1, assignment_idx.unsqueeze(-1))
        return ans

    # def save_model_weights(self, batch: dict, weight: torch.Tensor):
    #     system_pred = batch["system_pred"]
    #     correct_pred = batch["correct"]
    #     for i in range(weight.shape[1]):
    #         weight_data: dict[str, list[float]] = {
    #             f"system_{i}_wrong_prob": [],
    #             f"system_{i}_correct_prob": [],
    #             f"system_{i}_pred": [],
    #             "correct": [],
    #         }
    #         w_i = weight[i]
    #
    #         system_pred_i = system_pred[:, i]
    #         data_num, _ = w_i.shape
    #         for d_num in range(data_num):
    #             weight_data[f"system_{i}_wrong_prob"].append(w_i[d_num, 0].item())
    #             weight_data[f"system_{i}_correct_prob"].append(w_i[d_num, 1].item())
    #             weight_data[f"system_{i}_pred"].append(system_pred_i[d_num].item())
    #             weight_data["correct"].append(correct_pred[d_num].item())
    #         df = pd.DataFrame(weight_data)
    #         df.to_csv(f"{self.output_dir}/weight_{i}_{self.mode}.csv", index=False)

    def save_model(self):
        torch.save(self.weight_model.state_dict(), self.model_path)

    def load_model(self):
        self.weight_model.load_state_dict(
            torch.load(self.model_path, weights_only=True)
        )

    def loss(self):
        raise NotImplementedError

    def calc_metrics(self, ans, result) -> tuple[float, float, float, float]:
        if isinstance(ans, torch.Tensor):
            ans = ans.cpu()
        if isinstance(result, torch.Tensor):
            result = result.cpu()

        acc = accuracy_score(ans, result)
        pre = precision_score(ans, result, zero_division=0, average="macro")
        recall = recall_score(ans, result, zero_division=0, average="macro")
        f1 = f1_score(ans, result, zero_division=0, average="macro")

        return (acc, pre, recall, f1)

    # def random_model_predict(self, predictions, ground_truth):
    #     random_metrics = []
    #
    #     for i in tqdm(range(100), total=100, desc="random model predict"):
    #         match self.matching_model:
    #             case MaximumNumberConstraint():
    #                 res = self.random_select(predictions)
    #             case CostConstraint():
    #                 res = self.cost_select(predictions)
    #             case BothConstraint():
    #                 # res = self.both_select(predictions)
    #                 return None, None
    #
    #         random_met = self.calc_metrics(ground_truth, res)
    #         random_metrics.append(random_met)
    #     random_metrics_mean = np.mean(random_metrics, axis=0)
    #     random_metrics_std = np.std(random_metrics, axis=0)
    #     return random_metrics_mean, random_metrics_std

    # def both_select(self, predictions):
    #     target_count = self.matching_model.assign_num
    #     total_cost = self.matching_model.total_cost_per_annotator
    #     cost_per_task = self.matching_model.cost_per_annotator
    #     annotator_num = self.matching_model.annotator_num
    #     annotator_idx = [i for i in range(annotator_num)]
    #     assigned_cost = [0] * annotator_num
    #     assigned_num = [0] * annotator_num
    #     assign_ok = [True] * annotator_num
    #     data_num = predictions.shape[0]
    #     model_ans = []
    #     if isinstance(predictions, torch.Tensor):
    #         predictions = predictions.tolist()
    #     for i in range(data_num):
    #         annotator = np.random.choice(annotator_idx).item()
    #         while not assign_ok[annotator]:
    #             annotator = (annotator + 1) % annotator_num
    #
    #         assigned_cost[annotator] += cost_per_task[annotator]
    #         assigned_num[annotator] += 1
    #         model_ans.append(predictions[i][annotator])
    #         if (
    #             assigned_cost[annotator] >= total_cost[annotator]
    #             or assigned_num[annotator] > target_count[annotator]
    #         ):
    #             assign_ok[annotator] = False
    #
    #     return model_ans

    # def cost_select(self, predictions):
    #     total_cost = self.matching_model.total_cost_per_annotator
    #     cost_per_task = self.matching_model.cost_per_annotator
    #     annotator_num = self.matching_model.annotator_num
    #     annotator_idx = [i for i in range(annotator_num)]
    #     assigned_cost = [0] * annotator_num
    #     assign_ok = [True] * annotator_num
    #     data_num = predictions.shape[0]
    #
    #     model_ans = []
    #     if isinstance(predictions, torch.Tensor):
    #         predictions = predictions.tolist()
    #     for i in range(data_num):
    #         annotator = np.random.choice(annotator_idx).item()
    #         while not assign_ok[annotator]:
    #             annotator = (annotator + 1) % annotator_num
    #
    #         assigned_cost[annotator] += cost_per_task[annotator]
    #         model_ans.append(predictions[i][annotator])
    #         if assigned_cost[annotator] >= total_cost[annotator]:
    #             assign_ok[annotator] = False
    #
    #     return model_ans

    # def random_select(self, predictions):
    #     target_count = self.matching_model.get_assign_num()
    #     if isinstance(predictions, torch.Tensor):
    #         predictions = predictions.tolist()
    #
    #     model_ans = []
    #     indexes = [i for i in range(len(predictions))]
    #     sample_idxes = []
    #
    #     for i in range(len(target_count)):
    #         sample_idx = random.sample(indexes, target_count[i])
    #         sample_idxes.append(set(sample_idx))
    #         for sampled_idx in sample_idx:
    #             indexes.remove(sampled_idx)
    #
    #     for i in range(len(predictions)):
    #         for j in range(len(predictions[i])):
    #             if i not in sample_idxes[j]:
    #                 continue
    #             model_ans.append(predictions[i][j])
    #
    #     assert len(model_ans) == len(predictions)
    #     return model_ans

    def log_data(self, data: dict):
        if self.logger is not None:
            self.log_dict(data, on_epoch=True)

    def log_metrics(self, data: dict):
        if self.logger is not None:
            self.logger.log_metrics(data)

    def log_table(self, *args, **kwargs):
        if self.logger is not None and hasattr(self.logger, "log_table"):
            self.logger.log_table(*args, **kwargs)

    def configure_optimizers(self):
        if self.weight_model is None:
            return None
        optimizer = torch.optim.Adam(
            self.weight_model.parameters(),
            lr=self.learning_rate if self.mode == "train" else 0.0,
            weight_decay=self.weight_decay,
        )
        return optimizer


class ModelTrainerLearningToDefer(ModelTrainer):
    def __init__(
        self,
        weight_model: nn.Module,
        learning_rate: float,
        weight_decay: float,
        mode: str,
        seed: int,
        output_dir: Path,
        dataset_name: str,
        loss,
    ) -> None:
        super().__init__(
            weight_model=weight_model,
            matching_model=None,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            mode=mode,
            seed=seed,
            output_dir=output_dir,
            dataset_name=dataset_name,
            loss=loss,
        )
        self.loss = loss
        self.out_dim = self.weight_model.out_dim
        self.val_losses = []
        self.min_val_loss = 10**100

    def on_fit_start(self):
        super().on_fit_start()

    def training_step(self, x):
        feature = x["embedding"]
        label = x["label"]
        annotations = x["annotations"]
        out = self.weight_model(**x)
        logit = out["cls_logit"]
        prob = out["cls_prob"]
        loss = self.loss(prob, label, annotations)
        loss = loss["loss"]
        self.log_data(
            {
                "train/loss": loss.item(),
            }
        )
        self.annotations = torch.cat([self.annotations, annotations])
        self.gold_label = torch.cat([self.gold_label, label])
        self.cls_out = torch.cat([self.cls_out, prob])
        self.assignments = torch.cat([self.assignments, logit.argmax(dim=1)])

        return loss

    # def on_train_epoch_end(self):
    #     if self.current_epoch % self.assign_interval == 0:
    #         pred = torch.gather(
    #             self.annotations, 1, self.assignments.unsqueeze(-1).to(torch.long)
    #         )

    #         metrics = self.calc_metrics(self.gold_label, pred)
    #         assert self.logger is not None, f"logger is None in {self.__class__}"
    #         self.logger.log_metrics(
    #             {
    #                 "train/accuracy": metrics[0],
    #                 "train/precision": metrics[1],
    #                 "train/recall": metrics[2],
    #                 "train/f1": metrics[3],
    #             }
    #         )

    def validation_step(self, x):
        label = x["label"]
        annotations = x["annotations"]

        out = self.weight_model(**x)
        logit = out["cls_logit"]
        prob = out["cls_prob"]

        loss = self.loss(prob, label, annotations)
        loss = loss["loss"]
        self.log_data(
            {
                "valid/loss": loss.item(),
            }
        )
        self.annotations = torch.cat([self.annotations, annotations])
        self.gold_label = torch.cat([self.gold_label, label])
        self.cls_out = torch.cat([self.cls_out, prob])

        self.assignments = torch.cat([self.assignments, logit.argmax(dim=1)])
        self.val_losses.append(loss.item())

        return loss

    def on_validation_epoch_end(self):
        val_loss = np.mean(self.val_losses)

        if val_loss < self.min_val_loss:
            self.min_loss = val_loss
            self.save_model()

        # if self.current_epoch % self.assign_interval == 0:
        #     pred = torch.gather(
        #         self.annotations, 1, self.assignments.unsqueeze(-1).to(torch.long)
        #     )
        #     metrics = self.calc_metrics(self.gold_label, pred)
        #     assert self.logger is not None, f"logger is None in {self.__class__}"
        #     self.logger.log_metrics(
        #         {
        #             "valid/accuracy": metrics[0],
        #             "valid/precision": metrics[1],
        #             "valid/recall": metrics[2],
        #             "valid/f1": metrics[3],
        #         }
        #     )

    def predict_step(self, batch, _):
        feature = batch["embedding"]
        label = batch["label"]
        annotations = batch["annotations"]
        out = self.weight_model(**batch)
        logit = out["cls_logit"]
        prob = out["cls_prob"]

        self.annotations = torch.cat([self.annotations, annotations])
        self.gold_label = torch.cat([self.gold_label, label])
        self.cls_out = torch.cat([self.cls_out, prob])
        self.assignments = torch.cat([self.assignments, logit.argmax(dim=1)])

        for k, v in batch.items():
            self.batch[k].extend(v.tolist() if isinstance(v, torch.Tensor) else list(v))

    def on_predict_epoch_end(self):
        pred = torch.gather(
            self.annotations, 1, self.assignments.unsqueeze(-1).to(torch.long)
        )
        metrics = self.calc_metrics(self.gold_label, pred)
        data = {
            f"{self.mode}/accuracy": metrics[0],
            f"{self.mode}/precision": metrics[1],
            f"{self.mode}/recall": metrics[2],
            f"{self.mode}/f1": metrics[3],
        }

        self.log_metrics(data)
        # 制約違反の評価
        task_num = self.assignments.shape[0]
        ratio = [1 / self.out_dim for _ in range(self.out_dim)]
        maximum_assign_num = [int(task_num * ratio) for ratio in ratio]

        i = 0
        while sum(maximum_assign_num) < task_num:
            maximum_assign_num[i % len(maximum_assign_num)] += 1
            i += 1

        assigned_num = defaultdict(int)
        assignments = self.assignments.tolist()

        for assignmtnt in assignments:
            assigned_num[assignmtnt] += 1

        violations = []
        for i, maximum in enumerate(maximum_assign_num):
            violations.append(max(assigned_num[i] - maximum, 0))

        violations_data = {}
        violation_ratios = []
        for i, vi in enumerate(violations):
            violations_data[f"{self.mode}/violation_annotator_{i}"] = vi
            violations_data[f"{self.mode}/violation_ratio_annotator_{i}"] = (
                assigned_num[i] / maximum_assign_num[i]
            )
            violation_ratios.append(assigned_num[i] / maximum_assign_num[i])
        for k, v in assigned_num.items():
            k = int(k)
            violations_data[f"{self.mode}/assigned_annotator_{k}"] = v

        violations_data[f"{self.mode}/total_constraint_violation"] = sum(violations)
        violations_data[f"{self.mode}/total_violation_ratio"] = np.mean(
            violation_ratios
        )
        self.log_metrics(violations_data)


# class ClassificationTrainer(L.LightningModule):
#     def __init__(
#         self,
#         weight_model: nn.Module,
#         learning_rate: float,
#         weight_decay: float,
#         seed: int,
#         output_dir: str,
#         dataset_name: str,
#     ) -> None:
#         super().__init__()
#         seed_everything(seed)
#         self.model = weight_model
#         self.criterion = nn.CrossEntropyLoss()
#         self.learning_rate = learning_rate
#         self.weight_decay = weight_decay
#         self.dataset_name = dataset_name
#
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#
#         self.mode = "train"
#
#     def forward(self, **kwargs):
#         return self.model(**kwargs)
#
#     def training_step(self, batch, _):
#         out = self.forward(**batch)
#         logit = out["cls_logit"]
#         prob = out["cls_prob"]
#         loss = self.loss(logit, batch["label"])
#         pred = logit.argmax(dim=1)
#
#         acc = accuracy_score(batch["label"].detach().cpu(), pred.detach().cpu())
#         self.log_dict({"train_loss": loss, "train_accuracy": acc}, on_epoch=True)
#         return loss
#
#     def validation_step(self, batch, _):
#         out = self.forward(**batch)
#         logit = out["cls_logit"]
#         prob = out["cls_prob"]
#         loss = self.loss(logit, batch["label"])
#         pred = logit.argmax(dim=1)
#
#         acc = accuracy_score(batch["label"].detach().cpu(), pred.detach().cpu())
#         self.log_dict({"valid_loss": loss, "valid_accuracy": acc}, on_epoch=True)
#         return loss
#
#     def on_predict_epoch_start(self):
#         self.label = torch.tensor([]).to(self.device)
#         self.feature = torch.tensor([]).to(self.device)
#         self.pred = torch.tensor([]).to(self.device)
#
#     def predict_step(self, batch, _):
#         out = self.forward(**batch)
#         logit = out["cls_logit"]
#         prob = out["cls_prob"]
#         pred = logit.argmax(dim=1)
#         self.label = torch.cat([self.label, batch["label"]])
#         self.feature = torch.cat([self.feature, batch["embedding"]])
#         self.pred = torch.cat([self.pred, pred])
#
#     def on_predict_epoch_end(self):
#         xy = self.feature.cpu().numpy()
#         x = xy[:, 0]
#         y = xy[:, 1]
#         label = self.label.cpu().numpy()
#
#         df = pl.DataFrame(
#             {"x": x, "y": y, "label": label, "pred": self.pred.cpu().numpy()}
#         )
#         df.write_csv(self.output_dir / f"result_{self.mode}.csv")
#         acc = accuracy_score(self.label.detach().cpu(), self.pred.detach().cpu())
#         self.logger.log_metrics({f"result_{self.mode}_accuracy": acc})
#
#     def loss(self, out, label):
#         return self.criterion(out, label)
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(
#             self.model.parameters(),
#             lr=self.learning_rate,
#             weight_decay=self.weight_decay,
#         )
#         return optimizer
