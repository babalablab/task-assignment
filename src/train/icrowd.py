import polars as pl
import numpy as np
from sklearn.metrics import accuracy_score
from typing import Any
from model import MatchingBatchModel
from tqdm import tqdm


# class iCrowdTrainer:
#     def __init__(self, model, seed, logger, dataset_name) -> None:
#         self.model = model
#         self.seed = seed
#         self.system_num = self.model.system_num
#         self.logger = logger
#         self.logger.experiment.config["loss"] = "icrowd"
# 
#     def __call__(self, train_datset, test_dataset):
#         acc = self.model(train_datset, test_dataset)
#         train_acc = acc[1]
#         test_acc = acc[2]
#         res_acc, res_assignment = self.assign_tasks(test_acc)
#         self.logger.log_metrics({"test/accuracy": res_acc})
# 
#         # 制約違反の評価をする
#         annotator_num = test_acc.shape[1]
# 
#         violations_data = self.eval_violations(res_assignment, annotator_num)
#         self.logger.log_metrics(violations_data)
# 
#         # loggerを終了する
#         self.logger.finalize("success")
#         print("Done")
#         print(f"train_acc : {train_acc} test_acc : {res_acc}")
#         return {"train/acc": train_acc, "test/acc": test_acc}
# 
#     def eval_violations(self, assignment, annotator_num):
#         task_num = assignment.shape[0]
# 
#         ratio = [1 / annotator_num for _ in range(annotator_num)]
#         maximum_assign_num = [int(task_num * ratio) for ratio in ratio]
# 
#         i = 0
#         while sum(maximum_assign_num) < task_num:
#             maximum_assign_num[i % len(maximum_assign_num)] += 1
#             i += 1
# 
#         assigned_num = defaultdict(int)
#         assignment = np.array(assignment)
#         for idx in assignment:
#             assigned_num[idx] += 1
# 
#         violations = []
#         for i, maximum in enumerate(maximum_assign_num):
#             violations.append(max(assigned_num[i] - maximum, 0))
# 
#         violations_data = {}
#         violation_ratios = []
#         for i, vi in enumerate(violations):
#             violations_data[f"test/violation_annotator_{i}"] = vi
#             violations_data[f"test/violation_ratio_annotator_{i}"] = (
#                 assigned_num[i] / maximum_assign_num[i]
#             )
#             violation_ratios.append(assigned_num[i] / maximum_assign_num[i])
# 
#         for i, ai in assigned_num.items():
#             violations_data[f"test/assigned_annotator_{i}"] = ai
# 
#         violations_data["test/total_constraint_violation"] = sum(violations)
#         violations_data["test/total_violation_ratio"] = np.mean(violation_ratios)
#         return violations_data
# 
#     def assign_tasks(self, test_acc):
#         # 正解率が最も良いものに割り当る
#         predictions = self.model.test_annotations
#         assign_index = test_acc.argmax(1)
#         assigned_predictions = np.take_along_axis(
#             predictions, assign_index[:, np.newaxis], axis=1
#         )
#         ans = self.model.test_labels
#         acc = accuracy_score(ans, assigned_predictions)
#         return (acc, assign_index)
# 
#     def get_labels(self, df):
#         raise NotImplementedError
# 
#     def get_predictions(self, df):
#         raise NotImplementedError
# 
# 
# class NLPICrowdTrainer(iCrowdTrainer):
#     def get_labels(self, df):
#         return df.select(pl.col("label")).to_numpy()
# 
#     def get_predictions(self, df):
#         return df.select(pl.col("^annotator_.*$")).to_numpy()
# 
# 
# class SpiraliCrowdTrainer(iCrowdTrainer):
#     def get_labels(self, df):
#         return df.select(pl.col("label")).to_numpy()
# 
#     def get_predictions(self, df):
#         return df.select(pl.col("^pred_.*$")).to_numpy()
# 
# 
class iCrowdTaskAssignment:
    def __init__(
        self,
        matching_model: MatchingBatchModel,
        model: Any,
        seed: int,
        logger,
        dataset_name: str,
    ) -> None:
        self.model = model
        self.matching_model = matching_model
        self.seed = seed
        self.logger = logger
        self.train_df = model.train_df
        self.test_df = model.test_df
        assert self.matching_model.out_dim == self.model.system_num, (
            f"Matching model output dim {self.model.out_dim}"
            f"iCrowdTrainer system num {self.model.system_num}"
        )

        if self.logger is not None:
            self.logger.log_hyperparams({"method": "icrowd task assignment"})
            self.logger.log_hyperparams({"loss": "icrowd task assignment"})
            self.logger.log_hyperparams({"dataset": dataset_name})

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        accs = self.model()
        train_accs = accs[1]
        test_accs = accs[2]

        train_assignment, _ = self.matching_model(train_accs)
        train_assignment = train_assignment.argmax(1).cpu().numpy()
        test_assignment, _ = self.matching_model(test_accs)
        test_assignment = test_assignment.argmax(1).cpu().numpy()

        train_predictions = self.gather_prediction(self.train_df, train_assignment)
        test_predictions = self.gather_prediction(self.test_df, test_assignment)

        train_labels = self.get_labels(self.train_df)
        test_labels = self.get_labels(self.test_df)
        train_acc = accuracy_score(train_labels, train_predictions)
        test_acc = accuracy_score(test_labels, test_predictions)
        if self.logger is not None:
            self.logger.log_metrics(
                {"train_accuracy": train_acc, "test_accuracy": test_acc}
            )
            self.logger.finalize("success")

    def gather_prediction(self, df: pl.DataFrame, assignment: np.array):
        predictions = self.get_predictions(df)
        pred = []
        for i, assgn_idx in tqdm(
            enumerate(assignment), total=len(assignment), desc="Assigning tasks"
        ):
            pred.append(predictions[i][assgn_idx])
        return np.array(pred)

    def get_labels(self, df):
        raise NotImplementedError

    def get_predictions(self, df):
        raise NotImplementedError


class NLPICrowdTaskAssignment(iCrowdTaskAssignment):
    def get_labels(self, df):
        return df.select(pl.col("label")).to_numpy()

    def get_predictions(self, df):
        return df.select(pl.col("^annotator_.*$")).to_numpy()


# class SpiraliCrowdTaskAssignment(iCrowdTaskAssignment):
#     def get_labels(self, df):
#         return df.select(pl.col("label")).to_numpy()
# 
#     def get_predictions(self, df):
#         return df.select(pl.col("^pred_.*$")).to_numpy()
