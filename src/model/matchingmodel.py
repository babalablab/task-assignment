from typing import Union

import numpy as np
import torch
from pulp import PULP_CBC_CMD, LpBinary, LpMaximize, LpProblem, LpVariable, lpSum, value


class MatchingBatchModel:
    def __init__(
        self,
        annotator_num: int,
        seed: int = 10,
        mode="train",
    ):
        self.annotator_num = annotator_num
        self.seed = seed
        self.mode = mode
        self.task_num = None
        self.assign_num = None
        self.ratio: int

    def __call__(
        self, cost: np.ndarray, mode: str = "train"
    ) -> tuple[torch.Tensor, float]:
        task_num = cost.shape[0]

        x, objective = self.solve(mode, cost, task_num)

        return self.convert2tensor(x, task_num), objective

    def solve(self, mode: str, cost: np.ndarray, task_num: int) -> tuple[dict, float]:
        if self.mode == "test":
            return {"msg": "test"}, 0.0
        assert cost is not None, "costis empty"

        x = {
            (i, j): LpVariable(f"{i} {j}", cat=LpBinary)
            for i in range(task_num)
            for j in range(self.annotator_num)
        }  # 変数
        print("assignment started")
        m = LpProblem(sense=LpMaximize)  # 数理モデル
        solver = PULP_CBC_CMD(
            threads=32, timeLimit=300, options=[f"RandomS {self.seed}"]
        )

        assert cost.shape[0] == task_num, (
            f"cost shape {cost.shape} is not equal to task_num {task_num}"
        )

        assert cost.shape[1] == self.annotator_num, (
            f"cost shape {cost.shape[1]} is not equal to ratio {self.annotator_num}"
        )

        m += lpSum(
            [
                cost[i][j] * x[(i, j)]
                for i in range(task_num)
                for j in range(self.annotator_num)
            ]
        )  # 目的関数

        # set constraints
        self.set_constraint(m, x, task_num)

        m.solve(solver)
        v = value(m.objective)
        print("assignment finished")
        if isinstance(v, torch.Tensor):
            v = v.item()
        return x, v

    def set_constraint(self, m, x, task_num: int):
        raise NotImplementedError

    def convert2tensor(self, x, task_num):
        v = torch.tensor(
            [x[(0, j)].value() for j in range(self.annotator_num)]
        ).unsqueeze(dim=0)
        for i in range(1, task_num):
            vi = torch.tensor(
                [x[(i, j)].value() for j in range(self.annotator_num)]
            ).unsqueeze(dim=0)
            v = torch.cat([v, vi])
        return v

    def get_assign_num(self):
        return self.assign_num


# This may be fairness constraint
class MaximumNumberConstraint(MatchingBatchModel):
    def __init__(
        self,
        annotator_num: int,
        seed: int = 10,
        ratio: Union[None, list[float]] = None,
        mode="train",
    ):
        super().__init__(annotator_num, seed, mode)
        if ratio is None:
            self.ratio = [1 / annotator_num for _ in range(annotator_num)]

    def set_constraint(self, m, x, task_num):
        self.assign_num = [int(task_num * ratio) for ratio in self.ratio]

        i = 0
        assert task_num is not None, f"task_num is {task_num}"
        while sum(self.assign_num) < task_num:
            self.assign_num[i % len(self.assign_num)] += 1
            i += 1

        assert sum(self.assign_num) == task_num

        for i in range(task_num):
            m += lpSum([x[(i, j)] for j in range(self.annotator_num)]) == 1
        # 最大割り当て数の制約
        for j in range(len(self.assign_num)):
            m += lpSum(x[(i, j)] for i in range(task_num)) <= self.assign_num[j]


class CostConstraint(MatchingBatchModel):
    def __init__(
        self,
        annotator_num: int,
        cost_per_annotator: list[float],
        total_cost_per_annotator: list[float],
        seed: int = 10,
        mode="train",
    ):
        super().__init__(annotator_num, seed, mode)
        assert len(cost_per_annotator) == annotator_num, (
            f"cost_per_annotator length {len(cost_per_annotator)} is not equal to annotator_num {annotator_num}"
        )
        self.cost_per_annotator = cost_per_annotator
        self.total_cost_per_annotator = total_cost_per_annotator

    def set_constraint(self, m, x, task_num: int):
        for i in range(task_num):
            m += lpSum([x[(i, j)] for j in range(self.annotator_num)]) == 1
        for j in range(self.annotator_num):
            m += (
                lpSum(x[(i, j)] * self.cost_per_annotator[j] for i in range(task_num))
                <= self.total_cost_per_annotator[j]
            )


# class BothConstraint(MatchingBatchModel):
#     def __init__(
#         self,
#         annotator_num: int,
#         cost_per_annotator: list[float],
#         total_cost_per_annotator: list[float],
#         total_assign_num_per_annotator: list[int],
#         seed: int = 10,
#         mode="train",
#     ):
#         super().__init__(annotator_num, seed, mode)
#         self.cost_per_annotator = cost_per_annotator
#         self.total_cost_per_annotator = total_cost_per_annotator
#         self.assign_num = total_assign_num_per_annotator
#
#     def set_constraint(self, m, x, task_num: int):
#         for i in range(task_num):
#             m += lpSum([x[(i, j)] for j in range(self.annotator_num)]) == 1
#         # 最大割り当て数の制約
#         for j in range(self.annotator_num):
#             m += (
#                 lpSum(x[(i, j)] * self.cost_per_annotator[j] for i in range(task_num))
#                 <= self.total_cost_per_annotator[j]
#             )
#         for j in range(len(self.assign_num)):
#             m += lpSum(x[(i, j)] for i in range(task_num)) <= self.assign_num[j]
