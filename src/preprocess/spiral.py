# reference : https://github.com/oreilly-japan/deep-learning-from-scratch-2/blob/master/dataset/spiral.py
# coding: utf-8
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split


class SpiralPreprocessTemplate:
    def generate_artificial_labels(
        self, df: pl.DataFrame, CLS_NUM: int, annot_num: int
    ):
        class_idx = [i for i in range(CLS_NUM)]
        t = df.select("label").to_numpy().flatten()
        for i in range(annot_num):
            dist = self.accs[i]
            annot_i, filtered_annot_i = [], []
            for ti in t:
                acc = dist[ti]
                sampling_dist = np.array([(1 - acc) / (CLS_NUM - 1)] * CLS_NUM)
                sampling_dist[ti] = acc
                assert np.allclose(np.sum(sampling_dist), 1), (
                    f"the calculation of the distribution is failed {sampling_dist}"
                )
                anno = np.random.choice(class_idx, p=sampling_dist)
                annot_i.append(anno)

            df = df.with_columns(pl.Series(name=f"pred_{i}", values=annot_i))

        return df

    def generate_spiral_data(self, N, CLS_NUM, DIM):
        x = np.zeros((N * CLS_NUM, DIM))
        t = np.zeros((N * CLS_NUM), dtype=np.int32)

        for j in range(CLS_NUM):
            for i in range(N):  # N*j, N*(j+1)):
                rate = i / N
                radius = 1.0 * rate
                theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.7

                ix = N * j + i
                x[ix] = np.array(
                    [radius * np.sin(theta), radius * np.cos(theta)]
                ).flatten()
                t[ix] = j
        return x, t

    def generate_spiral_area_data(self, N, CLS_NUM, DIM):
        x = np.zeros((N * CLS_NUM, DIM))
        t = np.zeros((N * CLS_NUM), dtype=np.int32)

        def calcClassByTheta(theta):
            c = -1
            deg = np.degrees(theta) % 360
            if 90 < deg <= 210:
                c = 0
            elif 210 < deg <= 330:
                c = 1
            else:
                c = 2
            assert c != -1, f"theta {theta} is not contain in this class"
            return c

        for j in range(CLS_NUM):
            for i in range(N):  # N*j, N*(j+1)):
                rate = i / N
                radius = 1.0 * rate
                theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2

                ix = N * j + i
                x[ix] = np.array(
                    [radius * np.sin(theta), radius * np.cos(theta)]
                ).flatten()
                t[ix] = calcClassByTheta(theta)
        return x, t


class SpiralPreprocessTaskAssignment(SpiralPreprocessTemplate):
    def __init__(self, system_num=3, filter_ratio: float = 0.5, **kwargs):
        super().__init__()
        self.accs = [[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]]
        self.system_num = system_num
        self.filter_ratio: float = filter_ratio

    def __call__(self, seed: int, debug: bool, N: int):
        np.random.seed(seed)
        DIM = 2  # データの要素数
        CLS_NUM = self.system_num  # クラス数
        x, t = self.generate_spiral_data(N, CLS_NUM, DIM)

        train_x, test_x, train_t, test_t = train_test_split(
            x, t, test_size=0.3, stratify=t
        )
        val_x, test_x, val_t, test_t = train_test_split(
            test_x, test_t, test_size=0.5, stratify=test_t
        )
        train_df = pl.DataFrame(
            {
                "x": train_x[:, 0],
                "y": train_x[:, 1],
                "label": train_t,
                "idx": [i for i in range(len(train_t))],
            }
        )
        test_df = pl.DataFrame(
            {
                "x": test_x[:, 0],
                "y": test_x[:, 1],
                "label": test_t,
                "idx": [i for i in range(len(test_t))],
            }
        )
        val_df = pl.DataFrame(
            {
                "x": val_x[:, 0],
                "y": val_x[:, 1],
                "label": val_t,
                "idx": [i for i in range(len(val_t))],
            }
        )

        train_df = self.generate_artificial_labels(train_df, CLS_NUM, self.system_num)
        val_df = self.generate_artificial_labels(val_df, CLS_NUM, self.system_num)
        test_df = self.generate_artificial_labels(test_df, CLS_NUM, self.system_num)

        self.save_data(train_df, val_df, test_df, seed, debug)

    def check_accs(self, df, cls_num):
        _d = pl.DataFrame()
        for i in range(cls_num):
            for j in range(cls_num):
                _t = (
                    df.filter(pl.col("label") == i)
                    .group_by(pl.col("label") == pl.col(f"pred_{j}"))
                    .agg(pl.len())
                    .with_columns(
                        acc=pl.when(pl.col("label")).then(pl.col("len"))
                        / pl.sum("len"),
                        gt=pl.lit(i),
                        annotator=pl.lit(j),
                    )
                    .filter(pl.col("label"))
                    .select(pl.col("gt", "acc", "annotator"))
                )
                _d = _d.vstack(_t)
        _d = _d.pivot(index="annotator", columns="gt", values="acc")
        accs = _d.to_numpy()[:, 1:]
        accs = np.nan_to_num(accs, 0)
        diff = abs(accs - self.accs)

        assert (diff < 0.1).all(), (
            f"the difference of accuracy is {diff} and accuracy is {accs}"
        )

    def save_data(self, train_df, val_df, test_df, seed, debug):
        Path("./data/spiral").mkdir(exist_ok=True, parents=True)
        prefix = f"./data/spiral/seed_{seed}"
        if debug:
            train_df.write_csv(prefix + "_train_debug.csv")
            val_df.write_csv(prefix + "_valid_debug.csv")
            test_df.write_csv(prefix + "_test_debug.csv")
        else:
            train_df.write_csv(prefix + "_train.csv")
            val_df.write_csv(prefix + "_valid.csv")
            test_df.write_csv(prefix + "_test.csv")


class SpiralPreprocessTaskAssignmentDataNum(SpiralPreprocessTaskAssignment):
    def __init__(self, debug: bool, test_data_num: int, annotator_num: int):
        super().__init__()
        self.debug = debug
        assert test_data_num in [
            1_000,
            5_000,
            10_000,
            30_000,
            50_000,
            100_000,
            1_000_000,
        ], f"test data num {test_data_num} is not supported"
        self.test_data_num = test_data_num
        self.annotator_num = annotator_num
        self.mode = ""

        match test_data_num:
            case 1_000:
                self.mode = "test_1k"
            case 5_000:
                self.mode = "test_5k"
            case 10_000:
                self.mode = "test_10k"
            case 30_000:
                self.mode = "test_30k"
            case 50_000:
                self.mode = "test_50k"
            case 100_000:
                self.mode = "test_100k"
            case 1_000_000:
                self.mode = "test_1m"
            case _:
                raise ValueError(f"test data num {test_data_num} is not supported")
        self.accs = np.zeros((self.annotator_num, 3))
        for i in range(annotator_num):
            for j in range(3):
                self.accs[i][j] = 0.9 if i % 3 == j else 0.1

    def __call__(self, seed: int, debug: bool):
        np.random.seed(seed)

        N = (3_500 + self.test_data_num) // 3
        # if N % self.system_num != 0:
        #     N += self.system_num - (N % self.system_num)
        DIM = 2  # データの要素数
        CLS_NUM = 3  # クラス数

        x, t = self.generate_spiral_data(N, CLS_NUM, DIM)

        train_x, test_x, train_t, test_t = train_test_split(
            x, t, test_size=self.test_data_num, stratify=t
        )
        train_x, val_x, train_t, val_t = train_test_split(
            train_x, train_t, train_size=3_000, stratify=train_t
        )
        assert train_x.shape[0] == 3_000, (
            f"train data num is not 3000 but {train_x.shape[0]}"
        )

        train_df = pl.DataFrame(
            {
                "x": train_x[:, 0],
                "y": train_x[:, 1],
                "label": train_t,
                "idx": [i for i in range(len(train_t))],
            }
        )
        test_df = pl.DataFrame(
            {
                "x": test_x[:, 0],
                "y": test_x[:, 1],
                "label": test_t,
                "idx": [i for i in range(len(test_t))],
            }
        )
        val_df = pl.DataFrame(
            {
                "x": val_x[:, 0],
                "y": val_x[:, 1],
                "label": val_t,
                "idx": [i for i in range(len(val_t))],
            }
        )

        assert test_df.shape[0] == self.test_data_num

        train_df = self.generate_artificial_labels(
            train_df, CLS_NUM, self.annotator_num
        )
        val_df = self.generate_artificial_labels(val_df, CLS_NUM, self.annotator_num)
        test_df = self.generate_artificial_labels(test_df, CLS_NUM, self.annotator_num)

        self.save_data(train_df, val_df, test_df, seed, debug)

    def save_data(self, train_df, val_df, test_df, seed, debug):
        Path("./data/spiral_data_num").mkdir(exist_ok=True, parents=True)
        prefix = f"./data/spiral_data_num/seed_{seed}_{self.mode}_{self.annotator_num}"
        if debug:
            train_df.write_csv(prefix + "_train_debug.csv")
            val_df.write_csv(prefix + "_valid_debug.csv")
            test_df.write_csv(prefix + "_test_debug.csv")
        else:
            train_df.write_csv(prefix + "_train.csv")
            val_df.write_csv(prefix + "_valid.csv")
            test_df.write_csv(prefix + "_test.csv")


# class SpiralOverrapPreprocessTaskAssignment(SpiralPreprocessTaskAssignment):
#     def __init__(self, debug, system_num=3):
#         self.accs = [[0.9, 0.9, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]]
#         self.system_num = system_num
#         self.debug = debug

#     def save_data(self, train_df, val_df, test_df, seed):
#         Path("./data/spiral_overrap").mkdir(exist_ok=True, parents=True)
#         if self.debug:
#             train_df.write_csv(f"./data/spiral_overrap/train_seed_{seed}_debug.csv")
#             val_df.write_csv(f"./data/spiral_overrap/valid_seed_{seed}_debug.csv")
#             test_df.write_csv(f"./data/spiral_overrap/test_seed_{seed}_debug.csv")
#         else:
#             train_df.write_csv(f"./data/spiral_overrap/train_seed_{seed}.csv")
#             val_df.write_csv(f"./data/spiral_overrap/valid_seed_{seed}.csv")
#             test_df.write_csv(f"./data/spiral_overrap/test_seed_{seed}.csv")


# class SpiralPreprocessTaskAssignmentDifferentSystemNum(SpiralPreprocessTaskAssignment):
#     def __init__(self, debug, system_num):
#         self.accs = np.zeros((system_num, 3))
#         for i in range(system_num):
#             for j in range(3):
#                 self.accs[i][j] = 0.9 if i % 3 == j else 0.1
#
#         self.system_num = system_num
#         self.debug = debug
#
#     def save_data(self, train_df, val_df, test_df, seed, debug):
#         Path("./data/spiral_different_N").mkdir(exist_ok=True, parents=True)
#         if self.debug:
#             train_df.write_csv(
#                 f"./data/spiral_different_N/train_system_num_{self.system_num}_seed_{seed}_debug.csv"
#             )
#             val_df.write_csv(
#                 f"./data/spiral_different_N/valid_system_num_{self.system_num}_seed_{seed}_debug.csv"
#             )
#             test_df.write_csv(
#                 f"./data/spiral_different_N/test_system_num_{self.system_num}_seed_{seed}_debug.csv"
#             )
#         else:
#             train_df.write_csv(
#                 f"./data/spiral_different_N/train_system_num_{self.system_num}_seed_{seed}.csv"
#             )
#             val_df.write_csv(
#                 f"./data/spiral_different_N/valid_system_num_{self.system_num}_seed_{seed}.csv"
#             )
#             test_df.write_csv(
#                 f"./data/spiral_different_N/test_system_num_{self.system_num}_seed_{seed}.csv"
#             )
#
#     def __call__(self, seed: int, debug: bool):
#         np.random.seed(seed)
#
#         N = 5000  # クラスごとのサンプル数
#         DIM = 2  # データの要素数
#         CLS_NUM = 3
#
#         x, t = self.generate_spiral_data(N, CLS_NUM, DIM)
#
#         train_x, test_x, train_t, test_t = train_test_split(
#             x, t, test_size=0.3, stratify=t
#         )
#         val_x, test_x, val_t, test_t = train_test_split(
#             test_x, test_t, test_size=0.5, stratify=test_t
#         )
#
#         train_df = pl.DataFrame(
#             {
#                 "x": train_x[:, 0],
#                 "y": train_x[:, 1],
#                 "label": train_t,
#                 "idx": [i for i in range(len(train_t))],
#             }
#         )
#         test_df = pl.DataFrame(
#             {
#                 "x": test_x[:, 0],
#                 "y": test_x[:, 1],
#                 "label": test_t,
#                 "idx": [i for i in range(len(test_t))],
#             }
#         )
#         val_df = pl.DataFrame(
#             {
#                 "x": val_x[:, 0],
#                 "y": val_x[:, 1],
#                 "label": val_t,
#                 "idx": [i for i in range(len(val_t))],
#             }
#         )
#
#         train_df = self.generate_artificial_labels(train_df, CLS_NUM, self.system_num)
#         val_df = self.generate_artificial_labels(val_df, CLS_NUM, self.system_num)
#         test_df = self.generate_artificial_labels(test_df, CLS_NUM, self.system_num)
#
#         self.save_data(train_df, val_df, test_df, seed, debug)
#
#
# class SpiralAreaPreprocessTaskAssignment(SpiralPreprocessTemplate):
#     def __init__(self, debug, system_num=3):
#         super().__init__()
#         self.accs = [[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]]
#         self.system_num = system_num
#         self.debug = debug
#
#     def __call__(self, seed, debug):
#         np.random.seed(seed)
#         N = 5000  # クラスごとのサンプル数
#         DIM = 2  # データの要素数
#         CLS_NUM = self.system_num  # クラス数
#         x, t = self.generate_spiral_area_data(N, CLS_NUM, DIM)
#         train_x, test_x, train_t, test_t = train_test_split(
#             x, t, test_size=0.2, stratify=t
#         )
#         val_x, test_x, val_t, test_t = train_test_split(
#             test_x, test_t, test_size=0.5, stratify=test_t
#         )
#
#         train_df = pl.DataFrame(
#             {
#                 "x": train_x[:, 0],
#                 "y": train_x[:, 1],
#                 "label": train_t,
#                 "idx": [i for i in range(len(train_t))],
#             }
#         )
#         test_df = pl.DataFrame(
#             {
#                 "x": test_x[:, 0],
#                 "y": test_x[:, 1],
#                 "label": test_t,
#                 "idx": [i for i in range(len(test_t))],
#             }
#         )
#         val_df = pl.DataFrame(
#             {
#                 "x": val_x[:, 0],
#                 "y": val_x[:, 1],
#                 "label": val_t,
#                 "idx": [i for i in range(len(val_t))],
#             }
#         )
#
#         train_df = self.generate_artificial_labels(train_df, CLS_NUM)
#         val_df = self.generate_artificial_labels(val_df, CLS_NUM)
#         test_df = self.generate_artificial_labels(test_df, CLS_NUM)
#
#         self.save_data(train_df, val_df, test_df, seed)
#
#     def save_data(self, train_df, val_df, test_df, seed):
#         Path("./data/spiral_area").mkdir(exist_ok=True, parents=True)
#         if self.debug:
#             train_df.write_csv(f"./data/spiral_area/train_seed_{seed}_debug.csv")
#             val_df.write_csv(f"./data/spiral_area/valid_seed_{seed}_debug.csv")
#             test_df.write_csv(f"./data/spiral_area/test_seed_{seed}_debug.csv")
#         else:
#             train_df.write_csv(f"./data/spiral_area/train_seed_{seed}.csv")
#             val_df.write_csv(f"./data/spiral_area/valid_seed_{seed}.csv")
#             test_df.write_csv(f"./data/spiral_area/test_seed_{seed}.csv")


# class SpiralAreaOverrapPreprocessTaskAssignment(SpiralPreprocessTaskAssignment):
#     def __init__(self, debug, system_num=3):
#         super().__init__(debug=debug)
#         self.accs = [[0.9, 0.9, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]]
#         self.system_num = system_num
#         self.debug = debug

#     def save_data(self, train_df, val_df, test_df, seed):
#         Path("./data/spiral_area_overrap").mkdir(exist_ok=True, parents=True)
#         if self.debug:
#             train_df.write_csv(
#                 f"./data/spiral_area_overrap/train_seed_{seed}_debug.csv"
#             )
#             val_df.write_csv(f"./data/spiral_area_overrap/valid_seed_{seed}_debug.csv")
#             test_df.write_csv(f"./data/spiral_area_overrap/test_seed_{seed}_debug.csv")
#         else:
#             train_df.write_csv(f"./data/spiral_area_overrap/train_seed_{seed}.csv")
#             val_df.write_csv(f"./data/spiral_area_overrap/valid_seed_{seed}.csv")
#             test_df.write_csv(f"./data/spiral_area_overrap/test_seed_{seed}.csv")


# class SpiralPreprocessClassification(SpiralPreprocessTemplate):
#     def __call__(self, seed=1984, save=True):
#         np.random.seed(seed)
#         N = 5000  # クラスごとのサンプル数
#         DIM = 2  # データの要素数
#         CLS_NUM = 3  # クラス数
#         x, t = self.generate_spiral_data(N, CLS_NUM, DIM)
#         train_x, test_x, train_y, test_y = train_test_split(
#             x, t, test_size=0.3, stratify=t
#         )
#         val_x, test_x, val_y, test_y = train_test_split(
#             test_x, test_y, test_size=0.5, stratify=test_y
#         )
#         train = np.hstack((train_x, train_y[:, np.newaxis]))
#         val = np.hstack((val_x, val_y[:, np.newaxis]))
#         test = np.hstack((test_x, test_y[:, np.newaxis]))
#         if save:
#             Path("./data/spiral_classification").mkdir(exist_ok=True, parents=True)
#             np.save(f"./data/spiral_classification/train_seed_{seed}.npy", train)
#             np.save(f"./data/spiral_classification/valid_seed_{seed}.npy", val)
#             np.save(f"./data/spiral_classification/test_seed_{seed}.npy", test)
#         return x, t


# class SpiralAreaPreprocessClassification(SpiralPreprocessTemplate):
#     def __call__(self, seed=100, save=True):
#         np.random.seed(seed)
#         N = 5000  # クラスごとのサンプル数
#         DIM = 2  # データの要素数
#         CLS_NUM = 3  # クラス数
#         x, t = self.generate_spiral_area_data(N, CLS_NUM, DIM)
#         train_x, test_x, train_y, test_y = train_test_split(
#             x, t, test_size=0.3, stratify=t
#         )
#         val_x, test_x, val_y, test_y = train_test_split(
#             test_x, test_y, test_size=0.5, stratify=test_y
#         )
#         train = np.hstack((train_x, train_y[:, np.newaxis]))
#         val = np.hstack((val_x, val_y[:, np.newaxis]))
#         test = np.hstack((test_x, test_y[:, np.newaxis]))
#         if save:
#             Path("./data/spiral_by_area_classification").mkdir(
#                 exist_ok=True, parents=True
#             )
#             np.save(
#                 f"./data/spiral_by_area_classification/train_seed_{seed}.npy", train
#             )
#             np.save(f"./data/spiral_by_area_classification/test_seed_{seed}.npy", test)
#             np.save(f"./data/spiral_by_area_classification/valid_seed_{seed}.npy", val)
#         return x, t
