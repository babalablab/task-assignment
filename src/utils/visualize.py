# # from bisect import bisect_left
# # from pathlib import Path
# 
# # import matplotlib.pyplot as plt
# # import mlflow
# # import numpy as np
# # import pandas as pd
# # import polars as pl
# # import seaborn as sns
# 
# 
# # def calc_ms(data):
# #     return np.mean(data), np.std(data)
# 
# 
# # def count_dist(df, b):
# #     system_count = [0] * (101)
# #     crowd_count = [0] * (101)
# #     annotator_count = [0] * (101)
# 
# #     for _, d in df.iterrows():
# #         sp, cp, ap = -1, -1, -1
# #         if "system_weight" in d.keys():
# #             s1 = d["system_weight"]
# #             sp = bisect_left(b, s1)
# 
# #         if "crowd_weight" in d.keys():
# #             c1 = d["crowd_weight"]
# #             cp = bisect_left(b, c1)
# #         if "annotator_weight" in d.keys():
# #             a1 = d["annotator_weight"]
# #             ap = bisect_left(b, a1)
# #         if sp != -1:
# #             system_count[sp] += 1
# #         if cp != -1:
# #             crowd_count[cp] += 1
# #         if ap != -1:
# #             annotator_count[ap] += 1
# #     return system_count, crowd_count, annotator_count
# 
# 
# # def visualize_dist(df: pd.DataFrame, mode: str, img_base_path: Path, logger):
# #     if not img_base_path.exists():
# #         img_base_path.mkdir(parents=True)
# #     # current_dpi = plt.rcParams["figure.dpi"]
# #     b: list = [i / 100 for i in range(101)]
# #     fig, ax = plt.subplots()
# #     system_count, crowd_count, annotator_count = count_dist(df, b)
# #     ax.plot(b, system_count, label="Machine")
# #     ax.plot(b, crowd_count, label="Non-expert")
# #     ax.plot(b, annotator_count, label="Expert")
# #     plt.xlabel("Model weight", fontsize=15)
# #     plt.ylabel("Frequency", fontsize=15)
# #     plt.legend()
# #     # if img_base_path is not None:
# #     #     plt.savefig(
# #     #         img_base_path / (f"dist_{mode}.png"),
# #     #         dpi=current_dpi * 2,
# #     #         bbox_inches="tight",
# #     #     )
# #     #     plt.savefig(
# #     #         img_base_path / (f"dist_{mode}.eps"),
# #     #         dpi=current_dpi * 2,
# #     #         bbox_inches="tight",
# #     #     )
# 
# #     logger.experiment.log_figure(logger.run_id, fig, f"dist_{mode}.png")
# #     plt.close()
# #     return fig
# 
# 
# # def calc_acc_by_bin(method: str, target: str, df: pl.DataFrame):
# #     df = df.filter(pl.col("assignment") == target)
# #     row_name = f"{method}_weight"
# #     percentiles = df.with_columns(
# #         pl.when(pl.col(row_name) < 0.1)
# #         .then(pl.lit(1))
# #         .when(pl.col(row_name) < 0.2)
# #         .then(pl.lit(2))
# #         .when(pl.col(row_name) < 0.3)
# #         .then(pl.lit(3))
# #         .when(pl.col(row_name) < 0.4)
# #         .then(pl.lit(4))
# #         .when(pl.col(row_name) < 0.5)
# #         .then(pl.lit(5))
# #         .when(pl.col(row_name) < 0.6)
# #         .then(pl.lit(6))
# #         .when(pl.col(row_name) < 0.7)
# #         .then(pl.lit(7))
# #         .when(pl.col(row_name) < 0.8)
# #         .then(pl.lit(8))
# #         .when(pl.col(row_name) < 0.9)
# #         .then(pl.lit(9))
# #         .when(pl.col(row_name) < 1)
# #         .then(pl.lit(10))
# #         .otherwise(-1)
# #         .alias(f"{method}_weight_percentile")
# #     )
# #     accs = []
# #     data_num = []
# #     for i in range(1, 11):
# #         if method != "annotator":
# #             correct = percentiles.filter(
# #                 (pl.col(f"{method}_weight_percentile") == i)
# #                 & (pl.col(f"{method}_dicision") == pl.col("annotator"))
# #             ).select(pl.count())
# #         else:
# #             correct = percentiles.filter(
# #                 (pl.col(f"{method}_weight_percentile") == i)
# #             ).select(pl.count())
# #         all = percentiles.filter((pl.col(f"{method}_weight_percentile") == i)).select(
# #             pl.count()
# #         )
# #         if all.item() == 0:
# #             accs.append(0)
# #         else:
# #             accs.append(correct.item() / all.item())
# #         data_num.append((correct.item(), all.item()))
# #     return accs, data_num
# 
# 
# # def visualize_acc_each_bin(df: pl.DataFrame, mode: str, output_path: Path):
# #     fig, axs = plt.subplots(3, 3)
# #     method = ["system", "crowd", "annotator"]
# #     if "correct" in df.columns:
# #         df = df.rename({"correct": "annotator"})
# #     for i in range(3):
# #         for j in range(3):
# #             # 1つめの引数が対象となる手法、2つめの引数が問いあわされた手法
# #             accs, data_num = calc_acc_by_bin(method[j], method[i], df=df)
# #             df_dict = {"bin": [i / 10 for i in range(1, 11)], "acc": accs}
# #             fi = sns.barplot(
# #                 pl.DataFrame(df_dict).to_pandas(), x="bin", y="acc", ax=axs[i][j]
# #             )
# #             axs[i][j].set_ylim([0, 1.1])
# #             fi.set(xlabel=None)
# #             fi.set(ylabel=None)
# #             axs[i][j].figure.set_size_inches(10, 10)
# #             if j == 0:
# #                 axs[i][j].set_ylabel(f"割り当先： {method[i]}")
# #             if i == 2:
# #                 axs[i][j].set_xlabel(f"正解率を測定した手法： {method[j]}")
# #     s = ""
# #     if mode == "train":
# #         s = "学習"
# #     elif mode == "val":
# #         s = "検証"
# #     else:
# #         s = "テスト"
# #     plt.title(f"{s}データにおけるシステムへの重みと正解率の関係", y=-0.3, x=-0.8)
# #     mlflow.log_figure(fig, f"acc_each_bin_{mode}.png")
# #     plt.close()
# 
# 
# # def visualize_assignment_count(df: pl.DataFrame, mode: str, logger):
# #     if "correct" in df.columns:
# #         df = df.rename({"correct": "annotator"})
# #     _df = df.with_columns(
# #         pl.when(
# #             (pl.col("annotator_dicision") == pl.col("system_dicision"))
# #             & (pl.col("annotator_dicision") == pl.col("crowd_dicision"))
# #         )
# #         .then(pl.lit(1))
# #         .when(
# #             (pl.col("annotator_dicision") != pl.col("system_dicision"))
# #             & (pl.col("annotator_dicision") == pl.col("crowd_dicision"))
# #         )
# #         .then(pl.lit(2))
# #         .when(
# #             (pl.col("annotator_dicision") == pl.col("system_dicision"))
# #             & (pl.col("annotator_dicision") != pl.col("crowd_dicision"))
# #         )
# #         .then(pl.lit(3))
# #         .when(
# #             (pl.col("annotator_dicision") != pl.col("system_dicision"))
# #             & (pl.col("annotator_dicision") != pl.col("crowd_dicision"))
# #         )
# #         .then(pl.lit(4))
# #         .alias("pat")
# #     )
# 
# #     _d = pl.concat(
# #         [
# #             _df.filter(pl.col("pat") == 1)
# #             .group_by("assignment")
# #             .agg(pl.count())
# #             .with_columns(pl.lit("n=3")),
# #             _df.filter(pl.col("pat") == 2)
# #             .group_by("assignment")
# #             .agg(pl.count())
# #             .with_columns(pl.lit("n=2(crowd)")),
# #             _df.filter(pl.col("pat") == 3)
# #             .group_by("assignment")
# #             .agg(pl.count())
# #             .with_columns(pl.lit("n=2(system)")),
# #             _df.filter(pl.col("pat") == 4)
# #             .group_by("assignment")
# #             .agg(pl.count())
# #             .with_columns(pl.lit("n=1(only annotator)")),
# #         ]
# #     )
# 
# #     ax = sns.barplot(_d.to_pandas(), x="literal", y="count", hue="assignment")
# #     plt.title(f"{mode}データにおける割り当ての分布")
# #     logger.experiment.log_figure(logger.run_id, ax.figure, f"./{mode}_dist_data.png")
# #     plt.close()
# 
# 
# # def gather_weight_df(path: Path, epoch: int):
# #     train_df = pl.DataFrame()
# #     test_df = pl.DataFrame()
# #     for epoch in range(0, epoch, 10):
# #         dtypes = {
# #             "idx": pl.Int64,
# #             "text": pl.StringCache,
# #             "attribute": pl.StringCache,
# #             "around_text": pl.StringCache,
# #             "system_dicision": pl.Boolean,
# #             "crowd_dicision": pl.Boolean,
# #             "annotator_dicision": pl.Boolean,
# #             "assignment": pl.StringCache,
# #             "assignment_ans": pl.Boolean,
# #             "system_weight": pl.Float64,
# #             "crowd_weight": pl.Float64,
# #         }
# #         d = pl.read_csv(
# #             path / f"train_weights_{epoch}.csv",
# #             dtypes=dtypes,
# #         )
# 
# #         d = d.with_columns(epoch=pl.lit(epoch))
# #         train_df = pl.concat((train_df, d))
# #         d = pl.read_csv(
# #             path / f"val_weights_{epoch}.csv",
# #             dtypes=dtypes,
# #         )
# #         d = d.with_columns(epoch=pl.lit(epoch))
# #         test_df = pl.concat((test_df, d))
# #     return train_df, test_df
# 
# 
# # def vis_weight_transition(df: pl.DataFrame, mode: str):
# #     system_df = df.filter(
# #         (pl.col("system_dicision") == pl.col("annotator_dicision"))
# #         & (pl.col("crowd_dicision") != pl.col("annotator_dicision"))
# #     )
# #     plt.clf()
# #     s = ""
# #     if mode == "train":
# #         s = "学習"
# #     else:
# #         s = "検証"
# #     ax = plt.axes()
# #     if "system_weight" in system_df.columns:
# #         ax = sns.lineplot(
# #             system_df.to_pandas(), x="epoch", y="system_weight", label="system", ax=ax
# #         )
# #     if "crowd_weight" in system_df.columns:
# #         sns.lineplot(
# #             system_df.to_pandas(), x="epoch", y="crowd_weight", label="crowd", ax=ax
# #         )
# #     if "annotator_weight" in system_df.columns:
# #         sns.lineplot(
# #             system_df.to_pandas(),
# #             x="epoch",
# #             y="annotator_weight",
# #             label="annotator",
# #             ax=ax,
# #         )
# #     plt.ylabel("weight")
# #     plt.title(f"{s}データにおけるシステムのみ正解しているデータにおける重みの推移")
# #     plt.legend()
# #     mlflow.log_figure(ax.figure, f"./{mode}_weight_epoch_system.png")
# 
# #     plt.clf()
# #     crowd_df = df.filter(
# #         (pl.col("system_dicision") != pl.col("annotator_dicision"))
# #         & (pl.col("crowd_dicision") == pl.col("annotator_dicision"))
# #     )
# #     ax = plt.axes()
# #     if "system_weight" in crowd_df.columns:
# #         ax = sns.lineplot(
# #             crowd_df.to_pandas(), x="epoch", y="system_weight", label="system", ax=ax
# #         )
# #     if "crowd_weight" in crowd_df.columns:
# #         ax = sns.lineplot(
# #             crowd_df.to_pandas(), x="epoch", y="crowd_weight", label="crowd", ax=ax
# #         )
# #     if "annotator_weight" in crowd_df.columns:
# #         ax = sns.lineplot(
# #             crowd_df.to_pandas(),
# #             x="epoch",
# #             y="annotator_weight",
# #             label="annotator",
# #             ax=ax,
# #         )
# #     plt.ylabel("weight")
# #     plt.title(
# #         f"{s}データにおけるクラウドワーカーのみ正解しているデータにおける重みの推移"
# #     )
# #     mlflow.log_figure(ax.figure, f"./{mode}_weight_epoch_crowd.png")
# 
# #     plt.clf()
# #     ax = plt.axes()
# #     both_df = df.filter(
# #         (pl.col("system_dicision") == pl.col("annotator_dicision"))
# #         & (pl.col("crowd_dicision") == pl.col("annotator_dicision"))
# #     )
# #     if "system_weight" in both_df.columns:
# #         ax = sns.lineplot(
# #             both_df.to_pandas(), x="epoch", y="system_weight", label="system", ax=ax
# #         )
# #     if "crowd_weight" in both_df.columns:
# #         ax = sns.lineplot(
# #             both_df.to_pandas(), x="epoch", y="crowd_weight", label="crowd", ax=ax
# #         )
# #     if "annotator_weight" in both_df.columns:
# #         ax = sns.lineplot(
# #             both_df.to_pandas(),
# #             x="epoch",
# #             y="annotator_weight",
# #             label="annotator",
# #             ax=ax,
# #         )
# #     plt.ylabel("weight")
# #     plt.legend()
# #     plt.title(
# #         f"{s}データにおけるシステムとクラウドワーカーが正解しているデータにおける重みの推移"
# #     )
# #     mlflow.log_figure(ax.figure, f"./{mode}_weight_epoch_crowd_system.png")
# 
# #     plt.clf()
# #     ax = plt.axes()
# #     not_both_df = df.filter(
# #         (pl.col("system_dicision") != pl.col("annotator_dicision"))
# #         & (pl.col("crowd_dicision") != pl.col("annotator_dicision"))
# #     )
# #     if "system_weight" in not_both_df.columns:
# #         ax = sns.lineplot(
# #             not_both_df.to_pandas(), x="epoch", y="system_weight", label="system", ax=ax
# #         )
# #     if "crowd_weight" in not_both_df.columns:
# #         ax = sns.lineplot(
# #             not_both_df.to_pandas(), x="epoch", y="crowd_weight", label="crwod", ax=ax
# #         )
# #     if "annotator_weight" in not_both_df.columns:
# #         ax = sns.lineplot(
# #             not_both_df.to_pandas(),
# #             x="epoch",
# #             y="annotator_weight",
# #             label="annotator",
# #             ax=ax,
# #         )
# #     plt.ylabel("weight")
# #     plt.title(
# #         f"{s}データにおけるシステムとクラウドワーカーが間違えているデータにおける重みの推移"
# #     )
# #     mlflow.log_figure(ax.figure, f"./{mode}_weight_epoch_not_crowd_system.png")
# #     plt.close()
