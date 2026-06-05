# import pathlib
# import random
# import re
# import shutil
# from typing import Any
# 
# import pandas as pd
# import polars as pl
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
# 
# from config import Config
# from preprocess.tokenizer import JanomeBpeTokenizer
# 
# MAX_LEN = 512
# tokenizer = JanomeBpeTokenizer("../model/codecs.txt")
# 
# 
# class BasePreprocess:
#     def make_dicision(self, n):
#         if n >= 3:
#             return True
#         else:
#             return False
# 
#     def read_text(self, base_path, id):
#         path = base_path + "{}.txt".format(str(id))
#         with open(path, mode="r") as f:
#             text = f.readlines()
#         return text
# 
#     def text2pos(self, text_list):
#         # リストの中から、開始indexと終了indexを探す
#         # これをするために、リストと開始文字の文字indexの対応づけをしないといけない
#         place_list = [[] for _ in range(len(text_list) + 1)]
#         x, y = 0, 0
#         for _text in text_list:
#             if not _text:
#                 y += 1
#                 continue
#             for t in _text:
#                 place_list[y].append(x)
#                 x += len(t)
#             x = 0
#             y += 1
#         return place_list
# 
#     def calc_leftx(
#         self,
#         text_list,
#         place_list,
#         tokenized_target_att,
#         text_offset_start_line,
#         text_offset_start_offset,
#     ):
#         left_x = 0
#         try:
#             left_x = place_list[text_offset_start_line].index(text_offset_start_offset)
#         except ValueError:
#             for i, _token in enumerate(text_list[text_offset_start_line]):
#                 if tokenized_target_att[0] not in _token:
#                     continue
#                 left_x = i
#         return left_x
# 
#     def calc_rightx(
#         self,
#         text_list,
#         left_x,
#         left_y,
#         tokenized_target_text,
#         text_offset_start_line,
#         text_offset_end_line,
#     ):
#         right_x = 0
#         # ここの処理をする（列をまたいだ場合の処理）
#         if text_offset_start_line == text_offset_end_line:
#             right_x = left_x + len(tokenized_target_text) - 1
#         else:
#             right_x = left_x
#             tmp_y = left_y
#             for _ in range(len(tokenized_target_text)):
#                 # print(len(text_list), tmp_y, tokenized_target_text)
#                 if right_x == len(text_list[tmp_y]) - 1:
#                     right_x = 0
#                     tmp_y += 1
#                 else:
#                     right_x += 1
#         return right_x
# 
#     def extract_left_text(self, around_text_list, N, left_x, left_y, text_list):
#         while N > 0:
#             N -= 1
#             if left_x < 0:
#                 # 文章の先頭についた時終了する
#                 if left_y == 0:
#                     break
#                 # 文章が存在しない列の時、一つ上の列に行く
#                 if not text_list[left_y]:
#                     left_y -= 1
#                     left_x = len(text_list[left_y]) - 1
#                     continue
#                 else:
#                     left_y -= 1
#                     left_x = len(text_list[left_y]) - 1
#             if text_list[left_y]:
#                 if text_list[left_y][left_x] == "▁":
#                     continue
#                 around_text_list.insert(0, text_list[left_y][left_x])
#             else:
#                 break
#             left_x -= 1
#         return around_text_list, N
# 
#     def extract_right_text(self, around_text_list, N, right_x, right_y, text_list):
#         while N > 0:
#             N -= 1
#             if right_x > len(text_list[right_y]) - 1:
#                 if right_y == len(text_list) - 1:
#                     break
#                 if not text_list[right_y]:
#                     right_y += 1
#                     continue
#                 else:
#                     right_x = 0
#                     right_y += 1
#             if text_list[right_y]:
#                 if text_list[right_y][right_x] == "▁":
#                     continue
#                 around_text_list.append(text_list[right_y][right_x])
#             else:
#                 right_y += 1
#                 continue
#             right_x += 1
#         return around_text_list
# 
#     def preprocess(self):
#         zip_path = pathlib.Path("./data/Location.zip")
#         print("unzip file")
#         if zip_path.exists():
#             shutil.unpack_archive(zip_path, "./data/")
#         else:
#             print("zip file is not found.")
#             exit()
# 
#         system_df = pl.read_csv("./data/system_df.csv")
# 
#         system_df = system_df.with_row_count()
#         around_df = system_df.select(
#             pl.col(
#                 "row_nr",
#                 "page_id",
#                 "attribute",
#                 "text_offset_start_line_id",
#                 "text_offset_start_offset",
#                 "text_text",
#                 "text_offset_end_line_id",
#                 "text_offset_end_offset",
#             )
#         )
# 
#         ids = around_df["page_id"].unique().to_list()
#         around_texts, keys = [], []
#         for id in tqdm(ids, total=len(ids), desc="extracting around texts"):
#             text = self.read_text("./data/Location/plain/Lake/", id)
#             text_list = []
#             for t in text:
#                 text_list.append(self.tokenize_text(t))
# 
#             place_list = self.text2pos(text_list)
# 
#             _t = around_df.filter(pl.col("page_id") == id)
#             for i in range(len(_t)):
#                 text_offset_start_line = _t["text_offset_start_line_id"][i]
#                 text_offset_start_offset = _t["text_offset_start_offset"][i]
#                 text_offset_end_line = _t["text_offset_end_line_id"][i]
#                 text_id = _t["row_nr"][i]
#                 target_text = _t["text_text"][i].replace("\n", "")
#                 target_attribute = _t["attribute"][i]
# 
#                 tokenized_target_text = self.tokenize_text(target_text)
#                 tokenized_target_att = self.tokenize_text(target_attribute)
#                 token_length = (
#                     len(tokenized_target_att) + len(tokenized_target_text) + 2
#                 )
# 
#                 left_x = self.calc_leftx(
#                     text_list,
#                     place_list,
#                     tokenized_target_att,
#                     text_offset_start_line,
#                     text_offset_start_offset,
#                 )
#                 left_y = text_offset_start_line
#                 right_y = text_offset_end_line
#                 right_x = self.calc_rightx(
#                     text_list,
#                     left_x,
#                     left_y,
#                     tokenized_target_text,
#                     text_offset_start_line,
#                     text_offset_end_line,
#                 )
# 
#                 around_text_list = tokenized_target_text
#                 around_text_num = (MAX_LEN - token_length) // 2
#                 # 左側のテキストを追加する
#                 N = around_text_num
#                 around_text_list, n = self.extract_left_text(
#                     around_text_list, N, left_x, left_y, text_list
#                 )
# 
#                 around_text_list = self.extract_right_text(
#                     around_text_list, N + n, right_x, right_y, text_list
#                 )
# 
#                 around_texts.append("".join(around_text_list))
#                 keys.append(text_id)
#         tmp_df = pl.from_dict({"row_nr": keys, "around_text": around_texts})
#         tmp_df = tmp_df.with_columns(pl.col("row_nr").cast(pl.UInt32))
#         result = around_df.join(tmp_df, on="row_nr")
#         result = result.select(pl.col("row_nr", "around_text"))
# 
#         result.to_pandas().to_csv("./data/surrounding_texts.csv")
# 
#         df = pl.read_csv("./data/crowd.csv")
#         df = df.select(df.columns[3:])
#         idx = [1, 2, 3, 4, 5]
#         columns = ["title", "url", "value", "attr", "res"]
#         data = pl.DataFrame()
#         for i in idx:
#             c = [x + "0" + str(i) for x in columns]
#             c_dict = {name: x for name, x in zip(c, columns)}
#             _d = df.select(c)
#             _d = _d.rename(c_dict)
#             data = pl.concat([data, _d])
#         urls = data["url"].unique()
#         results = []
#         for url in urls:
#             _d = data.filter(data["url"] == url)
#             n = {"yes": 0, "no": 0, "nan": 0}
#             n["url"] = url
#             for i in range(len(_d)):
#                 _di = _d[i]
#                 n[_di["res"].item()] += 1
#             results.append(n)
#         res_df = pl.DataFrame(results)
# 
#         def extract_idx(url):
#             match = re.search(r"/(\d+)\.html", url)
#             return int(match.group(1))
# 
#         data = data.drop("res").unique("url").join(res_df, on="url")
#         data = data.with_columns(
#             (data["url"].apply(extract_idx).cast(pl.UInt32)).alias("idx")
#         ).sort("idx")
# 
#         count_df = system_df.select(
#             pl.col("^title|system_.*|attribute|correct|row_nr$")
#         )
#         count_df = count_df.with_row_count("idx").fill_null(False)
#         count_df = count_df.with_columns(
#             pl.col("^system_.*$").cast(pl.Int64, strict=False)
#         )
#         count_df = count_df.with_columns(
#             pl.concat_list(pl.col("^system_.*$")).alias("system_count")
#         ).with_columns(pl.col("system_count").list.sum())
# 
#         count_df = count_df.select(
#             pl.col("idx", "title", "attribute", "system_count", "correct", "row_nr")
#         )
#         data = count_df.join(data, on="idx", how="inner")
#         data = data.join(result, on="row_nr", how="inner")
# 
#         d = data.filter(pl.col("attr") != "座標・緯度")
#         data = d.filter(pl.col("attr") != "座標・経度")
# 
#         df = data.to_pandas()
#         return df
# 
#     def __call__(self, config):
#         df = self.preprocess(config)
#         system = df[["idx", "system_count"]]
#         system["system_dicision"] = system["system_count"].apply(self.make_dicision)
# 
#         crowd = df[["idx", "yes"]]
#         crowd["crowd_dicision"] = crowd["yes"].apply(lambda x: x >= 7)
# 
#         df = pd.merge(df, system, on="idx")
#         df = pd.merge(df, crowd, on="idx")
# 
#         df = df.rename(columns={"value": "text", "correct": "annotator"})
# 
#         train_df, validate = train_test_split(
#             df, test_size=0.2, stratify=df["attribute"], random_state=config.seed
#         )
#         validate_df, test_df = train_test_split(
#             validate,
#             test_size=0.5,
#             stratify=validate["attribute"],
#             random_state=config.seed,
#         )
#         seed = config.seed
#         train_df = train_df.reset_index()
#         validate_df = validate_df.reset_index()
#         test_df = test_df.reset_index()
# 
#         train_df.to_csv(f"./data/shinra_train_seed_{seed}.csv")
#         validate_df.to_csv(f"./data/shinra_validate_seed_{seed}.csv")
#         test_df.to_csv(f"./data/shinra_test_seed_{seed}.csv")
# 
#         df.to_csv(f"./data/data_seed_{seed}.csv", index=False)
# 
#     """
#     print("system")
#     system_score = calc_metrics(df["annotator"], df["system_dicision"])
#     system_score["kind"] = "system"
#     print("crowd")
# 
#     crowd_score = calc_metrics(df["annotator"], df["crowd_dicision"])
#     crowd_score["kind"] = "crowd"
#     print("annotator")
#     annotator_score = calc_metrics(df["annotator"], df["annotator"])
#     annotator_score["kind"] = "annotator"
#     scores = pd.DataFrame([system_score, crowd_score, annotator_score])
#     scores.to_csv(f"./outputs/only_scores_seed_{seed}.csv", index=False)
#     """
# 
#     def tokenize_text(self, text):
#         text = str(text)
#         return tokenizer.tokenize(text)[0]
# 
#     def remove_return(self, s):
#         s = str(s)
#         return s.replace("\n", "")
# 
#     def calc_metrics(self, ans, out):
#         acc = accuracy_score(ans, out)
#         pre = precision_score(ans, out, zero_division=0)
#         recall = recall_score(ans, out)
#         f1 = f1_score(ans, out)
#         print(
#             "accuracy: {:.3}, f1: {:.3}, precision: {:.3}, recall: {:.3}".format(
#                 acc, f1, pre, recall
#             )
#         )
#         return {"accuracy": acc, "f1": f1, "recall": recall, "precision": pre}
# 
# 
# class MultiSystemPreprocess(BasePreprocess):
#     def __call__(self, seed, system_num):
#         _ = self.preprocess()
#         system_df = pl.read_csv("./data/system_df.csv")
# 
#         system_df = system_df.with_row_count()
#         system_df = system_df.fill_null(False)
# 
#         system_df = system_df.select(
#             pl.col("^title|system_.*|row_nr|attribute|correct|text_text$")
#         )
#         columns = [col for col in system_df.columns if "system_" in col]
#         rename_dict = {name: f"system_{i}" for i, name in enumerate(columns)} | {
#             "correct": "annotator",
#             "text_text": "text",
#         }
#         result = pd.read_csv("./data/surrounding_texts.csv")
#         system_df = system_df.to_pandas()
#         system_df = system_df.merge(result, on="row_nr")
#         system_df = pl.from_pandas(system_df)
# 
#         df = system_df.rename(rename_dict)
# 
#         train_df, validate = train_test_split(
#             df, test_size=0.2, stratify=df["attribute"], random_state=seed
#         )
#         validate_df, test_df = train_test_split(
#             validate,
#             test_size=0.5,
#             stratify=validate["attribute"],
#             random_state=seed,
#         )
# 
#         train_df.write_csv("./data/shinra_all_system_train.csv")
#         validate_df.write_csv("./data/shinra_all_system_valid.csv")
#         test_df.write_csv("./data/shinra_all_system_test.csv")
#         df.write_csv(f"./data/all_system_data_seed_{seed}_system_{system_num}.csv")
# 
# 
# class ArtificialShinraLabelPreprocess(BasePreprocess):
#     def __call__(self, config):
#         df = self.preprocess(config)
#         attribute_count = df.attribute.value_counts().to_dict()
#         sys, crow = set(), set()
# 
#         for i, k in enumerate(attribute_count.keys()):
#             if i % 2 == 0:
#                 sys.add(k)
#             else:
#                 crow.add(k)
#         sys_d, crowd_d = [], []
#         for i in range(len(df)):
#             _d = df.iloc[i]
#             att = _d.attribute
#             if att in sys:
#                 sys_d.append(_d.correct if random.random() < 0.2 else not _d.correct)
#             else:
#                 sys_d.append(_d.correct if random.random() > 0.2 else not _d.correct)
#             if att in crow:
#                 crowd_d.append(_d.correct if random.random() < 0.2 else not _d.correct)
#             else:
#                 crowd_d.append(_d.correct if random.random() > 0.2 else not _d.correct)
#         df["system_dicision"] = sys_d
#         df["crowd_dicision"] = crowd_d
#         if not config.dataset.all_system:
#             df = df.rename(columns={"value": "text", "correct": "annotator"})
# 
#         self.split_data(df, config)
# 
#     def split_data(self, df, config):
#         train_df, validate = train_test_split(
#             df, test_size=0.2, stratify=df["attribute"], random_state=config.seed
#         )
#         validate_df, test_df = train_test_split(
#             validate,
#             test_size=0.5,
#             stratify=validate["attribute"],
#             random_state=config.seed,
#         )
#         seed = config.seed
#         if not config.dataset.all_system:
#             train_df = train_df.reset_index()
#             validate_df = validate_df.reset_index()
#             test_df = test_df.reset_index()
# 
#         train_df.to_csv(f"./data/shinra_artificial_train_seed_{seed}.csv")
#         validate_df.to_csv(f"./data/shinra_artificial_validate_seed_{seed}.csv")
#         test_df.to_csv(f"./data/shinra_artificial_test_seed_{seed}.csv")
#         df.to_csv(f"./data/shinra_artificial_data_seed_{seed}.csv", index=False)
# 
# 
# class MultiSystemPreprocessArtificial(BasePreprocess):
#     def __call__(self, config: Config):
#         df = self.preprocess(config)
#         attribute_count = df.attribute.value_counts().to_dict()
#         system_num = config.dataset.system_num
# 
#         system_d: dict[int, set[int]] = {i: set() for i in range(system_num)}
#         for i, k in enumerate(attribute_count.keys()):
#             system_d[i % system_num].add(k)
#         data: dict[str, list[Any]] = {f"system_{i}": [] for i in range(system_num)}
# 
#         for i in range(len(df)):
#             _d = df.iloc[i]
#             att = _d.attribute
#             for j in range(system_num):
#                 if att in system_d[j]:
#                     data[f"system_{j}"].append(
#                         _d.correct if random.random() < 0.2 else not _d.correct
#                     )
#                 else:
#                     data[f"system_{j}"].append(
#                         _d.correct if random.random() > 0.2 else not _d.correct
#                     )
# 
#         val_df = pd.DataFrame(data)
#         val_df["index"] = [i for i in val_df.index]
#         df["index"] = [i for i in df.index]
#         df = df.merge(val_df, on="index")
#         df = df.rename(columns={"value": "text", "correct": "annotator"})
# 
#         train_df, validate = train_test_split(
#             df, test_size=0.2, stratify=df["attribute"], random_state=config.seed
#         )
#         validate_df, test_df = train_test_split(
#             validate,
#             test_size=0.5,
#             stratify=validate["attribute"],
#             random_state=config.seed,
#         )
#         seed = config.seed
#         if not config.dataset.all_system:
#             train_df = train_df.reset_index()
#             validate_df = validate_df.reset_index()
#             test_df = test_df.reset_index()
# 
#         train_df.to_csv(
#             f"./data/shinra_all_system_artificial_train_seed_{seed}_system_{system_num}.csv"
#         )
#         validate_df.to_csv(
#             f"./data/shinra_all_system_artificial_validate_seed_{seed}_system_{system_num}.csv"
#         )
#         test_df.to_csv(
#             f"./data/shinra_all_system_artificial_test_seed_{seed}_system_{system_num}.csv"
#         )
#         df.to_csv(
#             f"./data/shinra_all_system_artificial_data_seed_{seed}_system_{system_num}.csv"
#         )
# 
# 
# # 1つのエージェントのみが正解しているデータを作成する。
# class MultiSystemPreprocessArtificialOneHot(BasePreprocess):
#     def __call__(self, config: Config):
#         df = self.preprocess(config)
#         attribute_count = df.attribute.value_counts().to_dict()
#         system_num = config.dataset.system_num
# 
#         system_d: dict[int, set[int]] = {i: set() for i in range(system_num)}
#         for i, k in enumerate(attribute_count.keys()):
#             system_d[i % system_num].add(k)
#         data: dict[str, list[Any]] = {f"system_{i}": [] for i in range(system_num)}
# 
#         for i in range(len(df)):
#             _d = df.iloc[i]
#             att = _d.attribute
#             for j in range(system_num):
#                 if att in system_d[j]:
#                     data[f"system_{j}"].append(_d.correct)
#                 else:
#                     data[f"system_{j}"].append(not _d.correct)
# 
#         val_df = pd.DataFrame(data)
#         val_df["index"] = [i for i in val_df.index]
#         df["index"] = [i for i in df.index]
#         df = df.merge(val_df, on="index")
#         df = df.rename(columns={"value": "text", "correct": "annotator"})
# 
#         train_df, validate = train_test_split(
#             df, test_size=0.2, stratify=df["attribute"], random_state=config.seed
#         )
#         validate_df, test_df = train_test_split(
#             validate,
#             test_size=0.5,
#             stratify=validate["attribute"],
#             random_state=config.seed,
#         )
#         seed = config.seed
#         if not config.dataset.all_system:
#             train_df = train_df.reset_index()
#             validate_df = validate_df.reset_index()
#             test_df = test_df.reset_index()
# 
#         train_df.to_csv(
#             f"./data/shinra_all_system_artificial_one_train_seed_{seed}_system_{system_num}.csv"
#         )
#         validate_df.to_csv(
#             f"./data/shinra_all_system_artificial_one_validate_seed_{seed}_system_{system_num}.csv"
#         )
#         test_df.to_csv(
#             f"./data/shinra_all_system_artificial_one_test_seed_{seed}_system_{system_num}.csv"
#         )
#         df.to_csv(
#             f"./data/shinra_all_system_artificial_one_data_seed_{seed}_system_{system_num}.csv"
#         )
