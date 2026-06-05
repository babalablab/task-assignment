# import polars as pl
# import numpy as np
# from sklearn.model_selection import train_test_split
# 
# from datasets import load_dataset
# from tqdm import tqdm
# 
# 
# def gen_labels(data):
#     datum = []
#     label2idx = {"neutral": 0, "entailment": 1, "contradiction": 2}
#     labels = list(label2idx.keys())
#     acc = {0: [0.9, 0.5, 0.5], 1: [0.4, 0.8, 0.4], 2: [0.3, 0.3, 0.7]}
#     anno = [0, 1, 2]
#     labels = list(label2idx.keys())
#     for d in tqdm(data, total=len(data), desc="preprocessing"):
#         gold_label = d["label"]
#         for a in anno:
#             c_prob = acc[a][gold_label]
#             w_prob = (1 - c_prob) / 2
#             p = [w_prob] * 3
#             p[gold_label] = c_prob
#             d[f"annotator_{a}"] = np.random.choice(labels, p=p)
#         datum.append(d)
# 
#     return datum
# 
# 
# def multinli_preprocess(seed: int):
#     ds = load_dataset("nyu-mll/multi_nli")
# 
#     train_df = pl.DataFrame(ds["train"].to_dict())
#     train_df = train_df.rename({"promptID": "idx"})
#     train_df = train_df.select(pl.col("idx", "label", "genre", "premise", "hypothesis"))
# 
#     train_data = train_df.to_dicts()
#     train_data = gen_labels(train_data)
#     train = pl.DataFrame(train_data)
# 
#     val_df = pl.DataFrame(ds["validation_matched"].to_dict())
#     val_df = val_df.rename({"promptID": "idx"})
#     val_df = val_df.select(pl.col("idx", "label", "genre", "premise", "hypothesis"))
# 
#     val_data = val_df.to_dicts()
#     val_data = gen_labels(val_data)
#     val_data = pl.DataFrame(val_data)
# 
#     val_y = val_data.select(pl.col("label"))
# 
#     val, test, _, _ = train_test_split(
#         val_data, val_y, stratify=val_y, random_state=seed, test_size=0.5
#     )
#     train.write_csv("./data/multinli_train_data.csv")
#     val.write_csv("./data/multinli_val_data.csv")
#     test.write_csv("./data/multinli_test_data.csv")
