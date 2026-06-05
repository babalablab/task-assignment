import polars as pl
from sklearn.model_selection import train_test_split
# import datasets
# import numpy as np


# def load_sst2_data(seed=10):
#     train_data = datasets.load_dataset(
#         "gpt3mix/sst2", split="train", trust_remote_code=True
#     )
#     validation_data = datasets.load_dataset(
#         "gpt3mix/sst2", split="validation", trust_remote_code=True
#     )
#     test_data = datasets.load_dataset(
#         "gpt3mix/sst2", split="test", trust_remote_code=True
#     )
#     train_llm_annotaion = pl.read_csv("data/llm_annotation/sst2/sst2_train.csv")
#     validataion_llm_annotation = pl.read_csv(
#         "data/llm_annotation/sst2/sst2_validation.csv"
#     )
#     test_llm_annotaion = pl.read_csv("data/llm_annotation/sst2/sst2_test.csv")
# 
#     def extract_data(data):
#         d = []
#         for i, di in enumerate(data):
#             token = di["text"]
#             label = 1 - di["label"]
#             d.append({"sentence": token, "label": label, "idx": i})
# 
#         return pl.DataFrame(d)
# 
#     train_df = extract_data(train_data)
#     valid_df = extract_data(validation_data)
#     test_df = extract_data(test_data)
# 
#     train_df = train_df.join(train_llm_annotaion, on="idx")
#     test_df = test_df.join(test_llm_annotaion, on="idx")
#     valid_df = valid_df.join(validataion_llm_annotation, on="idx")
# 
#     return {"train": train_df, "validation": valid_df, "test": test_df}
# 
# 
# def load_imdb_data(seed):
#     ds = datasets.load_dataset(
#         "stanfordnlp/imdb", split="train", trust_remote_code=True
#     )
#     text, label = [], []
#     for di in ds:
#         text.append(di["text"])
#         label.append(di["label"])
#     df = pl.DataFrame({"text": text, "label": label})
#     train_llm_annotaion = pl.read_csv("data/llm_annotation/imdb/imdb_train.csv")
#     df = df.join(
#         train_llm_annotaion, left_on="text", right_on="sentence", how="inner"
#     ).unique()
#     columns = df.columns
#     data = df.to_numpy()
#     label = data[:, 1]
#     train, val = train_test_split(
#         data, train_size=0.7, stratify=label, random_state=seed
#     )
#     train_label = train[:, 1]
#     train, _ = train_test_split(
#         train,
#         train_size=10000,
#         stratify=train_label,
#         random_state=seed,
#     )
#     train_d = {}
#     for i, c in enumerate(columns):
#         train_d[c] = list(train[:, i])
# 
#     val_d = {}
#     for i, c in enumerate(columns):
#         val_d[c] = list(val[:, i])
# 
#     train_df = pl.DataFrame(train_d)
#     val_df = pl.DataFrame(val_d)
# 
#     ds = datasets.load_dataset("stanfordnlp/imdb", split="test")
#     text, label = [], []
#     for di in ds:
#         text.append(di["text"])
#         label.append(di["label"])
#     test_df = pl.DataFrame({"text": text, "label": label})
#     test_llm_annotaion = pl.read_csv("data/llm_annotation/imdb/imdb_test.csv")
#     test_df = test_df.join(
#         test_llm_annotaion, left_on="text", right_on="sentence", how="inner"
#     ).unique()
#     test_data = test_df.to_numpy()
#     test_label = test_data[:, 1]
#     columns = test_df.columns
#     test, _ = train_test_split(
#         test_data, train_size=3000, stratify=test_label, random_state=seed
#     )
#     test_d = {}
#     for i, c in enumerate(columns):
#         test_d[c] = list(test[:, i])
#     test_df = pl.DataFrame(test_d)
# 
#     return {"train": train_df, "validation": val_df, "test": test_df}
# 
# 
# def load_poem_sentiment_data():
#     splits = {
#         "train": "data/train-00000-of-00001.parquet",
#         "validation": "data/validation-00000-of-00001.parquet",
#         "test": "data/test-00000-of-00001.parquet",
#     }
#     train_df = pl.read_parquet(
#         "hf://datasets/google-research-datasets/poem_sentiment/" + splits["train"]
#     ).rename({"id": "idx"})
#     validation_df = pl.read_parquet(
#         "hf://datasets/google-research-datasets/poem_sentiment/" + splits["validation"]
#     ).rename({"id": "idx"})
#     test_df = pl.read_parquet(
#         "hf://datasets/google-research-datasets/poem_sentiment/" + splits["test"]
#     ).rename({"id": "idx"})
#     train_llm_annotation = pl.read_csv(
#         "data/llm_annotation/poem_sentiment/poem_sentiment_train.csv"
#     ).with_columns(pl.col("idx").cast(pl.Int32))
#     validation_llm_annotation = pl.read_csv(
#         "data/llm_annotation/poem_sentiment/poem_sentiment_validation.csv"
#     ).with_columns(pl.col("idx").cast(pl.Int32))
#     test_llm_annotation = pl.read_csv(
#         "data/llm_annotation/poem_sentiment/poem_sentiment_test.csv"
#     ).with_columns(pl.col("idx").cast(pl.Int32))
# 
#     train_df = train_df.join(train_llm_annotation, on="idx")
#     validation_df = validation_df.join(validation_llm_annotation, on="idx")
#     test_df = test_df.join(test_llm_annotation, on="idx")
# 
#     return {"train": train_df, "validation": validation_df, "test": test_df}
# 
# 
def load_tweet_eval_data(seed):
    base_df = pl.read_csv("./data/tweet_eval/tweet_eval_annotated_with_llm.csv")
    # emoji2idx = {
    #     k[1]: k[0]
    #     for k in base_df.unique("emoji_label")
    #     .select(pl.col("annotation_label", "emoji_label"))
    #     .to_numpy()
    # }
    # llm_annotation = pl.read_csv(
    #     "./data/llm_annotation/tweet_eval/tweet_eval_train.csv"
    # ).with_columns(
    #     llm_emoji_annotation=pl.col("llm_annotation").replace_strict(emoji2idx)
    # )
    # base_df = base_df.join(llm_annotation, left_on="id", right_on="idx")
    _d = base_df.to_dicts()

    def split_data(data, train_size):
        label = [d["annotation_label"] for d in data]
        train_data, test_data, _, _ = train_test_split(
            data, label, stratify=label, train_size=train_size, random_state=seed
        )
        return train_data, test_data

    train_data, test_val_data = split_data(_d, train_size=0.7)
    test_data, val_data = split_data(test_val_data, train_size=0.5)
    train_df = pl.DataFrame(train_data, schema=base_df.schema)
    test_df = pl.DataFrame(test_data, schema=base_df.schema)
    validation_df = pl.DataFrame(val_data, schema=base_df.schema)
    # split data
    return {"train": train_df, "validation": validation_df, "test": test_df}


# def load_ethos_data():
#     data = datasets.load_dataset(
#         "iamollas/ethos", "binary", split="train", trust_remote_code=True
#     )
#     tokens, labels = [], []
#     for di in data:
#         tokens.append(di["text"])
#         labels.append(di["label"])
#     train_token, test_token, train_label, test_labels = train_test_split(
#         tokens, labels, train_size=0.8, stratify=labels
#     )
#     test_token, val_token, test_label, val_label = train_test_split(
#         test_token, test_labels, train_size=0.5, stratify=test_labels
#     )
#     train_df = pl.DataFrame({"text": train_token, "label": train_label})
#     validation_df = pl.DataFrame({"text": val_token, "label": val_label})
#     test_df = pl.DataFrame({"text": test_token, "label": test_label})
#     train_df = train_df.with_row_index(name="idx")
#     validation_df = validation_df.with_row_index(name="idx")
#     test_df = test_df.with_row_index(name="idx")
#     return {"train": train_df, "validation": validation_df, "test": test_df}
# 
# 
# def load_hatexplain_data():
#     train_data = datasets.load_dataset(
#         "Hate-speech-CNERG/hatexplain", split="train", trust_remote_code=True
#     )
#     test_data = datasets.load_dataset(
#         "Hate-speech-CNERG/hatexplain", split="test", trust_remote_code=True
#     )
#     validation_data = datasets.load_dataset(
#         "Hate-speech-CNERG/hatexplain", split="validation", trust_remote_code=True
#     )
# 
#     def extract_data(data):
#         d = []
#         for di in data:
#             token = " ".join(di["post_tokens"])
#             label = np.argmax(np.bincount(di["annotators"]["label"]))
#             d.append({"text": token, "label": label})
#         return pl.DataFrame(d)
# 
#     train_df = extract_data(train_data)
#     test_df = extract_data(test_data)
#     validation_df = extract_data(validation_data)
#     train_df = train_df.with_row_index(name="idx")
#     validation_df = validation_df.with_row_index(name="idx")
#     test_df = test_df.with_row_index(name="idx")
#     return {"train": train_df, "validation": validation_df, "test": test_df}
# 
# 
# def load_multi_nli_data(seed: int):
#     train_data = datasets.load_dataset(
#         "nyu-mll/multi_nli", split="train"
#     ).select_columns(["premise", "hypothesis", "label"])
#     test_data = (
#         datasets.load_dataset("nyu-mll/multi_nli", split="validation_matched")
#         .select_columns(["premise", "hypothesis", "label"])
#         .train_test_split(train_size=3000, stratify_by_column="label", seed=seed)[
#             "train"
#         ]
#     )
# 
#     data, label = [], []
#     for d in train_data:
#         data.append([d["premise"], d["hypothesis"], d["label"]])
#         label.append(d["label"])
#     train_data, valid_data, _, valid_label = train_test_split(
#         data, label, train_size=10000, stratify=label, random_state=seed
#     )
#     valid_data, _, _, _ = train_test_split(
#         valid_data,
#         valid_label,
#         train_size=10000,
#         stratify=valid_label,
#         random_state=seed,
#     )
# 
#     # train_data = {
#     #     "premise": [d[0] for d in train_data],
#     #     "hypothesis": [d[1] for d in train_data],
#     #     "label": [d[2] for d in train_data],
#     # }
#     # print(train_data)
#     train_df = pl.DataFrame(
#         {
#             "premise": [d[0] for d in train_data],
#             "hypothesis": [d[1] for d in train_data],
#             "label": [d[2] for d in train_data],
#         },
#     ).with_row_index(name="idx")
#     valid_df = pl.DataFrame(
#         {
#             "premise": [d[0] for d in valid_data],
#             "hypothesis": [d[1] for d in valid_data],
#             "label": [d[2] for d in valid_data],
#         },
#     ).with_row_index(name="idx")
#     test_df = pl.DataFrame(
#         {
#             "premise": [d["premise"] for d in test_data],
#             "hypothesis": [d["hypothesis"] for d in test_data],
#             "label": [d["label"] for d in test_data],
#         }
#     ).with_row_index(name="idx")
#     return {"train": train_df, "validation": valid_df, "test": test_df}
# 
# 
# def load_anli_data(seed=10):
#     train_dataset = datasets.load_dataset(
#         "facebook/anli", split="train_r1"
#     ).train_test_split(train_size=10000, stratify_by_column="label")["train"]
#     valid_dataset = datasets.load_dataset("facebook/anli", split="dev_r1")
#     test_dataset = datasets.load_dataset("facebook/anli", split="test_r1")
# 
#     def convert_dataset_to_df(dataset):
#         data = []
#         for d in dataset:
#             data.append(
#                 {
#                     "premise": d["premise"],
#                     "hypothesis": d["hypothesis"],
#                     "label": d["label"],
#                 }
#             )
#         return pl.DataFrame(data).with_row_index(name="idx")
# 
#     train_df = convert_dataset_to_df(train_dataset)
#     test_df = convert_dataset_to_df(test_dataset)
#     valid_df = convert_dataset_to_df(valid_dataset)
#     return {"train": train_df, "validation": valid_df, "test": test_df}
# 
# 
# def load_cifar10_data(seed: int):
#     train_dataset = datasets.load_dataset("uoft-cs/cifar10", split="train")
#     test_dataset = datasets.load_dataset(
#         "uoft-cs/cifar10", split="test"
#     ).train_test_split(train_size=3000, stratify_by_column="label")["train"]
#     tmp_dataset = train_dataset.train_test_split(
#         train_size=5000, test_size=3000, stratify_by_column="label", seed=seed
#     )
#     train_dataset = tmp_dataset["train"]
#     valid_dataset = tmp_dataset["test"]
#     return {"train": train_dataset, "validation": valid_dataset, "test": test_dataset}
# 
# 
# def load_mnist_data(seed: int):
#     tmp_dataset = datasets.load_dataset("ylecun/mnist", split="train").train_test_split(
#         train_size=5000, test_size=3000, stratify_by_column="label", seed=seed
#     )
#     test_dataset = datasets.load_dataset("ylecun/mnist", split="test").train_test_split(
#         train_size=3000, stratify_by_column="label"
#     )["train"]
#     train_dataset = tmp_dataset["train"]
#     valid_dataset = tmp_dataset["test"]
#     return {"train": train_dataset, "validation": valid_dataset, "test": test_dataset}
# 
# 
# def load_cifar100_data(seed: int):
#     train_dataset = datasets.load_dataset("uoft-cs/cifar100", split="train")
#     test_dataset = datasets.load_dataset(
#         "uoft-cs/cifar100", split="test"
#     ).train_test_split(train_size=3000, stratify_by_column="coarse_label")["train"]
#     tmp_dataset = train_dataset.train_test_split(
#         train_size=5000, test_size=3000, stratify_by_column="coarse_label", seed=seed
#     )
#     train_dataset = tmp_dataset["train"]
#     valid_dataset = tmp_dataset["test"]
#     return {"train": train_dataset, "validation": valid_dataset, "test": test_dataset}
