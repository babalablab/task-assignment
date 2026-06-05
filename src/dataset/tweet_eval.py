from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.model_selection import train_test_split
from utils.load_data import load_tweet_eval_data
import polars as pl
import pickle
import torch
import gc
from tqdm import tqdm
from pathlib import Path
import numpy as np


def device():
    device = ""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


class TweetEvalDataset(Dataset):
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        seed: int,
        mode: str,
        batch_size: int,
        debug: bool,
        sampling_rate: float = 1.0,
        filter_ratio: float = 0.0,
    ):
        super().__init__()
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.batch_size = batch_size
        self.debug = debug
        self.seed = seed
        self.device = device()
        # split data when the data does not exist
        csv_path = self.output_dir / f"tweet_eval_{self.seed}_{mode}.csv"
        data_path = self.output_dir / f"tweet_eval_{self.seed}_{mode}.pickle"
        if mode == "pytest":
            return
        if not csv_path.exists():
            self.preprocess()
        df = pl.read_csv(csv_path)
        if data_path.exists():
            self.data = self.load_pickle(data_path)

        else:
            self.data = self.extract_features(df)
            with open(data_path, "wb") as f:
                pickle.dump(self.data, f)
        if filter_ratio != 0.0 and mode != "test":
            self.data = self.filter_data(self.data, filter_ratio)
        if sampling_rate != 1.0 and mode == "train":
            idx = [i[0] for i in df.select("id").to_numpy()]
            labels = [l[0] for l in df.select("annotation_label").to_numpy()]
            sample_num = max(5, int(len(self.data) * sampling_rate))
            print(
                f"sampling_rate {sampling_rate}, data length {len(self.data)}, calculated sample num {sample_num}"
            )
            sampled_idx, _, _, _ = train_test_split(
                idx,
                labels,
                train_size=sample_num,
                stratify=labels,
                random_state=seed,
            )
            sampled_df = df.filter(pl.col("id").is_in(sampled_idx))

            self.data = self.extract_features(sampled_df)

        if debug:
            self.data = self.data[:10]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def get_dataloader(self):
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=True if self.mode == "train" else False,
        )

    def preprocess(self):
        datas = load_tweet_eval_data(self.seed)
        datas["train"].write_csv(f"./data/tweet_eval/tweet_eval_{self.seed}_train.csv")
        datas["validation"].write_csv(
            f"./data/tweet_eval/tweet_eval_{self.seed}_validation.csv"
        )
        datas["test"].write_csv(f"./data/tweet_eval/tweet_eval_{self.seed}_test.csv")

    def filter_data(self, data, filter_ratio):
        for i in range(len(data)):
            annotation = data[i]["annotations"]
            filter = np.random.uniform(size=annotation.shape) < filter_ratio
            annotation[filter] = -1
            data[i]["annotations"] = annotation
        return data

    def load_pickle(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def extract_features(self, data: pl.DataFrame):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(self.device)
        datas = []

        for i in tqdm(
            range(len(data)),
            total=len(data),
            desc=f"Preprocessing Texts mode: {self.mode}",
        ):
            text = data["text"][i]
            label = data["annotation_label"][i]
            idx = data["id"][i]
            annotations = (
                data.select(pl.col("^annotation_label_[0-3]|llm_annotation*$"))
                .select(pl.col(pl.Int64))[i]
                .to_numpy()
                .flatten()
            )
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                max_length=512,
                truncation=True,
            ).to(self.device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs["hidden_states"][-1][0][0].cpu()
            _d = {
                "idx": idx,
                "embedding": hidden_states,
                "label": torch.tensor(label),
                "annotations": torch.tensor(annotations),
            }
            datas.append(_d)
        del model, tokenizer
        gc.collect()
        return datas
