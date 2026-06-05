import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import polars as pl
import pickle
from tqdm import tqdm
import gc
from sklearn.model_selection import train_test_split
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


class SentimentAnalysisDataset(Dataset):
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        preprocess,
        data_name: str,
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
        if debug:
            data_path = Path(
                self.output_dir / f"{data_name}_{seed}_{mode}_debug.pickle"
            )
            csv_path = self.output_dir / f"{data_name}_{seed}_{mode}_debug.csv"
        else:
            csv_path = self.output_dir / f"{data_name}_{seed}_{mode}.csv"
            data_path = Path(self.output_dir / f"{data_name}_{seed}_{mode}.pickle")

        if not csv_path.exists():
            preprocess(debug=self.debug)
        df = pl.read_csv(csv_path)

        if data_path.exists():
            self.data = self.load_pickle(data_path)
        else:
            self.data = self._preprocess(df, data_path)
            with open(data_path, "wb") as f:
                pickle.dump(self.data, f)

        if filter_ratio != 0.0 and mode != "test":
            self.data = self.filter_data(self.data, filter_ratio)
        # サンプリングが必要な時の前処理
        if sampling_rate != 1.0 and mode == "train":
            # 存在しない時、作成する(indexのみのcsvと、埋め込み表現を含むpickleファイルまで作成する)
            idx = [i[0] for i in df.select("idx").to_numpy()]
            labels = [l[0] for l in df.select("label").to_numpy()]
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
            sampled_df = df.filter(pl.col("idx").is_in(sampled_idx))

            self.data = self._preprocess(sampled_df, data_path)
        if debug:
            self.data = self.data[:10]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_pickle(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def filter_data(self, data, filter_ratio):
        for i in range(len(data)):
            annotation = data[i]["annotations"]
            filter = np.random.uniform(size=annotation.shape) < filter_ratio
            annotation[filter] = -1
            data[i]["annotations"] = annotation
        return data

    def get_dataloader(self):
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=True if self.mode == "train" else False,
        )

    def _preprocess(self, df, path):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, output_hidden_states=True
        ).to(device())
        data = []

        for i in tqdm(range(len(df)), total=len(df), desc=f"Preprocessing {self.mode}"):
            text = df["text"][i]
            label = df["label"][i]
            idx = df["idx"][i]
            annotations = df.select(pl.col("^annotator_.*$"))[i].to_numpy().flatten()

            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                max_length=512,
                truncation=True,
            ).to(device())

            with torch.no_grad():
                outputs = model(**inputs)
            # BERTモデルの時に、cls_outを追加しないようにする
            hidden_states = outputs["hidden_states"][-1][0][0]
            _d = {
                "idx": idx,
                "embedding": hidden_states.cpu(),
                "label": torch.tensor(label),
                "annotations": torch.tensor(annotations),
            }

            data.append(_d)
        del model, tokenizer
        gc.collect()
        return data
