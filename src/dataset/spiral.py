from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class SpiralDataset(Dataset):
    def __init__(
        self,
        seed: int,
        mode: str,
        batch_size: int,
        system_num: int,
        preprocess,
        sampling_rate: float = 1.0,
        filter_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        prefix = f"./data/spiral/seed_{seed}"
        data_path = Path(prefix + f"_{mode}.csv")

        if not data_path.exists():
            preprocess(seed=seed, debug=False, N=3000)

        df = pl.read_csv(data_path)
        self.data = self._preprocess(df)
        if filter_ratio != 0.0 and mode != "test":
            print(f"filtering data filter ratio {filter_ratio}")
            self.data = self.filter_data(self.data, filter_ratio)

        if sampling_rate != 1.0 and mode == "train":
            idx = [i[0] for i in df.select("idx").to_numpy()]
            labels = [l[0] for l in df.select("label").to_numpy()]
            sample_num = max(5, int(len(self.data) * sampling_rate))
            sampled_idx, _, _, _ = train_test_split(
                idx, labels, train_size=sample_num, stratify=labels, random_state=seed
            )
            sampled_df = df.filter(pl.col("idx").is_in(sampled_idx))

            self.data = self._preprocess(sampled_df)

        self.batch_size = batch_size
        self.mode = mode
        self.data_num = len(self.data)

    def get_dataloader(self):
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=True if self.mode == "train" else False,
        )

    def _preprocess(self, data: pl.DataFrame):
        annot_num = len(data.select(pl.col("^pred_.*$")).columns)
        data = data.to_dict(as_series=False)
        # dictのリストに変換する
        length = len(data["x"])
        keys = data.keys()
        d = []
        for l in range(length):
            di = {k: torch.tensor(data[k][l]).to(torch.float32) for k in keys}
            di["embedding"] = torch.tensor([data["x"][l], data["y"][l]]).to(
                torch.float32
            )
            di["label"] = torch.tensor(data["label"][l]).to(torch.long)
            di["annotations"] = torch.tensor(
                [data[f"pred_{i}"][l] for i in range(annot_num)]
            ).to(torch.long)
            d.append(di)
        return d

    def filter_data(self, data, filter_ratio):
        for i in range(len(data)):
            annotation = data[i]["annotations"]
            filter = np.random.uniform(size=annotation.shape) < filter_ratio
            annotation[filter] = -1
            data[i]["annotations"] = annotation
        return data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class SpiralDatasetDifferentTestNum(SpiralDataset):
    def __init__(
        self,
        seed: int,
        mode: str,
        batch_size: int,
        test_data_num: int,
        annotator_num: int,
        preprocess,
    ) -> None:
        self.test_mode = ""

        match test_data_num:
            case 1_000:
                self.test_mode = "test_1k"
            case 5_000:
                self.test_mode = "test_5k"
            case 10_000:
                self.test_mode = "test_10k"
            case 30_000:
                self.test_mode = "test_30k"
            case 50_000:
                self.test_mode = "test_50k"
            case 100_000:
                self.test_mode = "test_100k"
            case 1_000_000:
                self.test_mode = "test_1m"
            case _:
                raise ValueError(f"test data num {test_data_num} is not supported")
        assert self.test_mode != "", f"test data num {test_data_num} is not supported"
        data_path: Path = Path(
            f"./data/spiral_data_num/seed_{seed}_{self.test_mode}_{annotator_num}_{mode}.csv"
        )

        if not data_path.exists():
            preprocess(seed=seed, debug=False)
        df = pl.read_csv(data_path)
        self.batch_size = batch_size
        self.mode = mode

        self.data = self._preprocess(df)
        self.data_num = len(self.data)
