# from utils import load_cifar10_data, load_cifar100_data, load_mnist_data
# import polars as pl
# from pathlib import Path
# import numpy as np
# from tqdm import tqdm
# from typing import Any
# from collections import defaultdict
# 
# 
# class BaseCVPreproess:
#     def __init__(self) -> None:
#         self.output_dir: Path
#         self.annotator_num: int
#         self.label_num: int
#         self.dist: np.ndarray[float]
#         self.data_name: str
#         self.seed: int
#         self.image: str
#         self.label: str
# 
#     def load_data(self, seed: int):
#         raise NotImplementedError
# 
#     def generate_labels(self, label):
#         assert hasattr(self, "dist"), "Set Distribution of artificial labels"
#         assert hasattr(self, "annotator_num"), "set number of annotators"
#         assert hasattr(self, "label_num"), "set number of labels"
# 
#         self.output_dir.mkdir(exist_ok=True, parents=True)
#         class_idx = np.arange(self.label_num)
#         labels = defaultdict(str)
#         for i in range(self.annotator_num):
#             acc = self.dist[i][label]
#             sampling_dist = np.array(
#                 [(1 - acc) / (self.label_num - 1)] * self.label_num
#             )
#             sampling_dist[label] = acc
#             anno = np.random.choice(class_idx, p=sampling_dist)
#             labels[f"annotator_{i}"] = anno
# 
#         return labels
# 
#     def __call__(self, debug: bool, seed: int, **kwargs) -> Any:
#         d = self.load_data(seed)
#         for k, v in d.items():
#             datas = []
#             file_output_path = self.output_dir / f"{self.data_name}_{self.seed}_{k}.csv"
#             img_output_dir = self.output_dir / f"img/{k}/{self.seed}"
#             if debug:
#                 file_output_path = (
#                     self.output_dir / f"{self.data_name}_{self.seed}_{k}_debug.csv"
#                 )
# 
#                 img_output_dir = self.output_dir / f"img/debug/{self.seed}"
#                 v = v.select([i for i in range(10)])
# 
#             img_output_dir.mkdir(parents=True, exist_ok=True)
# 
#             for i in tqdm(range(len(v)), total=len(v), desc=f"Processing {k}"):
#                 img = v[self.image][i]
#                 label: int = v[self.label][i]
#                 data = self.generate_labels(label)
#                 img.save(img_output_dir / f"{i}.png")
# 
#                 data["img_path"] = str(img_output_dir / f"{i}.png")
#                 data["label"] = label
#                 data["idx"] = i
#                 datas.append(data)
# 
#             df = pl.DataFrame(datas)
#             df.write_csv(file_output_path)
#             print(f"file is saved to {str(file_output_path)}")
# 
#     def gen_prob(self):
#         base_prob = np.full((self.annotator_num, self.label_num), 0.1)
#         j = 0
#         for i in range(self.label_num):
#             base_prob[j][i] = 0.9
#             j = (j + 1) % self.annotator_num
#         return base_prob
# 
# 
# class CIFAR10Preprocess(BaseCVPreproess):
#     def __init__(
#         self,
#         output_dir: Path,
#         annotator_num: int = 4,
#         label_num: int = 10,
#         seed: int = 10,
#     ) -> None:
#         super().__init__()
#         self.output_dir: Path = Path(output_dir)
#         self.annotator_num: int = annotator_num
#         self.label_num: int = label_num
#         self.dist: np.ndarray[float] = self.gen_prob()
#         self.data_name: str = "cifar10"
#         self.seed: int = seed
#         self.image: str = "img"
#         self.label: str = "label"
# 
#     def load_data(self, seed):
#         return load_cifar10_data(seed)
# 
# 
# class CIFAR100Preprocess(BaseCVPreproess):
#     def __init__(
#         self,
#         output_dir: Path,
#         annotator_num: int = 4,
#         label_num: int = 20,
#         seed: int = 10,
#     ) -> None:
#         super().__init__()
#         self.output_dir: Path = Path(output_dir)
#         self.annotator_num: int = annotator_num
#         self.label_num: int = label_num
#         self.dist: np.ndarray[float] = self.gen_prob()
#         self.data_name: str = "cifar100"
#         self.seed: int = seed
#         self.image: str = "img"
#         self.label: str = "coarse_label"
# 
#     def load_data(self, seed):
#         return load_cifar100_data(seed)
# 
# 
# class MNISTPreprocess(BaseCVPreproess):
#     def __init__(
#         self,
#         output_dir: str,
#         annotator_num: int = 4,
#         seed: int = 10,
#     ) -> None:
#         super().__init__()
#         self.output_dir: Path = Path(output_dir)
#         self.annotator_num: int = annotator_num
#         self.label_num: int = 10
#         self.dist: np.ndarray[float] = self.gen_prob()
#         self.data_name: str = "mnist"
#         self.seed: int = seed
#         self.image: str = "image"
#         self.label: str = "label"
# 
#     def load_data(self, seed):
#         return load_mnist_data(seed)
