# import torch
# from torch.utils.data import Dataset, DataLoader
# import polars as pl
# from transformers import AutoTokenizer, BertConfig, BertModel
# from tqdm import tqdm
# from pathlib import Path
# from preprocess import multinli_preprocess
# import pickle
# import gc
# 
# 
# class MultiNLIDataset(Dataset):
#     def __init__(
#         self,
#         data_path: str,
#         batch_size: int,
#         shuffle: bool,
#         seed: int,
#         debug: bool = False,
#         mode: str = "train",
#     ):
#         self.label2idx = {"entailment": 0, "contradiction": 1, "neutral": 2}
#         data_path = Path(data_path)
# 
#         if not data_path.exists():
#             multinli_preprocess(seed)
#         df = pl.read_csv(data_path)
#         self.mode = mode
#         if debug:
#             df = df[:32]
#         self.debug = debug
#         self.seed = seed
#         self.data = df.pipe(self.preprocess)
#         self.batch_size = batch_size
#         self.shuffle = shuffle
# 
#     def get_dataloader(self):
#         return DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle)
# 
#     def _device(self):
#         device = ""
#         if torch.cuda.is_available():
#             device = "cuda"
#         elif torch.backends.mps.is_available():
#             device = "mps"
#         else:
#             device = "cpu"
#         return device
# 
#     def __len__(self):
#         return len(self.data)
# 
#     def __getitem__(self, idx):
#         self.data[idx]["embedding"] = self.data[idx]["embedding"].to(self._device())
#         self.data[idx]["label"] = self.data[idx]["label"].to(self._device())
#         return self.data[idx]
# 
#     def preprocess(self, df: pl.DataFrame):
#         # pickle化するかどうかをしないと
#         path = (
#             Path(f"./data/multinli_{self.mode}_{self.seed}_debug.pickle")
#             if self.debug
#             else Path(f"./data/multinli_{self.mode}_{self.seed}.pickle")
#         )
#         dicts = []
#         if path.exists() and not self.debug:
#             with open(path, "rb") as f:
#                 dicts = pickle.load(f)
# 
#         else:
#             tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# 
#             df = df.with_columns(
#                 concat=pl.col("premise") + " [SEP] " + pl.col("hypothesis"),
#             )
#             data = df.to_numpy()
#             dicts = []
#             config = BertConfig()
#             model = BertModel(config)
#             model = model.to(self._device())
#             for di in tqdm(data, desc="preprocessing"):
#                 txt = tokenizer(
#                     di[-1], padding="max_length", max_length=512, return_tensors="pt"
#                 ).to(self._device())
#                 with torch.no_grad():
#                     emb = model(**txt)
# 
#                 emb = emb["last_hidden_state"][:, 0].squeeze(0).cpu()
# 
#                 label = torch.tensor(di[1])
# 
#                 annotatorA = self.label2idx[di[5]]
#                 annotatorB = self.label2idx[di[6]]
#                 annotatorC = self.label2idx[di[7]]
#                 annotator = torch.tensor([annotatorA, annotatorB, annotatorC])
# 
#                 dicts.append(
#                     {
#                         "idx": di[0],
#                         "embedding": emb,
#                         "label": label,
#                         "annotations": annotator,
#                     }
#                 )
# 
#             with open(path, "wb") as f:
#                 pickle.dump(dicts, f)
#             del tokenizer
#             del model
#             gc.collect()
#         return dicts
