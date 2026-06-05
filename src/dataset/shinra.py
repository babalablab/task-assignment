# import json
# from pathlib import Path
# import pickle
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# from transformers import RobertaConfig, RobertaModel
# 
# from dataset.tokenizer import JanomeBpeTokenizer
# from preprocess.preprocess import MultiSystemPreprocess
# 
# 
# class ShinraAgentAccuracyDataset(Dataset):
#     def __init__(
#         self,
#         system_num: int,
#         method: str,
#         batch_size: int,
#         seed,
#         around_text: bool = False,
#         num_tokens: int = 512,
#         debug: bool = False,
#         mode: str = "",
#     ) -> None:
#         assert mode != "", "mode should be specified!!"
#         # data = pd.read_csv(data_path)
#         super().__init__()
# 
#         self.system_num = system_num
# 
#         self.vocab = self.json_load("./model/vocab.json")
#         self.method = method
#         self.debug = debug
#         self.num_tokens = num_tokens
#         self.around_text = around_text
#         self.tokenizer = JanomeBpeTokenizer("../model/codecs.txt")
#         self.mode = mode
#         self.device = self._device()
# 
#         self.batch_size = batch_size
#         self.shuffle = True
#         data_path = Path(f"./data/shinra_all_system_{mode}.csv")
#         if not data_path.exists():
#             preprocess = MultiSystemPreprocess()
#             preprocess(seed, system_num)
# 
#         data = pd.read_csv(data_path, index_col=0)
#         if debug:
#             data = data[:32]
#         self.data = self.preprocess(data)
# 
#     def _load_bert(self):
#         config = RobertaConfig.from_pretrained("./model/config.json")
#         self.bert = RobertaModel.from_pretrained(
#             "./model/bert_model.pth", config=config, device_map="auto"
#         )
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
#         d = self.data[idx]
#         system_pred = torch.Tensor([d[f"system_{i}"] for i in range(self.system_num)])
#         bert_embedding = d["bert_embedding"].squeeze(0)
# 
#         correct = torch.tensor(d["annotator"])
# 
#         return {
#             "idx": idx,
#             "annotations": system_pred,
#             "label": correct,
#             "embedding": bert_embedding,
#         }
# 
#     def gen_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
#         with torch.no_grad():
#             out = self.bert(input_ids, attention_mask=attention_mask)
#             out = out["pooler_output"]
#             # out = out.cpu().numpy()
#         return out
# 
#     def preprocess(self, data):
#         data_path = Path()
#         if self.debug:
#             data_path = Path(f"./data/{self.method}_{self.mode}_debug.pickle")
#         else:
#             data_path = Path(f"./data/{self.method}_{self.mode}.pickle")
#         indexes = len(data)
#         data_list = []
#         if data_path.exists():
#             with open(data_path, "rb") as f:
#                 data_list = pickle.load(f)
#         else:
#             self._load_bert()
#             for i in tqdm(
#                 range(indexes), total=indexes, desc="preprocessing data ...."
#             ):
#                 d = data.iloc[i].to_dict()
# 
#                 id, mask = self.concat_text(d["text"], d["attribute"], d["around_text"])
#                 id = torch.tensor([id]).to(self.device)
#                 mask = torch.tensor([mask]).to(self.device)
#                 emb = self.gen_embedding(id, mask)
#                 d["token_id"] = id.to("cpu")
#                 d["attention_mask"] = mask.to("cpu")
#                 d["bert_embedding"] = emb.to("cpu")
#                 data_list.append(d)
#             with open(data_path, "wb") as f:
#                 pickle.dump(data_list, f)
#         for data in data_list:
#             data["token_id"] = data["token_id"].to(self.device)
#             data["attention_mask"] = data["attention_mask"].to(self.device)
#             data["bert_embedding"] = data["bert_embedding"].to(self.device)
#         assert len(data_list) != 0, "data is empty!!"
#         return data_list
# 
#     def json_load(self, file_path):
#         with open(file_path, "r") as f:
#             return json.load(f)
# 
#     def padding(self, array, pad, seq_len):
#         if len(array) >= seq_len:
#             return array
#         return array + [pad] * (seq_len - len(array))
# 
#     def concat_text(self, text, attribute, around_text):
#         text = self.tokenize_text(text)
#         attribute = self.tokenize_text(attribute)
#         around_text = self.tokenize_text(around_text)
#         if self.around_text:
#             text = (
#                 ["<s>"]
#                 + text
#                 + ["[SEP]"]
#                 + attribute
#                 + ["[SEP]"]
#                 + around_text
#                 + ["</s>"]
#             )
#         else:
#             text = ["<s>"] + text + ["[SEP]"] + attribute + ["</s>"]
#             if len(text) > self.num_tokens:
#                 text = text[: self.num_tokens - 1] + ["</s>"]
# 
#         attention_mask = [1] * len(text)
#         text = self.padding(text, "<pad>", self.num_tokens)
#         attention_mask = self.padding(attention_mask, 0, self.num_tokens)
#         token_id = [self.vocab.get(token, self.vocab["<unk>"]) for token in text]
# 
#         return token_id, attention_mask
# 
#     def tokenize_text(self, text):
#         text = str(text)
#         return self.tokenizer.tokenize(text)[0]
# 
# 
# # class ShinraDataset(Dataset):
# #     def __init__(
# #         self,
# #         data: pd.DataFrame,
# #         around_text: bool,
# #         my_config: MyConfig,
# #         num_tokens: int = 512,
# #         mode: str = "train",
# #         debug: bool = False,
# #     ) -> None:
# #         super().__init__()
# #         self.vocab = self.json_load("./model/vocab.json")
# #         self.num_tokens = num_tokens
# #         self.around_text = around_text
# #         self.tokenizer = JanomeBpeTokenizer("../model/codecs.txt")
# #         self.mode = mode
# #         self.device = self._device()
# #         self.config = my_config
# #         self.data = self.preprocess(data)
# #         self.batch_size = 100
# #         self.shuffle = True
# #
# #     def _load_bert(self):
# #         config = RobertaConfig.from_pretrained("./model/config.json")
# #         self.bert = RobertaModel.from_pretrained(
# #             "./model/bert_model.pth", config=config, device_map="auto"
# #         )
# #
# #     def get_dataloader(self):
# #         return DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle)
# #
# #     def _device(self):
# #         device = ""
# #         if torch.cuda.is_available():
# #             device = "cuda"
# #         elif torch.backends.mps.is_available():
# #             device = "mps"
# #         else:
# #             device = "cpu"
# #         return device
# #
# #     def __len__(self):
# #         return len(self.data)
# #
# #     def __getitem__(self, idx):
# #         d = self.data[idx]
# #         text = str(d["text"])
# #         system_decision = d["system_decision"]
# #         crowd_decision = d["crowd_decision"]
# #         annotator = d["annotator"]
# #         attribute = d["attribute"]
# #         token_id = d["token_id"]
# #         around_text = d["around_text"]
# #         attention_mask = d["attention_mask"]
# #         bert_embedding = d["bert_embedding"].squeeze(0)
# #         idx = d["idx"]
# #         n = sum(
# #             [
# #                 system_decision == annotator,
# #                 crowd_decision == annotator,
# #             ]
# #         )
# #         answer = torch.tensor([system_decision, crowd_decision])
# #         correct = torch.tensor([annotator])
# #         return {
# #             "idx": idx,
# #             "text": text,
# #             "attribute": "".join(attribute),
# #             "around_text": around_text,
# #             "system_decision": system_decision,
# #             "crowd_decision": crowd_decision,
# #             "annotator_decision": annotator,
# #             "decision": answer,
# #             "correct": correct,
# #             "tokens": token_id,
# #             "n": n,
# #             "attention_mask": attention_mask,
# #             "bert_embedding": bert_embedding,
# #         }
# #
# #     def tokenize_text(self, text):
# #         text = str(text)
# #         return self.tokenizer.tokenize(text)[0]
# #
# #     def concat_text(self, text, attribute, around_text):
# #         text = self.tokenize_text(text)
# #         attribute = self.tokenize_text(attribute)
# #         around_text = self.tokenize_text(around_text)
# #         if self.around_text:
# #             text = (
# #                 ["<s>"]
# #                 + text
# #                 + ["[SEP]"]
# #                 + attribute
# #                 + ["[SEP]"]
# #                 + around_text
# #                 + ["</s>"]
# #             )
# #         else:
# #             text = ["<s>"] + text + ["[SEP]"] + attribute + ["</s>"]
# #         if len(text) > self.num_tokens:
# #             text = text[: self.num_tokens - 1] + ["</s>"]
# #
# #         attention_mask = [1] * len(text)
# #         text = self.padding(text, "<pad>", self.num_tokens)
# #         attention_mask = self.padding(attention_mask, 0, self.num_tokens)
# #         token_id = [self.vocab.get(token, self.vocab["<unk>"]) for token in text]
# #
# #         return token_id, attention_mask
# #
# #     def gen_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
# #         with torch.no_grad():
# #             out = self.bert(input_ids, attention_mask=attention_mask)
# #             out = out["pooler_output"]
# #             # out = out.cpu().numpy()
# #         return out
# #
# #     def preprocess(self, data):
# #         data_path = Path()
# #         if self.config.debug:
# #             data_path = Path(f"./data/{self.config.method}_{self.mode}_debug.pickle")
# #         else:
# #             data_path = Path(f"./data/{self.config.method}_{self.mode}.pickle")
# #         indexes = len(data)
# #         data_list = []
# #         if data_path.exists():
# #             with open(data_path, "rb") as f:
# #                 data_list = pickle.load(f)
# #         else:
# #             self._load_bert()
# #             for i in tqdm(
# #                 range(indexes), total=indexes, desc="preprocessing data ...."
# #             ):
# #                 d = data.iloc[i].to_dict()
# #
# #                 id, mask = self.concat_text(d["text"], d["attribute"], d["around_text"])
# #                 id = torch.tensor([id]).to(self.device)
# #                 mask = torch.tensor([mask]).to(self.device)
# #                 emb = self.gen_embedding(id, mask)
# #                 d["token_id"] = id.to("cpu")
# #                 d["attention_mask"] = mask.to("cpu")
# #                 d["bert_embedding"] = emb
# #                 data_list.append(d)
# #             with open(data_path, "wb") as f:
# #                 pickle.dump(data_list, f)
# #         for data in data_list:
# #             data["token_id"] = data["token_id"].to(self.device)
# #             data["attention_mask"] = data["attention_mask"].to(self.device)
# #             data["bert_embedding"] = data["bert_embedding"].to(self.device)
# #         assert len(data_list) != 0, "data is empty!!"
# #         return data_list
# #
# #     def json_load(self, file_path):
# #         with open(file_path, "r") as f:
# #             return json.load(f)
# #
# #     def padding(self, array, pad, seq_len):
# #         if len(array) >= seq_len:
# #             return array
# #         return array + [pad] * (seq_len - len(array))
# #
# #
# # class ShinraAllSystemDataset(ShinraDataset):
# #     def __init__(
# #         self,
# #         data: pd.DataFrame,
# #         system_num: int,
# #         around_text: bool,
# #         num_tokens: int = 512,
# #         mode: str = "train",
# #     ) -> None:
# #         super().__init__(data, around_text, num_tokens, mode)
# #
# #         self.system_num = system_num
# #
# #     def __getitem__(self, idx):
# #         d = self.data[idx]
# #         text = str(d["text"])
# #
# #         system_pred = torch.Tensor([d[f"system_{i}"] for i in range(self.system_num)])
# #         annotator = d["annotator"]
# #         attribute = d["attribute"]
# #         token_id = d["token_id"]
# #         around_text = d["around_text"]
# #         attention_mask = d["attention_mask"]
# #         bert_embedding = d["bert_embedding"].squeeze(0)
# #
# #         n = sum(
# #             [
# #                 system_pred == annotator,
# #             ]
# #         )
# #         correct = torch.tensor([annotator])
# #         return {
# #             # "idx": idx,
# #             "text": text,
# #             "attribute": "".join(attribute),
# #             "around_text": around_text,
# #             "system_pred": system_pred,
# #             "annotator_decision": annotator,
# #             "decision": system_pred,
# #             "correct": correct,
# #             "tokens": token_id,
# #             "n": n,
# #             "attention_mask": attention_mask,
# #             "bert_embedding": bert_embedding,
# #         }
