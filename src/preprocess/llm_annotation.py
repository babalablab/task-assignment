# from transformers import pipeline
# from datasets import load_dataset
# import hydra
# from hydra.utils import instantiate
# import polars as pl
# from string import Template
# from tqdm import tqdm
# import re
# from pathlib import Path
# from datasets import load_dataset
# 
# from huggingface_hub._login import _login
# 
# _login(token="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", add_to_git_credential=False)
# 
# 
# class LLMAnnotation:
#     def __init__(
#         self, prompt_file: str, output_dir: str, data_name: str, model_name: str
#     ):
#         self.pipe = pipeline(
#             "text-generation",
#             model=model_name,
#             token=True,
#             truncation=True,
#             device_map="auto",
#         )
#         self.prompt_file = prompt_file
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#         self.data_name = data_name
#         self.data = data_name.split("/")[1]
# 
#     def __call__(self):
#         self.load_data()
#         prompt = self.load_prompt()
#         dataset = self.load_data()
#         assert hasattr(self, "source")
#         assert hasattr(self, "label")
#         assert hasattr(self, "idx")
#         for k, v in dataset.items():
#             datas = []
#             for i in tqdm(range(len(v)), total=len(v), desc=f"annotation {k}"):
#                 idx = v[self.idx][i]
#                 label = v[self.label][i]
#                 input_text = v[self.source][i]
#                 llama_prompt = self.build_prompt(prompt, input_text)
#                 res, raw_output = self.annotate(llama_prompt)
#                 data = {
#                     "idx": idx,
#                     "sentence": input_text,
#                     "label": label,
#                     "llm_annotation": res,
#                     "llm_output": raw_output,
#                 }
#                 datas.append(data)
#             self.save_result(datas, k)
#             self._post_process(datas)
# 
#     def _post_process(self, datas: list[dict]):
#         pass
# 
#     def load_prompt(self):
#         with open(self.prompt_file, "r") as f:
#             prompt = f.read()
#         return prompt
# 
#     def build_prompt(self, prompt, input_text):
#         prompt_template = Template(prompt)
#         return prompt_template.substitute(target_text=input_text)
# 
#     def load_data(self):
#         raise NotImplementedError
# 
#     def annotate(self, prompt):
#         messages = [
#             {"role": "user", "content": prompt},
#         ]
#         res = False
#         while not res:
#             output = self.pipe(
#                 messages, max_length=2048, pad_token_id=self.pipe.tokenizer.eos_token_id
#             )
#             response = output[0]["generated_text"][-1]["content"].replace("\n", "")
#             val_result = self.validate_result(response)
#             res = True if val_result is not None else False
#             if isinstance(val_result, str) and len(val_result) >= 3:
#                 res &= val_result[1].isdigit()
#             else:
#                 res = False
#         assert val_result[1].isdigit(), f"result is {val_result}"
#         return int(val_result[1]), response
# 
#     def save_result(self, data, k):
#         file_path = self.output_dir.absolute() / f"{self.data}_{k}.csv"
#         df = pl.DataFrame(data)
#         df.write_csv(file_path)
# 
#     def validate_result(self, content):
#         res = re.search(r"{[0-1]}", content)
#         return False if res is None else res.group()
# 
# 
# class SST2LLMAnnotation(LLMAnnotation):
#     def __init__(
#         self, prompt_file: str, output_dir: str, data_name: str, model_name: str
#     ):
#         super().__init__(prompt_file, output_dir, data_name, model_name)
#         self.source = "sentence"
#         self.label = "label"
#         self.idx = "idx"
# 
#     def load_data(self):
#         train_data = load_dataset("gpt3mix/sst2", split="train")
#         validation_data = load_dataset("gpt3mix/sst2", split="validation")
#         test_data = load_dataset("gpt3mix/sst2", split="test")
# 
#         def extract_data(data):
#             d = []
#             for i, di in enumerate(data):
#                 token = di["text"]
#                 label = di["label"]
#                 d.append({"sentence": token, "label": label, "idx": i})
# 
#             return pl.DataFrame(d)
# 
#         train_df = extract_data(train_data)
#         valid_df = extract_data(validation_data)
#         test_df = extract_data(test_data)
# 
#         return {"train": train_df, "validation": valid_df, "test": test_df}
# 
# 
# class IMDBLLMAnnotation(LLMAnnotation):
#     def __init__(
#         self, prompt_file: str, output_dir: str, data_name: str, model_name: str
#     ):
#         super().__init__(prompt_file, output_dir, data_name, model_name)
#         self.source = "text"
#         self.label = "label"
#         self.idx = "idx"
# 
#     def load_data(self):
#         splits = {
#             "train": "plain_text/train-00000-of-00001.parquet",
#             "test": "plain_text/test-00000-of-00001.parquet",
#             "unsupervised": "plain_text/unsupervised-00000-of-00001.parquet",
#         }
#         train_df = pl.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["train"])
#         train_df = train_df.with_row_index(name="idx")
#         test_df = pl.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["test"])
#         test_df = test_df.with_row_index(name="idx")
# 
#         return {"train": train_df, "test": test_df}
# 
# 
# class PoemSentimentLLMAnnotation(LLMAnnotation):
#     def __init__(
#         self, prompt_file: str, output_dir: str, data_name: str, model_name: str
#     ):
#         super().__init__(prompt_file, output_dir, data_name, model_name)
#         assert "poem_sentiment" in prompt_file, f"this prompt it invalid {prompt_file}"
#         self.source = "verse_text"
#         self.label = "label"
#         self.idx = "idx"
# 
#     def load_data(self):
#         splits = {
#             "train": "data/train-00000-of-00001.parquet",
#             "validation": "data/validation-00000-of-00001.parquet",
#             "test": "data/test-00000-of-00001.parquet",
#         }
#         train_df = pl.read_parquet(
#             "hf://datasets/google-research-datasets/poem_sentiment/" + splits["train"]
#         )
#         validation_df = pl.read_parquet(
#             "hf://datasets/google-research-datasets/poem_sentiment/"
#             + splits["validation"]
#         )
#         test_df = pl.read_parquet(
#             "hf://datasets/google-research-datasets/poem_sentiment/" + splits["test"]
#         )
#         train_df = train_df.with_row_index(name="idx")
#         validation_df = validation_df.with_row_index(name="idx")
#         test_df = test_df.with_row_index(name="idx")
# 
#         return {"train": train_df, "validation": validation_df, "test": test_df}
# 
#     def validate_result(self, content):
#         res = re.search(r"{[0-3]}", content)
#         return False if res is None else res.group()
# 
# 
# class TweetEvalLLMAnnotation(LLMAnnotation):
#     def __init__(
#         self, prompt_file: str, output_dir: str, data_name: str, model_name: str
#     ):
#         super().__init__(prompt_file, output_dir, data_name, model_name)
#         self.source = "text"
#         self.label = "annotation_label"
#         self.idx = "id"
# 
#     def load_data(self):
#         self.df = pl.read_csv("./data/tweet_eval_annotated.csv")
#         return {"train": self.df}
# 
#     def _post_process(self, datas: list[dict]):
#         llm_annotation_df = pl.DataFrame(datas)
#         self.df = self.df.join(
#             llm_annotation_df.select("llm_annotation", "idx"),
#             left_on="id",
#             right_on="idx",
#         )
#         assert len(self.df) == 3000
#         self.df.write_csv(self.output_dir / "tweet_eval_annotated_with_llm.csv")
# 
#     def validate_result(self, content):
#         res = re.search(r"{[0-3]}", content)
#         return False if res is None else res.group()
# 
# 
# @hydra.main(
#     version_base=None, config_path="../../config/", config_name="llm_annotation"
# )
# def main(config):
#     annotator = instantiate(config.annotator)
#     annotator()
# 
# 
# if __name__ == "__main__":
#     main()
