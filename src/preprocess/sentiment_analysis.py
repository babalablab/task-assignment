from utils import load_tweet_eval_data
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import polars as pl
from pathlib import Path
import json
from nltk import stem, word_tokenize, pos_tag


# class BasePreprocess:
#     def __init__(self) -> None:
#         pass
#
#     def load_data(self, seed: int = 10):
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
#     def __call__(self, debug, seed=10):
#         assert hasattr(self, "source"), "source must be set"
#         assert hasattr(self, "label"), "label must be set"
#         assert hasattr(self, "data_name"), "data_name must be set"
#         assert hasattr(self, "output_dir"), "output_dir must be set"
#         d = self.load_data()
#         for k, v in d.items():
#             datas = []
#             output_path = self.output_dir / f"{self.data_name}_{self.seed}_{k}.csv"
#
#             if debug:
#                 output_path = (
#                     self.output_dir / f"{self.data_name}_{self.seed}_{k}_debug.csv"
#                 )
#
#             for i in tqdm(range(len(v)), total=len(v), desc=f"Processing {k}"):
#                 text = v[self.source][i]
#                 label = v[self.label][i]
#                 data = self.generate_labels(label)
#                 if "llm_annotatinon" in v.columns:
#                     data["annotator_llm"] = v["llm_annotatinon"][i]
#                 data["text"] = text
#                 data["label"] = label
#                 data["idx"] = v["idx"][i]
#
#                 datas.append(data)
#
#             df = pl.DataFrame(datas)
#             df.write_csv(output_path)


# class SST2Preprocess(BasePreprocess):
#     def __init__(self, output_dir, seed: int = 10) -> None:
#         self.source = "sentence"
#         self.data_name = "sst2"
#         self.output_dir = Path(output_dir)
#         self.annotator_num = 2
#         self.label_num = 2
#         self.label = "label"
#         self.seed = seed
#         self.dist = [[0.9, 0.1], [0.1, 0.9]]
# 
#     def load_data(self, seed: int = 10):
#         return load_sst2_data(seed)
# 
# 
# class IMDBPreprocess(BasePreprocess):
#     def __init__(self, output_dir: Path, seed: int) -> None:
#         self.source = "text"
#         self.data_name = "imdb"
#         self.output_dir = Path(output_dir)
#         self.annotator_num = 2
#         self.label_num = 2
#         self.label = "label"
#         self.dist = [[0.9, 0.1], [0.1, 0.9]]
#         self.seed = seed
# 
#     def load_data(self):
#         return load_imdb_data(self.seed)
# 
# 
# class PoemSentimentPreprocess(BasePreprocess):
#     def __init__(self, output_dir, seed: int = 10) -> None:
#         self.model_kind = "AiManatee/RoBERTa_poem_sentiment"
#         self.output_dir = Path(output_dir)
#         self.source = "verse_text"
#         self.label = "label"
#         self.data_name = "poem_sentiment"
#         self.label_num = 4
#         self.annotator_num = 2
#         self.seed = 10
#         self.dist = [[0.9, 0.1, 0.1, 0.9], [0.1, 0.9, 0.9, 0.1]]
# 
#     def load_data(self):
#         return load_poem_sentiment_data()
# 
# 
# class TweetEvalPreprocess(BasePreprocess):
#     def __init__(self, output_dir, seed) -> None:
#         self.model_kind = "cardiffnlp/twitter-roberta-base-sentiment-latest"
#         self.source = "text"
#         self.data_name = "tweet_eval"
#         self.label = "label"
#         self.output_dir = Path(output_dir)
#         self.label_num = 3
#         self.annotator_num = 3
#         self.seed = seed
#         self.dist = [[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]]
# 
#     def load_data(self):
#         return load_tweet_eval_data()
# 
# 
# class EthosPreprocess(BasePreprocess):
#     def __init__(self, output_dir) -> None:
#         self.source = "text"
#         self.data_name = "ethos"
#         self.label = "label"
#         self.output_dir = Path(output_dir)
#         self.annotator_num = 3
#         self.label_num = 2
#         self.seed = 10
#         self.dist = [[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]]
# 
#     def load_data(self):
#         return load_ethos_data()
# 
# 
# class HateXplainPreprocess(BasePreprocess):
#     def __init__(self, output_dir) -> None:
#         self.source = "text"
#         self.data_name = "hatexplain"
#         self.output_dir = Path(output_dir)
#         self.label = "label"
#         self.annotator_num = 3
#         self.label_num = 3
#         self.seed = 10
#         self.dist = [[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]]
# 
#     def load_data(self):
#         return load_hatexplain_data()
# 
# 
# class NLIPreprocess(BasePreprocess):
#     def __init__(self, output_dir) -> None:
#         super().__init__()
#         self.output_dir = Path("xxx")
#         self.data_name = "xxx"
#         self.label = "label"
#         self.seed: int
# 
#     def __call__(self, debug: bool, seed: int = 10):
#         d = self.load_data()
#         for k, v in d.items():
#             datas = []
#             output_path = self.output_dir / f"{self.data_name}_{self.seed}_{k}.csv"
# 
#             if debug:
#                 output_path = (
#                     self.output_dir / f"{self.data_name}_{self.seed}_{k}_debug.csv"
#                 )
# 
#             for i in tqdm(range(len(v)), total=len(v), desc=f"Processing {k}"):
#                 premise = v["premise"][i]
#                 hypothesis = v["hypothesis"][i]
#                 label = v[self.label][i]
#                 data = self.generate_labels(label)
# 
#                 data["premise"] = premise
#                 data["hypothesis"] = hypothesis
#                 data["label"] = label
#                 data["idx"] = v["idx"][i]
# 
#                 datas.append(data)
# 
#             df = pl.DataFrame(datas)
#             df.write_csv(output_path)
# 
# 
# class MultiNLIPreprocess(NLIPreprocess):
#     def __init__(self, output_dir, seed) -> None:
#         self.data_name = "multinli"
#         self.output_dir = Path(output_dir)
#         self.label = "label"
#         self.annotator_num = 3
#         self.label_num = 3
#         self.seed = seed
#         self.dist = [[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]]
# 
#     def load_data(self):
#         return load_multi_nli_data(self.seed)
# 
# 
# class ANLIPreprocess(NLIPreprocess):
#     def __init__(self, output_dir: str, seed: int) -> None:
#         self.data_name = "anli"
#         self.output_dir = Path(output_dir)
#         self.label = "label"
#         self.annotator_num = 3
#         self.label_num = 3
#         self.seed = seed
#         self.dist = [[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]]
# 
#     def load_data(self):
#         return load_anli_data()
# 
# 
class BasePreprocessVocab:
    def __init__(self) -> None:
        self.load_vocab()
        self.source: str
        self.data_name: str
        self.output_dir: Path
        self.annotator_num: int
        self.label_num: int
        self.label: str
        self.seed: int
        self.dist: list[float]
        self.sentence_idx: int = 0

    def load_data(self):
        raise NotImplementedError

    def load_vocab(self):
        path = Path("./data/word-sets.json")
        with open(path, "r") as f:
            vocab = json.load(f)
        self.positive_words = set(vocab["attribute_sets"]["positive"]["set"])
        self.negative_words = set(vocab["attribute_sets"]["negative"]["set"])

    def eval_sentence(self, sentence):
        assert hasattr(self, "positive_words"), "positive_words must be set"
        assert hasattr(self, "negative_words"), "negative_words must be set"
        assert hasattr(self, "sentence_idx"), "sententce_idx must be set"

        stemmer2 = stem.WordNetLemmatizer()
        sentence = sentence[self.sentence_idx]

        res = [word for (word, tag) in pos_tag(word_tokenize(sentence)) if "NN" in tag]
        stemmed_res = set([stemmer2.lemmatize(r, pos="n") for r in res])
        res_pos = len(self.positive_words.intersection(stemmed_res))
        neg_pos = len(self.negative_words.intersection(stemmed_res))
        if res_pos >= neg_pos and res_pos != 0:
            return 0
        elif res_pos < neg_pos:
            return 1
        else:
            return 2

    def generate_labels(self, sentence_type, label):
        assert hasattr(self, "dist"), "Set Distribution of artificial labels"
        assert hasattr(self, "annotator_num"), "set number of annotators"
        assert hasattr(self, "label_num"), "set number of labels"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        class_idx = np.arange(self.label_num)
        labels = defaultdict(str)
        for i in range(self.annotator_num):
            acc = self.dist[i][sentence_type]
            sampling_dist = np.array(
                [(1 - acc) / (self.label_num - 1)] * self.label_num
            )
            sampling_dist[label] = acc
            anno = np.random.choice(class_idx, p=sampling_dist)
            labels[f"annotator_{i}"] = anno

        return labels

    def __call__(self, debug: bool, seed=10):
        assert hasattr(self, "source"), "source must be set"
        assert hasattr(self, "label"), "label must be set"
        assert hasattr(self, "data_name"), "data_name must be set"
        assert hasattr(self, "output_dir"), "output_dir must be set"
        d = self.load_data()

        for k, v in d.items():
            datas, filtered_datas = [], []
            kinds = v.map_rows(self.eval_sentence)
            v = v.hstack(kinds).rename({"map": "kind"})

            filename = self.output_dir / f"{self.data_name}_{self.seed}_{k}.csv"

            if debug:
                filename = (
                    self.output_dir / f"{self.data_name}_{self.seed}_{k}_debug.csv"
                )
                v = v[:100]
            for i in tqdm(range(len(v)), total=len(v), desc=f"Processing {k}"):
                text = v[self.source][i]
                label = v[self.label][i]
                kind = v["kind"][i]
                data = self.generate_labels(kind, label)
                if "llm_annotatinon" in v.columns:
                    data["annotator_llm"] = v["llm_annotatinon"][i]

                data["text"] = text
                data["label"] = label
                data["idx"] = v["idx"][i]
                data["kind"] = kind

                datas.append(data)

            df = pl.DataFrame(datas)
            print(f"file is saved {filename}")
            df.write_csv(filename)


# class SST2PreprocessVocab(BasePreprocessVocab):
#     def __init__(self, output_dir, seed=10) -> None:
#         super().__init__()
#         self.source = "sentence"
#         self.data_name = "sst2"
#         self.output_dir = Path(output_dir)
#         self.annotator_num = 2
#         self.label_num = 2
#         self.label = "label"
#         self.seed = 10
#         self.dist = [[0.9, 0.1, 0.5], [0.1, 0.9, 0.5]]
#         self.sentence_idx = 0
# 
#     def load_data(self):
#         return load_sst2_data()
# 
# 
# class IMDBPreprocessVocab(BasePreprocessVocab):
#     def __init__(self, output_dir, seed=10) -> None:
#         super().__init__()
#         self.source = "text"
#         self.data_name = "imdb"
#         self.output_dir = Path(output_dir)
#         self.annotator_num = 2
#         self.label_num = 2
#         self.label = "label"
#         self.dist = [[0.9, 0.1, 0.5], [0.1, 0.9, 0.5]]
#         self.sentence_idx = 0
#         self.seed = seed
# 
#     def load_data(self):
#         return load_imdb_data(seed=self.seed)
# 
# 
# class PoemSentimentPreprocessVocab(BasePreprocessVocab):
#     def __init__(self, output_dir) -> None:
#         super().__init__()
#         self.model_kind = "AiManatee/RoBERTa_poem_sentiment"
#         self.output_dir = Path(output_dir)
#         self.source = "verse_text"
#         self.label = "label"
#         self.data_name = "poem_sentiment"
#         self.label_num = 4
#         self.annotator_num = 2
#         self.dist = [[0.9, 0.1, 0.5], [0.1, 0.9, 0.5]]
#         self.sentence_idx = 1
#         self.seed = 10
# 
#     def load_data(self):
#         return load_poem_sentiment_data()
# 
# 
class TweetEvalPreprocessVocab(BasePreprocessVocab):
    def __init__(self, output_dir, seed) -> None:
        super().__init__()
        self.model_kind = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.source = "text"
        self.data_name = "tweet_eval"
        self.label = "label"
        self.output_dir = Path(output_dir)
        self.label_num = 3
        self.annotator_num = 3
        self.dist = [[0.9, 0.1, 0.5], [0.1, 0.9, 0.5], [0.3, 0.3, 0.7]]
        self.seed = seed
        self.sentence_idx = 1

    def load_data(self):
        return load_tweet_eval_data(self.seed)


# class EthosPreprocessVocab(BasePreprocessVocab):
#     def __init__(self, output_dir) -> None:
#         super().__init__()
#         self.model_kind = "cardiffnlp/twitter-roberta-base-sentiment-latest"
#         self.source = "text"
#         self.data_name = "ethos"
#         self.label = "label"
#         self.output_dir = Path(output_dir)
#         self.annotator_num = 3
#         self.label_num = 2
#         self.dist = [[0.9, 0.1, 0.5], [0.1, 0.9, 0.5], [0.3, 0.3, 0.7]]
#         self.sentence_idx = 1
#         self.seed = 10
# 
#     def load_data(self):
#         return load_ethos_data()
# 
# 
# class HateXplainPreprocessVocab(BasePreprocessVocab):
#     def __init__(self, output_dir) -> None:
#         super().__init__()
#         self.source = "text"
#         self.data_name = "hatexplain"
#         self.output_dir = Path(output_dir)
#         self.label = "label"
#         self.annotator_num = 3
#         self.label_num = 3
#         self.dist = [[0.9, 0.1, 0.5], [0.1, 0.9, 0.5], [0.3, 0.3, 0.7]]
#         self.sentence_idx = 1
#         self.seed = 10
# 
#     def load_data(self):
#         return load_hatexplain_data()
