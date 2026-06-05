import numpy as np
from typing import Any, Tuple
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
import jax.numpy as jnp
import jax
from jax import jit
from functools import partial

import gc


# def device():
#     device = ""
#     if torch.cuda.is_available():
#         device = "cuda"
#     elif torch.backends.mps.is_available():
#         device = "mps"
#     else:
#         device = "cpu"
#     return device


class ICrowd:
    def __init__(
        self,
        threshold: float,
        debug: bool,
        seed: int,
        output_dir: str,
        system_num: int,
        dataset_name: str,
        filter_ratio: float = 0.5,
        data_num: int = 0,
        sampling_rate: float = 1.0,
    ) -> None:
        self.threshold = threshold
        self.data_num = data_num
        self.sampling_rate = sampling_rate
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.system_num = system_num
        self.dataset_name = dataset_name
        self.filter_ratio = filter_ratio
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_annotations: jnp.ndarray
        self.train_labels: jnp.ndarray
        self.test_annotations: jnp.ndarray
        self.test_labels: jnp.ndarray

    def __call__(self, train_dataset, test_dataset) -> Any:
        N, x = self.preprocess(train_dataset, test_dataset)
        print(f"train annotations shape {self.train_annotations.shape}")
        print(f"test annotations shape {self.test_annotations.shape}")
        pt, pw = np.array([]), np.array([])
        print(f"N {N} x.shape {x.shape}")
        sd = self.calc_sd(N, x).block_until_ready()
        print(f"cals sd is done {sd.shape}")
        pt = self.calc_pt(N, sd).block_until_ready()

        pw = self.calc_pw(pt)
        pw = np.array(pw)

        accs = pw @ pt
        test_acc = accs[:, self.train_annotations.shape[0] :]
        train_acc = accs[:, : self.train_annotations.shape[0]]
        accs = accs.swapaxes(0, 1)
        train_acc = train_acc.swapaxes(0, 1)
        test_acc = test_acc.swapaxes(0, 1)
        return (accs, train_acc, test_acc)

    def preprocess(
        self, train_dataset: Dataset, test_dataset: Dataset
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError

    def extract_data(self, dataset: Dataset, train: bool = False):
        feats, annotations, labels = jnp.array([]), jnp.array([]), jnp.array([])

        def stack_jnp(arrays, array):
            return array if arrays.shape == (0,) else jnp.vstack([arrays, array])

        for i in tqdm(range(len(dataset)), total=len(dataset), desc="exracting data"):
            data = dataset.__getitem__(i)
            feat = data["embedding"].numpy()
            annotation = data["annotations"].numpy()
            label = data["label"].numpy()
            feats = stack_jnp(feats, feat)
            annotations = stack_jnp(annotations, annotation)
            labels = stack_jnp(labels, label)

        if train:
            self.train_annotations = annotations
            self.train_labels = labels
        else:
            self.test_annotations = annotations
            self.test_labels = labels
        return feats

    @partial(jit, static_argnums=(0,))
    def calc_sd(self, N: int, x: np.ndarray):
        print("calculating S")
        S = self.calc_S(N, x)
        S_mask = S <= self.threshold
        S = jnp.where(S_mask, 0, S)
        del S_mask
        gc.collect()

        print("calculating D")
        D_vec = jnp.sum(S, axis=1)

        print("expanding D")
        D = jnp.diag(D_vec)
        print("calculating D_sqrt")
        D_sqrt = jnp.sqrt(jnp.linalg.inv(D))

        del D, D_vec
        gc.collect()
        print("calculating Sd")
        Sd = D_sqrt @ S @ D_sqrt
        print("checking nan")
        Sd = jnp.nan_to_num(Sd)
        return Sd

    @partial(jit, static_argnums=(0, 1))
    def calc_pt(self, N: int, Sd: np.ndarray):
        print(f"calculating pt {N}")
        print(f"shape of Sd {Sd.shape}")

        def eval_cond(*args):
            args = args[0]
            prev_p = args["prev_p"]
            p = args["p"]
            diff = jnp.abs(jnp.sum(prev_p - p))
            return diff > 1e-3

        def calc_p(*args):
            args = args[0]
            p = args["p"]
            alpha = args["alpha"]
            q = args["q"]
            Sd = args["Sd"]
            next_p = 1 / (1 + alpha) * p @ Sd + alpha / (1 + alpha) * q
            out = {"prev_p": p, "p": next_p, "alpha": alpha, "q": q, "Sd": Sd}
            return out

        def calc_pt(q, i):
            q_i = q[i]
            p = q_i.copy()
            alpha = 1
            prev_p = p + 1e4
            pt = jax.lax.while_loop(
                eval_cond,
                calc_p,
                {"prev_p": prev_p, "p": p, "alpha": alpha, "q": q_i, "Sd": Sd},
            )
            return q, pt["p"]

        _, pt = jax.lax.scan(calc_pt, jnp.eye(N), jnp.arange(N))

        pt = jnp.nan_to_num(pt)
        print("calt pt is done")
        return pt

    def calc_S(self, N: int, x: jnp.ndarray):
        raise NotImplementedError

    def calc_pw(self, pt: jnp.ndarray):
        raise NotImplementedError

    def calc_distance(self, x1: jnp.ndarray, x2: jnp.ndarray):
        raise NotImplementedError


class NLPICrowd(ICrowd):
    def __init__(
        self,
        threshold: float,
        debug: bool,
        seed: int,
        output_dir: str,
        system_num: int,
        dataset_name: str,
        sampling_rate: float = 1.0,
    ) -> None:
        super().__init__(
            threshold=threshold,
            debug=debug,
            seed=seed,
            output_dir=output_dir,
            system_num=system_num,
            dataset_name=dataset_name,
            data_num=0,
            sampling_rate=sampling_rate,
        )

    def preprocess(self, train_dataset: Dataset, test_dataset: Dataset):
        train_data = self.extract_data(train_dataset, train=True)
        test_data = self.extract_data(test_dataset)
        N = len(train_data) + len(test_data)
        x = jnp.vstack([train_data, test_data])
        return N, x

    @partial(jit, static_argnums=(0,))
    def calc_S(self, N: int, x: jnp.ndarray):
        def dist_tensor(x, xi):
            output = []
            for i in range(x.shape[0]):
                output.append(self.calc_distance(xi, x[i]))
            return (x, jnp.array(output))

        _, S = jax.lax.scan(dist_tensor, x, x, length=x.shape[0])
        return S

    @partial(jit, static_argnums=(0,))
    def calc_distance(self, x1: jnp.ndarray, x2: jnp.ndarray):
        return jnp.dot(x1, x2) / (jnp.linalg.norm(x1) * jnp.linalg.norm(x2))

    def calc_pw(self, pt: jnp.ndarray):
        pw = jnp.array([])
        annotator_num = self.train_annotations.shape[1]
        for i in tqdm(range(annotator_num), total=annotator_num, desc="calculating pw"):
            pwi = (self.train_annotations[:, i] == self.train_labels[:, 0]).astype(
                jnp.int16
            )
            pwi = jnp.hstack((pwi, jnp.zeros(pt.shape[0] - pwi.shape[0])))
            pwi = pwi @ pt
            pw = pwi if pw.shape == (0,) else np.vstack((pw, pwi))
        return pw


# class CVICrowd(NLPICrowd):
#     def preprocess(self, train_dataset: Dataset, test_dataset: Dataset):
#         train_data = self.extract_data(train_dataset, train=True)
#         test_data = self.extract_data(test_dataset)
#         N = len(train_data) + len(test_data)
#         x = jnp.vstack([train_data, test_data])
#         return N, x


# class SpiraliCrowd(ICrowd):
#     def preprocess(self, train_dataset: Dataset, test_dataset: Dataset):
#         train_data = self.extract_data(train_dataset, train=True)
#         test_data = self.extract_data(test_dataset)
#         N = len(train_data) + len(test_data)
#         x = jnp.vstack([train_data, test_data])
#         return N, x
#
#     @partial(jit, static_argnums=(0,))
#     def calc_S(self, N: int, x: jnp.ndarray):
#         def dist_tensor(x, xi):
#             output = []
#             for i in range(x.shape[0]):
#                 output.append(self.calc_distance(xi, x[i]))
#             return (x, jnp.array(output))
#
#         _, S = jax.lax.scan(dist_tensor, x, x, length=x.shape[0])
#         maxS = S.max()
#
#         def normalized_dist_tensor(x, xi):
#             output = []
#             for i in range(x.shape[0]):
#                 output.append(1 - self.calc_distance(xi, x[i]) / maxS)
#             return (x, jnp.array(output))
#
#         _, S = jax.lax.scan(normalized_dist_tensor, x, x, length=x.shape[0])
#
#         return S
#
#     def calc_pw(self, pt: np.ndarray):
#         pw = jnp.array([])
#         annotator_num = self.train_annotations.shape[1]
#         for i in tqdm(range(annotator_num), total=annotator_num, desc="calculating pw"):
#             pwi = (self.train_annotations[:, i] == self.train_labels[:, 0]).astype(
#                 jnp.int16
#             )
#             pwi = jnp.hstack((pwi, jnp.zeros(pt.shape[0] - pwi.shape[0])))
#             pwi = pwi @ pt
#             pw = pwi if pw.shape == (0,) else np.vstack((pw, pwi))
#         return pw
#
#     @partial(jit, static_argnums=(0,))
#     def calc_distance(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
#         return jnp.sqrt(jnp.sum((x1 - x2) ** 2))
