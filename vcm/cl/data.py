"""Benchmarks: Permuted MNIST (domain-incremental), Split CIFAR-10/100 (task-incremental).

Raw files are downloaded once into `data/raw/` and decoded into `data/*.npz`.
MNIST comes from the PyTorch S3 mirror (checksums verified); CIFAR comes from the
fast.ai S3 mirror, which stores PNG folders (class order = sorted folder names, which
matches the official label order for both CIFAR-10 and CIFAR-100).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import tarfile
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

MNIST_URLS = ["https://ossci-datasets.s3.amazonaws.com/mnist/",
              "https://storage.googleapis.com/cvdf-datasets/mnist/"]
MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
CIFAR_URLS = {"cifar10": "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz",
              "cifar100": "https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz"}


@dataclass
class Task:
    id: int
    name: str
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    classes: list[str]                 # name of each local label
    groups: list[str] = field(default_factory=list)  # coarse category of each label, if known

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def release_train(self) -> None:
        """Drop the training set once it has been learned (keeps memory flat)."""
        self.x_train, self.y_train = self.x_train[:0].clone(), self.y_train[:0].clone()


class LazyTasks(Sequence):
    """Tasks are materialised on first access; everything random is fixed up front."""

    def __init__(self, n: int, make: Callable[[int], Task]):
        self._n, self._make, self._cache = n, make, {}

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(self._n))]
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError(i)
        if i not in self._cache:
            self._cache[i] = self._make(i)
        return self._cache[i]


@dataclass
class Benchmark:
    name: str
    tasks: Sequence[Task]
    input_shape: tuple[int, ...]
    n_outputs: int                     # outputs per head
    task_id_at_test: bool              # True: task-incremental; False: domain-incremental


# -- download helpers ---------------------------------------------------------------
def _download(urls: list[str], dest: Path, md5: str | None = None) -> Path:
    if dest.exists() and (md5 is None or _md5(dest) == md5):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if md5 is None or _md5(dest) == md5:
                return dest
            last = RuntimeError(f"checksum mismatch for {url}")
        except OSError as e:  # try the next mirror
            last = e
    raise RuntimeError(f"could not download {dest.name}: {last}")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    shape = tuple(int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load_mnist() -> dict[str, np.ndarray]:
    cache = DATA_DIR / "mnist.npz"
    if not cache.exists():
        raw = {}
        for name, md5 in MNIST_FILES.items():
            path = _download([u + name for u in MNIST_URLS], DATA_DIR / "raw" / name, md5)
            raw[name.split("-")[0] + "_" + name.split("-")[1]] = _read_idx(path)
        np.savez_compressed(cache, x_train=raw["train_images"], y_train=raw["train_labels"],
                            x_test=raw["t10k_images"], y_test=raw["t10k_labels"])
    with np.load(cache) as d:
        return {k: d[k] for k in d.files}


def load_cifar(which: str) -> dict[str, np.ndarray]:
    cache = DATA_DIR / f"{which}.npz"
    if not cache.exists():
        from PIL import Image

        archive = _download([CIFAR_URLS[which]], DATA_DIR / "raw" / f"{which}.tgz")
        items = {"train": [], "test": []}
        with tarfile.open(archive) as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".png"):
                    continue
                parts = member.name.split("/")
                split, cls, group = parts[1], parts[-2], parts[2] if len(parts) == 5 else parts[-2]
                img = np.asarray(Image.open(io.BytesIO(tar.extractfile(member).read())).convert("RGB"))
                items[split].append((cls, group, member.name, img))
        classes = sorted({c for c, _, _, _ in items["train"]})
        group_of = {c: g for c, g, _, _ in items["train"]}
        out = {"classes": np.array(classes), "groups": np.array([group_of[c] for c in classes])}
        for split, rows in items.items():
            rows.sort(key=lambda r: r[2])
            out[f"x_{split}"] = np.stack([r[3] for r in rows])
            out[f"y_{split}"] = np.array([classes.index(r[0]) for r in rows], dtype=np.int64)
        np.savez_compressed(cache, **out)
    with np.load(cache) as d:
        return {k: d[k] for k in d.files}


# -- benchmarks ---------------------------------------------------------------------
def _split_val(n: int, val_fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    idx = rng.permutation(n)
    n_val = int(round(n * val_fraction))
    return idx[n_val:], idx[:n_val]


def permuted_mnist(
    num_tasks: int = 10,
    seed: int = 0,
    val_fraction: float = 0.1,
    train_per_task: int | None = None,
    keep_first_full: bool = False,
) -> Benchmark:
    """Task 0 is plain MNIST; task t>0 applies a fixed random pixel permutation."""
    d = load_mnist()
    rng = np.random.default_rng(seed)
    x_tr = torch.tensor(d["x_train"].reshape(-1, 784), dtype=torch.float32) / 255.0
    x_te = torch.tensor(d["x_test"].reshape(-1, 784), dtype=torch.float32) / 255.0
    y_tr = torch.tensor(d["y_train"], dtype=torch.long)
    y_te = torch.tensor(d["y_test"], dtype=torch.long)
    plan = []
    for t in range(num_tasks):
        perm = torch.arange(784) if t == 0 else torch.tensor(rng.permutation(784))
        tr, va = _split_val(len(x_tr), val_fraction, rng)
        if train_per_task is not None and not (keep_first_full and t == 0):
            tr = tr[:train_per_task]
        plan.append((perm, torch.tensor(tr), torch.tensor(va)))

    def make(t: int) -> Task:
        perm, tr, va = plan[t]
        return Task(
            id=t, name=f"perm{t}",
            x_train=x_tr[tr][:, perm], y_train=y_tr[tr],
            x_val=x_tr[va][:, perm], y_val=y_tr[va],
            x_test=x_te[:, perm], y_test=y_te,
            classes=[str(i) for i in range(10)], groups=["digit"] * 10,
        )

    return Benchmark("pmnist", LazyTasks(num_tasks, make), (784,), 10, task_id_at_test=False)


def split_cifar(
    which: str = "cifar10",
    classes_per_task: int | None = None,
    seed: int = 0,
    val_fraction: float = 0.1,
    train_per_task: int | None = None,
    num_tasks: int | None = None,
    keep_first_full: bool = False,
) -> Benchmark:
    """Consecutive blocks of classes in the official label order; labels are local per task."""
    d = load_cifar(which)
    classes_per_task = classes_per_task or (2 if which == "cifar10" else 10)
    names, groups = list(d["classes"]), list(d["groups"])
    mean = d["x_train"].reshape(-1, 3).mean(0) / 255.0
    std = d["x_train"].reshape(-1, 3).std(0) / 255.0

    def prep(x: np.ndarray) -> torch.Tensor:
        x = (x.astype(np.float32) / 255.0 - mean) / std
        return torch.tensor(x.transpose(0, 3, 1, 2).copy())

    rng = np.random.default_rng(seed)
    n_tasks = len(names) // classes_per_task
    if num_tasks is not None:
        n_tasks = min(n_tasks, num_tasks)
    tasks = []
    for t in range(n_tasks):
        cls = list(range(t * classes_per_task, (t + 1) * classes_per_task))
        local = {c: i for i, c in enumerate(cls)}
        tr_mask = np.isin(d["y_train"], cls)
        te_mask = np.isin(d["y_test"], cls)
        x_all, y_all = d["x_train"][tr_mask], d["y_train"][tr_mask]
        tr, va = _split_val(len(x_all), val_fraction, rng)
        if train_per_task is not None and not (keep_first_full and t == 0):
            tr = tr[:train_per_task]
        to_local = np.vectorize(local.get)
        tasks.append(Task(
            id=t, name=f"{which}-{t}",
            x_train=prep(x_all[tr]), y_train=torch.tensor(to_local(y_all[tr])),
            x_val=prep(x_all[va]), y_val=torch.tensor(to_local(y_all[va])),
            x_test=prep(d["x_test"][te_mask]), y_test=torch.tensor(to_local(d["y_test"][te_mask])),
            classes=[names[c] for c in cls], groups=[groups[c] for c in cls],
        ))
    return Benchmark(f"split_{which}", tasks, (3, 32, 32), classes_per_task, task_id_at_test=True)


def make_benchmark(name: str, **kw) -> Benchmark:
    if name == "pmnist":
        return permuted_mnist(**kw)
    if name in ("split_cifar10", "split_cifar100"):
        return split_cifar(name.split("_")[1], **kw)
    raise ValueError(f"unknown benchmark {name!r}")
