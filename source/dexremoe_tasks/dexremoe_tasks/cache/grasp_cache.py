"""Validated grasp cache.

Reads `assets/grasp_cache_isaac_validated/<split>/grasp_<scale>_<shape>.npz`.
Each file has been filtered by the 60-step stability check.

Contracts:
  - The file a row came from determines which physics geometry to use.
    `shape_id` in the npz is a random preprocess label and MUST NOT be used
    to select geometry or to enter observations.
  - Train sampling is stratified across 3 size buckets: small / medium / large.
  - Eval sampling is without replacement, fixed seed, per (scale, shape) file.
  - `b3` / `b0` size encodings are open-ended; `data_bucket` only supports
    the 3 training buckets.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


# 11 scales: 9 train + 2 test (s066, s09 are out-of-distribution)
SCALE_NAMES: tuple[str, ...] = (
    "s07", "s072", "s074", "s076", "s078", "s08", "s082", "s084", "s086",
    "s066", "s09",
)
TRAIN_SCALES: tuple[str, ...] = SCALE_NAMES[:9]
TEST_SCALES: tuple[str, ...] = SCALE_NAMES[9:]

GEOM_NAMES: tuple[str, ...] = ("cuboid", "cylinder", "sphere")


def parse_scale_name(name: str) -> float:
    """`s07`->0.70, `s072`->0.72, `s08`->0.80, `s09`->0.90, `s066`->0.66."""
    s = name[1:]
    if len(s) == 2 and s[0] == "0":
        s = s[1] + "0"
    return int(s) / 100.0


SCALE_VALUES: dict[str, float] = {n: parse_scale_name(n) for n in SCALE_NAMES}

# 3-bucket partition used only for `data_bucket` (expert training data selection)
BUCKET_3: dict[str, str] = {
    "s07": "small", "s072": "small", "s074": "small",
    "s076": "medium", "s078": "medium", "s08": "medium",
    "s082": "large", "s084": "large", "s086": "large",
}

HORA_BASE_SIZE_M: float = 0.08
"""Object characteristic size = HORA_BASE_SIZE_M * scale."""


@dataclass
class _FileEntry:
    path: str
    scale_name: str
    shape_name: str
    bucket: str | None
    n_rows: int


def _parse_filename(path: str) -> tuple[str, str]:
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("_")
    if len(parts) != 3 or parts[0] != "grasp":
        raise ValueError(f"Unexpected cache filename: {path}")
    scale_name, shape_name = parts[1], parts[2]
    if shape_name not in GEOM_NAMES:
        raise ValueError(f"Unknown shape '{shape_name}' in {path}")
    return scale_name, shape_name


def _scan(root_dir: str, split: str) -> list[_FileEntry]:
    pattern = os.path.join(root_dir, split, "grasp_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No npz files found at {pattern}")
    out = []
    for path in files:
        scale_name, shape_name = _parse_filename(path)
        bucket = BUCKET_3.get(scale_name)
        with np.load(path) as d:
            n = int(d["joint_isaac"].shape[0])
        out.append(_FileEntry(path, scale_name, shape_name, bucket, n))
    return out


class GraspCache:
    """Validated grasp pool for one split."""

    def __init__(self, root_dir: str, split: Literal["train", "test"]):
        self.root_dir = root_dir
        self.split = split
        self.entries = _scan(root_dir, split)

        joint_l, pos_l, quat_l, mass_l, scale_l, geom_l, scaleidx_l, file_l = \
            [], [], [], [], [], [], [], []
        for file_i, e in enumerate(self.entries):
            s_idx = SCALE_NAMES.index(e.scale_name)
            g_idx = GEOM_NAMES.index(e.shape_name)
            with np.load(e.path) as d:
                n = int(d["joint_isaac"].shape[0])
                joint_l.append(d["joint_isaac"].astype(np.float32))
                pos_l.append(d["obj_pos_isaac"].astype(np.float32))
                quat_l.append(d["obj_quat_wxyz"].astype(np.float32))
                mass_l.append(d["mass"].astype(np.float32))
                scale_l.append(d["scale"].astype(np.float32))
                geom_l.append(np.full(n, g_idx, dtype=np.int64))
                scaleidx_l.append(np.full(n, s_idx, dtype=np.int64))
                file_l.append(np.full(n, file_i, dtype=np.int64))

        self.joint = torch.from_numpy(np.concatenate(joint_l))         # (N, 16)
        self.pos = torch.from_numpy(np.concatenate(pos_l))             # (N, 3)
        self.quat = torch.from_numpy(np.concatenate(quat_l))           # (N, 4) wxyz
        self.mass = torch.from_numpy(np.concatenate(mass_l))           # (N,)
        self.scale = torch.from_numpy(np.concatenate(scale_l))         # (N,)
        self.geom_idx = torch.from_numpy(np.concatenate(geom_l))       # (N,)
        self.scale_idx = torch.from_numpy(np.concatenate(scaleidx_l))  # (N,)
        self.src_file = torch.from_numpy(np.concatenate(file_l))       # (N,)

        # Per-bucket row id cache (train only)
        self._bucket_ids: dict[str, torch.Tensor] = {}
        if split == "train":
            bucket_of_file = [e.bucket for e in self.entries]
            row_bucket = np.array([bucket_of_file[i] for i in self.src_file.tolist()], dtype=object)
            for b in ("small", "medium", "large"):
                ids = np.where(row_bucket == b)[0]
                if ids.size == 0:
                    raise RuntimeError(f"Bucket '{b}' has no rows in train split")
                self._bucket_ids[b] = torch.from_numpy(ids)

        # Per-file row ids (eval)
        self._file_ids: list[torch.Tensor] = []
        for file_i in range(len(self.entries)):
            ids = np.where(self.src_file.numpy() == file_i)[0]
            self._file_ids.append(torch.from_numpy(ids))

    def __len__(self) -> int:
        return int(self.joint.shape[0])

    def num_per_bucket(self) -> dict[str, int]:
        return {b: int(ids.numel()) for b, ids in self._bucket_ids.items()}

    def describe(self) -> str:
        lines = [f"GraspCache(split={self.split}, N={len(self)})"]
        for e in self.entries:
            lines.append(f"  {os.path.basename(e.path):32s} n={e.n_rows:5d} "
                         f"scale={e.scale_name:5s} bucket={e.bucket or '-'}")
        if self.split == "train":
            lines.append(f"  bucket totals: {self.num_per_bucket()}")
        return "\n".join(lines)

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
        bucket: str | None = None,
        device: str = "cuda:0",
    ) -> dict[str, torch.Tensor]:
        """Training sample. `bucket=None` -> stratified 1/3 across 3 buckets.
        `bucket in {small,medium,large}` -> draw only from that bucket."""
        if self.split != "train":
            raise RuntimeError("GraspCache.sample() is for train split; use sample_eval().")

        if bucket is None:
            per = n // 3
            rem = n - per * 3
            parts = []
            for i, b in enumerate(("small", "medium", "large")):
                ids = self._bucket_ids[b]
                k = per + (1 if i < rem else 0)
                pick = rng.choice(ids.numel(), size=k, replace=True)
                parts.append(ids.numpy()[pick])
            row_ids = np.concatenate(parts, axis=0)
        else:
            if bucket not in self._bucket_ids:
                raise ValueError(f"Unknown bucket '{bucket}'. Choose from small/medium/large.")
            ids = self._bucket_ids[bucket]
            pick = rng.choice(ids.numel(), size=n, replace=True)
            row_ids = ids.numpy()[pick]

        return self._gather(row_ids, device)

    def sample_eval(
        self,
        n_per_file: int,
        seed: int = 0,
        device: str = "cuda:0",
    ) -> dict[str, torch.Tensor]:
        """Fixed-seed, without-replacement eval sampling."""
        rng = np.random.default_rng(seed)
        parts = []
        counts = []
        for ids in self._file_ids:
            n_avail = ids.numel()
            k = min(n_per_file, n_avail)
            pick = rng.choice(n_avail, size=k, replace=False)
            parts.append(ids.numpy()[pick])
            counts.append(k)
        row_ids = np.concatenate(parts, axis=0)
        out = self._gather(row_ids, device)
        out["rows_per_file"] = torch.tensor(counts, dtype=torch.int64)
        return out

    def _gather(self, row_ids: np.ndarray, device: str) -> dict[str, torch.Tensor]:
        idx = torch.from_numpy(np.asarray(row_ids, dtype=np.int64))
        joint = self.joint[idx].to(device, non_blocking=True)
        pos = self.pos[idx].to(device, non_blocking=True)
        quat = self.quat[idx].to(device, non_blocking=True)
        mass = self.mass[idx].to(device, non_blocking=True)
        scale = self.scale[idx].to(device, non_blocking=True)
        geom_idx = self.geom_idx[idx].to(device, non_blocking=True)
        scale_idx = self.scale_idx[idx].to(device, non_blocking=True)
        size_m = (HORA_BASE_SIZE_M * scale).to(device, non_blocking=True)
        return {
            "joint": joint,
            "pos": pos,
            "quat": quat,
            "mass": mass,
            "scale": scale,
            "geom_idx": geom_idx,
            "scale_idx": scale_idx,
            "size_m": size_m,
            "row_ids": idx.to(device),
        }
