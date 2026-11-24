"""内置旧卷样例检查点。

- ckpt-demo-3rank    : 3 个 rank，长度 [10, 7, 13]，step=42000，数据确定可复现
- ckpt-step-mismatch : 2 个 rank，训练步不一致（用于演示失败保留旧发布）
- ckpt-missing-rank  : manifest 声明 3 个 rank，缺 rank_2.npz（演示缺片）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from storage import OldVolume

SAMPLE_VOLUME_ID = "oldvol-7f3a9c21"


def _write_rank(ckpt_dir: Path, r: int, param, m, v, step: int) -> None:
    np.savez(ckpt_dir / f"rank_{r}.npz",
             param=param, adam_m=m, adam_v=v, step=np.int64(step))


def _manifest(ckpt_dir: Path, ckpt_id: str, ranks: int, step, lengths, dtype) -> None:
    (ckpt_dir / "manifest.json").write_text(json.dumps({
        "ckpt_id": ckpt_id,
        "ranks": ranks,
        "step": step,
        "lengths": lengths,
        "dtype": dtype,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_demo(base: Path) -> None:
    ckpt = base / "ckpt-demo-3rank"
    if (ckpt / "manifest.json").is_file():
        return
    ckpt.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261123)
    lengths = [10, 7, 13]
    total = sum(lengths)
    param = rng.standard_normal(total).astype(np.float32)
    m = (rng.standard_normal(total) * 0.01).astype(np.float32)
    v = (rng.random(total) * 0.001).astype(np.float32)  # 二阶状态非负
    off = 0
    for r, ln in enumerate(lengths):
        _write_rank(ckpt, r, param[off:off + ln], m[off:off + ln],
                    v[off:off + ln], step=42000)
        off += ln
    _manifest(ckpt, "ckpt-demo-3rank", 3, 42000, lengths, "float32")


def _build_step_mismatch(base: Path) -> None:
    ckpt = base / "ckpt-step-mismatch"
    if (ckpt / "manifest.json").is_file():
        return
    ckpt.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for r, step in enumerate((1000, 1001)):  # 训练步故意不一致
        n = 8
        _write_rank(ckpt, r,
                    rng.standard_normal(n).astype(np.float32),
                    rng.standard_normal(n).astype(np.float32),
                    (rng.random(n) * 0.001).astype(np.float32),
                    step=step)
    _manifest(ckpt, "ckpt-step-mismatch", 2, "不一致", [8, 8], "float32")


def _build_missing_rank(base: Path) -> None:
    ckpt = base / "ckpt-missing-rank"
    if (ckpt / "manifest.json").is_file():
        return
    ckpt.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    for r in (0, 1):  # manifest 声明 3 个 rank，故意缺 rank_2.npz
        n = 6
        _write_rank(ckpt, r,
                    rng.standard_normal(n).astype(np.float32),
                    rng.standard_normal(n).astype(np.float32),
                    (rng.random(n) * 0.001).astype(np.float32),
                    step=9000)
    _manifest(ckpt, "ckpt-missing-rank", 3, 9000, [6, 6, 6], "float32")


def ensure_samples(old: OldVolume) -> None:
    """初始化旧卷样例与默认挂载状态（幂等）。"""
    old.dir.mkdir(parents=True, exist_ok=True)
    vid = old.dir / "volume.id"
    if not vid.is_file():
        vid.write_text(SAMPLE_VOLUME_ID, encoding="utf-8")
    ckpts = old.checkpoints_dir()
    ckpts.mkdir(parents=True, exist_ok=True)
    _build_demo(ckpts)
    _build_step_mismatch(ckpts)
    _build_missing_rank(ckpts)
    if not old.mount_file.is_file():
        old.mount()  # 默认挂载同一卷
