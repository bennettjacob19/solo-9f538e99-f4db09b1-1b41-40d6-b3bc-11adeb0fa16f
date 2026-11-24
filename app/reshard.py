"""核心重分片逻辑。

旧卷上的检查点按 rank 存放：每个 rank 一个 npz，内含
  - param   : 参数段
  - adam_m  : 同段的 Adam 一阶状态
  - adam_v  : 同段的 Adam 二阶状态
  - step    : 训练步（标量）

重分片规则：把各 rank 的数组按 rank 顺序拼接成全局数组，
三个数组使用同一套全局下标连续等分，余数依次分给低 rank，
每个新分片都带上训练步。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARRAY_KEYS = ("param", "adam_m", "adam_v")


class ReshardError(Exception):
    """重分片失败基类，message 面向用户。"""


class SourceUnreadableError(ReshardError):
    """源检查点不可读（旧卷未挂载或挂载了不同的卷）。"""


class MissingShardError(ReshardError):
    """缺片：manifest 声明的 rank 文件或数组不存在。"""


class StepMismatchError(ReshardError):
    """训练步不一致。"""


class SegmentMismatchError(ReshardError):
    """同一 rank 内三个数组长度不一致，或 dtype 不一致。"""


class OverCapacityError(ReshardError):
    """新卷容量超限（上一套发布 + 本次暂存都计入）。"""


class SimulatedWriteError(ReshardError):
    """模拟写失败。"""


class VerifyError(ReshardError):
    """暂存回读校验失败。"""


def split_sizes(total: int, n: int) -> list[int]:
    """total 个元素连续等分成 n 段，余数依次分给低 rank。

    例: split_sizes(10, 3) == [4, 3, 3]
    """
    if n < 1:
        raise ValueError("n 必须 >= 1")
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def split_offsets(total: int, n: int) -> list[tuple[int, int]]:
    """返回每段的 (起始全局下标, 长度)。"""
    offsets, off = [], 0
    for size in split_sizes(total, n):
        offsets.append((off, size))
        off += size
    return offsets


@dataclass
class SourceCheckpoint:
    ckpt_id: str
    step: int
    dtype: str
    arrays: dict  # key -> list[np.ndarray]，按 rank 排列

    @property
    def ranks(self) -> int:
        return len(self.arrays["param"])

    @property
    def total(self) -> int:
        return int(sum(a.size for a in self.arrays["param"]))


def load_source_checkpoint(ckpt_dir: Path) -> SourceCheckpoint:
    """读取并校验旧卷上的源检查点。缺片 / 训练步不一致在此抛出。"""
    ckpt_dir = Path(ckpt_dir)
    manifest_path = ckpt_dir / "manifest.json"
    if not manifest_path.is_file():
        raise MissingShardError(f"缺片: {ckpt_dir.name}/manifest.json 不存在")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ranks = int(manifest["ranks"])
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in ARRAY_KEYS}
    steps: list[int] = []
    for r in range(ranks):
        shard = ckpt_dir / f"rank_{r}.npz"
        if not shard.is_file():
            raise MissingShardError(
                f"缺片: rank_{r}.npz 不存在（manifest 声明 {ranks} 个 rank）"
            )
        with np.load(shard) as z:
            names = set(z.files)
            for k in ARRAY_KEYS:
                if k not in names:
                    raise MissingShardError(f"缺片: rank_{r}.npz 缺少数组 {k}")
                arrays[k].append(np.ascontiguousarray(z[k]).reshape(-1))
            if "step" not in names:
                raise MissingShardError(f"缺片: rank_{r}.npz 缺少训练步 step")
            steps.append(int(z["step"]))
    if len(set(steps)) != 1:
        raise StepMismatchError(f"训练步不一致: 各 rank step = {steps}")
    for r in range(ranks):
        lens = {k: int(arrays[k][r].size) for k in ARRAY_KEYS}
        if len(set(lens.values())) != 1:
            raise SegmentMismatchError(f"rank {r} 三个数组长度不一致: {lens}")
    dtypes = {str(arrays[k][r].dtype) for k in ARRAY_KEYS for r in range(ranks)}
    if len(dtypes) != 1:
        raise SegmentMismatchError(f"dtype 不一致: {sorted(dtypes)}")
    return SourceCheckpoint(
        ckpt_id=str(manifest.get("ckpt_id", ckpt_dir.name)),
        step=steps[0],
        dtype=dtypes.pop(),
        arrays=arrays,
    )


def reshard_arrays(per_rank: list[np.ndarray], n_new: int) -> list[np.ndarray]:
    """拼接后按全局下标连续等分成 n_new 段（余数给低 rank）。"""
    full = np.concatenate(per_rank) if per_rank else np.empty(0)
    return [full[off:off + size].copy() for off, size in split_offsets(full.size, n_new)]


def write_shard(path: Path, arrays: dict, step: int) -> int:
    """写一个新 rank 分片（带训练步），返回文件字节数。"""
    path = Path(path)
    np.savez(path, **arrays, step=np.int64(step))
    return path.stat().st_size


def verify_shards(shard_paths: list[Path],
                  expected_sizes: list[int],
                  step: int,
                  source: SourceCheckpoint) -> None:
    """回读全部暂存分片：缺片 / 训练步 / 长度 / 与源拼接一致性。"""
    if len(shard_paths) != len(expected_sizes):
        raise VerifyError(
            f"回读缺片: 期望 {len(expected_sizes)} 片，实际 {len(shard_paths)} 片"
        )
    concat: dict[str, list[np.ndarray]] = {k: [] for k in ARRAY_KEYS}
    for r, (path, size) in enumerate(zip(shard_paths, expected_sizes)):
        path = Path(path)
        if not path.is_file():
            raise VerifyError(f"回读缺片: {path.name} 不存在")
        try:
            with np.load(path) as z:
                data = {k: z[k] for k in z.files}
        except Exception as exc:  # noqa: BLE001 - 统一转成 VerifyError
            raise VerifyError(f"回读失败 {path.name}: {exc}") from exc
        for k in ARRAY_KEYS:
            if k not in data:
                raise VerifyError(f"回读缺片: {path.name} 缺少数组 {k}")
        if "step" not in data:
            raise VerifyError(f"回读缺片: {path.name} 缺少训练步 step")
        if int(data["step"]) != step:
            raise VerifyError(
                f"回读训练步不一致: {path.name} step={int(data['step'])}，期望 {step}"
            )
        for k in ARRAY_KEYS:
            arr = np.ascontiguousarray(data[k]).reshape(-1)
            if arr.size != size:
                raise VerifyError(
                    f"回读长度不符: {path.name} {k} 长度 {arr.size}，期望 {size}"
                )
            concat[k].append(arr)
    for k in ARRAY_KEYS:
        merged = np.concatenate(concat[k]) if concat[k] else np.empty(0)
        original = np.concatenate(source.arrays[k]) if source.arrays[k] else np.empty(0)
        if not np.array_equal(merged, original):
            raise VerifyError(f"数组 {k} 拼接后与源检查点不一致")
