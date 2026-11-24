"""卷模拟与发布流程。

- 旧卷：目录模拟，可卸载 / 挂回。只有挂回同一卷（volume id 相同）才可读。
- 新卷：published/（当前发布的一套）+ staging/（本次暂存）。
  容量按字节限制，上一套发布与本次暂存都计入占用；记录分片文件占用峰值。
- 发布流程：读源 -> 校验（缺片 / 训练步）-> 重分片 -> 逐片暂存并累计容量
  -> 在指定挂载条件下回读校验 -> 原子替换发布。
  任何一步失败：清理暂存、保留上次发布结果、在历史里记录原因。
"""
from __future__ import annotations

import json
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path

from reshard import (ARRAY_KEYS, OverCapacityError, ReshardError,
                     SimulatedWriteError, SourceUnreadableError, VerifyError,
                     load_source_checkpoint, reshard_arrays, split_sizes,
                     verify_shards, write_shard)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _dir_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


class OldVolume:
    """旧卷：mount.json 记录挂载状态；挂载的卷 id 与盘上 id 一致才可读。"""

    def __init__(self, root: Path):
        self.dir = Path(root) / "old_volume"
        self.mount_file = Path(root) / "mount.json"

    # ---- 挂载状态 ----
    def _read_mount(self) -> dict:
        if self.mount_file.is_file():
            try:
                return json.loads(self.mount_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {"mounted": False, "volume_id": None}

    def _write_mount(self, mounted: bool, volume_id: str | None) -> None:
        self.mount_file.parent.mkdir(parents=True, exist_ok=True)
        self.mount_file.write_text(
            json.dumps({"mounted": mounted, "volume_id": volume_id}, indent=2),
            encoding="utf-8",
        )

    def stored_id(self) -> str | None:
        f = self.dir / "volume.id"
        return f.read_text(encoding="utf-8").strip() if f.is_file() else None

    def status(self) -> dict:
        m = self._read_mount()
        sid = self.stored_id()
        return {
            "mounted": bool(m.get("mounted")),
            "volume_id": m.get("volume_id"),
            "stored_id": sid,
            "accessible": bool(m.get("mounted")) and m.get("volume_id") == sid,
        }

    def accessible(self) -> bool:
        return self.status()["accessible"]

    def mount(self, volume_id: str | None = None) -> None:
        """挂载；不传 id 表示挂回盘上那一块（同一卷）。"""
        self._write_mount(True, volume_id if volume_id is not None else self.stored_id())

    def unmount(self) -> None:
        self._write_mount(False, None)

    # ---- 内容 ----
    def checkpoints_dir(self) -> Path:
        return self.dir / "checkpoints"

    def list_checkpoints(self) -> list[dict] | None:
        """返回检查点摘要列表；旧卷不可读时返回 None。"""
        if not self.accessible():
            return None
        base = self.checkpoints_dir()
        out = []
        if base.is_dir():
            for d in sorted(base.iterdir()):
                mf = d / "manifest.json"
                if d.is_dir() and mf.is_file():
                    try:
                        m = json.loads(mf.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        m = {"ckpt_id": d.name, "error": "manifest 损坏"}
                    m.setdefault("ckpt_id", d.name)
                    out.append(m)
        return out


class NewVolume:
    """新卷：容量限制、published/staging 占用、峰值与历史。"""

    def __init__(self, root: Path):
        self.dir = Path(root) / "new_volume"
        self.published_root = self.dir / "published"
        self.staging_root = self.dir / "staging"
        self.state_file = self.dir / "state.json"
        for d in (self.published_root, self.staging_root):
            d.mkdir(parents=True, exist_ok=True)
        self._state = self._load()

    def _load(self) -> dict:
        if self.state_file.is_file():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {
            "capacity_bytes": 1 << 20,  # 默认 1 MiB
            "published": None,
            "peak_bytes": 0,
            "history": [],
        }

    def save(self) -> None:
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.state_file)

    # ---- 状态访问 ----
    @property
    def capacity(self) -> int:
        return int(self._state["capacity_bytes"])

    def set_capacity(self, capacity: int) -> None:
        self._state["capacity_bytes"] = int(capacity)

    @property
    def published(self) -> dict | None:
        return self._state["published"]

    @property
    def peak_bytes(self) -> int:
        return int(self._state["peak_bytes"])

    @property
    def history(self) -> list[dict]:
        return self._state["history"]

    def published_bytes(self) -> int:
        return _dir_bytes(self.published_root)

    def staging_bytes(self) -> int:
        return _dir_bytes(self.staging_root)

    def usage_bytes(self) -> int:
        """分片文件当前占用：上一套发布 + 本次暂存。"""
        return self.published_bytes() + self.staging_bytes()

    def record_peak(self, value: int) -> None:
        if value > self.peak_bytes:
            self._state["peak_bytes"] = int(value)

    def append_history(self, entry: dict) -> None:
        self._state["history"].append(entry)
        del self._state["history"][:-50]

    # ---- 发布 ----
    def new_staging_dir(self) -> Path:
        d = self.staging_root / f"stg-{uuid.uuid4().hex[:12]}"
        d.mkdir(parents=True, exist_ok=False)
        return d

    def publish(self, staging_dir: Path, record: dict) -> None:
        """用暂存目录整体替换当前发布；旧套先挪为备份，新套落入后再删，
        移动中途失败可回滚到旧套。"""
        target = self.published_root / record["id"]
        backup = self.dir / f".backup-{uuid.uuid4().hex[:8]}"
        had_old = self.published_root.exists() and any(self.published_root.iterdir())
        if had_old:
            shutil.move(str(self.published_root), str(backup))
        self.published_root.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(staging_dir), str(target))
        except Exception:
            if had_old and backup.exists():
                shutil.rmtree(self.published_root, ignore_errors=True)
                shutil.move(str(backup), str(self.published_root))
            raise
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        (target / "manifest.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._state["published"] = record


class ReshardService:
    """把 OldVolume / NewVolume 串起来的业务流程。"""

    def __init__(self, data_dir: Path):
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        self.old = OldVolume(data_dir)
        self.new = NewVolume(data_dir)
        self.lock = threading.Lock()

    # ---- 可读性 ----
    def published_readable(self) -> bool | None:
        """已发布结果当前是否可读；无发布返回 None。

        依赖旧卷的结果：旧卷卸载或挂了不同的卷 -> 不可读；挂回同一卷 -> 恢复。
        """
        pub = self.new.published
        if pub is None:
            return None
        if not pub.get("requires_old_volume"):
            return True
        st = self.old.status()
        return bool(st["mounted"] and st["volume_id"] == pub.get("old_volume_id"))

    # ---- 重分片主流程 ----
    def run(self, ckpt_id: str, n_new: int, capacity: int,
            can_mount_old: bool, simulate_write_fail: bool) -> dict:
        with self.lock:
            op = {
                "time": _now(),
                "ckpt_id": ckpt_id,
                "n_new": n_new,
                "capacity": capacity,
                "can_mount_old": can_mount_old,
                "simulate_write_fail": simulate_write_fail,
            }
            staging_dir = self.new.new_staging_dir()
            try:
                if capacity < 0:
                    raise ReshardError("新卷容量必须 >= 0")
                self.new.set_capacity(capacity)
                op.update(self._do_run(ckpt_id, n_new, can_mount_old,
                                       simulate_write_fail, staging_dir))
                op["status"] = "published"
            except ReshardError as exc:
                shutil.rmtree(staging_dir, ignore_errors=True)
                op["status"] = "failed"
                op["reason"] = str(exc)
            except Exception as exc:  # noqa: BLE001 - 未预期错误同样保留上次发布
                shutil.rmtree(staging_dir, ignore_errors=True)
                op["status"] = "failed"
                op["reason"] = f"内部错误: {exc!r}"
            self.new.append_history(op)
            self.new.save()
            return op

    def _do_run(self, ckpt_id: str, n_new: int, can_mount_old: bool,
                simulate_write_fail: bool, staging_dir: Path) -> dict:
        if n_new < 1:
            raise ReshardError("新 rank 数必须 >= 1")
        if not self.old.accessible():
            raise SourceUnreadableError("旧卷未挂载或挂载了不同的卷，源检查点不可读")

        # 1) 读源 + 校验（缺片 / 训练步不一致 / 段长度）
        src = load_source_checkpoint(self.old.checkpoints_dir() / ckpt_id)

        # 2) 三个数组按同一全局下标连续等分，余数给低 rank
        shards = {k: reshard_arrays(src.arrays[k], n_new) for k in ARRAY_KEYS}
        sizes = split_sizes(src.total, n_new)

        # 3) 逐片暂存；上一套发布 + 本次暂存一起计入容量
        published_before = self.new.published_bytes()
        staged = 0
        shard_files: list[Path] = []
        fail_at = n_new // 2 if simulate_write_fail else -1
        for r in range(n_new):
            if r == fail_at:
                raise SimulatedWriteError(f"模拟写失败: 写入 rank_{r}.npz 时出错")
            path = staging_dir / f"rank_{r}.npz"
            staged += write_shard(path, {k: shards[k][r] for k in ARRAY_KEYS}, src.step)
            shard_files.append(path)
            if published_before + staged > self.new.capacity:
                raise OverCapacityError(
                    f"新卷容量超限: 上一套发布 {published_before} B + "
                    f"本次暂存 {staged} B > 容量 {self.new.capacity} B"
                )

        # 4) 在指定挂载条件下回读校验全套分片，可读才发布
        old_st = self.old.status()
        if can_mount_old and not old_st["accessible"]:
            raise VerifyError("声明续训时可挂旧卷，但旧卷当前不可读，"
                              "无法在指定挂载条件下验证全套分片")
        verify_shards(shard_files, sizes, src.step, src)

        # 5) 发布：记录峰值（上一套 + 本次暂存），原子替换
        op_peak = published_before + staged
        self.new.record_peak(op_peak)
        pub_id = f"pub-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        record = {
            "id": pub_id,
            "source_ckpt": src.ckpt_id,
            "step": src.step,
            "ranks": n_new,
            "total_elems": src.total,
            "dtype": src.dtype,
            "shard_sizes": sizes,
            "bytes": staged,
            "requires_old_volume": bool(can_mount_old),
            "old_volume_id": old_st["volume_id"] if can_mount_old else None,
            "created_at": _now(),
        }
        self.new.publish(staging_dir, record)
        self.new.record_peak(self.new.published_bytes())
        return {
            "publish_id": pub_id,
            "step": src.step,
            "shard_sizes": sizes,
            "staged_bytes": staged,
            "prev_published_bytes": published_before,
            "op_peak": op_peak,
        }
