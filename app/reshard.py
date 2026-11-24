"""核心重分片逻辑（流式，内存有界）。

旧卷上的检查点按 rank 存放：每个 rank 一个 npz，内含
  - param   : 参数段
  - adam_m  : 同段的 Adam 一阶状态
  - adam_v  : 同段的 Adam 二阶状态
  - step    : 训练步（标量）

重分片规则：把各 rank 的数组按 rank 顺序拼接成全局数组，
三个数组使用同一套全局下标连续等分，余数依次分给低 rank，
每个新分片都带上训练步。

实现要点
--------
* 读源与写新片都是流式的：每条流每次只在内存里保留至多
  ``READ_CHUNK_BYTES``（1 MiB）的数组正文，因此单个源片 / 新片超过
  16 MiB（乃至 2 GiB 以上）也能处理；磁盘上不产生额外的解压副本。
* 源 npz 可以是 ``np.savez``（未压缩）或 ``np.savez_compressed``（压缩），
  两者都用 zipfile 按成员流式读取；压缩成员也按块解压，不会整块展开。
* 源数组正文读取量：写新片时每个数组读一遍（1x），回读校验时再读一遍
  （1x），合计恰好 2x；文件头预读与 ``step`` 标量不计入正文。
  校验回读的是暂存 npz，不再额外读源。
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ARRAY_KEYS = ("param", "adam_m", "adam_v")

# 单次读取源/暂存数组正文的块大小。写片时三个数组各有一条流（3 个块），
# 回读校验时一条源流 + 一条暂存流；加上精确读取时的临时拼接，
# 同时在内存里的数组正文远小于 16 MiB。
MAX_LIVE_ARRAY_BYTES = 16 << 20
READ_CHUNK_BYTES = MAX_LIVE_ARRAY_BYTES // 16  # 1 MiB

# 写暂存的 npy 成员时总用 zip64 本地头（成员可能超过 2 GiB）；
# zip64 的 npz 仍可被普通 np.load 正确读取。


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


class SourceReadError(ReshardError):
    """源片读坏（npz 损坏、截断、CRC 错误、格式不支持等）。"""


class OverCapacityError(ReshardError):
    """新卷容量超限（上一套发布 + 本次暂存都计入）。"""


class WriteError(ReshardError):
    """写新片失败。"""


class SimulatedWriteError(WriteError):
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


# ---------------------------------------------------------------------------
# 读取量统计
# ---------------------------------------------------------------------------
class SourceBodyCounter:
    """累计源数组（param / adam_m / adam_v）正文读取字节数。

    文件头预读、step 标量不计入“正文”统计。重复解压读到的字节同样计入。
    """

    def __init__(self) -> None:
        self.body_bytes = 0

    def add(self, nbytes: int) -> None:
        self.body_bytes += int(nbytes)


# ---------------------------------------------------------------------------
# 源检查点元数据（只读文件头 + step，不读大数组正文）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RankMeta:
    path: Path
    count: int           # 每个数组的段长度（元素数）
    dtype: np.dtype
    ndim: int
    fortran: bool


@dataclass
class SourceLayout:
    """源检查点布局。数组正文按 rank 顺序拼接即为全局数组。"""

    ckpt_id: str
    step: int
    dtype: np.dtype
    ranks: int
    metas: list[RankMeta] = field(default_factory=list)
    per_rank_counts: list[int] = field(default_factory=list)

    @property
    def total(self) -> int:
        return int(sum(self.per_rank_counts))

    @property
    def itemsize(self) -> int:
        return int(self.dtype.itemsize)

    @property
    def body_bytes(self) -> int:
        """三个数组的正文总字节数（不含 npz 文件头 / zip 元数据 / step）。"""
        return 3 * self.total * self.itemsize


def _member_map(zf: zipfile.ZipFile) -> dict[str, str]:
    """npz 成员名 -> zip 内实际名字（去掉 .npy 后缀）。"""
    out: dict[str, str] = {}
    for name in zf.namelist():
        base = name[:-4] if name.endswith(".npy") else name
        out.setdefault(base, name)
    return out


def _read_header(fp) -> tuple[tuple[int, ...], bool, np.dtype]:
    magic = np.lib.format.read_magic(fp)
    if magic == (1, 0):
        return np.lib.format.read_array_header_1_0(fp)
    if magic == (2, 0):
        return np.lib.format.read_array_header_2_0(fp)
    if magic == (3, 0):
        # npy 3.0 头是 utf8 编码的 4 字节长度 + 头文本，与 2.0 布局一致。
        import ast
        import struct
        raw_len = fp.read(4)
        if len(raw_len) != 4:
            raise SourceReadError("源片被截断（npy 3.0 头长度不足）")
        header_len = struct.unpack("<I", raw_len)[0]
        if header_len > 1 << 20:
            raise SourceReadError("不支持的 npy 头（过长）")
        header = fp.read(header_len)
        if len(header) != header_len:
            raise SourceReadError("源片被截断（npy 3.0 头不足）")
        try:
            d = ast.literal_eval(header.decode("utf8"))
            shape = tuple(int(x) for x in d["shape"])
            return shape, bool(d["fortran_order"]), np.dtype(d["descr"])
        except (SyntaxError, ValueError, KeyError, TypeError) as exc:
            raise SourceReadError(f"npy 3.0 头无法解析（{exc}）") from exc
    raise SourceReadError(f"不支持的 npy 格式版本: {magic}")


def _read_exact(fp, nbytes: int, error=SourceReadError,
                message: str = "源片被截断（数组正文不足）") -> bytes:
    """从文件对象精确读 nbytes；不足视为截断。"""
    parts: list[bytes] = []
    got = 0
    while got < nbytes:
        chunk = fp.read(nbytes - got)
        if not chunk:
            raise error(message)
        parts.append(chunk)
        got += len(chunk)
    return parts[0] if len(parts) == 1 else b"".join(parts)


def _check_numeric(dtype: np.dtype, where: str) -> None:
    if dtype.hasobject or dtype.kind not in "biufc":
        raise SourceReadError(f"{where}: 仅支持一维数值数组，实际 dtype={dtype}")


def _inspect_rank(path: Path, rank: int) -> tuple[RankMeta, int]:
    """读取一个 rank 的元数据：三个数组的 shape/dtype + 训练步。"""
    try:
        with zipfile.ZipFile(path) as zf:
            names = _member_map(zf)
            for key in ARRAY_KEYS:
                if key not in names:
                    raise MissingShardError(f"缺片: rank_{rank}.npz 缺少数组 {key}")
            steps_name = names.get("step")
            if steps_name is None:
                raise MissingShardError(f"缺片: rank_{rank}.npz 缺少训练步 step")

            shapes: list[tuple[int, ...]] = []
            dtypes: list[np.dtype] = []
            ndim = 0
            fortran = False
            for key in ARRAY_KEYS:
                with zf.open(names[key]) as fp:
                    shape, order, dtype = _read_header(fp)
                dtype = np.dtype(dtype)
                _check_numeric(dtype, f"rank_{rank}.npz {key}")
                if order and len(shape) > 1:
                    raise SourceReadError(
                        f"rank_{rank}.npz {key}: 仅支持一维数组（或 C 连续多维数组）"
                    )
                shapes.append(tuple(int(x) for x in shape))
                dtypes.append(dtype)
                ndim = max(ndim, len(shape))
                fortran = fortran or order

            counts = {int(np.prod(s, dtype=np.int64)) if s else 1 for s in shapes}
            if len(counts) != 1:
                lens = {k: int(np.prod(s)) if s else 1
                        for k, s in zip(ARRAY_KEYS, shapes)}
                raise SegmentMismatchError(
                    f"rank {rank} 三个数组长度不一致: {lens}"
                )
            if len({d.str for d in dtypes}) != 1:
                raise SegmentMismatchError(
                    f"rank {rank} 三个数组 dtype 不一致: "
                    f"{[str(d) for d in dtypes]}"
                )

            with zf.open(steps_name) as fp:
                step_shape, _, step_dtype = _read_header(fp)
                step_count = int(np.prod(step_shape, dtype=np.int64)) if step_shape else 1
                if step_count != 1:
                    raise MissingShardError(
                        f"缺片: rank_{rank}.npz 的 step 不是标量"
                    )
                step_raw = _read_exact(fp, int(np.dtype(step_dtype).itemsize))
                if fp.read(1):
                    raise MissingShardError(
                        f"缺片: rank_{rank}.npz 的 step 含多余数据"
                    )
            step = int(np.frombuffer(step_raw, dtype=np.dtype(step_dtype))[0])
    except ReshardError:
        raise
    except (zipfile.BadZipFile, OSError, ValueError, EOFError) as exc:
        raise SourceReadError(f"源片读坏: rank_{rank}.npz 无法读取（{exc}）") from exc

    meta = RankMeta(path=path, count=counts.pop(), dtype=dtypes[0],
                    ndim=ndim, fortran=fortran)
    return meta, step


def inspect_source_checkpoint(ckpt_dir: Path) -> SourceLayout:
    """只读文件头与 step，校验缺片 / 训练步 / 段长度 / dtype。"""
    ckpt_dir = Path(ckpt_dir)
    manifest_path = ckpt_dir / "manifest.json"
    if not manifest_path.is_file():
        raise MissingShardError(f"缺片: {ckpt_dir.name}/manifest.json 不存在")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ranks = int(manifest["ranks"])
    except (OSError, ValueError, KeyError) as exc:
        raise SourceReadError(f"源片读坏: {ckpt_dir.name}/manifest.json（{exc}）") from exc

    metas: list[RankMeta] = []
    steps: list[int] = []
    for r in range(ranks):
        shard = ckpt_dir / f"rank_{r}.npz"
        if not shard.is_file():
            raise MissingShardError(
                f"缺片: rank_{r}.npz 不存在（manifest 声明 {ranks} 个 rank）"
            )
        meta, step = _inspect_rank(shard, r)
        metas.append(meta)
        steps.append(step)

    if not steps:
        raise MissingShardError("缺片: manifest 声明 0 个 rank，没有可读数组")
    if len(set(steps)) != 1:
        raise StepMismatchError(f"训练步不一致: 各 rank step = {steps}")

    dtypes = {m.dtype.str for m in metas}
    if len(dtypes) != 1:
        raise SegmentMismatchError(f"dtype 不一致: {sorted(dtypes)}")

    return SourceLayout(
        ckpt_id=str(manifest.get("ckpt_id", ckpt_dir.name)),
        step=steps[0],
        dtype=metas[0].dtype,
        ranks=ranks,
        metas=metas,
        per_rank_counts=[m.count for m in metas],
    )


# ---------------------------------------------------------------------------
# 流式读取源数组
# ---------------------------------------------------------------------------
class SourceCursor:
    """按 rank 顺序流式读取某个数组键，跨 rank 自动衔接。

    每个源 rank 成员只打开一次；压缩 npz 也只解压一次。
    读取正文时把字节数记入 ``counter``。
    """

    def __init__(self, layout: SourceLayout, key: str,
                 counter: SourceBodyCounter | None = None,
                 chunk_bytes: int = READ_CHUNK_BYTES) -> None:
        self.layout = layout
        self.key = key
        self.counter = counter
        self.chunk_bytes = max(1, int(chunk_bytes))
        self.itemsize = layout.itemsize
        if self.itemsize <= 0:  # pragma: no cover - 数值 dtype 不会出现
            raise SourceReadError(f"不支持的 dtype: {layout.dtype}")
        self.chunk_elems = max(1, self.chunk_bytes // self.itemsize)

        self._rank = 0
        self._zf: zipfile.ZipFile | None = None
        self._fp = None
        self._member_remaining = 0
        self._pending = memoryview(b"")
        self.remaining_elems = layout.total
        self._closed = False

    # -- 打开 / 关闭当前 rank 成员 --
    def _open_member(self) -> bool:
        while self._rank < self.layout.ranks:
            meta = self.layout.metas[self._rank]
            if meta.count == 0:
                self._rank += 1
                continue
            try:
                self._zf = zipfile.ZipFile(meta.path)
                names = _member_map(self._zf)
                if self.key not in names:
                    raise MissingShardError(
                        f"缺片: rank_{self._rank}.npz 缺少数组 {self.key}"
                    )
                self._fp = self._zf.open(names[self.key])
                shape, order, dtype = _read_header(self._fp)
                count = int(np.prod(shape, dtype=np.int64)) if shape else 1
                if count != meta.count or np.dtype(dtype).str != meta.dtype.str:
                    raise SourceReadError(
                        f"源片读坏: rank_{self._rank}.npz {self.key} 元数据与预读不符"
                    )
                self._member_remaining = count
                return True
            except ReshardError:
                self._close_member()
                raise
            except (zipfile.BadZipFile, OSError, ValueError, EOFError) as exc:
                self._close_member()
                raise SourceReadError(
                    f"源片读坏: rank_{self._rank}.npz {self.key}（{exc}）"
                ) from exc
        return False

    def _close_member(self) -> None:
        if self._fp is not None:
            try:
                self._fp.close()
            except Exception:  # noqa: BLE001 - 关闭失败不掩盖原始错误
                pass
            self._fp = None
        if self._zf is not None:
            try:
                self._zf.close()
            except Exception:  # noqa: BLE001
                pass
            self._zf = None

    def close(self) -> None:
        self._close_member()
        self._pending = memoryview(b"")
        self._closed = True

    def __enter__(self) -> "SourceCursor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- 读一块，长度不超过 chunk_elems 个元素 --
    def _pull_chunk(self) -> memoryview:
        if self._fp is None:
            if not self._open_member():
                return memoryview(b"")
        n = min(self.chunk_elems, self._member_remaining)
        nbytes = n * self.itemsize
        try:
            raw = _read_exact(self._fp, nbytes)
            self._member_remaining -= n
            if self._member_remaining == 0:
                # 读到成员末尾，触发 CRC 校验；多出数据说明 npz 损坏。
                tail = self._fp.read(1)
                if tail:
                    raise SourceReadError(
                        f"源片读坏: rank_{self._rank}.npz {self.key} 有多余数据"
                    )
                self._rank += 1
                self._close_member()
        except ReshardError:
            raise
        except (zipfile.BadZipFile, OSError, ValueError, EOFError) as exc:
            raise SourceReadError(
                f"源片读坏: rank_{self._rank}.npz {self.key}（{exc}）"
            ) from exc
        if self.counter is not None:
            self.counter.add(len(raw))
        return memoryview(raw)

    def iter_bytes(self, nbytes: int):
        """精确产出 nbytes 个正文字节（元素边界对齐）。"""
        if nbytes < 0:
            raise ValueError("nbytes 必须 >= 0")
        left = int(nbytes)
        while left:
            if not self._pending:
                self._pending = self._pull_chunk()
                if not self._pending:
                    raise SourceReadError(
                        f"源片读坏: {self.key} 正文不足（还差 {left} B）"
                    )
            take = min(len(self._pending), left)
            yield self._pending[:take]
            self._pending = self._pending[take:]
            left -= take

    def iter_all(self):
        """顺序产出整个数组的正文块。"""
        yield from self.iter_bytes(self.layout.total * self.itemsize)


# ---------------------------------------------------------------------------
# 写暂存 npz（流式；不整块载入数组）
# ---------------------------------------------------------------------------
def write_shards_streaming(staging_dir: Path, layout: SourceLayout,
                           sizes: list[int],
                           simulate_write_fail: bool = False,
                           counter: SourceBodyCounter | None = None,
                           member_done=None) -> list[Path]:
    """按新 rank 写整套暂存 npz（流式，内存有界）。

    三个源数组各开一个游标，按新 rank 顺序交替推进：每个新分片一个
    zip 会话顺序写入 param / adam_m / adam_v / step。源正文在每个数组上
    只读一遍（压缩源也只解压一遍），内存中每条流只驻留一个块。
    ``member_done`` 在每片写完后回调，用于容量检查。
    """
    staging_dir = Path(staging_dir)
    paths = [staging_dir / f"rank_{r}.npz" for r in range(len(sizes))]
    fail_at = len(sizes) // 2 if simulate_write_fail else -1
    cursors = {key: SourceCursor(layout, key, counter=counter)
               for key in ARRAY_KEYS}
    try:
        for r, size in enumerate(sizes):
            if r == fail_at:
                raise SimulatedWriteError(f"模拟写失败: 写入 rank_{r}.npz 时出错")
            path = paths[r]
            nbytes = size * layout.itemsize
            try:
                with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED,
                                     allowZip64=True) as zf:
                    for key in ARRAY_KEYS:
                        # 始终用 zip64 本地头：成员可超过 2 GiB，numpy 读取不受影响。
                        with zf.open(key + ".npy", "w", force_zip64=True) as fp:
                            np.lib.format.write_array_header_1_0(fp, {
                                "descr": np.lib.format.dtype_to_descr(layout.dtype),
                                "fortran_order": False,
                                "shape": (size,),
                            })
                            for chunk in cursors[key].iter_bytes(nbytes):
                                fp.write(chunk)
                    with zf.open("step.npy", "w", force_zip64=True) as fp:
                        np.lib.format.write_array_header_1_0(fp, {
                            "descr": np.lib.format.dtype_to_descr(np.dtype(np.int64)),
                            "fortran_order": False,
                            "shape": (),
                        })
                        fp.write(np.int64(layout.step).tobytes())
            except ReshardError:
                raise
            except (OSError, zipfile.BadZipFile, ValueError) as exc:
                raise WriteError(f"写入失败: {path.name}（{exc}）") from exc
            if member_done is not None:
                member_done(path)
    finally:
        for cursor in cursors.values():
            cursor.close()

    # 写完后确认三个数组 + step 都在
    for r, path in enumerate(paths):
        try:
            with zipfile.ZipFile(path) as zf:
                names = _member_map(zf)
        except (zipfile.BadZipFile, OSError) as exc:
            raise WriteError(f"写入失败: {path.name} 无法回读（{exc}）") from exc
        for key in (*ARRAY_KEYS, "step"):
            if key not in names:
                raise WriteError(f"写入失败: {path.name} 缺少 {key}")
    return paths


# ---------------------------------------------------------------------------
# 回读暂存 npz 并与源检查点逐字节核对
# ---------------------------------------------------------------------------
def _iter_member_body(path: Path, key: str, expected_count: int,
                      expected_dtype: np.dtype,
                      chunk_bytes: int = READ_CHUNK_BYTES):
    """流式读回 npz 某成员正文，并校验 shape / dtype。"""
    itemsize = max(1, expected_dtype.itemsize)
    try:
        with zipfile.ZipFile(path) as zf:
            names = _member_map(zf)
            if key not in names:
                raise VerifyError(f"回读缺片: {path.name} 缺少数组 {key}")
            with zf.open(names[key]) as fp:
                shape, _order, dtype = _read_header(fp)
                dtype = np.dtype(dtype)
                if tuple(shape) != (expected_count,):
                    raise VerifyError(
                        f"回读长度不符: {path.name} {key} shape {tuple(shape)}，"
                        f"期望 ({expected_count},)"
                    )
                if dtype.hasobject or dtype.kind not in "biufc":
                    raise VerifyError(f"回读 dtype 非法: {path.name} {key} {dtype}")
                if dtype.str != expected_dtype.str:
                    raise VerifyError(
                        f"回读 dtype 不符: {path.name} {key} {dtype}，"
                        f"期望 {expected_dtype}"
                    )
                elems = max(1, chunk_bytes // max(1, itemsize))
                left = expected_count
                while left:
                    n = min(elems, left)
                    raw = _read_exact(
                        fp, n * itemsize, error=VerifyError,
                        message=f"回读截断: {path.name} {key} 正文不足",
                    )
                    left -= n
                    yield memoryview(raw)
                tail = fp.read(1)
                if tail:
                    raise VerifyError(f"回读多余数据: {path.name} {key}")
    except ReshardError:
        raise
    except (zipfile.BadZipFile, OSError, ValueError, EOFError) as exc:
        raise VerifyError(f"回读失败 {path.name} {key}: {exc}") from exc


def _read_member_scalar(path: Path, key: str) -> int:
    try:
        with zipfile.ZipFile(path) as zf:
            names = _member_map(zf)
            if key not in names:
                raise VerifyError(f"回读缺片: {path.name} 缺少训练步 step")
            with zf.open(names[key]) as fp:
                shape, _order, dtype = _read_header(fp)
                dtype = np.dtype(dtype)
                count = int(np.prod(shape, dtype=np.int64)) if shape else 1
                if count != 1:
                    raise VerifyError(f"回读 step 不是标量: {path.name}")
                raw = _read_exact(
                    fp, dtype.itemsize, error=VerifyError,
                    message=f"回读截断: {path.name} step 正文不足",
                )
                return int(np.frombuffer(raw, dtype=dtype)[0])
    except ReshardError:
        raise
    except (zipfile.BadZipFile, OSError, ValueError, EOFError) as exc:
        raise VerifyError(f"回读失败 {path.name} step: {exc}") from exc


def _streams_equal(a_iter, b_iter) -> bool:
    """比较两个字节块流是否完全一致（逐块比较，跨块边界对齐）。"""
    a_buf = memoryview(b"")
    b_buf = memoryview(b"")
    a_done = b_done = False
    while True:
        if not a_buf and not a_done:
            try:
                a_buf = memoryview(next(a_iter))
            except StopIteration:
                a_done = True
        if not b_buf and not b_done:
            try:
                b_buf = memoryview(next(b_iter))
            except StopIteration:
                b_done = True
        if not a_buf and not b_buf:
            return a_done and b_done
        if not a_buf or not b_buf:
            return False
        n = min(len(a_buf), len(b_buf))
        if a_buf[:n] != b_buf[:n]:
            return False
        a_buf = a_buf[n:]
        b_buf = b_buf[n:]


def verify_staging_streaming(staging_dir: Path, layout: SourceLayout,
                             sizes: list[int],
                             counter: SourceBodyCounter | None = None) -> None:
    """回读整套暂存 npz，逐片与源检查点核对三个数组的值和训练步。

    源正文在校验阶段再读 1x（写阶段已读 1x，合计 2x）。回读校验只读
    暂存文件，不产生额外的源读取。
    """
    staging_dir = Path(staging_dir)
    paths = [staging_dir / f"rank_{r}.npz" for r in range(len(sizes))]
    for r, path in enumerate(paths):
        if not path.is_file():
            raise VerifyError(f"回读缺片: {path.name} 不存在")

    for key in ARRAY_KEYS:
        with SourceCursor(layout, key, counter=counter) as cursor:
            for r, (path, size) in enumerate(zip(paths, sizes)):
                expected = size * layout.itemsize
                if not _streams_equal(
                        cursor.iter_bytes(expected),
                        _iter_member_body(path, key, size, layout.dtype)):
                    raise VerifyError(
                        f"数组 {key} 在 {path.name} 与源检查点不一致"
                    )

    for r, path in enumerate(paths):
        step = _read_member_scalar(path, "step")
        if step != layout.step:
            raise VerifyError(
                f"回读训练步不一致: {path.name} step={step}，期望 {layout.step}"
            )



# ---------------------------------------------------------------------------
# 兼容接口（整块读写；服务流程不调用，仅保留给外部脚本/旧调用方）
# 注意：这些函数会把数组整块读入内存，不满足流式内存约束。
# ---------------------------------------------------------------------------
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
    """整块读取并校验源检查点（兼容接口）。"""
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
    """整块写一个新 rank 分片（兼容接口），返回文件字节数。"""
    path = Path(path)
    np.savez(path, **arrays, step=np.int64(step))
    return path.stat().st_size


def verify_shards(shard_paths: list[Path],
                  expected_sizes: list[int],
                  step: int,
                  source: SourceCheckpoint) -> None:
    """整块回读全部暂存分片（兼容接口）。"""
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
