"""交付前自测：重分片逻辑 + 换卡 + 卸载/挂回 + 失败保留旧发布。

运行: python3 tests/selftest.py
退出码 0 = 全部通过。
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import reshard  # noqa: E402
from reshard import (ARRAY_KEYS, MissingShardError, OverCapacityError,  # noqa: E402
                     SimulatedWriteError, StepMismatchError, split_sizes)
from sample_data import ensure_samples  # noqa: E402
from storage import ReshardService  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, fn) -> None:
    try:
        fn()
        results.append((PASS, name, ""))
        print(f"[PASS] {name}")
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc(limit=3)
        results.append((FAIL, name, tb))
        print(f"[FAIL] {name}\n{tb}")


def make_service(tmp: str) -> ReshardService:
    svc = ReshardService(Path(tmp))
    ensure_samples(svc.old)
    return svc


def load_source_arrays(data_dir: Path, ckpt_id: str):
    """直接从旧卷目录读源数组，用于和发布结果比对。"""
    ckpt = data_dir / "old_volume" / "checkpoints" / ckpt_id
    per_rank = {k: [] for k in ARRAY_KEYS}
    r = 0
    while (ckpt / f"rank_{r}.npz").is_file():
        with np.load(ckpt / f"rank_{r}.npz") as z:
            for k in ARRAY_KEYS:
                per_rank[k].append(z[k])
        r += 1
    return {k: np.concatenate(v) for k, v in per_rank.items()}


def published_arrays(svc: ReshardService):
    pub = svc.new.published
    out = {k: [] for k in ARRAY_KEYS}
    steps = []
    for r in range(pub["ranks"]):
        with np.load(svc.new.published_root / pub["id"] / f"rank_{r}.npz") as z:
            for k in ARRAY_KEYS:
                out[k].append(z[k])
            steps.append(int(z["step"]))
    return {k: np.concatenate(v) for k, v in out.items()}, steps


# ---------- 1. 纯逻辑：等分规则 ----------
def t_split_sizes():
    assert split_sizes(30, 5) == [6, 6, 6, 6, 6]
    assert split_sizes(10, 3) == [4, 3, 3], "余数应给低 rank"
    assert split_sizes(31, 4) == [8, 8, 8, 7]
    assert split_sizes(2, 5) == [1, 1, 0, 0, 0], "rank 多于元素时低 rank 先分"
    assert split_sizes(7, 1) == [7]
    assert sum(split_sizes(100, 9)) == 100


# ---------- 2. 换卡 3 -> 5：拼接一致 + 每片带训练步 ----------
def t_reshard_3_to_5():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        op = svc.run("ckpt-demo-3rank", 5, 1 << 20, True, False)
        assert op["status"] == "published", op
        assert op["shard_sizes"] == [6, 6, 6, 6, 6]
        src = load_source_arrays(Path(tmp), "ckpt-demo-3rank")
        got, steps = published_arrays(svc)
        for k in ARRAY_KEYS:
            assert np.array_equal(got[k], src[k]), f"{k} 拼接后与源不一致"
        assert steps == [42000] * 5, "每片都应带训练步"
        assert svc.new.published["step"] == 42000


# ---------- 3. 换卡 3 -> 2：余数给低 rank ----------
def t_reshard_3_to_2():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        op = svc.run("ckpt-demo-3rank", 2, 1 << 20, False, False)
        assert op["status"] == "published", op
        assert op["shard_sizes"] == [15, 15]
        src = load_source_arrays(Path(tmp), "ckpt-demo-3rank")
        got, _ = published_arrays(svc)
        for k in ARRAY_KEYS:
            assert np.array_equal(got[k], src[k])


# ---------- 4. 奇数总量余数分布 ----------
def t_remainder_to_low_ranks():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        op = svc.run("ckpt-demo-3rank", 7, 1 << 20, False, False)  # 30 = 7*4+2
        assert op["status"] == "published", op
        assert op["shard_sizes"] == [5, 5, 4, 4, 4, 4, 4], op["shard_sizes"]


# ---------- 5. 缺片：失败并保留上次发布 ----------
def t_missing_shard_keeps_published():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        ok = svc.run("ckpt-demo-3rank", 2, 1 << 20, False, False)
        assert ok["status"] == "published"
        before = svc.new.published["id"]
        bad = svc.run("ckpt-missing-rank", 4, 1 << 20, False, False)
        assert bad["status"] == "failed"
        assert "缺片" in bad["reason"], bad["reason"]
        assert svc.new.published["id"] == before, "失败后应保留上次发布"
        assert svc.new.staging_bytes() == 0, "失败后暂存应清理"


# ---------- 6. 训练步不一致：失败并保留上次发布 ----------
def t_step_mismatch_keeps_published():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        svc.run("ckpt-demo-3rank", 2, 1 << 20, False, False)
        before = svc.new.published["id"]
        bad = svc.run("ckpt-step-mismatch", 4, 1 << 20, False, False)
        assert bad["status"] == "failed"
        assert "训练步不一致" in bad["reason"], bad["reason"]
        assert svc.new.published["id"] == before


# ---------- 7. 容量超限（上一套 + 本次暂存都计入） ----------
def t_over_capacity_keeps_published():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        svc.run("ckpt-demo-3rank", 2, 1 << 20, False, False)
        before = svc.new.published["id"]
        prev_bytes = svc.new.published_bytes()
        # 容量只比上一套大一点，必然装不下上一套 + 本次暂存
        bad = svc.run("ckpt-demo-3rank", 4, prev_bytes + 10, False, False)
        assert bad["status"] == "failed"
        assert "容量超限" in bad["reason"], bad["reason"]
        assert svc.new.published["id"] == before
        assert svc.new.staging_bytes() == 0


# ---------- 8. 模拟写失败：保留上次发布，暂存清理 ----------
def t_simulated_write_failure():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        svc.run("ckpt-demo-3rank", 2, 1 << 20, False, False)
        before = svc.new.published["id"]
        bad = svc.run("ckpt-demo-3rank", 6, 1 << 20, False, True)
        assert bad["status"] == "failed"
        assert "模拟写失败" in bad["reason"], bad["reason"]
        assert svc.new.published["id"] == before
        assert svc.new.staging_bytes() == 0


# ---------- 9. 峰值 = 上一套 + 本次暂存 ----------
def t_peak_accounting():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        op1 = svc.run("ckpt-demo-3rank", 2, 1 << 20, False, False)
        assert op1["op_peak"] == 0 + op1["staged_bytes"]
        peak1 = svc.new.peak_bytes
        op2 = svc.run("ckpt-demo-3rank", 5, 1 << 20, False, False)
        assert op2["op_peak"] == op2["prev_published_bytes"] + op2["staged_bytes"]
        assert op2["prev_published_bytes"] > 0, "第二套发布时上一套应计入"
        assert svc.new.peak_bytes == max(peak1, op2["op_peak"])


# ---------- 10. 卸载 -> 依赖旧卷的结果不可读；挂回同一卷恢复；挂不同卷不恢复 ----------
def t_unmount_dependency_cycle():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        op = svc.run("ckpt-demo-3rank", 4, 1 << 20, True, False)  # 依赖旧卷
        assert op["status"] == "published"
        assert svc.published_readable() is True
        svc.old.unmount()
        assert svc.published_readable() is False, "卸载后应不可读"
        svc.old.mount("oldvol-different-000")
        assert svc.published_readable() is False, "挂不同卷不应恢复"
        svc.old.mount()  # 挂回同一卷
        assert svc.published_readable() is True, "挂回同一卷应恢复"


# ---------- 11. 不依赖旧卷的结果：卸载后仍可读 ----------
def t_independent_result_survives_unmount():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        svc.run("ckpt-demo-3rank", 4, 1 << 20, False, False)  # 不依赖旧卷
        svc.old.unmount()
        assert svc.published_readable() is True


# ---------- 12. 卸载时源不可读，不能重分片 ----------
def t_unmounted_source_unreadable():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        svc.old.unmount()
        bad = svc.run("ckpt-demo-3rank", 4, 1 << 20, False, False)
        assert bad["status"] == "failed"
        assert "源检查点不可读" in bad["reason"], bad["reason"]
        assert svc.new.published is None


# ---------- 13. HTTP 冒烟：页面 / 重分片 / 下载 / 卸载后 409 / 挂回恢复 ----------
def t_http_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["DATA_DIR"] = tmp
        # 延迟导入，让 server 用临时 DATA_DIR
        for mod in ("server",):
            sys.modules.pop(mod, None)
        import server  # noqa: WPS433

        c = server.app.test_client()
        assert c.get("/").status_code == 200

        r = c.post("/reshard", data={
            "ckpt_id": "ckpt-demo-3rank", "n_new": "5",
            "capacity": str(1 << 20), "can_mount_old": "on",
        }, follow_redirects=False)
        assert r.status_code == 302
        pub = server.service.new.published
        assert pub and pub["ranks"] == 5

        # 下载 rank_3 并核对内容 = 全局下标 [18, 24)
        resp = c.get("/download/3")
        assert resp.status_code == 200
        data = np.load(io.BytesIO(resp.data))
        src = load_source_arrays(Path(tmp), "ckpt-demo-3rank")
        for k in ARRAY_KEYS:
            assert np.array_equal(data[k], src[k][18:24]), f"下载分片 {k} 内容错误"
        assert int(data["step"]) == 42000

        # 卸载 -> 依赖旧卷，下载应 409
        c.post("/mount", data={"action": "unmount"})
        assert c.get("/download/3").status_code == 409
        page = c.get("/").data.decode()
        assert "不可读" in page

        # 挂回同一卷 -> 恢复
        c.post("/mount", data={"action": "mount_same"})
        assert c.get("/download/3").status_code == 200


def main() -> int:
    tests = [
        ("等分规则（余数给低 rank）", t_split_sizes),
        ("换卡 3->5：拼接一致、每片带训练步", t_reshard_3_to_5),
        ("换卡 3->2：连续等分", t_reshard_3_to_2),
        ("余数分布 30->7 片", t_remainder_to_low_ranks),
        ("缺片：失败并保留上次发布", t_missing_shard_keeps_published),
        ("训练步不一致：失败并保留上次发布", t_step_mismatch_keeps_published),
        ("容量超限：失败并保留上次发布", t_over_capacity_keeps_published),
        ("模拟写失败：保留上次发布、清理暂存", t_simulated_write_failure),
        ("峰值 = 上一套 + 本次暂存", t_peak_accounting),
        ("卸载/挂不同卷/挂回同一卷的可读性", t_unmount_dependency_cycle),
        ("不依赖旧卷的结果卸载后仍可读", t_independent_result_survives_unmount),
        ("卸载时源不可读，重分片失败", t_unmounted_source_unreadable),
        ("HTTP 冒烟：页面/发布/下载/409/恢复", t_http_smoke),
    ]
    for name, fn in tests:
        check(name, fn)
    failed = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failed)}/{len(results)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
