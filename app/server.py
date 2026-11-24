"""Flask 入口：检查点重分片网页。

路由：
  GET  /                 状态页（旧卷 / 新卷 / 发布结果 / 历史）
  POST /mount            模拟卸载、挂回同一卷、挂载不同卷
  POST /reshard          选检查点 + 新 rank 数 + 新卷容量 -> 重分片并发布
  GET  /download/<rank>  按新 rank 下载已发布的 npz 分片
"""
from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, abort, flash, redirect, render_template, request, send_file

from sample_data import ensure_samples
from storage import ReshardService

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
OTHER_VOLUME_ID = "oldvol-different-000"

service = ReshardService(DATA_DIR)
ensure_samples(service.old)


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("SECRET_KEY", "reshard-demo-secret")

    @app.get("/")
    def index():
        pub = service.new.published
        readable = service.published_readable()
        return render_template(
            "index.html",
            old=service.old.status(),
            checkpoints=service.old.list_checkpoints(),
            new=service.new,
            published=pub,
            readable=readable,
            usage=service.new.usage_bytes(),
            history=list(reversed(service.new.history)),
        )

    @app.post("/mount")
    def mount():
        action = request.form.get("action")
        if action == "unmount":
            service.old.unmount()
            flash("旧卷已卸载。依赖旧卷的发布结果现在不可读。", "warn")
        elif action == "mount_same":
            service.old.mount()
            flash(f"已挂回同一卷 {service.old.stored_id()}，依赖结果恢复可读。", "ok")
        elif action == "mount_other":
            service.old.mount(OTHER_VOLUME_ID)
            flash(f"已挂载另一块卷 {OTHER_VOLUME_ID}：卷 ID 不同，"
                  "旧卷内容与依赖它的结果仍不可读。", "warn")
        else:
            flash("未知挂载操作。", "err")
        return redirect("/")

    @app.post("/reshard")
    def reshard():
        ckpt_id = request.form.get("ckpt_id", "")
        try:
            n_new = int(request.form.get("n_new", ""))
            capacity = int(request.form.get("capacity", ""))
        except ValueError:
            flash("新 rank 数和新卷容量必须是整数。", "err")
            return redirect("/")
        if n_new < 1 or capacity < 0:
            flash("新 rank 数必须 >= 1，容量必须 >= 0。", "err")
            return redirect("/")
        result = service.run(
            ckpt_id=ckpt_id,
            n_new=n_new,
            capacity=capacity,
            can_mount_old=request.form.get("can_mount_old") == "on",
            simulate_write_fail=request.form.get("simulate_write_fail") == "on",
        )
        if result["status"] == "published":
            flash(
                f"发布成功 {result['publish_id']}：{n_new} 个新分片，"
                f"训练步 {result['step']}，本次峰值 {result['op_peak']} B"
                f"（上一套 {result['prev_published_bytes']} B + "
                f"本次暂存 {result['staged_bytes']} B）。",
                "ok",
            )
        else:
            flash(f"重分片失败：{result['reason']}。已保留上次发布结果。", "err")
        return redirect("/")

    @app.get("/download/<int:rank>")
    def download(rank: int):
        pub = service.new.published
        if pub is None:
            abort(404, description="尚未发布任何结果")
        if service.published_readable() is False:
            abort(409, description="结果依赖旧卷，旧卷未挂载（或挂载了不同的卷），当前不可读")
        if not 0 <= rank < int(pub["ranks"]):
            abort(404, description=f"rank {rank} 超出范围")
        path = service.new.published_root / pub["id"] / f"rank_{rank}.npz"
        if not path.is_file():
            abort(404, description="分片文件缺失")
        return send_file(path, as_attachment=True,
                         download_name=f"{pub['id']}_rank_{rank}.npz")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
