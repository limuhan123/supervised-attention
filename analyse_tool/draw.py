import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_training_curves(save_dir):
    """从train.py逐轮CSV绘图，不需要重新读取大权重文件。"""
    save_dir = Path(save_dir)
    with (save_dir / "metrics.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return
    with (save_dir / "best_metrics.json").open(encoding="utf-8") as stream:
        best_epoch = json.load(stream)["epoch"]
    panels = [("Detection loss", [("train_det_loss", "Train"), ("val_det_loss", "Validation")]),
              ("Validation AP", [("map50", "AP50"), ("map50_95", "AP50-95")]),
              ("Residual coefficient", [("gate", "g")]),
              ("Auxiliary loss", [("train_aux_loss", "Unweighted auxiliary loss")]),
              ("Attention response", [("attention_positive", "Positive region"),
                                      ("attention_background", "Background")]),
              ("Learning rate", [("lr", "Learning rate")])]
    from matplotlib.ticker import MaxNLocator
    with plt.rc_context({"figure.facecolor": "white", "axes.facecolor": "white",
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9}):
        figure, axes = plt.subplots(2, 3, figsize=(12, 6.3), layout="constrained")
        epochs = [int(row["epoch"]) for row in rows]
        for axis, (title, series) in zip(axes.flat, panels):
            present = False
            for (key, label), color in zip(series, ("#76a5c6", "#85b69c")):
                valid = [(int(row["epoch"]), float(row[key])) for row in rows
                         if row.get(key) not in (None, "", "None")]
                if valid:
                    present = True
                    axis.plot(*zip(*valid), label=label, color=color, linewidth=1.8,
                              marker="o" if len(valid) < 3 else None, markersize=3)
            axis.set_title(title)
            if not present:
                axis.text(.5, .5, "Not used", ha="center", transform=axis.transAxes, color="#888")
                axis.set_axis_off()
                continue
            axis.axvline(best_epoch, color="#aaa", linestyle=":", linewidth=1)
            axis.set(xlabel="Epoch", xlim=(.5, max(2, epochs[-1] + .5)))
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(alpha=.15)
            axis.legend(frameon=False, fontsize=8)
            if title in ("Validation AP", "Attention response"):
                axis.set_ylim(0, 1)
        figure.suptitle(save_dir.name + " | selected by validation AP50-95", fontsize=11)
        figure.savefig(save_dir / "training_curves.png", dpi=180)
        figure.savefig(save_dir / "training_curves.pdf")
        plt.close(figure)
    if "effective_gate_positive" in rows[0]:
        save_attention_diagnostic_curves(save_dir, rows)


def save_attention_diagnostic_curves(save_dir, rows):
    panels = [("Actual modulation gA", [("effective_gate_positive", "Positive region"),
                                         ("effective_gate_background", "Background")]),
              ("Positive-region energy fraction", [("positive_energy_ratio", "Mean over images with sites")]),
              ("Shared P3 gradient norms before clipping", [("gradient_det_norm", "Detection"),
                                                              ("gradient_aux_norm", "Weighted auxiliary")]),
              ("Detection / auxiliary gradient cosine", [("gradient_cosine", "Cosine")])]
    from matplotlib.ticker import MaxNLocator
    with plt.rc_context({"figure.facecolor": "white", "axes.facecolor": "white",
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9}):
        figure, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
        for axis, (title, series) in zip(axes.flat, panels):
            present = False
            for (key, label), color in zip(series, ("#76a5c6", "#85b69c")):
                valid = [(int(row["epoch"]), float(row[key])) for row in rows
                    if row.get(key) not in (None, "", "None")
                    and ("gradient norms" not in title or float(row[key]) > 0)]
                if valid:
                    present = True
                    axis.plot(*zip(*valid), marker="o" if len(valid) < 3 else None,
                              markersize=3, linewidth=1.7, color=color, label=label)
            axis.set_title(title)
            if not present:
                axis.text(.5, .5, "Not used / not sampled", ha="center", transform=axis.transAxes)
                axis.set_axis_off()
                continue
            axis.set_xlabel("Epoch")
            axis.set_xlim(.5, max(2, max(int(row["epoch"]) for row in rows) + .5))
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(alpha=.15)
            axis.legend(frameon=False, fontsize=8)
            if "gradient norms" in title:
                axis.set_yscale("log")
            elif "cosine" in title:
                axis.set_ylim(-1.05, 1.05)
                axis.axhline(0, color="#aaa", linewidth=.7)
            elif "energy fraction" in title:
                axis.set_ylim(0, 1)
            elif title == "Actual modulation gA":
                lower, upper = axis.get_ylim()
                axis.set_ylim(min(0, lower), max(0, upper))
                axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        figure.suptitle(Path(save_dir).name, fontsize=11)
        figure.savefig(Path(save_dir) / "attention_diagnostics.png", dpi=180)
        figure.savefig(Path(save_dir) / "attention_diagnostics.pdf")
        plt.close(figure)


def save_experiment_comparison(rows, output):
    rows = [row for row in rows if row["state"] == "complete"]
    if not rows:
        return
    figure, axis = plt.subplots(figsize=(9, 4.5), layout="constrained", facecolor="white")
    groups = {}
    for row in rows:
        groups.setdefault(row["experiment"], []).append(row)
    names = list(groups)
    x = list(range(len(names)))
    for offset, key, label, color in ((-.18, "AP50", "AP50", "#76a5c6"),
                                      (.18, "AP50_95", "AP50-95", "#85b69c")):
        values = [100 * statistics.mean(r[key] for r in groups[name]) for name in names]
        errors = [100 * statistics.stdev(r[key] for r in groups[name]) if len(groups[name]) > 1 else 0
                  for name in names]
        bars = axis.bar([v + offset for v in x], values, .34,
                        yerr=errors, capsize=2, color=color, label=label)
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    axis.set(xticks=x, xticklabels=[name + f"\nn={len(groups[name])}" for name in names], ylim=(0, 105),
             ylabel="Validation AP (%)", title="Single target 0 | mean, sample SD for repeated runs")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(Path(output) / "comparison.png", dpi=200)
    figure.savefig(Path(output) / "comparison.pdf")
    plt.close(figure)


def load_loss_data(model_path: Path,dataset,data_select):
    """
    从模型权重中读取训练总损失。

    Returns:
        x: epoch序号
        y: 训练损失
    """
    checkpoint = torch.load(
        model_path,
        weights_only=False,
        map_location="cpu",
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"权重文件不是字典格式：{model_path}"
        )

    train_result = checkpoint.get(dataset)

    if train_result is None:
        raise ValueError(
            f"{model_path} 中的 {dataset} 为 None 或不存在"
        )

    if not isinstance(train_result, dict):
        raise TypeError(
            f"{model_path} 中的 {dataset} 不是字典"
        )

    y = train_result.get(data_select)

    if y is None:
        raise ValueError(
            f"{model_path} 中不存在有效的 {data_select}"
        )

    # Tensor转为Python列表
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().flatten().tolist()
    else:
        y = list(y)

    # 删除可能存在的None值，并转成float
    valid_y = []

    for value in y:
        if value is None:
            continue

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()

        valid_y.append(float(value))

    if not valid_y:
        raise ValueError(
            f"{model_path} 中的 {data_select} 没有有效数据"
        )

    # 直接根据损失数据长度生成横坐标，避免epoch和数据长度不一致
    x = list(range(1, len(valid_y) + 1))

    return x, valid_y


def draw_single_model(model_path: Path,dataset,data_select):
    """
    绘制单个模型的损失曲线。
    图片保存在模型文件所在目录。
    """
    x, y = load_loss_data(model_path,dataset,data_select)

    # 使用模型所在文件夹作为图例名
    model_name = model_path.parent.name

    # 如果模型文件直接放在当前目录，则使用模型文件名
    if not model_name:
        model_name = model_path.stem

    save_path = model_path.parent / f"{data_select}.png"

    plt.figure(figsize=(10, 7))

    plt.plot(
        x,
        y,
        linewidth=2,
        label=model_name,
    )

    plt.title(f"{data_select} - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel(data_select)

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"模型：{model_name}")
    print(f"权重路径：{model_path}")
    print(f"曲线已保存：{save_path}")


def find_model_files(root_dir: Path):
    """
    搜索大文件夹中各子文件夹内的模型权重。

    优先级：
        1. last.pt
        2. last.pth
        3. best.pt
        4. best.pth

    每个子文件夹只选取一个权重文件。
    """
    model_files = []

    candidate_names = [
        "last.pt",
        "last.pth",
        "best.pt",
        "best.pth",
    ]

    # 只遍历大文件夹的直接子文件夹
    for model_dir in sorted(root_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        selected_model = None

        for candidate_name in candidate_names:
            candidate_path = model_dir / candidate_name

            if candidate_path.exists():
                selected_model = candidate_path
                break

        if selected_model is not None:
            model_files.append(selected_model)
        else:
            print(
                f"跳过文件夹：{model_dir}，"
                f"没有找到 last.pt、last.pth、best.pt 或 best.pth"
            )

    return model_files


def draw_multiple_models(root_dir: Path,dataset,data_select):
    """
    将大文件夹下所有模型的损失曲线绘制在同一张图中。
    """
    model_files = find_model_files(root_dir)

    if not model_files:
        raise FileNotFoundError(
            f"在 {root_dir} 的子文件夹中没有找到模型权重"
        )

    plt.figure(figsize=(12, 8))

    success_count = 0

    for model_path in model_files:
        model_name = model_path.parent.name

        try:
            x, y = load_loss_data(model_path,dataset,data_select)

            # Matplotlib会自动为不同曲线分配不同颜色
            plt.plot(
                x,
                y,
                linewidth=2,
                label=model_name,
            )

            success_count += 1

            print(
                f"已读取：{model_name}，"
                f"epoch数量：{len(y)}，"
                f"权重：{model_path.name}"
            )

        except Exception as error:
            print(
                f"读取模型失败：{model_path}\n"
                f"错误信息：{error}"
            )

    if success_count == 0:
        plt.close()
        raise RuntimeError("所有模型的数据读取均失败")

    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel(data_select)

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    plt.legend(
        loc="best",
        frameon=True,
    )

    plt.tight_layout()

    # 对比图保存在输入的大文件夹中
    save_path = root_dir / f"{data_select}.png"

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("-" * 60)
    print(f"成功绘制模型数量：{success_count}")
    print(f"多模型对比图已保存：{save_path}")


def main_draw(input_path: str,dataset="train",data_select="loss_total"):
    """
    根据输入路径自动判断：
        文件路径：绘制单模型
        文件夹路径：绘制文件夹内所有模型
    """
    path = Path(input_path).expanduser().resolve()
    data_select = f"{dataset}_{data_select}"
    dataset = f"{dataset}_result"

    if not path.exists():
        raise FileNotFoundError(
            f"输入路径不存在：{path}"
        )

    # 输入的是权重文件
    if path.is_file():
        if path.suffix.lower() not in {".pt", ".pth"}:
            raise ValueError(
                f"输入文件不是.pt或.pth权重文件：{path}"
            )

        draw_single_model(path,dataset,data_select)

    # 输入的是文件夹
    elif path.is_dir():
        draw_multiple_models(path,dataset,data_select)

    else:
        raise ValueError(
            f"无法识别输入路径：{path}"
        )


if __name__ == "__main__":
    path = "weights/1"
    main_draw(path)
