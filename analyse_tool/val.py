from network import *
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast
from thop import profile
from pathlib import Path
import json
import csv


def export_attention_diagnostics(model_path, output_dir, device="cuda:0", gate_values=(None,)):
    """复用train.py验证程序，读取已有权重；不训练、不改写检查点。"""
    from train import validate_one_epoch, set_seed
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
    config = dict(checkpoint["config"])
    set_seed(config.get("seed", 42))
    config["use_amp"] = config["use_amp"] and str(device).startswith("cuda")
    config["collect_attention_details"] = True
    config["attention_context_classes"] = config.get("attention_context_classes", [4])
    dataset = YOLO_Dataset(config["data_path"], mode="val", image_size=config["image_size"],
                           label_filter_classes=config["label_filter_classes"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                        num_workers=0, pin_memory=str(device).startswith("cuda"), collate_fn=dataset.collate_fn)
    device = torch.device(device)
    model = globals()[config["model_name"]](nc=config["num_classes"], scales=config["depth"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    original_gate = float(model.branch_scale.detach()) if hasattr(model, "branch_scale") else None
    summaries = []
    for gate in gate_values:
        if gate is not None and original_gate is None:
            raise ValueError("该模型没有可进行推理干预的branch_scale")
        used_gate = original_gate if gate is None else float(gate)
        if used_gate is not None:
            with torch.no_grad():
                model.branch_scale.fill_(used_gate)
        result = validate_one_epoch(model, loader, Loss(model), device, 0, config)
        label = "saved_gate" if gate is None else f"gate_{gate:g}"
        rows = []
        for values in config.get("last_attention_details", []):
            row = dict(values)
            row["image"] = Path(dataset.image_paths[row["image_index"]]).name
            row["gate"] = used_gate
            for name in ("positive_mean", "background_mean"):
                row["gA_" + name] = used_gate * row[name] if used_gate is not None and row[name] is not None else None
            rows.append(row)
        if rows:
            with (output_dir / f"{label}_per_image.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        summary = {"checkpoint": str(model_path), "intervention": label, "gate": used_gate,
            "AP50": result[4], "AP50_95": result[6], "validation_images": len(dataset),
            "attention_positive": config.get("last_attention_positive"),
            "attention_background": config.get("last_attention_background"),
            "positive_energy_ratio": config.get("last_positive_energy_ratio"),
            "empty_attention_mean": config.get("last_empty_attention_mean"),
            "gA_positive": used_gate * config["last_attention_positive"]
                if used_gate is not None and config.get("last_attention_positive") is not None else None,
            "gA_background": used_gate * config["last_attention_background"]
                if used_gate is not None and config.get("last_attention_background") is not None else None,
            "attention_evaluation_regions": {"shrink": .85, "expand": 1.15}}
        summaries.append(summary)
        with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
            json.dump(summaries, stream, ensure_ascii=False, indent=2)
        print("INFERENCE_DIAGNOSTIC", json.dumps(summary), flush=True)
    return summaries



# 将标签也放在指定的设备上
def move_targets_to_device(targets, device):
    if not isinstance(targets, dict):
        return targets

    for key, value in targets.items():
        if isinstance(value, torch.Tensor):
            targets[key] = value.to(device)
    return targets

def infer_num_classes(checkpoint):
    cfg = checkpoint.get("config", {})
    if isinstance(cfg, dict) and "num_classes" in cfg:
        return int(cfg["num_classes"])

    state_dict = checkpoint.get("model_state_dict", {})
    for k, v in state_dict.items():
        # Detect 头最后分类卷积输出通道就是类别数
        if k.endswith("detect.cv3.0.2.weight"):
            return int(v.shape[0])
    return 2

def get_loss_preds(model_output):
    return model_output[0] if isinstance(model_output, tuple) else model_output

def get_eval_preds(model_output):
    preds = get_loss_preds(model_output)
    return preds[0] if isinstance(preds, tuple) else preds

def val_main(model_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载权重
    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
    config = checkpoint["config"]
    # print(config["data_path"])
    network_name = config['model_name']
    # num_classes = infer_num_classes(checkpoint)
    num_classes = config["num_classes"]
    # 数据读取
    dataset = YOLO_Dataset(config["data_path"], 
                        mode="val", image_size=config["image_size"],
                        label_filter_classes=config["label_filter_classes"])
    pin_memory = device.type == "cuda"
    dataloader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=pin_memory,
            collate_fn=dataset.collate_fn,
            drop_last=False,
        )

    cls = globals()[network_name]
    model = cls(nc=num_classes,scales=config["depth"]).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    loss_fn = Loss(model)
    model.eval()
    if hasattr(model, "branch_scale"):
        print(model.branch_scale.item())
    use_amp = config["use_amp"] and device.type == "cuda"
    epoch_sums = [0.0, 0.0, 0.0]
    total_images = 0
    pred_records = []
    gt_records = []
    image_offset = 0
    with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Val", leave=False)
            for batch in pbar:
                images = batch["images"].to(device, non_blocking=True)
                targets = move_targets_to_device(batch["targets"], device)

                with autocast("cuda",enabled=use_amp):
                    preds = get_loss_preds(model(images, return_loss=True))
                    _, loss_detach = loss_fn(preds, targets)
                    preds_eval = get_eval_preds(model(images))
                
                batch_pred_records, batch_gt_records = collect_map_records(
                    preds_eval,
                    targets,
                    config["num_classes"],
                    img_size=config["image_size"],
                    image_offset=image_offset,
                    conf_threshold=config["val_conf_threshold"],
                    nms_iou_threshold=config["val_nms_iou_threshold"],
                    pre_nms_topk=config["val_pre_nms_topk"],
                    max_det=config["val_max_det"],
                )
                pred_records.extend(batch_pred_records)
                gt_records.extend(batch_gt_records)

                batch_size = images.size(0)
                image_offset += batch_size
                current_sums = [
                    l.item() * batch_size if isinstance(l, torch.Tensor) else float(l) * batch_size
                    for l in loss_detach
                ]

                for j in range(3):
                    epoch_sums[j] += current_sums[j]
                total_images += batch_size

                pbar.set_postfix({"v_loss": f"{sum(current_sums) / max(batch_size, 1):.4f}"})

    avg_box = epoch_sums[0] / total_images
    avg_cls = epoch_sums[1] / total_images
    avg_dfl = epoch_sums[2] / total_images
    avg_total = sum(epoch_sums) / total_images
    map_metrics = compute_map_from_records(pred_records, gt_records, num_classes, iou_thresholds=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95])
    map50 = map_metrics["map50"]
    ap50 = map_metrics["ap50_per_class"]
    ap50_95 = map_metrics["ap50_95_per_class"]
    map50_95 = map_metrics["map50_95"]
    map_per = map_metrics["map_per_iou"]
    from network.ap_profile import summarize_ap_profile
    ap_profile = summarize_ap_profile(map_metrics)

    print(
            f"Valid | Box: {avg_box:.4f}, Cls: {avg_cls:.4f}, "
            f"DFL: {avg_dfl:.4f}, Total: {avg_total:.4f}, map50: {map50:.4f}, map50_95: {map50_95:.4f}"
    )
    print(f"AP50 per class: {ap50}")
    print(f"AP50_95 per class: {ap50_95}")
    print(f"map_per in thresholds: {map_per}")


    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    macs, params = profile(model, inputs=(dummy_input, ),verbose=False)
    print(f"总参数量: {params}")
    print(f"GFLOPs: {macs * 2 / 1e9:.4f}")

# 保存验证结果到 JSON

    save_dir = Path(model_path).parent
    json_path = save_dir / "val_result.json"

    val_result = {

        # 模型基本信息
        "model": {
            "model_name": network_name,
            "model_path": str(model_path),
            "num_classes": int(num_classes),
            "image_size": int(config["image_size"]),
            "branch_scale": float(model.branch_scale.item()) if hasattr(model, "branch_scale") else None
        },

        #验证损失
        "loss": {
            "box": round(float(avg_box), 3),
            "cls": round(float(avg_cls), 3),
            "dfl": round(float(avg_dfl), 3),
            "total": round(float(avg_total), 3),
        },

        # 检测指标
        "metrics": {
            "full_precision_profile": ap_profile,
            "map50": round(float(map50), 3),
            "map50_95": round(float(map50_95), 3),

            "ap50_per_class": [
                round(float(x), 3) for x in ap50
            ],

            "ap50_95_per_class": [
                round(float(x), 3) for x in ap50_95
            ],

            "iou_thresholds": [
                round(float(x), 2)
                for x in map_metrics["iou_thresholds"]
            ],

            "map_per_iou": [
                round(float(x), 3) for x in map_per
            ],
        },

        # 模型复杂度
        "complexity": {
            "params": int(params),
            "params_M": round(float(params / 1e6), 3),
            "gflops": round(float(macs * 2 / 1e9), 3),
        },

        # 验证参数
        "validation_config": {
            "conf_threshold": float(config["val_conf_threshold"]),
            "nms_iou_threshold": float(config["val_nms_iou_threshold"]),
            "pre_nms_topk": int(config["val_pre_nms_topk"]),
            "max_det": int(config["val_max_det"]),
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            val_result,
            f,
            ensure_ascii=False,
            indent=4
        )

    print(f"验证结果已保存到: {json_path}")


if __name__ == "__main__":
    model_path="weights/one-low_yolov8_se/best.pth"
    val_main(model_path)
