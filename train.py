import os
import copy
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm
from network import *
from analyse_tool import *

# 有分支的网络名称
branch_network = ["Net_learn_branch", "Net_learn_branch_film","Net_learn_self_branch","YOLOv8_self_res","YOLOv8_small_self_branch"]

def get_eval_preds(model_output):
    preds = model_output[0] if isinstance(model_output, tuple) else model_output
    return preds[0] if isinstance(preds, tuple) else preds

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_targets_to_device(targets, device):
    if not isinstance(targets, dict):
        return targets

    for key, value in targets.items():
        if isinstance(value, torch.Tensor):
            targets[key] = value.to(device)
    return targets


def parse_label_filter_classes(value):
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple, set)):
        return [int(cls_id) for cls_id in value]
    return [int(cls_id.strip()) for cls_id in value.split(",") if cls_id.strip() != ""]


def summarize_dataset_targets(dataset, num_classes, split_name="train"):
    total_images = len(dataset)
    total_targets = 0
    empty_images = 0
    missing_labels = 0
    class_counts = [0 for _ in range(max(num_classes, 1))]
    out_of_range_classes = {}

    for img_path in dataset.image_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(
            os.path.dirname(img_path).replace("images", "labels"),
            img_name + ".txt",
        )

        if not os.path.exists(label_path):
            missing_labels += 1
            empty_images += 1
            continue

        labels_np = dataset._load_label(label_path)
        if labels_np.shape[0] == 0:
            empty_images += 1
            continue

        total_targets += int(labels_np.shape[0])
        for cls_id in labels_np[:, 0].astype(int).tolist():
            if 0 <= cls_id < len(class_counts):
                class_counts[cls_id] += 1
            else:
                out_of_range_classes[cls_id] = out_of_range_classes.get(cls_id, 0) + 1

    print(f"[{split_name}] images={total_images}, targets={total_targets}, empty_images={empty_images}, missing_labels={missing_labels}")
    print(f"[{split_name}] class_counts={class_counts}")
    if out_of_range_classes:
        print(f"[{split_name}] 警告: 发现越界类别ID -> {out_of_range_classes}")


def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, best_val_loss, config, train_result=None, val_result=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_dir = os.path.join(path,config["save_name"])
    number = 0
    while os.path.exists(save_dir):
        number += 1
        save_dir = os.path.join(path,f"{config['save_name']}-{number}")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_loss": best_val_loss,
            "config": config,
            "train_result":train_result,
            "val_result" : val_result,
            "model_name": config["model_name"]
        },
        path,
    )

# 训练参数设置
def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device="cpu"):
    checkpoint = torch.load(path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_val_loss = float(checkpoint.get("best_loss", float("inf")))
    ckpt_config = checkpoint.get("config", {})
    train_dict = checkpoint["train_result"]
    val_dict = checkpoint["val_result"]
    return start_epoch, best_val_loss, ckpt_config,train_dict,val_dict

# 训练
def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, epoch, config,best_loss):
    model.train()
    use_amp = config["use_amp"] 
    accumulate = max(config["accumulate"], 1)
    branch = config["model_name"] in branch_network
    if branch:
        epoch_sums = [0.0, 0.0, 0.0,0.0]
    else:
        epoch_sums = [0.0, 0.0, 0.0]
    total_images = 0
    optimizer_steps = 0

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/Train , best_loss {round(best_loss,4)}")

    for i, batch in enumerate(pbar):
        images = batch["images"].to(device, non_blocking=True)
        targets = move_targets_to_device(batch["targets"], device)

        with autocast("cuda",enabled=use_amp):
            
            # 如果有分支，把分支结果拿出来
            if branch:   
                targets_branch = move_targets_to_device(batch["other_targets"], device)
                preds,branch_pred = model(images)
                loss_all, loss_detach = loss_fn(preds, targets)
                branch_loss = loss_fn.attention_loss(branch_pred,targets_branch)
                total_loss = (loss_all.sum() + branch_loss*config["branch_loss"])/ accumulate

            else:
                preds = model(images)
                loss_all, loss_detach = loss_fn(preds, targets)
                total_loss = loss_all.sum() / accumulate

        # 反向传播
        scaler.scale(total_loss).backward()

        should_step = (i + 1) % accumulate == 0 or (i + 1) == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        batch_size = images.size(0)
        current_sums = [
            l.item() * batch_size if isinstance(l, torch.Tensor) else float(l) * batch_size
            for l in loss_detach
        ]
        if branch:
            epoch_sums[3] += branch_loss.item() * batch_size

        for j in range(3):
            epoch_sums[j] += current_sums[j]
        total_images += batch_size

        # 更新进度条
        pbar.set_postfix(
            {
                "loss": f"{sum(current_sums) / max(batch_size, 1):.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
            }
        )

    if total_images == 0:
        return 0.0, 0.0, 0.0, 0.0

    # 计算平均损失
    avg_box = epoch_sums[0] / total_images
    avg_cls = epoch_sums[1] / total_images
    avg_dfl = epoch_sums[2] / total_images
    if branch:
        avg_branch = epoch_sums[3] * config["branch_loss"] / total_images
    avg_total = sum(epoch_sums) / total_images

    if branch:
        return avg_box, avg_cls, avg_dfl, avg_branch,avg_total, optimizer_steps
    else:
        return avg_box, avg_cls, avg_dfl, avg_total, optimizer_steps

# 验证
def validate_one_epoch(model, loader, loss_fn, device, epoch, config):
    model.eval()
    use_amp = config["use_amp"]
    accumulate = max(config["accumulate"], 1)

    epoch_sums = [0.0, 0.0, 0.0]
    total_images = 0
    pred_records = []
    gt_records = []
    image_offset = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/Val", leave=False)
        for batch in pbar:
            images = batch["images"].to(device, non_blocking=True)
            targets = move_targets_to_device(batch["targets"], device)

            with autocast("cuda",enabled=use_amp):   
                preds = model(images, return_loss=True)
                preds_map = get_eval_preds(model(images))
                _, loss_detach = loss_fn(preds, targets)

            # 计算map50
            batch_pred_records, batch_gt_records = collect_map_records(
                preds_map,
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

    if total_images == 0:
        return 0.0, 0.0, 0.0, 0.0

    # 计算各个部分的损失值
    avg_box = epoch_sums[0] / total_images
    avg_cls = epoch_sums[1] / total_images
    avg_dfl = epoch_sums[2] / total_images
    avg_total = sum(epoch_sums) / total_images
    map_metrics = compute_map_from_records(pred_records, gt_records, config["num_classes"], iou_thresholds=[0.5])
    map50 = map_metrics["map50"]
    ap50 = map_metrics["ap50_per_class"]
    return avg_box, avg_cls, avg_dfl, avg_total,map50,ap50


def main(args):

    config = {
        "data_path": args.data_path,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": args.device,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "accumulate": args.accumulate,
        "seed": args.seed,
        "num_classes": args.num_classes,
        "save_dir": args.save_dir,
        "save_name": args.save_name,
        "use_amp": args.use_amp,
        "pre_model": args.pre_model,
        "label_filter_classes": parse_label_filter_classes(args.label_filter_classes),
        "depth":args.depth,
        "val_conf_threshold": args.val_conf_threshold,
        "val_nms_iou_threshold": args.val_nms_iou_threshold,
        "val_pre_nms_topk": args.val_pre_nms_topk,
        "val_max_det": args.val_max_det,
        "use_albumentations": args.use_albumentations,
        "model_name": args.model_name,
        "save_ratio": args.save_ratio,
        "para_number": -1,
        "gflops":-1,
        "branch_loss": args.branch_loss
    }

    # 训练数据保存路径
    os.makedirs(os.path.dirname(config["save_dir"]), exist_ok=True)
    save_dir = os.path.join(config["save_dir"],config["save_name"])
    number = 0
    while os.path.exists(save_dir):
        number += 1
        save_dir = os.path.join(config["save_dir"],f"{config['save_name']}-{number}")
    os.makedirs(save_dir)

    set_seed(config["seed"])

    # 选择训练硬件
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if config["device"]=="0":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif config["device"]=="1":
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    elif config["device"]=="2":
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    elif config["device"]=="3":
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    elif config["device"]=="cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")


    print(f"Device: {device}")
    if config["label_filter_classes"] is not None:
        print(f"标签选择: {config['label_filter_classes']}")

    # 数据集处理
    train_dataset = YOLO_Dataset(
        config["data_path"],
        mode="train",
        image_size=config["image_size"],
        label_filter_classes=config["label_filter_classes"],
        use_albumentations=config["use_albumentations"],
    )
    val_dataset = YOLO_Dataset(
        config["data_path"],
        mode="val",
        image_size=config["image_size"],
        label_filter_classes=config["label_filter_classes"],
    )

    # 计算数据集大小
    summarize_dataset_targets(train_dataset, config["num_classes"], split_name="train")
    summarize_dataset_targets(val_dataset, config["num_classes"], split_name="val")

    # 数据集加载
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
        collate_fn=train_dataset.collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
        collate_fn=val_dataset.collate_fn,
        drop_last=False,
    )

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # 判断输入名称是否存在
    if config["model_name"] in globals():
        model = globals()[config["model_name"]](nc=config["num_classes"],scales=config["depth"]).to(device)
    else:
        print("没有对应的神经网络，请检查名称")
        return
    
    # 是否使用预训练权重
    if not args.resume and config["pre_model"] != "":
        model.load_state_dict(torch.load(config["pre_model"])["model_state_dict"])
    
    # 计算神经网络参数量
    profile_model = copy.deepcopy(model).to(device).eval()
    config["para_number"],config["gflops"] = para_cumulate(config,profile_model,device)

    loss_fn = Loss(model)

    # 梯度优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999),
    )

    # 学习率优化器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config["epochs"], 1),
        eta_min=config["lr"] * 0.05,
    )

    scaler = GradScaler("cuda",enabled=config["use_amp"] and device.type == "cuda")

    best_loss = 99999
    no_improve_count = 0
    start_epoch = 0

    # 训练结果数据   
    train_dict = {}
    val_dict = {}

    train_dict["train_loss_box"] = []
    train_dict["train_loss_cls"] = []
    train_dict["train_loss_dfl"] = []
    train_dict["train_loss_total"] = []

    val_dict["val_loss_box"] = []
    val_dict["val_loss_cls"] = []
    val_dict["val_loss_dfl"] = []
    val_dict["val_loss_total"] = []
    val_dict["val_map50"] = []

    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"恢复训练失败,未找到checkpoint: {args.resume}")

        # 记忆数据重新赋值
        start_epoch, best_loss, ckpt_config,train_dict,val_dict = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )

        if isinstance(ckpt_config, dict):
            for key in ("num_classes", "image_size", "batch_size"):
                if key in ckpt_config and config.get(key) != ckpt_config[key]:
                    print(f"警告: 当前配置 {key}={config.get(key)},checkpoint记录为 {ckpt_config[key]}")

        print(f"恢复训练的权重路径: {args.resume}")
        print(f"恢复训练的轮次: {start_epoch}, 最佳训练损失值: {best_loss:.4f}")


    for epoch in range(start_epoch, config["epochs"]):

        if config["model_name"] in branch_network:
            train_box, train_cls, train_dfl, train_branch,train_total, optimizer_steps = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, epoch, config,best_loss
            )
        else:
            train_box, train_cls, train_dfl,train_total, optimizer_steps = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, epoch, config,best_loss
            )
        
        train_dict["train_loss_box"].append(train_box)
        train_dict["train_loss_cls"].append(train_cls)
        train_dict["train_loss_dfl"].append(train_dfl)
        train_dict["train_loss_total"].append(train_total)

        val_box, val_cls, val_dfl, val_total,map50,ap50 = validate_one_epoch(
            model, val_loader, loss_fn, device, epoch, config
        )
        
        val_dict["val_loss_box"].append(val_box)
        val_dict["val_loss_cls"].append(val_cls)
        val_dict["val_loss_dfl"].append(val_dfl)
        val_dict["val_loss_total"].append(val_total)
        val_dict["val_map50"].append(map50)
        if optimizer_steps > 0:
            scheduler.step()

        print(f"\n--- Epoch {epoch + 1}/{config['epochs']} ---")
        if config["model_name"] in branch_network:
            print(
                f"Train | Box: {train_box:.4f}, Cls: {train_cls:.4f}, "
                f"DFL: {train_dfl:.4f}, Branch: {train_branch:.4f}, Total: {train_total:.4f}"
            )
        else:
            print(
                f"Train | Box: {train_box:.4f}, Cls: {train_cls:.4f}, "
                f"DFL: {train_dfl:.4f}, Total: {train_total:.4f}"
            )
        print(
            f"Valid | Box: {val_box:.4f}, Cls: {val_cls:.4f}, "
            f"DFL: {val_dfl:.4f}, Total: {val_total:.4f}, "
            f"map50: {map50:.4f}"
        )

        last_path = os.path.join(save_dir, "last.pth")
        save_checkpoint(last_path, epoch, model, optimizer, scheduler, scaler, best_loss, config, train_dict, val_dict)
        # 损失值
        best_loss_temple = round(config["save_ratio"]*val_total + (1-config["save_ratio"])*train_total,4)

        if best_loss_temple < best_loss:
            best_loss = best_loss_temple
            no_improve_count = 0
            best_path = os.path.join(save_dir, "best.pth")
            save_checkpoint(best_path, epoch, model, optimizer, scheduler, scaler, best_loss, config)
            print(f"New best model saved. Best_Loss: {best_loss:.4f}")
        else:
            no_improve_count += 1
            print(f"No improvement. Patience: {no_improve_count}/{config['patience']}, loss_temple: {best_loss_temple}")

        if no_improve_count >= config["patience"]:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"\n训练完成,文件保存路径为：{save_dir}")
    print(f"Best val loss: {best_loss:.4f}")

# 训练参数设置
def parser_set():
    parser = argparse.ArgumentParser(description="YOLO training with resume support")
    parser.add_argument("--resume", type=str, default="", help="需要恢复训练的权重路径,如果不需要不输入即可")
    parser.add_argument("--pre_model", type=str, default="", help="预训练权重路径")
    parser.add_argument("--num_classes", type=int, default=1, help="类别数量")
    parser.add_argument("--data_path", type=str, default="data_all/car_with_ground", help="数据集路径")
    parser.add_argument("--image_size", type=int, default=640, help="训练时的输入尺寸")
    parser.add_argument("--batch_size", type=int, default=32, help="批处理数量")
    parser.add_argument("--num_workers", type=int, default=0, help="数据集加载的核心数量")
    parser.add_argument("--device",type=str,default="1",help="数字代表显卡编号,输入cpu表示用cpu跑,如果都不是默认cpu")
    parser.add_argument("--epochs", type=int, default=300, help="训练轮次")
    parser.add_argument("--lr", type=float, default=5e-3, help="初始学习率")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="学习率衰减权重")
    parser.add_argument("--patience", type=int, default=30, help="早停轮次")
    parser.add_argument("--accumulate", type=int, default=2, help="梯度累计轮次")
    parser.add_argument("--seed", type=int, default=42, help="随机数种子")
    parser.add_argument("--save_dir", type=str, default="weights/1/high", help="权重保存路径")
    parser.add_argument("--save_name",type=str,default="1-yolov8_small_self_branch_p4_4.5",help="训练结果保存名称")
    parser.add_argument("--use_amp",type=bool,default=True,help="是否在训练时使用AMP")
    parser.add_argument("--val_conf_threshold", type=float, default=0.001, help="验证mAP使用的置信度阈值,越低越准但越慢")
    parser.add_argument("--val_nms_iou_threshold", type=float, default=0.7, help="验证mAP使用的NMS IoU阈值")
    parser.add_argument("--val_pre_nms_topk", type=int, default=1000, help="每张图NMS前最多保留的候选框数量")
    parser.add_argument("--val_max_det", type=int, default=300, help="每张图NMS后最多保留的检测框数量")
    parser.add_argument("--use_albumentations", action="store_true", help="启用Albumentations高级增强;若训练中出现段错误，建议不要开启")
    parser.add_argument(
        "--label_filter_classes",
        type=str,
        default="0",
        help="只处理这些类别的标签,例如: 0,2,3; 不传则处理全部标签",
    )
    parser.add_argument("--model_name",type=str,default="YOLOv8_small_self_branch",help="使用的神经网络名称")
    parser.add_argument("--branch_loss",type=float,default=4.5,help="分支损失占总损失的比例,为0时表示不计算分支损失")
    parser.add_argument("--save_ratio",type=float,default=0.7,help="训练保存时,验证集和训练集的损失值比例")
    parser.add_argument("--depth",type=tuple,default=(0.33, 0.25, 1024),
                        help="神经网络的深度参数,第一个是每层重复数量,第二个是通道倍率(至少是64的公因数倒数),第三个是最大通道限制")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    arguments = parser_set()
    main(arguments)
