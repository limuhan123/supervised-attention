"""Select context annotations for the auxiliary spatial attention branch."""

import torch
import math


@torch.no_grad()
def select_context_targets(context_targets, context_classes=(4,)):
    """
    保留上下文框，而不是位于其中的对象框。
    在同步图像/标签变换后调用。这些标注仅用于监督损失计算。
    在训练和推理过程中，前向传播都会根据图像特征预测一个热力图，而无需真实标签输入.
    """
    if context_classes is None or len(context_classes) == 0:
        raise ValueError("At least one context class must be specified")
    classes = context_targets["cls"].reshape(-1)
    keep = torch.zeros_like(classes, dtype=torch.bool)
    for class_id in context_classes:
        keep |= classes == class_id
    return {key: value[keep.to(value.device)]
            for key, value in context_targets.items()}


@torch.no_grad()
def attention_region_sums(logits, targets, include_diagnostics=False, return_records=False):
    """统一以0.85/1.15区域评价，独立于训练标签构造，保证硬框/忽略带组可比较。"""
    response = logits.float().sigmoid()
    batch, _, height, width = response.shape
    sums = [0.0] * (10 if include_diagnostics else 4)
    records = []
    for index in range(batch):
        positive = torch.zeros((height, width), dtype=torch.bool, device=logits.device)
        expanded = torch.zeros_like(positive)
        boxes = targets["bboxes"][targets["batch_id"].view(-1) == index].cpu().tolist()
        for cx, cy, bw, bh in boxes:
            for scale, mask in ((0.85, positive), (1.15, expanded)):
                x1 = max(0, min(width, math.floor((cx - bw * scale / 2) * width)))
                x2 = max(0, min(width, math.ceil((cx + bw * scale / 2) * width)))
                y1 = max(0, min(height, math.floor((cy - bh * scale / 2) * height)))
                y2 = max(0, min(height, math.ceil((cy + bh * scale / 2) * height)))
                mask[y1:y2, x1:x2] = True
        attention = response[index, 0]
        background = ~expanded
        values = [attention[positive].sum().item(), positive.sum().item(),
                  attention[background].sum().item(), background.sum().item()]
        total = attention.sum().item()
        energy_ratio = values[0] / (total + 1e-6) if values[1] else None
        if include_diagnostics:
            values += [total, attention.numel(), energy_ratio or 0.0, int(values[1] > 0),
                       total if not boxes else 0.0, attention.numel() if not boxes else 0]
        sums = [a + b for a, b in zip(sums, values)]
        if return_records:
            records.append({"batch_index": index,
                "positive_mean": values[0] / values[1] if values[1] else None,
                "background_mean": values[2] / values[3] if values[3] else None,
                "attention_mean": total / attention.numel(),
                "positive_energy_ratio": energy_ratio, "site_boxes": len(boxes)})
    return (sums, records) if return_records else sums
