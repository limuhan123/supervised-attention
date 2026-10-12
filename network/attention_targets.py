"""Select context annotations for the auxiliary spatial attention branch."""

import torch


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
