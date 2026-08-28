import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, Tuple ,Union, Any 
import torch.nn.functional as F





class Loss():
    def __init__(self,model,tal_topk=10):
        device = next(model.parameters()).device
        m = model.detect# Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none") #分类使用BCE，不带sigmoid 
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        #如果设置了reg_max，则需要使用dfl方法进行解码
        self.use_dfl = m.reg_max > 1

        self.assigner = Calculator(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
        )

        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
    
    def __call__(
        self,
        preds: Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算位置、分类和dfl的损失之和乘以批量"""
        return self.loss(self.parse_output(preds), batch)
    
    #解析模型的推理结果
    def parse_output(
        self, preds: Union[Dict[str, torch.Tensor],Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        return preds[0] if isinstance(preds, Tuple) else preds
    
    #计算损失
    def loss(self, preds: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        batch_size = preds["boxes"].shape[0]
        loss, loss_detach = self.loss_calculate(preds, batch)[1:]
        return loss * batch_size, loss_detach

    #返回xyxy格式的坐标
    @staticmethod
    def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
        lt, rb = distance.chunk(2, dim)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return torch.cat([c_xy, wh], dim)  # xywh 
        return torch.cat((x1y1, x2y2), dim)

    #生成锚框
    @staticmethod
    def make_anchors(feats, strides, grid_cell_offset=0.5):
        anchor_points, stride_tensor = [], []
        assert feats is not None
        dtype, device = feats[0].dtype, feats[0].device
        #创建网格及其索引
        for i in range(len(feats)):  
            stride = strides[i]
            h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
            sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
            sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
            sy, sx = torch.meshgrid(sy, sx, indexing="ij") 
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_points), torch.cat(stride_tensor)

    # 检测结果解码
    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return self.dist2bbox(pred_dist, anchor_points, xywh=False)
    
    # 通过转换为张量格式和缩放坐标对目标进行预处理
    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = self.assigner.cul.xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def loss_calculate(self, preds: Dict[str,torch.Tensor], label: Dict[str, Any]):
        loss = torch.zeros(3, device=self.device)  
        #分别拿出检测框和置信度
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )

        #获取网格索引和步长
        anchor_points, stride_tensor = self.make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0] #计算图像的真实尺寸

        #读取标签
        targets = torch.cat((label["batch_id"].view(-1, 1), label["cls"].view(-1, 1), label["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        #解码检测结果
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        #对标签进行处理，便于计算损失值
        """
        target_labels : [bs,num_anchor] 每一个anchor对应的物体的类别编号是什么
        target_bboxes : [bs,num_anchor,4] 每一个anchor对应的检测框的xyxy
        target_scores : [bs,num_anchor,num_cls] 每一个anchor对应的置信度分布
        fg_mask : [bs,num_anchor] 针对每一个anchor,所有标签数量的和,反应该anchor是否有标注框
        target_gt_idx :[bs,num_anchor] 针对每一个anchor,iou最大的那个标签,表示的是每一个anchor分别被分配给了哪个标签
        """
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        #使用BCE计算分类
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        #计算边界框损失
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= 7.5  # box gain
        loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5  # dfl gain

        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )  # loss(box, cls, dfl)
    
    @staticmethod
    def dice_loss_from_prob(pred_prob, target, valid_mask=None, eps=1e-6):
        """
        pred_prob: [H, W]
        target:    [H, W]
        valid_mask:[H, W] or None
        """
        if valid_mask is not None:
            pred_prob = pred_prob * valid_mask
            target = target * valid_mask

        inter = (pred_prob * target).sum()
        union = pred_prob.sum() + target.sum()

        return 1.0 - (2.0 * inter + eps) / (union + eps)

    @staticmethod
    def focal_bce_with_logits(logits, target, valid_mask=None, alpha=0.25, gamma=2.0, eps=1e-6):
        """
        logits: [H, W]
        target: [H, W]
        """
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        prob = torch.sigmoid(logits)

        pt = prob * target + (1.0 - prob) * (1.0 - target)
        alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)

        focal_weight = alpha_t * (1.0 - pt).pow(gamma)
        loss = focal_weight * bce

        if valid_mask is not None:
            loss = loss * valid_mask
            return loss.sum() / (valid_mask.sum() + eps)

        return loss.mean()

    @staticmethod
    def attention_loss(
        pred_logits,
        targets,
        lambda_dice=1.0,
        lambda_out=1.0,
        lambda_topk=0.5,
        lambda_sparse=0.03,
        lambda_ratio=0.5,
        target_ratio=0.7,
        shrink_ratio=0.85,
        expand_ratio=1.15,
        topk_ratio=0.1,
        eps=1e-6,
    ):
        """
        场地区域监督注意力损失。

        pred_logits:
            [B, 1, H, W]，注意力分支输出的 logits，未经过 sigmoid。

        targets:
            {
                "batch_id": [N],
                "bboxes":   [N, 4]  # YOLO归一化格式 [cx, cy, w, h]
            }
        """

        if pred_logits.dim() != 4:
            raise ValueError(f"pred_logits should be [B, 1, H, W], got {pred_logits.shape}")

        if pred_logits.shape[1] != 1:
            raise ValueError(
                f"建议 attention loss 输入单通道注意力 logits [B,1,H,W]，"
                f"当前输入为 {pred_logits.shape}。不要直接对多通道特征求平均。"
            )

        B, _, H, W = pred_logits.shape
        device = pred_logits.device
        dtype = pred_logits.dtype

        batch_ids = targets["batch_id"].view(-1).long().to(device)
        bboxes = targets["bboxes"].to(device=device, dtype=dtype)

        total_loss = pred_logits.new_tensor(0.0)

        for i in range(B):
            logits = pred_logits[i, 0]      # [H, W]
            att = torch.sigmoid(logits)    # [H, W]

            target_mask = batch_ids == i

            pos_mask = torch.zeros((H, W), device=device, dtype=dtype)
            ignore_mask = torch.zeros((H, W), device=device, dtype=dtype)

            if target_mask.sum() > 0:
                for box in bboxes[target_mask]:
                    cx, cy, bw, bh = box

                    cx = cx * W
                    cy = cy * H
                    bw = bw * W
                    bh = bh * H

                    # 正样本区域：内缩框
                    pos_bw = bw * shrink_ratio
                    pos_bh = bh * shrink_ratio

                    px1 = int(torch.floor(torch.clamp(cx - pos_bw / 2, 0, W)).item())
                    py1 = int(torch.floor(torch.clamp(cy - pos_bh / 2, 0, H)).item())
                    px2 = int(torch.ceil(torch.clamp(cx + pos_bw / 2, 0, W)).item())
                    py2 = int(torch.ceil(torch.clamp(cy + pos_bh / 2, 0, H)).item())

                    if px2 > px1 and py2 > py1:
                        pos_mask[py1:py2, px1:px2] = 1.0

                    # ignore 区域：外扩框内部
                    ign_bw = bw * expand_ratio
                    ign_bh = bh * expand_ratio

                    ix1 = int(torch.floor(torch.clamp(cx - ign_bw / 2, 0, W)).item())
                    iy1 = int(torch.floor(torch.clamp(cy - ign_bh / 2, 0, H)).item())
                    ix2 = int(torch.ceil(torch.clamp(cx + ign_bw / 2, 0, W)).item())
                    iy2 = int(torch.ceil(torch.clamp(cy + ign_bh / 2, 0, H)).item())

                    if ix2 > ix1 and iy2 > iy1:
                        ignore_mask[iy1:iy2, ix1:ix2] = 1.0

            # 外扩框之外是明确背景
            neg_mask = 1.0 - ignore_mask

            # 有效监督区域：正样本区域 + 明确背景区域
            # 中间 ignore 区域不参与 BCE
            valid_mask = torch.clamp(pos_mask + neg_mask, 0, 1)

            # 监督目标：正样本区域为1，背景区域为0
            target_map = pos_mask

            # 如果没有场地标注，则整图作为背景
            if target_mask.sum() == 0:
                neg_mask = torch.ones((H, W), device=device, dtype=dtype)
                valid_mask = torch.ones((H, W), device=device, dtype=dtype)
                target_map = torch.zeros((H, W), device=device, dtype=dtype)

            # 1. Focal BCE：像素级正负监督
            loss_focal = Loss.focal_bce_with_logits(
                logits=logits,
                target=target_map,
                valid_mask=valid_mask,
            )

            # 2. Dice：增强区域完整性
            if pos_mask.sum() > 0:
                loss_dice = Loss.dice_loss_from_prob(
                    pred_prob=att,
                    target=target_map,
                    valid_mask=valid_mask,
                )
            else:
                loss_dice = pred_logits.new_tensor(0.0)

            # 3. 框外平均抑制
            if neg_mask.sum() > 0:
                outside_values = att[neg_mask.bool()]
                loss_outside_mean = outside_values.mean()

                # 4. 框外 top-k 抑制，防止局部背景高亮
                k = max(1, int(topk_ratio * outside_values.numel()))
                loss_outside_topk = outside_values.topk(k).values.mean()
            else:
                loss_outside_mean = pred_logits.new_tensor(0.0)
                loss_outside_topk = pred_logits.new_tensor(0.0)

            # 5. 背景稀疏，而不是全图稀疏
            if neg_mask.sum() > 0:
                loss_sparse = (att * neg_mask).sum() / (neg_mask.sum() + eps)
            else:
                loss_sparse = pred_logits.new_tensor(0.0)

            # 6. 激活能量比例约束
            if pos_mask.sum() > 0:
                inside_energy = (att * pos_mask).sum()
                total_energy = att.sum() + eps
                inside_ratio = inside_energy / total_energy
                loss_ratio = F.relu(target_ratio - inside_ratio)
            else:
                loss_ratio = pred_logits.new_tensor(0.0)

            loss = (
                loss_focal
                + lambda_dice * loss_dice
                + lambda_out * loss_outside_mean
                + lambda_topk * loss_outside_topk
                + lambda_sparse * loss_sparse
                + lambda_ratio * loss_ratio
            )

            total_loss += loss

        return total_loss / max(B, 1)

    @staticmethod
    def focal_scene_context_loss(scene_logits, targets, context_cls_id=4, alpha=0.25, gamma=2.0):
        """
        基于 Focal Loss 的纯辅助场坪场景监督。
        """
        B = scene_logits.size(0)
        device = scene_logits.device
        
        batch_ids = targets["batch_id"].view(-1).long().to(device)
        cls_ids = targets["cls"].view(-1).long().to(device)
        
        # 构造图像级正负样本 (1=有场坪, 0=无场坪)
        scene_labels = torch.zeros((B, 1), device=device, dtype=scene_logits.dtype)
        for i in range(B):
            if (cls_ids[batch_ids == i] == context_cls_id).any():
                scene_labels[i, 0] = 1.0
                
        # 计算 Focal Loss
        bce_loss = F.binary_cross_entropy_with_logits(scene_logits, scene_labels, reduction='none')
        p = torch.sigmoid(scene_logits)
        p_t = p * scene_labels + (1 - p) * (1 - scene_labels)
        
        focal_weight = (1 - p_t) ** gamma
        alpha_weight = scene_labels * alpha + (1 - scene_labels) * (1 - alpha)
        
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        return focal_loss.mean()      


#用于计算标签与真实值之间关系
class Calculator(nn.Module):
    def __init__(self,        
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        stride: list = [8, 16, 32],
        eps: float = 1e-9,
        topk2=None,):
        super().__init__()

        self.topk = topk
        self.topk2 = topk2 or topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.stride = stride
        self.stride_val = self.stride[1] if len(self.stride) > 1 else self.stride[0]
        self.eps = eps

        self.cul = Yolo_tool()

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        输入：
        pd_score: 置信度预测 (bs,num_anchors,num_cls)
        pd_bboxes: 检测框预测 (bs,num_anchors,4)
        anc_point: 锚框 (num_anchors,2)
        gt_labels: 标签类别预测 (bs,max,1)
        gt_bboxes: 标签检测框预测 (bs,max,4)
        mask_gt: 标签模板，用于去除掉不需要的数据 (bs,max,1)
         
        返回值：
        针对标签
        target_labels : [bs,num_anchor] 每一个anchor对应的物体的类别编号是什么
        target_bboxes : [bs,num_anchor,4] 每一个anchor对应的检测框的xyxy
        target_scores : [bs,num_anchor,num_cls] 每一个anchor对应的置信度分布
        fg_mask : [bs,num_anchor] 针对每一个anchor,所有标签数量的和,反应该anchor是否有标注框
        target_gt_idx :[bs,num_anchor] 针对每一个anchor,iou最大的那个标签,表示的是每一个anchor分别被分配给了哪个标签
        """

        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        #如果标签值为空直接返回    
        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        #使用显存进行计算，如果显存不够则使用cpu
        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Move tensors to CPU, compute, then move back to original device
                print.warning("显卡内存不够用，使用cpu进行计算")
                cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
                result = self._forward(*cpu_tensors)
                return Tuple(t.to(device) for t in result)
            raise

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        
        #筛选候选框和锚点
        # mask_pos 标记了所有需要的候选锚点[bs,max,num_anchor]
        # alin_metric 计算了每一个真实标签框和预测结果的广义iou [bs,max,num_anchor]
        # overlaps 计算了每一个真实标签框和预测结果的iou [bs,max.num_anchor]
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes, align_metric
        )

        #计算正锚点的目标标签、目标边界框和目标分数
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # 归一化
        align_metric *= mask_pos #只保留正样本的广义iou
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # 找到每一个框对应的广义iou最佳的anchor [bs,max]
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # 找到每一个框对应的iou最佳的anchor [bs,max]
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    
    #锚点筛选
    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):

        #计算标签与anchor的关系，返回值为[bs,max,num_anchor], 删掉检测框没有包含的锚点
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        
        #计算iou以及iou和置信度的乘积 [bs,max,num_anchor]
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        
        #计算候选框，从align_metric中的每一个真值框中，找到值最大的前topk个框，并做标记。[bs,max,num_anchor]
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        
        #相乘，将不需要的数据全部置0，只保留有用的候选框和有用的锚点
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    #选择iou最高的标签框
    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
        """
        输入:
            mask_pos : 标记了所有需要的候选锚点[bs,max,num_anchor]
            overlaps : 计算了每一个真实标签框和预测结果的iou [bs,max,num_anchor]
            max : 该batch中标注框最多的图像的标注框数量.
            align_metric : alin_metric 计算了每一个真实标签框和预测结果的广义iou [bs,max,num_anchor]

        输出:
            target_gt_idx :[bs,num_anchor] 针对每一个anchor,iou最大的那个标签,表示的是每一个anchor分别被分配给了哪个标签
            fg_mask : [bs,num_anchor] 针对每一个anchor,所有标签数量的和
            mask_pos : [bs,max,num_anchor]最终的anchor掩模
        """
        # [bs,max,num_anchor] -> [bs,num_anchor]
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # 有一个锚点被分配给了多个检测框
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # [bs,max,num_anchor]

            max_overlaps_idx = overlaps.argmax(1)  # [bs,num_anchor] 针对每一个anchor，找到最大的那个iou框
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            #is_max_overlaps, mask_pos二者都是0,1掩模，最后的到也是掩模
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # [bs,max,num_anchor]

            fg_mask = mask_pos.sum(-2)

        if self.topk2 != self.topk:
            align_metric = align_metric * mask_pos  # 重新得到广义iou张量[bs,max,num_anchor]
            max_overlaps_idx = torch.topk(align_metric, self.topk2, dim=-1, largest=True).indices  # 仅返回索引[bs,max]
            topk_idx = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)  # update mask_pos
            topk_idx.scatter_(-1, max_overlaps_idx, 1.0)
            mask_pos *= topk_idx
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # [bs,num_anchor]
        return target_gt_idx, fg_mask, mask_pos
    
    #计算正锚点的目标标签、目标边界框和目标分数
    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
            target_gt_idx :[bs,num_anchor] 针对每一个anchor,iou最大的那个标签,表示的是每一个anchor分别被分配给了哪个标签
            fg_mask : [bs,num_anchor] 针对每一个anchor,所有标签数量的和，如果为0，表示这个anchor处是没有标签的
        Returns:
            都是标签给出的信息
            target_labels : [bs,num_anchor] 每一个anchor对应的物体的类别编号是什么
            target_bboxes : [bs,num_anchor,4] 每一个anchor对应的检测框的xyxy
            target_scores : [bs,num_anchor,num_cls] 每一个anchor对应的置信度分布
        """
        # [bs,1]
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        #乘上max，相当于把索引按照不同的batch展平了
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # [bs,num_anchor]
        #每一个anchor对应的物体的类别编号是什么
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # [bs,num_anchor]

        # gt_bboxes [bs,max,4] -> [bs*max,4],把第一个按照flatte展平了,然后再索引
        # 得到[bs,num_anchor]，表示每一个anchor对应的检测框的xyxy
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # 把所有小于0的强制变成0
        target_labels.clamp_(0)

        # [bs,num_anchor,num_cls]
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  

        # 因为推理的时候，scroe给出的是一个长度的num_cls的向量，所以这里是模仿这个分布，利用target_labels中的序号作为索引，
        # 在对应位置填入1
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # [bs,num_anchor,num_cls]
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0) #把所有背景处置零

        return target_labels, target_bboxes, target_scores

    #预处理标签候选框
    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
        
        gt_bboxes_xywh = self.cul.xyxy2xywh(gt_bboxes)
        wh_mask = gt_bboxes_xywh[..., 2:] < self.stride[0]  # the smallest stride
        gt_bboxes_xywh[..., 2:] = torch.where(
            (wh_mask * mask_gt).bool(),
            torch.tensor(self.stride_val, dtype=gt_bboxes_xywh.dtype, device=gt_bboxes_xywh.device),
            gt_bboxes_xywh[..., 2:],
        )
        gt_bboxes = self.cul.xywh2xyxy(gt_bboxes_xywh)

        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(eps) #储存的是布尔值 
        
    #iou计算
    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        #mask_gt [bs,max,num_anchor],没有被检测框包含的锚点被置0，没有达到max的部分也被置0

        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # [bs, max, mum_anchor]
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # [2, bs, max]
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # [bs, max]
        ind[1] = gt_labels.squeeze(-1)  # [bs, max]
        # 计算每一个batch中的每一个真实检测框，相较于每一个anchor的置信度
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # [bs, max, num_anchor]

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        #广播操作
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt] #[bs,max,num_anchor,4]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt] #[bs,max,num_anchor,4]
        #[bs,max,num_anchor,4]计算标签框和所有anchor的交并比
        overlaps[mask_gt] = self.cul.bbox_iou(gt_boxes, pd_boxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    #候选框选择
    def select_topk_candidates(self, metrics, topk_mask=None):
        #metrics 置信度和iou的乘积，[bs,max,num_anchor]
        #topk指取前多少个
        # 从每一个标签框中的num_anchor个值中取出前topk个最大的值，[b, max, topk]，并记录这个最大值在metrics中的位置索引
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)

        #筛选低分,topk_mask是索引,[b,max,topk]
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # [b, max, topk]
        topk_idxs.masked_fill_(~topk_mask, 0) #~是取反，即将False部分，不满足要求的部分置0

        #[bs,max,num_anchor]
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        #[bs,max,1]
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # 根据索引，进行累加
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)

        # 如果有多个topk都加在了一个锚点上，就把这个锚点清0
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

class Yolo_tool():
    def __init__(self):
        pass
    #一种比clone更快的复制方法
    @staticmethod
    def empty_like(x):
        return torch.empty_like(x, dtype=x.dtype) if isinstance(x, torch.Tensor) else np.empty_like(x, dtype=x.dtype)

    def xyxy2xywh(self,x):
        assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
        y = self.empty_like(x)#复制
        x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        y[..., 0] = (x1 + x2) / 2  # x c
        y[..., 1] = (y1 + y2) / 2  # y 
        y[..., 2] = x2 - x1  # w
        y[..., 3] = y2 - y1  # h
        return y
    
    def xywh2xyxy(self,x):
        assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
        y = self.empty_like(x)  # faster than clone/copy
        xy = x[..., :2]  # centers
        wh = x[..., 2:] / 2  # half width-height
        y[..., :2] = xy - wh  # top left xy
        y[..., 2:] = xy + wh  # bottom right xy
        return y

    #计算iou的方法
    @staticmethod
    def bbox_iou(
        box1: torch.Tensor,
        box2: torch.Tensor,
        xywh: bool = True,
        GIoU: bool = False,
        DIoU: bool = False,
        CIoU: bool = False,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        Args:
            box1 : 标签检测框坐标 [bs,max,8400,4]
            box2 : 预测检测框坐标 [bs,max,8400,4]
            xywh : 如果是true表示输入的框是xywh,反之为xyxy
            GIoU : 若为True，则计算广义交并比
            DIoU : 若为True，则计算距离交并比。
            CIoU : 若设为True，则计算完整交并比。
            eps : 用于避免除以零的小数值。

        Returns:
            IoU, GIoU, DIoU, or CIoU 
        """
        # 获取边界框的坐标
        # 检测框转换
        if xywh:  
            (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
            w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
            b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
            b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
        else:  
            b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
            b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        # 计算相交面积 [bs,max,num_anchor,1],真实值和每一个预测值都相减
        inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
            b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
        ).clamp_(0)

        # 区域面积
        union = w1 * h1 + w2 * h2 - inter + eps

        # IoU
        iou = inter / union
        if CIoU or DIoU or GIoU:
            cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # 最小外接矩形的宽
            ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # 最小外接矩形的高
            if CIoU or DIoU:  
                c2 = cw.pow(2) + ch.pow(2) + eps  # 平方和
                rho2 = (
                    (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
                ) / 4  #求两个框中点的距离
                if CIoU:  
                    v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + eps))
                    return iou - (rho2 / c2 + v * alpha)  # CIoU
                return iou - rho2 / c2  # DIoU
            c_area = cw * ch + eps  # 最小外接矩形的面积
            return iou - (c_area - union) / c_area  #计算惩罚项，根据两个框的重叠程度
        return iou  # IoU
    
    @staticmethod
    def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: Union[int , None] = None) -> torch.Tensor:
        x1y1, x2y2 = bbox.chunk(2, -1)
        dist = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
        if reg_max is not None:
            dist = dist.clamp_(0, reg_max - 0.01)  # dist (lt, rb)
        return dist

class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max: int = 16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.cul = Yolo_tool()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    # 计算边界框的IoU和DFL损失
    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = self.cul.bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = self.cul.bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = self.cul.bbox2dist(anchor_points, target_bboxes)
            # normalize ltrb by image size
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)
    