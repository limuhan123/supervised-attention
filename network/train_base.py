import os
import glob
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms as T

cv2.setNumThreads(0)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

try:
    from torchvision.ops import nms as torchvision_nms
except Exception:
    torchvision_nms = None
try:
    import albumentations as A
except ImportError:
    A = None


# 可监督数据加


# 数据集加载
class YOLO_Dataset(Dataset):
    def __init__(self, data_path, mode='train', image_size=640, label_filter_classes=None, use_albumentations=False):
        super().__init__()
        self.mode = mode
        self.image_size = image_size
        self.use_aug = mode == 'train'
        self.use_albumentations = use_albumentations
        self.label_filter_classes = None
        if label_filter_classes is not None:
            self.label_filter_classes = {int(cls_id) for cls_id in label_filter_classes}
        
        # 路径设置
        img_dir = os.path.join(data_path, 'images', mode)
        label_dir = os.path.join(data_path, 'labels', mode)
        
        # 检查路径是否存在
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"图像路径不存在: {img_dir}")
        if not os.path.exists(label_dir):
            # 如果是测试模式且没有标签，可能允许，但训练模式必须有
            if mode == 'train':
                raise FileNotFoundError(f"标签路径不存在: {label_dir}")
            else:
                print(f"警告: 标签路径不存在: {label_dir}，将返回空标签")
                label_dir = None 
        
        # 获取所有图像路径
        self.image_paths = sorted(glob.glob(os.path.join(img_dir, '*.*')))
        # 过滤支持的图片格式
        self.image_paths = [p for p in self.image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        if len(self.image_paths) == 0:
            raise ValueError(f"没有图像： {img_dir}")

        #暂时不对图像进行复杂变换，仅做Tensor转换
        self.transform = T.Compose([
            T.ToTensor(), 
        ])

        # 优先使用 albumentations 做图像与标签同步增强
        self.albu_aug = None
        if self.use_aug and self.use_albumentations and A is not None:
            self.albu_aug = A.Compose(
                [
                    A.HorizontalFlip(p=0.5), 
                    A.VerticalFlip(p=0.1),
                    A.Affine(
                        scale=(0.9, 1.1),
                        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                        rotate=(-15, 15),
                        shear=(-5, 5),
                        p=0.6,
                    ), # 仿射变换
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
                    A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=30, val_shift_limit=20, p=0.4), # 颜色改变
                ],
                bbox_params=A.BboxParams(
                    format="yolo",
                    label_fields=["class_labels"],
                    min_visibility=0.2,
                ),# 对YOLO数据集中的标签，按照图像变化的方法自适应变换
            )

    def __len__(self):
        return len(self.image_paths)

    def _load_label_raw(self, label_path):
        labels = []
        if label_path and os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = list(map(float, line.split()))
                    if len(parts) >= 5:
                        # YOLO 格式: class, x_center, y_center, w, h
                        labels.append(parts[:5]) 
        # 返回 numpy 数组，如果没有标签则返回 (0, 5)
        return np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)

    def _split_labels(self, labels):
        if self.label_filter_classes is None or labels.shape[0] == 0:
            return labels, np.zeros((0, 5), dtype=np.float32)

        keep_mask = np.array(
            [int(cls_id) in self.label_filter_classes for cls_id in labels[:, 0]],
            dtype=bool,
        )
        return labels[keep_mask], labels[~keep_mask]

    def _load_label(self, label_path):
        labels, _ = self._split_labels(self._load_label_raw(label_path))
        return labels

    def __getitem__(self, idx):
        # 获取路径
        img_path = self.image_paths[idx]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 构造对应的标签路径
        try:
            label_path = os.path.join(
                os.path.dirname(img_path).replace('images', 'labels'), 
                img_name + '.txt'
            )
        except Exception:
            label_path = "" # 防止路径替换失败

        # 读取图像 (BGR)
        image = cv2.imread(img_path)
        if image is None:
            # 如果图片损坏，随机返回另一张，防止训练中断
            print(f"图像加载失败： {img_path}")
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # 读取全量标签，完成图像同步变换后再拆分为训练标签和其他标签
        labels_all = self._load_label_raw(label_path)
        image, labels_all = self.letterbox(image, labels_all, new_shape=self.image_size, color=(114, 114, 114))
        if self.use_aug:
            if self.albu_aug is not None:
                image, labels_all = self.apply_albumentations(image, labels_all)
            else:
                image, labels_all = self.basic_augment(image, labels_all)

        labels_np, other_labels_np = self._split_labels(labels_all)

        # 图像预处理
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        resized_image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        image_tensor = self.transform(resized_image)

        cls_tensor, bboxes_tensor = self._labels_to_tensors(labels_np)
        other_cls_tensor, other_bboxes_tensor = self._labels_to_tensors(other_labels_np)
          
        # 构建返回字典
        # __getitem__ 只返回当前图的信息，batch 索引由 collate_fn 添加
        
        target = {
            'cls': cls_tensor,       # (N, 1)
            'bboxes': bboxes_tensor  # (N, 4)
        }
        other_target = {
            'cls': other_cls_tensor,
            'bboxes': other_bboxes_tensor
        }
        
        return {
            'image': image_tensor,   # Tensor图像
            'target': target,        # 指定类别标签信息
            'other_target': other_target  # 非指定类别标签信息
        }

    @staticmethod
    def _labels_to_tensors(labels_np):
        if labels_np.shape[0] == 0:
            cls_tensor = torch.zeros((0,), dtype=torch.long)
            bboxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        else:
            # labels_np 格式: [class, x, y, w, h]
            cls_tensor = torch.from_numpy(labels_np[:, 0]).long()
            bboxes_tensor = torch.from_numpy(labels_np[:, 1:5])
        return cls_tensor, bboxes_tensor

    @staticmethod
    def _empty_targets_dict():
        return {
            'batch_id': torch.zeros((0, 1), dtype=torch.long),
            'cls': torch.zeros((0,), dtype=torch.long),
            'bboxes': torch.zeros((0, 4), dtype=torch.float32)
        }

    def apply_albumentations(self, image, labels):
        """使用 Albumentations 去做数据增强"""
        eps = 1e-6
        bboxes = []
        class_labels = []

        if labels.size > 0:
            # 数值清洗：修复因浮点误差出现的微小负值，并过滤非法框
            for row in labels:
                cls, x, y, w, h = row.tolist()
                x = float(np.clip(x, 0.0, 1.0))
                y = float(np.clip(y, 0.0, 1.0))
                w = float(np.clip(w, 0.0, 1.0))
                h = float(np.clip(h, 0.0, 1.0))

                # YOLO bbox 转成边界后保证仍在 [0,1] 内
                x1 = x - w / 2.0
                y1 = y - h / 2.0
                x2 = x + w / 2.0
                y2 = y + h / 2.0
                x1 = float(np.clip(x1, 0.0, 1.0))
                y1 = float(np.clip(y1, 0.0, 1.0))
                x2 = float(np.clip(x2, 0.0, 1.0))
                y2 = float(np.clip(y2, 0.0, 1.0))

                w2 = x2 - x1
                h2 = y2 - y1
                if w2 <= eps or h2 <= eps:
                    continue

                x2c = (x1 + x2) / 2.0
                y2c = (y1 + y2) / 2.0
                bboxes.append([x2c, y2c, w2, h2])
                class_labels.append(int(cls))

        try:
            augmented = self.albu_aug(image=image, bboxes=bboxes, class_labels=class_labels)
        except Exception:
            # 增强库偶发数值边界异常时退回基础增强，避免训练中断
            return self.basic_augment(image, labels)
        image_aug = augmented["image"]
        bboxes_aug = augmented["bboxes"]
        class_aug = augmented["class_labels"]

        if len(bboxes_aug) == 0:
            labels_aug = np.zeros((0, 5), dtype=np.float32)
        else:
            labels_aug = np.zeros((len(bboxes_aug), 5), dtype=np.float32)
            labels_aug[:, 0] = np.array(class_aug, dtype=np.float32)
            labels_aug[:, 1:5] = np.array(bboxes_aug, dtype=np.float32)
            labels_aug[:, 1:5] = np.clip(labels_aug[:, 1:5], 0.0, 1.0)

            # 再次过滤极小框，避免后续流程出现数值边界问题
            keep = (labels_aug[:, 3] > eps) & (labels_aug[:, 4] > eps)
            labels_aug = labels_aug[keep]

        return image_aug, labels_aug

    # 色彩空间变换
    @staticmethod
    def random_hsv(image, hgain=0.015, sgain=0.7, vgain=0.4):
        r = np.random.uniform(-1, 1, 3) * np.array([hgain, sgain, vgain]) + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        dtype = image.dtype

        x = np.arange(0, 256, dtype=np.float32)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 基础数据增强
    @staticmethod
    def basic_augment(image, labels, p_hflip=0.5, p_vflip=0.1):
        """
        进行基础的数据增强
        随机翻转和色彩空间变换
        """
        image = YOLO_Dataset.random_hsv(image)

        if labels.size > 0 and random.random() < p_hflip:
            image = cv2.flip(image, 1)
            labels[:, 1] = 1.0 - labels[:, 1]

        if labels.size > 0 and random.random() < p_vflip:
            image = cv2.flip(image, 0)
            labels[:, 2] = 1.0 - labels[:, 2]

        return image, labels
    
    @staticmethod
    def collate_fn(batch):
        """
        输入: batch 是一个列表，包含多个 __getitem__ 的返回值
            [{'image': tensor, 'target': {'cls': tensor, 'bboxes': tensor}}, ...]
        
        输出: 
            {
                'images': tensor,          # (B, 3, H, W)
                'targets': {
                    'batch_id': tensor,    # (all_boxes, 1) 表示每一个标签属于哪一张图
                    'cls': tensor,         # (all_boxes, 1)
                    'bboxes': tensor       # (all_boxes, 4)
                },
                'other_targets': {
                    'batch_id': tensor,
                    'cls': tensor,
                    'bboxes': tensor
                }
            }
        """
        images = []
        all_cls = []
        all_bboxes = []
        all_batch_idx = []
        other_all_cls = []
        other_all_bboxes = []
        other_all_batch_idx = []
        
        for batch_idx, sample in enumerate(batch):
            images.append(sample['image'])
            
            target = sample['target']
            num_objs = target['cls'].shape[0]

            if num_objs > 0:
                all_cls.append(target['cls'])
                all_bboxes.append(target['bboxes'])
                # 创建当前图片的 batch 索引列，形状 (N, 1)，值全为 batch_idx
                all_batch_idx.append(torch.full((num_objs, 1), batch_idx, dtype=torch.long))

            other_target = sample.get('other_target')
            if other_target is not None:
                num_other_objs = other_target['cls'].shape[0]
                if num_other_objs > 0:
                    other_all_cls.append(other_target['cls'])
                    other_all_bboxes.append(other_target['bboxes'])
                    other_all_batch_idx.append(torch.full((num_other_objs, 1), batch_idx, dtype=torch.long))
        
        # 堆叠图像
        images_batch = torch.stack(images, dim=0)
        
        # 拼接标签
        if len(all_cls) == 0:
            # 如果整个 batch 都没有目标
            targets_dict = YOLO_Dataset._empty_targets_dict()
        else:
            targets_dict = {
                'batch_id': torch.cat(all_batch_idx, dim=0),   # (all_boxes, 1)
                'cls': torch.cat(all_cls, dim=0),           # (all_boxes, 1)
                'bboxes': torch.cat(all_bboxes, dim=0)      # (all_boxesth, 4)
            }

        if len(other_all_cls) == 0:
            other_targets_dict = YOLO_Dataset._empty_targets_dict()
        else:
            other_targets_dict = {
                'batch_id': torch.cat(other_all_batch_idx, dim=0),
                'cls': torch.cat(other_all_cls, dim=0),
                'bboxes': torch.cat(other_all_bboxes, dim=0)
            }
        
        return {
            'images': images_batch,
            'targets': targets_dict,
            'other_targets': other_targets_dict
        }
    
    #图像与标签预处理函数
    @staticmethod
    def letterbox(image, labels, new_shape=640, color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        """
        将图像 Resize 并保持宽高比，剩余部分用 color 填充。
        同时调整 labels 中的坐标以匹配新的图像。
        
        参数:
            image: numpy array (H, W, C)
            labels: numpy array (N, 5) -> [class, x, y, w, h] (归一化坐标)
            new_shape: 目标尺寸 (int 或 tuple)
            color: 填充颜色 (RGB)
            scaleup: 是否允许放大图像 (如果原图小于 new_shape)
        """
        # 当前形状
        shape = image.shape[:2]  # [h, w]
        
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        # 计算缩放比例 (scale ratio)
        # 目标是让图像适应 new_shape，取宽和高缩放比例中较小的那个，以保证能放得下
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        if not scaleup:  # 如果只允许缩小，不允许放大
            r = min(r, 1.0)
        
        # 计算缩放后的新尺寸
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        
        # 计算需要填充的宽高 (padding)
        dw = new_shape[1] - new_unpad[0]  # 宽度需要补多少
        dh = new_shape[0] - new_unpad[1]  # 高度需要补多少
        
        if auto:  # 最小矩形填充 (可选，通常正方形固定尺寸不需要)
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # 确保能被32整除
        
        # 填充平分到两边 (居中填充)
        dw /= 2
        dh /= 2
        
        # 如果原图已经是正方形且尺寸匹配，直接返回
        if shape[::-1] == new_unpad and dw == 0 and dh == 0:
            return image, labels

        # 执行 Resize
        # INTER_AREA 适合缩小，INTER_LINEAR 适合放大
        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # 创建画布并填充 (Top, Bottom, Left, Right)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        # 调整 Labels 坐标
        if labels.size > 0:
            # labels 格式: [class, x, y, w, h]
            # 缩放影响了所有坐标 (x, y, w, h) 都要乘以缩放率 r
            # 平移只影响中心点 (x, y)，不影响宽高 (w, h)
            
            # 注意：dw 和 dh 是像素值，需要转换为归一化值 (除以新图像的尺寸 new_shape)
            # 新的归一化偏移量 = 填充像素 / 新图像总尺寸
            pad_w_norm = left / new_shape[1]
            pad_h_norm = top / new_shape[0]
            
            # 缩放后的归一化系数 = 原图尺寸 * r / 新图尺寸          
            scale_x = new_unpad[0] / new_shape[1]
            scale_y = new_unpad[1] / new_shape[0]
            
            labels[:, 1] = labels[:, 1] * scale_x + pad_w_norm  # x
            labels[:, 2] = labels[:, 2] * scale_y + pad_h_norm  # y
            labels[:, 3] = labels[:, 3] * scale_x               # w
            labels[:, 4] = labels[:, 4] * scale_y               # h
            
            # 防止因浮点数误差导致坐标超出 [0, 1]
            labels[:, 1:] = np.clip(labels[:, 1:], 0.0, 1.0)

        return image, labels


# 交并比计算
def detect_compute_iou(box, boxes):
    # box: [x1, y1, x2, y2, score, class]
    # boxes: N x 6
    xx1 = torch.maximum(box[0], boxes[:, 0])
    yy1 = torch.maximum(box[1], boxes[:, 1])
    xx2 = torch.minimum(box[2], boxes[:, 2])
    yy2 = torch.minimum(box[3], boxes[:, 3])

    w = torch.clamp(xx2 - xx1, min=0.0)
    h = torch.clamp(yy2 - yy1, min=0.0)
    inter = w * h

    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = area1 + area2 - inter + 1e-6  # 防止除零
    iou = inter / union
    return iou


def box_iou(box1, box2):
    """
    计算两组框的 IoU
    box1: [N, 4]  -> x1, y1, x2, y2
    box2: [M, 4]  -> x1, y1, x2, y2
    return: [N, M]
    """
    # 交集左上角
    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])

    # 交集右下角
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])

    # 交集宽高
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)

    # 交集面积
    inter_area = inter_w * inter_h

    # 各自面积
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)

    # 并集面积
    union = area1[:, None] + area2 - inter_area + 1e-16

    return inter_area / union


# ap计算
def compute_ap(recall, precision):
    """
    根据 PR 曲线计算 AP
    recall: [N]
    precision: [N]
    """
    # 两端补点
    mrec = torch.cat([torch.tensor([0.0], device=recall.device), recall, torch.tensor([1.0], device=recall.device)])
    mpre = torch.cat([torch.tensor([0.0], device=precision.device), precision, torch.tensor([0.0], device=precision.device)])

    # precision envelope
    for i in range(mpre.shape[0] - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

    # 找 recall 变化点
    idx = torch.where(mrec[1:] != mrec[:-1])[0]

    # 积分
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap.item()


# 非极大值抑制
def detect_nms(
    results: torch.Tensor,
    conf_threshold=0.001,
    iou_threshold=0.7,
    pre_nms_topk=1000,
    max_det=300,
    device=torch.device('cpu'),
):
    output = []
    results_new = results.contiguous().permute(0, 2, 1)  # [B, A, 4+nc]

    for b in range(results_new.shape[0]):
        pred = results_new[b].to(device)

        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:]
        max_scores, max_indices = torch.max(class_scores, dim=1)

        # 先按置信度过滤
        mask = max_scores > conf_threshold
        boxes_xywh = boxes_xywh[mask]
        max_scores = max_scores[mask]
        max_indices = max_indices[mask]

        if boxes_xywh.size(0) == 0:
            output.append(torch.zeros((0, 6), device=device))
            continue

        # 先做 top-k 预筛，降低后续 NMS 的候选量
        if pre_nms_topk is not None and boxes_xywh.size(0) > pre_nms_topk:
            topk_idx = torch.topk(max_scores, k=pre_nms_topk, largest=True).indices
            boxes_xywh = boxes_xywh[topk_idx]
            max_scores = max_scores[topk_idx]
            max_indices = max_indices[topk_idx]

        # xywh -> xyxy
        cx = boxes_xywh[:, 0]
        cy = boxes_xywh[:, 1]
        w = boxes_xywh[:, 2]
        h = boxes_xywh[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        dets = torch.stack((x1, y1, x2, y2, max_scores, max_indices.float()), dim=1)

        if torchvision_nms is not None:
            keep_all = []
            for cls in torch.unique(max_indices):
                cls_mask = max_indices == cls
                cls_idx = torch.where(cls_mask)[0]
                keep = torchvision_nms(dets[cls_idx, :4], dets[cls_idx, 4], iou_threshold)
                keep_all.append(cls_idx[keep])

            if keep_all:
                keep_all = torch.cat(keep_all, dim=0)
                keep_all = keep_all[torch.argsort(dets[keep_all, 4], descending=True)]
                if max_det is not None:
                    keep_all = keep_all[:max_det]
                output.append(dets[keep_all])
            else:
                output.append(torch.zeros((0, 6), device=device))
        else:
            final_dets = []
            for cls in torch.unique(max_indices):
                cls_mask = max_indices == cls
                cls_dets = dets[cls_mask]
                scores = cls_dets[:, 4]
                _, order = torch.sort(scores, descending=True)

                keep = []
                while order.numel() > 0:
                    idx = order[0]
                    keep.append(idx.item())
                    if max_det is not None and len(keep) >= max_det:
                        break
                    if order.numel() == 1:
                        break
                    ious = detect_compute_iou(cls_dets[idx], cls_dets[order[1:]])
                    order = order[1:][ious <= iou_threshold]

                if keep:
                    final_dets.append(cls_dets[keep])

            if final_dets:
                final_dets = torch.cat(final_dets, dim=0)
                final_dets = final_dets[torch.argsort(final_dets[:, 4], descending=True)]
                if max_det is not None:
                    final_dets = final_dets[:max_det]
                output.append(final_dets)
            else:
                output.append(torch.zeros((0, 6), device=device))

    # 兼容单张推理流程，batch=1时直接返回tensor
    return output[0] if len(output) == 1 else output


def collect_map_records(
    result,
    target,
    num_classes,
    img_size=640,
    image_offset=0,
    conf_threshold=0.001,
    nms_iou_threshold=0.7,
    pre_nms_topk=1000,
    max_det=300,
):
    """Collect prediction and ground-truth records for dataset-level AP calculation."""
    if not isinstance(result, torch.Tensor):
        raise TypeError("计算Map需要输入的数据格式为：[B, 4+nc, A].")

    device = result.device
    nms_result = detect_nms(
        result,
        conf_threshold=conf_threshold,
        iou_threshold=nms_iou_threshold,
        pre_nms_topk=pre_nms_topk,
        max_det=max_det,
        device=device,
    )
    if isinstance(nms_result, torch.Tensor):
        nms_result = [nms_result]
    batch_size = len(nms_result)

    gt_boxes_all = target["bboxes"].to(device)
    if gt_boxes_all.numel():
        gt_xy = gt_boxes_all[:, :2]
        gt_wh = gt_boxes_all[:, 2:4] / 2.0
        gt_boxes_all = torch.cat((gt_xy - gt_wh, gt_xy + gt_wh), dim=1).clamp_(0.0, 1.0)
    else:
        gt_boxes_all = gt_boxes_all.view(0, 4)

    batch_ids = target["batch_id"].view(-1).long().to(device)
    gt_cls_all = target["cls"].view(-1).long().to(device)

    gt_records = []
    for i in range(gt_cls_all.numel()):
        cls_id = int(gt_cls_all[i].item())
        if 0 <= cls_id < num_classes:
            gt_records.append(
                {
                    "img_id": int(batch_ids[i].item()) + int(image_offset),
                    "cls": cls_id,
                    "box": gt_boxes_all[i].detach().cpu(),
                }
            )

    pred_abs = False
    for det in nms_result:
        if det.numel() and det[:, :4].amax() > 1.5:
            pred_abs = True
            break

    pred_records = []
    for b in range(batch_size):
        pred_b = nms_result[b]
        if pred_b.numel() == 0:
            continue

        if pred_abs:
            pred_b = pred_b.clone()
            pred_b[:, :4] = (pred_b[:, :4] / float(img_size)).clamp_(0.0, 1.0)

        for p in pred_b:
            cls_id = int(p[5].item())
            if 0 <= cls_id < num_classes:
                pred_records.append(
                    {
                        "img_id": b + int(image_offset),
                        "cls": cls_id,
                        "conf": float(p[4].item()),
                        "box": p[:4].detach().cpu(),
                    }
                )

    return pred_records, gt_records


def compute_map_from_records(pred_records, gt_records, num_classes, iou_thresholds=None):
    """Calculate AP/mAP from records collected across the whole validation set."""
    if iou_thresholds is None:
        iou_thresholds = [0.5]
    iou_thresholds = [float(t) for t in iou_thresholds]
    if len(iou_thresholds) == 0:
        raise ValueError("iou_thresholds 不能为空")

    device = torch.device("cpu")
    gt_by_class_image = {}
    gt_count_by_class = [0 for _ in range(num_classes)]
    for gt in gt_records:
        cls_id = int(gt["cls"])
        if 0 <= cls_id < num_classes:
            key = (cls_id, int(gt["img_id"]))
            gt_by_class_image.setdefault(key, []).append(gt["box"].to(device))
            gt_count_by_class[cls_id] += 1

    preds_by_class = [[] for _ in range(num_classes)]
    for pred in pred_records:
        cls_id = int(pred["cls"])
        if 0 <= cls_id < num_classes:
            preds_by_class[cls_id].append(pred)

    valid_classes = [i for i, count in enumerate(gt_count_by_class) if count > 0]
    ap_per_class_by_iou = []
    map_per_iou = []

    for iou_thr in iou_thresholds:
        ap_per_class = []

        # 依次计算每一个类别
        for cls_id in range(num_classes):
            gt_count = gt_count_by_class[cls_id]
            pred_cls_records = preds_by_class[cls_id]

            if gt_count == 0 or len(pred_cls_records) == 0:
                ap_per_class.append(0.0)
                continue

            pred_cls_records = sorted(pred_cls_records, key=lambda x: x["conf"], reverse=True)
            tp = torch.zeros(len(pred_cls_records), device=device)
            fp = torch.zeros(len(pred_cls_records), device=device)
            matched_gts = {}

            for i, pred in enumerate(pred_cls_records):
                img_id = int(pred["img_id"])
                gt_boxes_list = gt_by_class_image.get((cls_id, img_id), [])

                if len(gt_boxes_list) == 0:
                    fp[i] = 1
                    continue

                gt_boxes = torch.stack(gt_boxes_list, dim=0).to(device)
                pred_box = pred["box"].to(device).unsqueeze(0)
                ious = box_iou(pred_box, gt_boxes).squeeze(0)
                max_iou, max_idx = torch.max(ious, dim=0)

                matched_key = (cls_id, img_id)
                if matched_key not in matched_gts:
                    matched_gts[matched_key] = set()

                if max_iou >= iou_thr and max_idx.item() not in matched_gts[matched_key]:
                    tp[i] = 1
                    matched_gts[matched_key].add(max_idx.item())
                else:
                    fp[i] = 1

            tp_cum = torch.cumsum(tp, dim=0)
            fp_cum = torch.cumsum(fp, dim=0)
            recall = tp_cum / (gt_count + 1e-16)
            precision = tp_cum / (tp_cum + fp_cum + 1e-16)
            ap_per_class.append(compute_ap(recall, precision))

        if valid_classes:
            map_value = sum(ap_per_class[i] for i in valid_classes) / len(valid_classes)
        else:
            map_value = 0.0

        ap_per_class_by_iou.append(ap_per_class)
        map_per_iou.append(map_value)

    ap50_95_per_class = []
    for c in range(num_classes):
        ap_c = [ap_per_class_by_iou[k][c] for k in range(len(iou_thresholds))]
        ap50_95_per_class.append(sum(ap_c) / len(ap_c))

    return {
        "iou_thresholds": iou_thresholds,
        "valid_classes": valid_classes,
        "gt_count_per_class": gt_count_by_class,
        "ap_per_class_by_iou": ap_per_class_by_iou,
        "map_per_iou": map_per_iou,
        "ap_per_class": ap_per_class_by_iou[0],
        "map": map_per_iou[0],
        "ap50_per_class": ap_per_class_by_iou[0],
        "map50": map_per_iou[0],
        "ap50_95_per_class": ap50_95_per_class,
        "map50_95": sum(map_per_iou) / len(map_per_iou),
    }

def Map(result, target, num_classes, img_size=640, iou_thresholds=None):
    """
    通用 mAP 评估函数。
    iou_thresholds:
        - None: 默认 [0.5]
        - list/tuple: 例如 [0.5, 0.55, ..., 0.95]
    """
    pred_records, gt_records = collect_map_records(result, target, num_classes, img_size=img_size)
    return compute_map_from_records(pred_records, gt_records, num_classes, iou_thresholds=iou_thresholds)


def Map_50(result, target, num_classes, img_size=640):
    """兼容旧接口，只返回 IoU=0.5 的 AP/mAP。"""
    metrics = Map(
        result=result,
        target=target,
        num_classes=num_classes,
        img_size=img_size,
        iou_thresholds=[0.5],
    )
    return {
        'ap_per_class': metrics['ap50_per_class'],
        'map': metrics['map50'],
    }


def Map_50_95(result, target, num_classes, img_size=640):
    """COCO风格 mAP@[0.5:0.95]，步长 0.05。"""
    iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50,0.55,...,0.95
    metrics = Map(
        result=result,
        target=target,
        num_classes=num_classes,
        img_size=img_size,
        iou_thresholds=iou_thresholds,
    )
    return {
        'iou_thresholds': metrics['iou_thresholds'],
        'ap50_per_class': metrics['ap50_per_class'],
        'map50': metrics['map50'],
        'ap50_95_per_class': metrics['ap50_95_per_class'],
        'map50_95': metrics['map50_95'],
    }
import os
import glob
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms as T

cv2.setNumThreads(0)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

try:
    from torchvision.ops import nms as torchvision_nms
except Exception:
    torchvision_nms = None
try:
    import albumentations as A
except ImportError:
    A = None


# 可监督数据加


# 数据集加载
class YOLO_Dataset(Dataset):
    def __init__(self, data_path, mode='train', image_size=640, label_filter_classes=None, use_albumentations=False):
        super().__init__()
        self.mode = mode
        self.image_size = image_size
        self.use_aug = mode == 'train'
        self.use_albumentations = use_albumentations
        self.label_filter_classes = None
        if label_filter_classes is not None:
            self.label_filter_classes = {int(cls_id) for cls_id in label_filter_classes}
        
        # 路径设置
        img_dir = os.path.join(data_path, 'images', mode)
        label_dir = os.path.join(data_path, 'labels', mode)
        
        # 检查路径是否存在
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"图像路径不存在: {img_dir}")
        if not os.path.exists(label_dir):
            # 如果是测试模式且没有标签，可能允许，但训练模式必须有
            if mode == 'train':
                raise FileNotFoundError(f"标签路径不存在: {label_dir}")
            else:
                print(f"警告: 标签路径不存在: {label_dir}，将返回空标签")
                label_dir = None 
        
        # 获取所有图像路径
        self.image_paths = sorted(glob.glob(os.path.join(img_dir, '*.*')))
        # 过滤支持的图片格式
        self.image_paths = [p for p in self.image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        if len(self.image_paths) == 0:
            raise ValueError(f"没有图像： {img_dir}")

        #暂时不对图像进行复杂变换，仅做Tensor转换
        self.transform = T.Compose([
            T.ToTensor(), 
        ])

        # 优先使用 albumentations 做图像与标签同步增强
        self.albu_aug = None
        if self.use_aug and self.use_albumentations and A is not None:
            self.albu_aug = A.Compose(
                [
                    A.HorizontalFlip(p=0.5), 
                    A.VerticalFlip(p=0.1),
                    A.Affine(
                        scale=(0.9, 1.1),
                        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                        rotate=(-15, 15),
                        shear=(-5, 5),
                        p=0.6,
                    ), # 仿射变换
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
                    A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=30, val_shift_limit=20, p=0.4), # 颜色改变
                ],
                bbox_params=A.BboxParams(
                    format="yolo",
                    label_fields=["class_labels"],
                    min_visibility=0.2,
                ),# 对YOLO数据集中的标签，按照图像变化的方法自适应变换
            )

    def __len__(self):
        return len(self.image_paths)

    def _load_label_raw(self, label_path):
        labels = []
        if label_path and os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = list(map(float, line.split()))
                    if len(parts) >= 5:
                        # YOLO 格式: class, x_center, y_center, w, h
                        labels.append(parts[:5]) 
        # 返回 numpy 数组，如果没有标签则返回 (0, 5)
        return np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)

    def _split_labels(self, labels):
        if self.label_filter_classes is None or labels.shape[0] == 0:
            return labels, np.zeros((0, 5), dtype=np.float32)

        keep_mask = np.array(
            [int(cls_id) in self.label_filter_classes for cls_id in labels[:, 0]],
            dtype=bool,
        )
        return labels[keep_mask], labels[~keep_mask]

    def _load_label(self, label_path):
        labels, _ = self._split_labels(self._load_label_raw(label_path))
        return labels

    def __getitem__(self, idx):
        # 获取路径
        img_path = self.image_paths[idx]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 构造对应的标签路径
        try:
            label_path = os.path.join(
                os.path.dirname(img_path).replace('images', 'labels'), 
                img_name + '.txt'
            )
        except Exception:
            label_path = "" # 防止路径替换失败

        # 读取图像 (BGR)
        image = cv2.imread(img_path)
        if image is None:
            # 如果图片损坏，随机返回另一张，防止训练中断
            print(f"图像加载失败： {img_path}")
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # 读取全量标签，完成图像同步变换后再拆分为训练标签和其他标签
        labels_all = self._load_label_raw(label_path)
        image, labels_all = self.letterbox(image, labels_all, new_shape=self.image_size, color=(114, 114, 114))
        if self.use_aug:
            if self.albu_aug is not None:
                image, labels_all = self.apply_albumentations(image, labels_all)
            else:
                image, labels_all = self.basic_augment(image, labels_all)

        labels_np, other_labels_np = self._split_labels(labels_all)

        # 图像预处理
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        resized_image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        image_tensor = self.transform(resized_image)

        cls_tensor, bboxes_tensor = self._labels_to_tensors(labels_np)
        other_cls_tensor, other_bboxes_tensor = self._labels_to_tensors(other_labels_np)
          
        # 构建返回字典
        # __getitem__ 只返回当前图的信息，batch 索引由 collate_fn 添加
        
        target = {
            'cls': cls_tensor,       # (N, 1)
            'bboxes': bboxes_tensor  # (N, 4)
        }
        other_target = {
            'cls': other_cls_tensor,
            'bboxes': other_bboxes_tensor
        }
        
        return {
            'image': image_tensor,   # Tensor图像
            'target': target,        # 指定类别标签信息
            'other_target': other_target  # 非指定类别标签信息
        }

    @staticmethod
    def _labels_to_tensors(labels_np):
        if labels_np.shape[0] == 0:
            cls_tensor = torch.zeros((0,), dtype=torch.long)
            bboxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        else:
            # labels_np 格式: [class, x, y, w, h]
            cls_tensor = torch.from_numpy(labels_np[:, 0]).long()
            bboxes_tensor = torch.from_numpy(labels_np[:, 1:5])
        return cls_tensor, bboxes_tensor

    @staticmethod
    def _empty_targets_dict():
        return {
            'batch_id': torch.zeros((0, 1), dtype=torch.long),
            'cls': torch.zeros((0,), dtype=torch.long),
            'bboxes': torch.zeros((0, 4), dtype=torch.float32)
        }

    def apply_albumentations(self, image, labels):
        """使用 Albumentations 去做数据增强"""
        eps = 1e-6
        bboxes = []
        class_labels = []

        if labels.size > 0:
            # 数值清洗：修复因浮点误差出现的微小负值，并过滤非法框
            for row in labels:
                cls, x, y, w, h = row.tolist()
                x = float(np.clip(x, 0.0, 1.0))
                y = float(np.clip(y, 0.0, 1.0))
                w = float(np.clip(w, 0.0, 1.0))
                h = float(np.clip(h, 0.0, 1.0))

                # YOLO bbox 转成边界后保证仍在 [0,1] 内
                x1 = x - w / 2.0
                y1 = y - h / 2.0
                x2 = x + w / 2.0
                y2 = y + h / 2.0
                x1 = float(np.clip(x1, 0.0, 1.0))
                y1 = float(np.clip(y1, 0.0, 1.0))
                x2 = float(np.clip(x2, 0.0, 1.0))
                y2 = float(np.clip(y2, 0.0, 1.0))

                w2 = x2 - x1
                h2 = y2 - y1
                if w2 <= eps or h2 <= eps:
                    continue

                x2c = (x1 + x2) / 2.0
                y2c = (y1 + y2) / 2.0
                bboxes.append([x2c, y2c, w2, h2])
                class_labels.append(int(cls))

        try:
            augmented = self.albu_aug(image=image, bboxes=bboxes, class_labels=class_labels)
        except Exception:
            # 增强库偶发数值边界异常时退回基础增强，避免训练中断
            return self.basic_augment(image, labels)
        image_aug = augmented["image"]
        bboxes_aug = augmented["bboxes"]
        class_aug = augmented["class_labels"]

        if len(bboxes_aug) == 0:
            labels_aug = np.zeros((0, 5), dtype=np.float32)
        else:
            labels_aug = np.zeros((len(bboxes_aug), 5), dtype=np.float32)
            labels_aug[:, 0] = np.array(class_aug, dtype=np.float32)
            labels_aug[:, 1:5] = np.array(bboxes_aug, dtype=np.float32)
            labels_aug[:, 1:5] = np.clip(labels_aug[:, 1:5], 0.0, 1.0)

            # 再次过滤极小框，避免后续流程出现数值边界问题
            keep = (labels_aug[:, 3] > eps) & (labels_aug[:, 4] > eps)
            labels_aug = labels_aug[keep]

        return image_aug, labels_aug

    # 色彩空间变换
    @staticmethod
    def random_hsv(image, hgain=0.015, sgain=0.7, vgain=0.4):
        r = np.random.uniform(-1, 1, 3) * np.array([hgain, sgain, vgain]) + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        dtype = image.dtype

        x = np.arange(0, 256, dtype=np.float32)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 基础数据增强
    @staticmethod
    def basic_augment(image, labels, p_hflip=0.5, p_vflip=0.1):
        """
        进行基础的数据增强
        随机翻转和色彩空间变换
        """
        image = YOLO_Dataset.random_hsv(image)

        if labels.size > 0 and random.random() < p_hflip:
            image = cv2.flip(image, 1)
            labels[:, 1] = 1.0 - labels[:, 1]

        if labels.size > 0 and random.random() < p_vflip:
            image = cv2.flip(image, 0)
            labels[:, 2] = 1.0 - labels[:, 2]

        return image, labels
    
    @staticmethod
    def collate_fn(batch):
        """
        输入: batch 是一个列表，包含多个 __getitem__ 的返回值
            [{'image': tensor, 'target': {'cls': tensor, 'bboxes': tensor}}, ...]
        
        输出: 
            {
                'images': tensor,          # (B, 3, H, W)
                'targets': {
                    'batch_id': tensor,    # (all_boxes, 1) 表示每一个标签属于哪一张图
                    'cls': tensor,         # (all_boxes, 1)
                    'bboxes': tensor       # (all_boxes, 4)
                },
                'other_targets': {
                    'batch_id': tensor,
                    'cls': tensor,
                    'bboxes': tensor
                }
            }
        """
        images = []
        all_cls = []
        all_bboxes = []
        all_batch_idx = []
        other_all_cls = []
        other_all_bboxes = []
        other_all_batch_idx = []
        
        for batch_idx, sample in enumerate(batch):
            images.append(sample['image'])
            
            target = sample['target']
            num_objs = target['cls'].shape[0]

            if num_objs > 0:
                all_cls.append(target['cls'])
                all_bboxes.append(target['bboxes'])
                # 创建当前图片的 batch 索引列，形状 (N, 1)，值全为 batch_idx
                all_batch_idx.append(torch.full((num_objs, 1), batch_idx, dtype=torch.long))

            other_target = sample.get('other_target')
            if other_target is not None:
                num_other_objs = other_target['cls'].shape[0]
                if num_other_objs > 0:
                    other_all_cls.append(other_target['cls'])
                    other_all_bboxes.append(other_target['bboxes'])
                    other_all_batch_idx.append(torch.full((num_other_objs, 1), batch_idx, dtype=torch.long))
        
        # 堆叠图像
        images_batch = torch.stack(images, dim=0)
        
        # 拼接标签
        if len(all_cls) == 0:
            # 如果整个 batch 都没有目标
            targets_dict = YOLO_Dataset._empty_targets_dict()
        else:
            targets_dict = {
                'batch_id': torch.cat(all_batch_idx, dim=0),   # (all_boxes, 1)
                'cls': torch.cat(all_cls, dim=0),           # (all_boxes, 1)
                'bboxes': torch.cat(all_bboxes, dim=0)      # (all_boxesth, 4)
            }

        if len(other_all_cls) == 0:
            other_targets_dict = YOLO_Dataset._empty_targets_dict()
        else:
            other_targets_dict = {
                'batch_id': torch.cat(other_all_batch_idx, dim=0),
                'cls': torch.cat(other_all_cls, dim=0),
                'bboxes': torch.cat(other_all_bboxes, dim=0)
            }
        
        return {
            'images': images_batch,
            'targets': targets_dict,
            'other_targets': other_targets_dict
        }
    
    #图像与标签预处理函数
    @staticmethod
    def letterbox(image, labels, new_shape=640, color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        """
        将图像 Resize 并保持宽高比，剩余部分用 color 填充。
        同时调整 labels 中的坐标以匹配新的图像。
        
        参数:
            image: numpy array (H, W, C)
            labels: numpy array (N, 5) -> [class, x, y, w, h] (归一化坐标)
            new_shape: 目标尺寸 (int 或 tuple)
            color: 填充颜色 (RGB)
            scaleup: 是否允许放大图像 (如果原图小于 new_shape)
        """
        # 当前形状
        shape = image.shape[:2]  # [h, w]
        
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        # 计算缩放比例 (scale ratio)
        # 目标是让图像适应 new_shape，取宽和高缩放比例中较小的那个，以保证能放得下
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        if not scaleup:  # 如果只允许缩小，不允许放大
            r = min(r, 1.0)
        
        # 计算缩放后的新尺寸
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        
        # 计算需要填充的宽高 (padding)
        dw = new_shape[1] - new_unpad[0]  # 宽度需要补多少
        dh = new_shape[0] - new_unpad[1]  # 高度需要补多少
        
        if auto:  # 最小矩形填充 (可选，通常正方形固定尺寸不需要)
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # 确保能被32整除
        
        # 填充平分到两边 (居中填充)
        dw /= 2
        dh /= 2
        
        # 如果原图已经是正方形且尺寸匹配，直接返回
        if shape[::-1] == new_unpad and dw == 0 and dh == 0:
            return image, labels

        # 执行 Resize
        # INTER_AREA 适合缩小，INTER_LINEAR 适合放大
        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # 创建画布并填充 (Top, Bottom, Left, Right)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        # 调整 Labels 坐标
        if labels.size > 0:
            # labels 格式: [class, x, y, w, h]
            # 缩放影响了所有坐标 (x, y, w, h) 都要乘以缩放率 r
            # 平移只影响中心点 (x, y)，不影响宽高 (w, h)
            
            # 注意：dw 和 dh 是像素值，需要转换为归一化值 (除以新图像的尺寸 new_shape)
            # 新的归一化偏移量 = 填充像素 / 新图像总尺寸
            pad_w_norm = left / new_shape[1]
            pad_h_norm = top / new_shape[0]
            
            # 缩放后的归一化系数 = 原图尺寸 * r / 新图尺寸          
            scale_x = new_unpad[0] / new_shape[1]
            scale_y = new_unpad[1] / new_shape[0]
            
            labels[:, 1] = labels[:, 1] * scale_x + pad_w_norm  # x
            labels[:, 2] = labels[:, 2] * scale_y + pad_h_norm  # y
            labels[:, 3] = labels[:, 3] * scale_x               # w
            labels[:, 4] = labels[:, 4] * scale_y               # h
            
            # 防止因浮点数误差导致坐标超出 [0, 1]
            labels[:, 1:] = np.clip(labels[:, 1:], 0.0, 1.0)

        return image, labels


# 交并比计算
def detect_compute_iou(box, boxes):
    # box: [x1, y1, x2, y2, score, class]
    # boxes: N x 6
    xx1 = torch.maximum(box[0], boxes[:, 0])
    yy1 = torch.maximum(box[1], boxes[:, 1])
    xx2 = torch.minimum(box[2], boxes[:, 2])
    yy2 = torch.minimum(box[3], boxes[:, 3])

    w = torch.clamp(xx2 - xx1, min=0.0)
    h = torch.clamp(yy2 - yy1, min=0.0)
    inter = w * h

    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = area1 + area2 - inter + 1e-6  # 防止除零
    iou = inter / union
    return iou


def box_iou(box1, box2):
    """
    计算两组框的 IoU
    box1: [N, 4]  -> x1, y1, x2, y2
    box2: [M, 4]  -> x1, y1, x2, y2
    return: [N, M]
    """
    # 交集左上角
    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])

    # 交集右下角
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])

    # 交集宽高
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)

    # 交集面积
    inter_area = inter_w * inter_h

    # 各自面积
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)

    # 并集面积
    union = area1[:, None] + area2 - inter_area + 1e-16

    return inter_area / union


# ap计算
def compute_ap(recall, precision):
    """
    根据 PR 曲线计算 AP
    recall: [N]
    precision: [N]
    """
    # 两端补点
    mrec = torch.cat([torch.tensor([0.0], device=recall.device), recall, torch.tensor([1.0], device=recall.device)])
    mpre = torch.cat([torch.tensor([0.0], device=precision.device), precision, torch.tensor([0.0], device=precision.device)])

    # precision envelope
    for i in range(mpre.shape[0] - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

    # 找 recall 变化点
    idx = torch.where(mrec[1:] != mrec[:-1])[0]

    # 积分
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap.item()


# 非极大值抑制
def detect_nms(
    results: torch.Tensor,
    conf_threshold=0.001,
    iou_threshold=0.7,
    pre_nms_topk=1000,
    max_det=300,
    device=torch.device('cpu'),
):
    output = []
    results_new = results.contiguous().permute(0, 2, 1)  # [B, A, 4+nc]

    for b in range(results_new.shape[0]):
        pred = results_new[b].to(device)

        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:]
        max_scores, max_indices = torch.max(class_scores, dim=1)

        # 先按置信度过滤
        mask = max_scores > conf_threshold
        boxes_xywh = boxes_xywh[mask]
        max_scores = max_scores[mask]
        max_indices = max_indices[mask]

        if boxes_xywh.size(0) == 0:
            output.append(torch.zeros((0, 6), device=device))
            continue

        # 先做 top-k 预筛，降低后续 NMS 的候选量
        if pre_nms_topk is not None and boxes_xywh.size(0) > pre_nms_topk:
            topk_idx = torch.topk(max_scores, k=pre_nms_topk, largest=True).indices
            boxes_xywh = boxes_xywh[topk_idx]
            max_scores = max_scores[topk_idx]
            max_indices = max_indices[topk_idx]

        # xywh -> xyxy
        cx = boxes_xywh[:, 0]
        cy = boxes_xywh[:, 1]
        w = boxes_xywh[:, 2]
        h = boxes_xywh[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        dets = torch.stack((x1, y1, x2, y2, max_scores, max_indices.float()), dim=1)

        if torchvision_nms is not None:
            keep_all = []
            for cls in torch.unique(max_indices):
                cls_mask = max_indices == cls
                cls_idx = torch.where(cls_mask)[0]
                keep = torchvision_nms(dets[cls_idx, :4], dets[cls_idx, 4], iou_threshold)
                keep_all.append(cls_idx[keep])

            if keep_all:
                keep_all = torch.cat(keep_all, dim=0)
                keep_all = keep_all[torch.argsort(dets[keep_all, 4], descending=True)]
                if max_det is not None:
                    keep_all = keep_all[:max_det]
                output.append(dets[keep_all])
            else:
                output.append(torch.zeros((0, 6), device=device))
        else:
            final_dets = []
            for cls in torch.unique(max_indices):
                cls_mask = max_indices == cls
                cls_dets = dets[cls_mask]
                scores = cls_dets[:, 4]
                _, order = torch.sort(scores, descending=True)

                keep = []
                while order.numel() > 0:
                    idx = order[0]
                    keep.append(idx.item())
                    if max_det is not None and len(keep) >= max_det:
                        break
                    if order.numel() == 1:
                        break
                    ious = detect_compute_iou(cls_dets[idx], cls_dets[order[1:]])
                    order = order[1:][ious <= iou_threshold]

                if keep:
                    final_dets.append(cls_dets[keep])

            if final_dets:
                final_dets = torch.cat(final_dets, dim=0)
                final_dets = final_dets[torch.argsort(final_dets[:, 4], descending=True)]
                if max_det is not None:
                    final_dets = final_dets[:max_det]
                output.append(final_dets)
            else:
                output.append(torch.zeros((0, 6), device=device))

    # 兼容单张推理流程，batch=1时直接返回tensor
    return output[0] if len(output) == 1 else output


def collect_map_records(
    result,
    target,
    num_classes,
    img_size=640,
    image_offset=0,
    conf_threshold=0.001,
    nms_iou_threshold=0.7,
    pre_nms_topk=1000,
    max_det=300,
):
    """Collect prediction and ground-truth records for dataset-level AP calculation."""
    if not isinstance(result, torch.Tensor):
        result = result[0]

    device = result.device
    nms_result = detect_nms(
        result,
        conf_threshold=conf_threshold,
        iou_threshold=nms_iou_threshold,
        pre_nms_topk=pre_nms_topk,
        max_det=max_det,
        device=device,
    )
    if isinstance(nms_result, torch.Tensor):
        nms_result = [nms_result]
    batch_size = len(nms_result)

    gt_boxes_all = target["bboxes"].to(device)
    if gt_boxes_all.numel():
        gt_xy = gt_boxes_all[:, :2]
        gt_wh = gt_boxes_all[:, 2:4] / 2.0
        gt_boxes_all = torch.cat((gt_xy - gt_wh, gt_xy + gt_wh), dim=1).clamp_(0.0, 1.0)
    else:
        gt_boxes_all = gt_boxes_all.view(0, 4)

    batch_ids = target["batch_id"].view(-1).long().to(device)
    gt_cls_all = target["cls"].view(-1).long().to(device)

    gt_records = []
    for i in range(gt_cls_all.numel()):
        cls_id = int(gt_cls_all[i].item())
        if 0 <= cls_id < num_classes:
            gt_records.append(
                {
                    "img_id": int(batch_ids[i].item()) + int(image_offset),
                    "cls": cls_id,
                    "box": gt_boxes_all[i].detach().cpu(),
                }
            )

    pred_abs = False
    for det in nms_result:
        if det.numel() and det[:, :4].amax() > 1.5:
            pred_abs = True
            break

    pred_records = []
    for b in range(batch_size):
        pred_b = nms_result[b]
        if pred_b.numel() == 0:
            continue

        if pred_abs:
            pred_b = pred_b.clone()
            pred_b[:, :4] = (pred_b[:, :4] / float(img_size)).clamp_(0.0, 1.0)

        for p in pred_b:
            cls_id = int(p[5].item())
            if 0 <= cls_id < num_classes:
                pred_records.append(
                    {
                        "img_id": b + int(image_offset),
                        "cls": cls_id,
                        "conf": float(p[4].item()),
                        "box": p[:4].detach().cpu(),
                    }
                )

    return pred_records, gt_records


def compute_map_from_records(pred_records, gt_records, num_classes, iou_thresholds=None):
    """Calculate AP/mAP from records collected across the whole validation set."""
    if iou_thresholds is None:
        iou_thresholds = [0.5]
    iou_thresholds = [float(t) for t in iou_thresholds]
    if len(iou_thresholds) == 0:
        raise ValueError("iou_thresholds 不能为空")

    device = torch.device("cpu")
    gt_by_class_image = {}
    gt_count_by_class = [0 for _ in range(num_classes)]
    for gt in gt_records:
        cls_id = int(gt["cls"])
        if 0 <= cls_id < num_classes:
            key = (cls_id, int(gt["img_id"]))
            gt_by_class_image.setdefault(key, []).append(gt["box"].to(device))
            gt_count_by_class[cls_id] += 1

    preds_by_class = [[] for _ in range(num_classes)]
    for pred in pred_records:
        cls_id = int(pred["cls"])
        if 0 <= cls_id < num_classes:
            preds_by_class[cls_id].append(pred)

    valid_classes = [i for i, count in enumerate(gt_count_by_class) if count > 0]
    ap_per_class_by_iou = []
    map_per_iou = []

    for iou_thr in iou_thresholds:
        ap_per_class = []

        # 依次计算每一个类别
        for cls_id in range(num_classes):
            gt_count = gt_count_by_class[cls_id]
            pred_cls_records = preds_by_class[cls_id]

            if gt_count == 0 or len(pred_cls_records) == 0:
                ap_per_class.append(0.0)
                continue

            pred_cls_records = sorted(pred_cls_records, key=lambda x: x["conf"], reverse=True)
            tp = torch.zeros(len(pred_cls_records), device=device)
            fp = torch.zeros(len(pred_cls_records), device=device)
            matched_gts = {}

            for i, pred in enumerate(pred_cls_records):
                img_id = int(pred["img_id"])
                gt_boxes_list = gt_by_class_image.get((cls_id, img_id), [])

                if len(gt_boxes_list) == 0:
                    fp[i] = 1
                    continue

                gt_boxes = torch.stack(gt_boxes_list, dim=0).to(device)
                pred_box = pred["box"].to(device).unsqueeze(0)
                ious = box_iou(pred_box, gt_boxes).squeeze(0)
                max_iou, max_idx = torch.max(ious, dim=0)

                matched_key = (cls_id, img_id)
                if matched_key not in matched_gts:
                    matched_gts[matched_key] = set()

                if max_iou >= iou_thr and max_idx.item() not in matched_gts[matched_key]:
                    tp[i] = 1
                    matched_gts[matched_key].add(max_idx.item())
                else:
                    fp[i] = 1

            tp_cum = torch.cumsum(tp, dim=0)
            fp_cum = torch.cumsum(fp, dim=0)
            recall = tp_cum / (gt_count + 1e-16)
            precision = tp_cum / (tp_cum + fp_cum + 1e-16)
            ap_per_class.append(round(compute_ap(recall, precision),3))

        if valid_classes:
            map_value = sum(ap_per_class[i] for i in valid_classes) / len(valid_classes)
        else:
            map_value = 0.0

        ap_per_class_by_iou.append(ap_per_class)
        map_per_iou.append(round(map_value,3))

    ap50_95_per_class = []
    for c in range(num_classes):
        ap_c = [ap_per_class_by_iou[k][c] for k in range(len(iou_thresholds))]
        ap50_95_per_class.append(round(sum(ap_c) / len(ap_c),3))

    return {
        "iou_thresholds": iou_thresholds,
        "valid_classes": valid_classes,
        "gt_count_per_class": gt_count_by_class,
        "ap_per_class_by_iou": ap_per_class_by_iou,
        "map_per_iou": map_per_iou,
        "ap_per_class": ap_per_class_by_iou[0],
        "map": map_per_iou[0],
        "ap50_per_class": ap_per_class_by_iou[0],
        "map50": map_per_iou[0],
        "ap50_95_per_class": ap50_95_per_class,
        "map50_95": sum(map_per_iou) / len(map_per_iou),
    }

def Map(result, target, num_classes, img_size=640, iou_thresholds=None):
    """
    通用 mAP 评估函数。
    iou_thresholds:
        - None: 默认 [0.5]
        - list/tuple: 例如 [0.5, 0.55, ..., 0.95]
    """
    pred_records, gt_records = collect_map_records(result, target, num_classes, img_size=img_size)
    return compute_map_from_records(pred_records, gt_records, num_classes, iou_thresholds=iou_thresholds)


def Map_50(result, target, num_classes, img_size=640):
    """兼容旧接口，只返回 IoU=0.5 的 AP/mAP。"""
    metrics = Map(
        result=result,
        target=target,
        num_classes=num_classes,
        img_size=img_size,
        iou_thresholds=[0.5],
    )
    return {
        'ap_per_class': metrics['ap50_per_class'],
        'map': metrics['map50'],
    }


def Map_50_95(result, target, num_classes, img_size=640):
    """COCO风格 mAP@[0.5:0.95]，步长 0.05。"""
    iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50,0.55,...,0.95
    metrics = Map(
        result=result,
        target=target,
        num_classes=num_classes,
        img_size=img_size,
        iou_thresholds=iou_thresholds,
    )
    return {
        'iou_thresholds': metrics['iou_thresholds'],
        'ap50_per_class': metrics['ap50_per_class'],
        'map50': metrics['map50'],
        'ap50_95_per_class': metrics['ap50_95_per_class'],
        'map50_95': metrics['map50_95'],
    }
