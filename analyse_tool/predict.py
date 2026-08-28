import torch
from network.model_yolo import YOLOv8
import cv2
import numpy as np

def compute_iou(box, boxes):
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

def nms(results: torch.Tensor, conf_threshold=0.25, iou_threshold=0.4, pre_nms_topk=1000):
    output = []
    results_new = results.contiguous().permute(0, 2, 1)  # [B, A, 4+nc]

    for b in range(results_new.shape[0]):
        pred = results_new[b].cpu()

        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:]
        max_scores, max_indices = torch.max(class_scores, dim=1)

        # 先按置信度过滤
        mask = max_scores > conf_threshold
        boxes_xywh = boxes_xywh[mask]
        max_scores = max_scores[mask]
        max_indices = max_indices[mask]

        if boxes_xywh.size(0) == 0:
            output.append(torch.zeros((0, 6)))
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
                if order.numel() == 1:
                    break
                ious = compute_iou(cls_dets[idx], cls_dets[order[1:]])
                order = order[1:][ious <= iou_threshold]

            if keep:
                final_dets.append(cls_dets[keep])

        if final_dets:
            output.append(torch.cat(final_dets, dim=0))
        else:
            output.append(torch.zeros((0, 6)))

    # 兼容单张推理流程，batch=1时直接返回tensor
    return output[0] if len(output) == 1 else output


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


checkpoint = torch.load("weights/visdrone-2/best_yolov8.pth", weights_only=False, map_location="cpu")
num_classes = infer_num_classes(checkpoint)
print("checkpoint num_classes:", num_classes)

model = YOLOv8(nc=num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
print(checkpoint["epoch"])
model.eval()

image = cv2.imread("visdrone/images/train/0000002_00005_d_0000014.jpg")
if image is None:
    raise FileNotFoundError("无法读取图像: visdrone/images/train/0000002_00005_d_0000014.jpg")

orig_h, orig_w = image.shape[:2]
input_size = 640
img_bgr = cv2.resize(image, (input_size, input_size))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
test_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

with torch.no_grad():
    result = model(test_tensor)
    final_result = nms(result[0], conf_threshold=0.25, iou_threshold=0.4, pre_nms_topk=1000)
print("nms后的数据维度:", final_result.shape)

#绘制检测结果
draw_img = image.copy()
scale_x = orig_w / float(input_size)
scale_y = orig_h / float(input_size)

#遍历所有检测结果
for det in final_result.detach().numpy():
    x1 = int(np.clip(det[0] * scale_x, 0, orig_w - 1))
    y1 = int(np.clip(det[1] * scale_y, 0, orig_h - 1))
    x2 = int(np.clip(det[2] * scale_x, 0, orig_w - 1))
    y2 = int(np.clip(det[3] * scale_y, 0, orig_h - 1))

    cv2.rectangle(draw_img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

cv2.imwrite("test_img.jpg", draw_img)
