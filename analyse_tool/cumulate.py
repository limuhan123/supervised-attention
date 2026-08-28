import torch
from network import YOLOv8,YOLO_Dataset
from thop import profile
import os


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

def para_cumulate(config,model,device):

    num_classes = config["num_classes"]
    train_dataset = YOLO_Dataset(config["data_path"], mode="train", image_size=640)
    summarize_dataset_targets(train_dataset, num_classes, split_name="train")
    print("目标类别 :", num_classes)

    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    macs, params = profile(model, inputs=(dummy_input, ),verbose=False)
    print(f"总参数量: {params}")
    print(f"GFLOPs: {macs * 2 / 1e9:.4f}")

    return params,float(round(macs * 2 / 1e9,4))