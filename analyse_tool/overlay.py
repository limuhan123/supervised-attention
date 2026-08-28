import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from network import *
from pathlib import Path

def visualize_overlay(image, features, save_path=None):
    """
    将热力图叠加在预处理后的原图上
    image: np.array格式的图像 (预处理并pad好的)
    features: 模型输出的 branch
    """
    # 提取并处理热力图
    feat_logits = features[0].detach().cpu()
    att_map = torch.sigmoid(feat_logits).squeeze().numpy()  # [40, 40]
    
    # 将输入图片(tensor反向处理或直接用预处理时的padded图)转为 uint8
    img_show = (image.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    h, w = img_show.shape[:2]
    
    # 将热力图放大到原图尺寸
    att_map_resized = cv2.resize(att_map, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 转为伪彩色
    heatmap = np.uint8(255 * att_map_resized)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 叠加
    overlay = cv2.addWeighted(img_show, 0.5, heatmap_color, 0.5, 0)
    
    if save_path:
        cv2.imwrite(save_path,overlay)

def preprocess_image(image, img_size=640):
    # 调整大小
    h, w = image.shape[:2]
    scale = min(img_size / h, img_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    
    # 填充到正方形
    padded = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    
    # 转换为tensor并归一化
    tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)  # 添加batch维度
    
    return tensor

def main_branch(model_path,image_path):
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
    config = checkpoint["config"]
    cls = globals()[config["model_name"]]
    my_model = cls(nc=config["num_classes"],scales=config["depth"]).to(device)
    my_model.load_state_dict(checkpoint['model_state_dict'])
    my_model.eval()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR转RGB
    input_tensor = preprocess_image(image)
    _,result = my_model(input_tensor)
    print(f"Backbone 输出形状: {result.shape}")

    #获取保存路径
    save_dir = Path(model_path).parent
    img_name = Path(image_path).parent.name
    save_path = save_dir / f"{img_name}-overlay"

    #可视化所有通道的平均激活
    visualize_overlay(input_tensor, result, save_path=save_path)


if __name__ == "__main__":
    model_path = "weights/one-low_no_Net_learn_self_branch/best.pth"
    img_path = "data_all/car_with_ground/images/train/0013.png"
    main_branch(model_path,img_path)