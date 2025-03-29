import os
import torch
from torch.utils.data import DataLoader
from dataset import FloorPlanDataset, get_transforms
from model import UNet
from config import config
from utils import load_checkpoint, visualize_predictions
import numpy as np


def calculate_iou(preds, labels, num_classes):
    ious = []
    preds = torch.argmax(preds, dim=1)

    for cls in range(1, num_classes):  # 跳过背景类
        pred_inds = (preds == cls)
        target_inds = (labels == cls)

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            ious.append(float('nan'))  # 如果没有真值或预测，设为NaN
        else:
            ious.append((intersection / union).item())

    return np.nanmean(ious)


def evaluate(model, loader, device, num_classes):
    model.eval()
    total_iou = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            batch_iou = calculate_iou(preds, y, num_classes)

            if not np.isnan(batch_iou):
                total_iou += batch_iou
                total_samples += 1

    mean_iou = total_iou / total_samples if total_samples > 0 else 0
    return mean_iou


def main():
    # 加载数据
    _, val_transform = get_transforms()
    val_ds = FloorPlanDataset(
        image_dir=config.VAL_IMAGES,
        annotation_path=config.VAL_ANNOTATIONS,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # 评估时batch_size=1以便计算IoU
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 加载模型
    model = UNet(n_channels=3, n_classes=config.NUM_CLASSES).to(config.DEVICE)
    load_checkpoint(torch.load(os.path.join(config.SAVE_DIR, config.MODEL_NAME)), model)

    # 计算mIoU
    miou = evaluate(model, val_loader, config.DEVICE, config.NUM_CLASSES)
    print(f"Mean IoU: {miou:.4f}")

    # 可视化一些预测结果
    visualize_predictions(model, val_loader, config.DEVICE, num_examples=3)


if __name__ == "__main__":
    main()
