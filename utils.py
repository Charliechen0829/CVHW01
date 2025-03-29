import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer=None):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])


def visualize_predictions(model, loader, device, num_examples=3):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        if idx >= num_examples:
            break

        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.softmax(model(x), dim=1)
            preds = torch.argmax(preds, dim=1).squeeze(0)

        # 转换回CPU numpy数组
        x = x.cpu().squeeze(0).permute(1, 2, 0).numpy()
        y = y.cpu().squeeze(0).numpy()
        preds = preds.cpu().numpy()

        # 反归一化图像
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        x = std * x + mean
        x = np.clip(x, 0, 1)

        # 可视化
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.imshow(x)
        ax1.set_title("Original Image")
        ax2.imshow(y, cmap="jet")
        ax2.set_title("Ground Truth")
        ax3.imshow(preds, cmap="jet")
        ax3.set_title("Prediction")
        plt.show()


def save_predictions_as_images(loader, model, folder="saved_images", device="cuda"):
    model.eval()
    if not os.path.exists(folder):
        os.makedirs(folder)

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.softmax(model(x), dim=1)
            preds = torch.argmax(preds, dim=1).squeeze(0)

        # 保存预测结果
        preds = preds.float().cpu().numpy()
        preds = (preds * 255 / preds.max()).astype(np.uint8)  # 归一化到0-255
        preds = Image.fromarray(preds)
        preds.save(os.path.join(folder, f"pred_{idx}.png"))
