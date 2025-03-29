import os
import torch
from PIL import Image
from torchvision import transforms as T
from model import UNet
from config import config
from utils import load_checkpoint
import matplotlib.pyplot as plt


def predict_single_image(model, image_path, device):
    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((416, 416)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 预测
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax2.imshow(pred, cmap="jet")
    ax2.set_title("Prediction")
    plt.show()

    return pred


def main():
    # 加载模型
    model = UNet(n_channels=3, n_classes=config.NUM_CLASSES).to(config.DEVICE)
    load_checkpoint(torch.load(os.path.join(config.SAVE_DIR, config.MODEL_NAME)), model)

    # 预测单张图像
    image_path = r".\data\test\1116_png.rf.1b192683036a98e0a3328ad9cc5d885a.jpg"
    predict_single_image(model, image_path, config.DEVICE)


if __name__ == "__main__":
    main()
