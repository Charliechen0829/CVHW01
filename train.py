import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import FloorPlanDataset
from model import UNet
from config import config
from torchvision import transforms as T


def get_transforms():
    """获取数据增强变换"""
    train_transform = T.Compose([
        T.Resize((416, 416)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = T.Compose([
        T.Resize((416, 416)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    """单个epoch的训练循环"""
    model.train()
    total_loss = 0.0
    loop = tqdm(loader, desc="Training", leave=False)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.to(device)

        # 验证标签合法性
        assert targets.min() >= 0 and targets.max() < config.NUM_CLASSES, \
            f"Invalid labels: min={targets.min()}, max={targets.max()}"

        # 混合精度训练
        with torch.amp.autocast(device_type='cuda'):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # 反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 更新统计量
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def evaluate_fn(loader, model, loss_fn, device):
    """单个epoch的验证循环"""
    model.eval()
    total_loss = 0.0
    loop = tqdm(loader, desc="Validation", leave=False)

    with torch.no_grad():
        for data, targets in loop:
            data = data.to(device)
            targets = targets.to(device)

            # 前向传播
            with torch.amp.autocast(device_type='cuda'):
                predictions = model(data)
                loss = loss_fn(predictions, targets)

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def save_checkpoint(state, filename):
    """保存模型检查点"""
    torch.save(state, filename)
    print(f"\n✅ Checkpoint saved to {filename}")


def main():
    # 初始化配置
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    train_transform, val_transform = get_transforms()

    # 创建数据集
    train_ds = FloorPlanDataset(
        image_dir=config.TRAIN_IMAGES,
        annotation_path=config.TRAIN_ANNOTATIONS,
        transform=train_transform
    )
    val_ds = FloorPlanDataset(
        image_dir=config.VAL_IMAGES,
        annotation_path=config.VAL_ANNOTATIONS,
        transform=val_transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # 初始化模型
    model = UNet(n_channels=3, n_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()

    # 训练状态跟踪
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    best_epoch = -1

    # 训练循环
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS} {'-' * 30}")

        # 训练
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE)
        train_losses.append(train_loss)

        # 验证
        val_loss = evaluate_fn(val_loader, model, loss_fn, config.DEVICE)
        val_losses.append(val_loss)

        # 更新
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            }
            save_checkpoint(
                checkpoint,
                filename=os.path.join(config.SAVE_DIR, config.MODEL_NAME)
            )

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # 可视化保存逻辑
        if epoch == config.NUM_EPOCHS - 1:
            plot_loss_curves(
                train_losses,
                val_losses,
                best_epoch,
                best_loss,
                save_path=os.path.join(config.SAVE_DIR, f"loss_epoch_{epoch + 1:03d}.png")
            )

    print("\n🏆 Training completed!")
    print(f"Best validation loss: {best_loss:.4f} at epoch {best_epoch + 1}")


def plot_loss_curves(train_losses, val_losses, best_epoch, best_val_loss, save_path):
    plt.figure(figsize=(12, 8))

    # 绘制损失曲线
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-o', linewidth=2, markersize=8, label='Training Loss')
    plt.plot(epochs, val_losses, 'r-s', linewidth=2, markersize=8, label='Validation Loss')

    # 标记最佳验证点
    plt.scatter(best_epoch + 1, best_val_loss, s=300, marker='*',
                color='gold', edgecolors='black', zorder=10,
                label=f'Best Val: {best_val_loss:.4f} @ Epoch {best_epoch + 1}')

    plt.title('Training Progress', fontsize=16, pad=20)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Cross Entropy Loss', fontsize=12)
    plt.xticks(epochs, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=10)

    all_losses = train_losses + val_losses
    plt.ylim(min(all_losses) * 0.9, max(all_losses) * 1.1)

    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
