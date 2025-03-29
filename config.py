import torch
import os


class Config:
    # 数据路径
    DATA_DIR = "./data"
    TRAIN_IMAGES = os.path.join(DATA_DIR, "train")
    VAL_IMAGES = os.path.join(DATA_DIR, "valid")
    TEST_IMAGES = os.path.join(DATA_DIR, "test")

    TRAIN_ANNOTATIONS = os.path.join(TRAIN_IMAGES, "_annotations.coco.json")
    VAL_ANNOTATIONS = os.path.join(VAL_IMAGES, "_annotations.coco.json")
    TEST_ANNOTATIONS = os.path.join(TEST_IMAGES, "_annotations.coco.json")

    # 训练参数
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 5  # 4类(门、窗、区域、门窗区域) + 背景

    # 模型保存路径
    SAVE_DIR = "./models"
    MODEL_NAME = "unet.pth"

    # 设备配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


config = Config()
