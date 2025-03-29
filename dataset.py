import os
import json
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class FloorPlanDataset(Dataset):
    def __init__(self, image_dir, annotation_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # 加载COCO格式的标注文件
        with open(annotation_path) as f:
            self.coco_data = json.load(f)

        # 创建image_id到annotations的映射
        self.image_anns = defaultdict(list)
        for ann in self.coco_data['annotations']:
            self.image_anns[ann['image_id']].append(ann)

        # 创建category_id到标签索引的映射
        self.cat_to_label = {}
        self.num_classes = len(self.coco_data['categories']) + 1  # +1 for background

        for idx, cat in enumerate(self.coco_data['categories']):
            self.cat_to_label[cat['id']] = idx + 1  # 0表示背景

        # 存储图像信息列表
        self.images = self.coco_data['images']

    # 返回数据集中的图像数量
    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        image_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        # 创建分割mask (初始化为全0，表示背景)
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

        # 为每个标注对象填充mask
        for ann in self.image_anns.get(img_info['id'], []):
            cat_id = ann['category_id']
            if cat_id not in self.cat_to_label:
                continue  # 跳过无效类别
            bbox = ann['bbox']
            x, y, w, h = map(int, bbox)
            mask[y:y + h, x:x + w] = self.cat_to_label[cat_id]

        # 确保mask中没有超出类别的值
        mask = np.clip(mask, 0, self.num_classes - 1)

        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(mask).long()  # 转换为torch tensor

        return image, mask


def get_transforms():
    # 训练数据增强
    train_transform = T.Compose([
        T.Resize((416, 416)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 验证/测试数据增强
    val_transform = T.Compose([
        T.Resize((416, 416)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform