import torch
from torchvision import transforms
from datasets import load_dataset

# ========================
# CUSTOM DATASET WRAPPER
# ========================
class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["img"]   # PIL Image
        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_transforms():
    # CNN giữ nguyên resolution 32x32
    # ========================
    # TRANSFORMS
    # ========================
    transform_cnn = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    transform_vit = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform_cnn, transform_vit