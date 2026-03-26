import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from dataset import CIFAR10Dataset, get_transforms
from models import get_model
from utils import evaluate
import time, pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 10

# ========================
# DATALOADER
# ========================
def get_dataloaders(transform, train_data, test_data):
    full_train = CIFAR10Dataset(train_data, transform)
    test_dataset = CIFAR10Dataset(test_data, transform)

    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size

    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, name, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_acc = 0
    total_train_time = 0

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        print(f"\n{name} - Epoch {epoch + 1}/{EPOCHS}")
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        epoch_duration = time.time() - start_time  # Tính thời gian 1 epoch
        total_train_time += epoch_duration
        _, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        print(f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Time: {epoch_duration:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"models/{name}_best.pth")
    avg_time = total_train_time / EPOCHS
    print(f"\n{name} Training Complete. Average Time per Epoch: {avg_time:.2f}s")
    return best_acc

def plot_distribution(split_name, title):
    # Lấy nhãn từ dataset
    labels = list(dataset[split_name]['label'])

    # Đếm số lượng mẫu cho mỗi lớp
    df = pd.DataFrame(labels, columns=['label'])
    counts = df['label'].value_counts().sort_index()

    # Vẽ biểu đồ
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=[classes[i] for i in counts.index], y=counts.values, palette='magma')

    # Thêm tiêu đề và nhãn
    plt.title(title, fontsize=15)
    plt.xlabel('Class Name', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.ylim(0, max(counts.values) * 1.1)  # Tạo khoảng trống phía trên để hiển thị số

    # Hiển thị số lượng cụ thể trên mỗi cột
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    plt.savefig(f"{split_name}_distribution.png")
    plt.show()


def calculate_class_similarity():
    class_means = []

    for i in range(10):
        class_images = [np.array(x) for x, y in zip(dataset['train']['img'], dataset['train']['label']) if y == i]
        mean_color = np.mean(class_images, axis=(0, 1, 2))
        class_means.append(mean_color)

    class_means = np.array(class_means)

    norm_means = class_means / np.linalg.norm(class_means, axis=1, keepdims=True)
    similarity_matrix = np.dot(norm_means, norm_means.T)

    # Vẽ Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                xticklabels=classes, yticklabels=classes)
    plt.title("Figure 4: Semantic & Visual Similarity Potential")
    plt.tight_layout()
    plt.savefig("visual_similarity_heatmap.png")
    plt.show()

# ========================
# MAIN
# ========================
if __name__ == "__main__":
    dataset = load_dataset("uoft-cs/cifar10")
    t_cnn, t_vit = get_transforms()
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    train_data = dataset["train"]
    test_data = dataset["test"]

    # --- Training ResNet50 ---
    print("\n=== Training ResNet50 ===")
    train_l, val_l, test_l = get_dataloaders(train_data, test_data, t_cnn)
    resnet = get_model("resnet50").to(device)
    train_model(resnet, train_l, val_l, "resnet50", lr=1e-4)

    # --- Training ViT ---
    print("\n=== Training ViT-Base (State-of-the-Art) ===")
    train_l_vit, val_l_vit, test_l_vit = get_dataloaders(train_data, test_data, t_vit)
    vit = get_model("vit").to(device)
    train_model(vit, train_l_vit, val_l_vit, "vit_base", lr=5e-5)

    # --- Final Test Evaluation ---
    resnet.load_state_dict(torch.load("models/resnet50_best.pth"))
    vit.load_state_dict(torch.load("models/vit_base_best.pth"))
    _, test_acc1, _ = evaluate(resnet, test_l, nn.CrossEntropyLoss(), device)
    print(f"\nFinal ResNet Test Accuracy: {test_acc1:.4f}")
    _, test_acc2, _ = evaluate(vit, test_l_vit, nn.CrossEntropyLoss(), device)
    print(f"\nFinal ViT Test Accuracy: {test_acc2:.4f}")