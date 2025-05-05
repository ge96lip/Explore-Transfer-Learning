import os, glob, random, copy
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights
from sklearn.metrics import classification_report
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from active_learning import ActiveLearning, cat_breeds_lower, dog_breeds_lower

# === Config ===
ROUNDS = 14
INITIAL_LABELED_PERCENTAGE = 0.05
ROUND_ADDITION_PERCENTAGE = 0.025
BATCH_SIZE = 64
EPOCHS = 40
LR = 0.01
SEEDS = [3, 12]
RESULT_LOG = []

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LabeledFromPathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        class_name = "_".join(os.path.basename(path).split("_")[:-1]).lower()
        label = 0 if class_name in cat_breeds_lower else 1
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.image_paths)

def evaluate_model(model, dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total

def train_model(model, train_dataset, val_dataset, loss_fn, epochs=EPOCHS):
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_model = copy.deepcopy(model)
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = evaluate_model(model, val_dataset)
        if acc > best_acc:
            best_model = copy.deepcopy(model)
            best_model.load_state_dict(model.state_dict())
            best_acc = acc
    return best_model

def get_label_from_path(path):
    class_name = "_".join(os.path.basename(path).split("_")[:-1]).lower()
    return 0 if class_name in cat_breeds_lower else 1

def log_results_to_csv(csv_path, seed_results):
    df = pd.DataFrame(seed_results)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

# === Begin Seed Loop ===
for seed in SEEDS:
    print(f"\n=== Running for SEED {seed} ===")
    set_seed(seed)

    # === Load Data ===
    all_images = glob.glob("data/oxford-iiit-pet/images/*.jpg")
    all_images = [f for f in all_images if "_".join(os.path.basename(f).split("_")[:-1]).lower()
                  in (cat_breeds_lower + dog_breeds_lower)]
    labels = [get_label_from_path(p) for p in all_images]
    train_imgs, temp_imgs, _, temp_labels = train_test_split(all_images, labels, test_size=0.3, stratify=labels, random_state=seed)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, stratify=temp_labels, random_state=seed)

    val_cats = [f for f in val_imgs if get_label_from_path(f) == 0][:10]
    val_dogs = [f for f in val_imgs if get_label_from_path(f) == 1][:10]
    val_imgs_balanced = val_cats + val_dogs
    val_dataset = LabeledFromPathDataset(val_imgs_balanced, transform=transform)
    test_dataset = LabeledFromPathDataset(test_imgs, transform=transform)

    train_size = len(train_imgs)
    initial_size = int(INITIAL_LABELED_PERCENTAGE * train_size)
    round_size = int(ROUND_ADDITION_PERCENTAGE * train_size)

    initial_labeled = random.sample(train_imgs, initial_size)
    pool = list(set(train_imgs) - set(initial_labeled))

    strategies = ['entropy_random', 'hybrid_random', 'random_only']
    models = {s: resnet34(weights=ResNet34_Weights.DEFAULT).to(device) for s in strategies}
    for m in models.values():
        m.fc = nn.Linear(m.fc.in_features, 2)
        m.to(device)
    labels_per_strategy = {s: copy.deepcopy(initial_labeled) for s in strategies}
    pools = {s: copy.deepcopy(pool) for s in strategies}

    for s in strategies:
        acc = evaluate_model(models[s], test_dataset)
        num_cats = sum(get_label_from_path(f) == 0 for f in labels_per_strategy[s])
        num_dogs = sum(get_label_from_path(f) == 1 for f in labels_per_strategy[s])
        RESULT_LOG.append({'seed': seed, 'round': 0, 'strategy': s, 'accuracy': acc, 'num_cats': num_cats, 'num_dogs': num_dogs})

    for r in range(1, ROUNDS + 1):
        for s in strategies:
            model = models[s]
            feature_extractor = copy.deepcopy(model)
            feature_extractor.fc = torch.nn.Identity()
            feature_extractor.to(device)
            al_engine = ActiveLearning(k=5, model=model, feature_extractor=feature_extractor)
            al_engine.set_training_pool(pools[s])

            if s == 'lc_random':
                al_samples = al_engine.query_least_confidence(None)
            elif s == 'entropy_random':
                al_samples = al_engine.query_max_entropy(None)
            elif s == 'kmeans_random':
                al_samples = al_engine.query_kmeans(None)
            elif s == 'hybrid_random':
                al_samples = al_engine.query_hybrid()
            else:
                al_samples = []

            al_samples = al_samples[:round_size // 2] if s != 'random_only' else []
            random_samples = random.sample(pools[s], round_size - len(al_samples))
            new_samples = al_samples + random_samples

            labels_per_strategy[s].extend(new_samples)
            pools[s] = list(set(pools[s]) - set(new_samples))

            train_set = LabeledFromPathDataset(labels_per_strategy[s], transform=transform)
            models[s] = train_model(model, train_set, val_dataset, nn.CrossEntropyLoss())
            acc = evaluate_model(models[s], test_dataset)
            print(f"Round {r}, Strategy {s}, Accuracy: {acc:.4f}")
            num_cats = sum(get_label_from_path(f) == 0 for f in labels_per_strategy[s])
            num_dogs = sum(get_label_from_path(f) == 1 for f in labels_per_strategy[s])
            RESULT_LOG.append({'seed': seed, 'round': r, 'strategy': s, 'accuracy': acc,
                               'num_cats': num_cats, 'num_dogs': num_dogs})

# === Save CSV ===
log_results_to_csv("active_learning_eval.csv", RESULT_LOG)
