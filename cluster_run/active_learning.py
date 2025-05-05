import os
import glob
import numpy as np
from PIL import Image
from typing import List
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights
from torch.nn import functional as F
from sklearn.cluster import KMeans
from torch import nn

# Define breed categories
cat_breeds = [
    'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
    'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
    'Siamese', 'Sphynx'
]
dog_breeds = [
    'American_Bulldog', 'American_Pit_Bull_Terrier', 'Basset_Hound', 'Beagle',
    'Boxer', 'Chihuahua', 'English_Cocker_Spaniel', 'English_Setter', 'German_Shorthaired',
    'Great_Pyrenees', 'Havanese', 'Japanese_Chin', 'Keeshond', 'Leonberger', 'Miniature_Pinscher',
    'Newfoundland', 'Pomeranian', 'Pug', 'Saint_Bernard', 'Samoyed', 'Scottish_Terrier',
    'Shiba_Inu', 'Staffordshire_Bull_Terrier', 'Wheaten_Terrier', 'Yorkshire_Terrier'
]
cat_breeds_lower = [b.lower() for b in cat_breeds]
dog_breeds_lower = [b.lower() for b in dog_breeds]


class PetDataset(Dataset):
    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        class_name = "_".join(os.path.basename(path).split("_")[:-1]).lower()
        if class_name in cat_breeds_lower:
            label = 0
        elif class_name in dog_breeds_lower:
            label = 1
        else:
            raise ValueError(f"Unknown breed: {path}")
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.image_paths)


class ActiveLearning:
    def __init__(self, k=20, beta=10, image_size=224, model=None, feature_extractor=None):
        self.k = k
        self.beta = beta
        self.train_image_paths = []

        # Use passed-in model if provided
        self.model = model if model is not None else resnet34(weights=ResNet34_Weights.DEFAULT)
        self.model.eval()

        # Use passed-in feature extractor or default
        self.feature_extractor = feature_extractor if feature_extractor is not None else resnet34(weights=ResNet34_Weights.DEFAULT)
        if feature_extractor is None:
            self.feature_extractor.fc = torch.nn.Identity()
        self.feature_extractor.eval()

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def set_training_pool(self, image_paths):
        self.train_image_paths = image_paths
    
    def preprocess(self, path):
        img = Image.open(path).convert("RGB")
        return torch.unsqueeze(self.transform(img), 0)

    def softmax_probs(self, img_tensor):
        with torch.no_grad():
            return F.softmax(self.model(img_tensor), dim=1).squeeze().numpy()

    def extract_feature(self, img_tensor):
        with torch.no_grad():
            return self.feature_extractor(img_tensor).squeeze().numpy()

    def query_least_confidence(self, folder: str) -> List[str]:
        files = self.train_image_paths
        scores = []
        for f in files:
            x = self.preprocess(f)
            p = self.softmax_probs(x)
            lc = 1 - np.max(p)
            scores.append((f, lc))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [f for f, _ in scores[:self.k]]

    def query_max_entropy(self, folder: str) -> List[str]:
        files = self.train_image_paths
        scores = []
        for f in files:
            x = self.preprocess(f)
            p = self.softmax_probs(x)
            entropy = -np.sum(p * np.log(p + 1e-12))
            scores.append((f, entropy))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [f for f, _ in scores[:self.k]]

    def query_kmeans(self, folder: str) -> List[str]:
        files = self.train_image_paths
        features = []
        valid_files = []

        for f in files:
            try:
                img_tensor = self.preprocess(f)
                feature = self.extract_feature(img_tensor)
                if feature.ndim == 1:
                    features.append(feature)
                    valid_files.append(f)
            except Exception as e:
                print(f"Skipping {f} due to error: {e}")

        if len(features) < self.k:
            raise ValueError(f"Not enough valid features to form {self.k} clusters. Only got {len(features)}.")

        features = np.vstack(features)  # Shape: (N, D)
        kmeans = KMeans(n_clusters=self.k, random_state=0).fit(features)
        centers = kmeans.cluster_centers_

        selected = []
        for center in centers:
            dists = np.linalg.norm(features - center, axis=1)
            idx = np.argmin(dists)
            selected.append(valid_files[idx])
        return selected

    def query_hybrid(self) -> List[str]:
        files = self.train_image_paths
        margins = []
        features = []
        for f in files:
            x = self.preprocess(f)
            p = self.softmax_probs(x)
            p_sorted = np.sort(p)
            margin = 1 - (p_sorted[-1] - p_sorted[-2])
            margins.append((f, margin))
            features.append(self.extract_feature(x))
        top_beta = sorted(margins, key=lambda x: x[1], reverse=True)[:self.k * self.beta]
        top_files = [f for f, _ in top_beta]
        top_features = [features[files.index(f)] for f in top_files]
        kmeans = KMeans(n_clusters=self.k, random_state=0).fit(top_features)
        selected = []
        for i in range(self.k):
            dists = np.linalg.norm(np.array(top_features) - kmeans.cluster_centers_[i], axis=1)
            selected.append(top_files[np.argmin(dists)])
        return selected
    
if __name__ == "__main__":
    
    
    al = ActiveLearning(k=5, beta=10)
    img_path = "data/oxford-iiit-pet/images/Abyssinian_1.jpg"
    img_tensor = al.preprocess(img_path)
    feature = al.extract_feature(img_tensor)
    print("Feature shape:", feature.shape)

    # Query 10 least confident samples
    print("Querying least confident samples...")
    selected_lc = al.query_least_confidence("./data/oxford-iiit-pet/images/")

    # Query 10 most entropic samples
    print("Querying most entropic samples...")
    selected_entropy = al.query_max_entropy("./data/oxford-iiit-pet/images/")

    # Query 10 cluster centers
    print("Querying KMeans cluster centers...")
    selected_kmeans = al.query_kmeans("./data/oxford-iiit-pet/images/")

    # Hybrid method
    selected_hybrid = al.query_hybrid("./data/oxford-iiit-pet/images/")