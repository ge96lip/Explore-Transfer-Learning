import torch
import random
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm


torch.manual_seed(42)
random.seed(42)

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
TARGET_SIZE = 224

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations
transform = transforms.Compose([
    transforms.Resize(256),           
    transforms.CenterCrop(TARGET_SIZE),      
    transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load dataset
full_dataset = datasets.OxfordIIITPet(
    root='./data',
    download=True,
    transform=transform,
    target_types='category'
)

# Convert 37-class labels to binary: 0 = cat, 1 = dog
dataset = [(img, 1 if label >= 25 else 0) for img, label in tqdm(full_dataset, desc="Converting labels")]
# Dataset label 0-36, classes 0-24 are cats, 25-36 are dogs
# dataset = [(img, 1 if label >= 25 else 0) for img, label in dataset]

# Only train and val set -> if we want test set change the code to: 

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

#train_size = int(0.8 * len(dataset))
#val_size = len(dataset) - train_size
#train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
# Load pretrained ResNet18 and modify the final layer
# load standard resent18 which is trained on ImageNet (which originally has 1000 output classes

# Load pretrained ResNet18 with modern API
# weights = ResNet18_Weights.DEFAULT  # or .IMAGENET1K_V1 explicitly
# model = resnet18(weights=weights)
model = resnet34(weights=ResNet34_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)
# deprecated: 
# model = models.resnet18(pretrained=True)
# the fc is the final fully connected layer of the original resent18 model -> now we replace it 
model.fc = nn.Linear(model.fc.in_features, 1)  # Binary output

"""
#freez all but the output layer: 
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 1)  # This stays trainable
"""
model = model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()  # Sigmoid + Binary Cross-Entropy
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
best_acc = 0.0 
# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = (torch.sigmoid(outputs) > 0.5).squeeze(1).long()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    print(f"Validation Accuracy: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")
        print("Improved model saved")
        
    # scheduler.step()
# Load best model after training
model.load_state_dict(torch.load("best_model.pt"))

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predictions = (torch.sigmoid(outputs) > 0.5).squeeze(1).long()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")