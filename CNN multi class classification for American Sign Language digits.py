import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image




# Define a custom dataset class
class ASLDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []

        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                image = Image.open(image_path)
                image = image.resize((64, 64))  # Resize to your desired size
                image_array = np.array(image)
                self.images.append(image_array)
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define data transforms
transform = transforms.Compose([
    transforms.ToTensor()
])


import matplotlib.pyplot as plt

# Create dataset and data loaders for training and testing
train_dataset = ASLDataset(root_dir="C:\\Users\\sha\\Desktop\\ASL Digits gray\\asl_dataset_digits_gray", transform=transform) 
test_dataset = ASLDataset(root_dir="C:\\Users\\sha\\Desktop\\ASL Digits gray\\test_gray", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Get the number of examples in the train and test datasets
num_train_examples = len(train_dataset)
num_test_examples = len(test_dataset)

print("Number of examples in train dataset:", num_train_examples)
print("Number of examples in test dataset:", num_test_examples)



class_counts = [0] * 10  # Initialize a list to store the counts for each class

# Count the occurrences of each class in the train dataset
for _, label in train_dataset:
    class_counts[label] += 1

# Print the counts for each class
for class_idx, count in enumerate(class_counts):
    print(f"Class {class_idx}: {count} examples")



test_class_counts = [0] * 10  # Initialize a list to store the counts for each class in the test dataset

# Count the occurrences of each class in the test dataset
for _, label in test_dataset:
    test_class_counts[label] += 1

# Print the counts for each class in the test dataset
for class_idx, count in enumerate(test_class_counts):
    print(f"Class {class_idx}: {count} examples")



import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for digits 0-9
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model and other training settings
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Compute training loss
        running_loss += loss.item()
        
        # Compute training accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total
    
    print(f"Epoch {epoch+1}/{num_epochs} completed. Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}")




# Evaluate the model on the test set
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {correct/total:.4f}")

