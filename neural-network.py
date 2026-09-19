"""
Original file is located at
    https://colab.research.google.com/drive/1cjMiPSSHo__0Fm1AUGo4EY835jgnsqxa
"""

import matplotlib.pyplot as plt
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms



""" get dataset from kaggle """
import kagglehub
path = kagglehub.dataset_download("sumn2u/garbage-classification-v2")
print("Path to dataset files:", path)



""" data preprocessing """
#divide training, validation, and test data
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

#resize images to 224 x 224 for EfficientNet
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

data_root = os.path.join(path, "original")

def remove_folder(folder): #removing the clothing and shoes folder
  if os.path.exists(folder):
      shutil.rmtree(folder)
remove_folder("original/clothes")
remove_folder("original/shoes")

full_dataset = ImageFolder(root=data_root, transform=transform) #this puts all of the images into one big folder

train_size = int(0.8 * len(full_dataset)) #80/10/10 split cus it's nice
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size]) #this divides all the images into smaller folders

#dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)



""" creating model """
#use EfficentNet B0 for transfer learning
from torchvision import models

if torch.backends.mps.is_available():
    device = torch.device("mps")       # Apple GPU
elif torch.cuda.is_available():
    device = torch.device("cuda")      # NVIDIA GPU
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

#load the EfficientNet model
model = models.efficientnet_b0(pretrained=True)

num_classes = len(full_dataset.classes)

for param in model.parameters():
    param.requires_grad = False
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes) #changing the # of features to how many classes there are
model = model.to(device)

#loss function & optimizer
criterion = nn.CrossEntropyLoss() #cross-entropy for multiple classes
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001) #might add weight decay...?



""" model training """
from tqdm.auto import tqdm # progress bar for notebooks and regular Python scripts

# training loop
num_epochs = 10 #adjust this?

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

print("starting training")

# Prepare figure for dynamic plotting
plt.figure(figsize=(12, 6))

def train_epoch(model, train_loader, optimizer, criterion, device, epoch_num, num_epochs, total_train_samples):
    model.train()
    running_loss = 0.0
    correct_train_predictions = 0
    total_train_predictions = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch_num}/{num_epochs} (Training)"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train_predictions += labels.size(0)
        correct_train_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_train_samples
    epoch_accuracy = correct_train_predictions / total_train_predictions
    return epoch_loss, epoch_accuracy

def validate_epoch(model, val_loader, criterion, device, epoch_num, num_epochs, total_val_samples):
    model.eval()
    val_running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch_num}/{num_epochs} (Validation)"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    val_epoch_loss = val_running_loss / total_val_samples
    val_accuracy = correct_predictions / total_predictions
    return val_epoch_loss, val_accuracy

for epoch in range(num_epochs):
    epoch_num = epoch + 1

    # Training step
    epoch_loss, epoch_train_accuracy = train_epoch(model, train_loader, optimizer, criterion, device, epoch_num, num_epochs, len(train_dataset))
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_train_accuracy)

    # Validation step
    val_epoch_loss, val_accuracy = validate_epoch(model, val_loader, criterion, device, epoch_num, num_epochs, len(val_dataset))
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch_num}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    #plotting after each epoch
    # This script may run outside Jupyter, where IPython's ``clear_output`` is
    # unavailable. Clearing the Matplotlib figure gives the same refreshed plot
    # without adding a notebook-only dependency.
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_accuracies) + 1), [acc * 100 for acc in train_accuracies], label='Train Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), [acc * 100 for acc in val_accuracies], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Train and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

print("training complete")

# save the trained model
torch.save(model.state_dict(), 'efficientnet_garbage_classifier.pth')
print("model is saved as efficientnet_garbage_classifier.pth")



""" model testing """
print("starting testing")

model.eval()

test_running_loss = 0.0
correct_test_predictions = 0
total_test_predictions = 0

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_running_loss += loss.item() * inputs.size(0)

        _, predicted = torch.max(outputs.data, 1)
        total_test_predictions += labels.size(0)
        correct_test_predictions += (predicted == labels).sum().item()

test_loss = test_running_loss / len(test_dataset)
test_accuracy = correct_test_predictions / total_test_predictions

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print("testing complete.")
