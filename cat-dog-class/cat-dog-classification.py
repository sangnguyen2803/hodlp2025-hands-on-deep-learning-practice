# 1. Imports
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets

# 2. Paths
train_data_path = "./data/train/"
val_data_path = "./data/val/"
test_data_path = "./data/test/"

# 3. Transforms (data preprocessing)
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # resize to 64x64
    transforms.ToTensor(),        # convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 4. Datasets (supervised)
# ImageFolder expects structure:
# train/
#   ├── cats/
#   ├── dogs/
# val/
#   ├── cats/
#   ├── dogs/
# test/
#   ├── cats/
#   ├── dogs/
train_data = datasets.ImageFolder(root=train_data_path, transform=transform)
val_data = datasets.ImageFolder(root=val_data_path, transform=transform)
test_data = datasets.ImageFolder(root=test_data_path, transform=transform)

# 5. Data Loaders
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 6. Neural Network (simple example)
class SimpleCNN(nn.Module):
    # do any setup required in init(), in this case calling our superclass constructor
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),  # since image 64x64 → after 2 pools → 16x16
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    # The forward() method describes how data flows through the network in both training and making predictions (inference)
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 7. Initialize model, loss, optimizer
num_classes = len(train_data.classes)  # automatically detects number of labels
model = SimpleCNN(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 8. Training loop (supervised)
for epoch in range(50):  # run 3 epochs for demo
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:  # <-- labels are used here => supervised
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader):.4f}")

print("✅ Training done (supervised mode).")

# 9. (Optional) Validation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Validation Accuracy: {100 * correct / total:.2f}%")
