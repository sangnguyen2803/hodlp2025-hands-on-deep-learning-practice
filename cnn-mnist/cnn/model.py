import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        n_images = int.from_bytes(f.read(4), 'big')
        n_rows = int.from_bytes(f.read(4), 'big')
        n_cols = int.from_bytes(f.read(4), 'big')
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(n_images, 1, n_rows, n_cols).astype(np.float32) / 255.0
        return data

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        n_labels = int.from_bytes(f.read(4), 'big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        # convert python datatype to tensor of PyTorch 0 - 255
        img = torch.tensor(img) 
        if self.transform:
            # convert to float32 ranging from 0.0 to 1.0, applying normalization, resizing and augmenting to prepare the image for neural network
            # transform method includes the torch.tensor()
            img = self.transform(img)
        return img, label

class CNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5):
        # super(CNN, self).__init__()
        super().__init__()
        # Layer 1:
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Layer 2:
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Layer 3:
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # Fully-connected
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(loader, desc=f"Epoch [{epoch}/{total_epochs}] Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# def test_model(model, loader, device, criterion=None, num_classes=None):
#     model.eval()
#     correct = 0
#     total = 0
#     all_preds = []
#     all_labels = []
#     total_loss = 0.0
    
#     with torch.no_grad():
#         for images, labels in loader:
#             images, labels = images.to(device), labels.to(device)
            
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
            
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
            
#             all_preds.extend(predicted.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
            
#             if criterion is not None:
#                 loss = criterion(outputs, labels)
#                 total_loss += loss.item() * labels.size(0)  # nhân với batch size
    
#     accuracy = 100 * correct / total
#     avg_loss = total_loss / total if criterion is not None else None
    
#     conf_matrix = None
#     if num_classes is not None:
#         conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    
#     return {
#         'accuracy': accuracy,
#         'avg_loss': avg_loss,
#         'conf_matrix': conf_matrix
#     }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_images = load_mnist_images('../dataset/train-images.idx3-ubyte')
    train_labels = load_mnist_labels('../dataset/train-labels.idx1-ubyte')

    test_images = load_mnist_images('../dataset/t10k-images.idx3-ubyte')
    test_labels = load_mnist_labels('../dataset/t10k-labels.idx1-ubyte')

    print('Shape of train images: ', train_images.shape)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5,), (0.5,)) # normalize to [-1, 1]
    ])
    
    train_dataset = MNISTDataset(
        train_images, 
        train_labels,
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset, #available to use for NN
        batch_size=64,
        shuffle=True
    )
    test_dataset = MNISTDataset(
        test_images,
        test_labels,
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False
    )

    model = CNN(num_classes=10, dropout=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f'CNN Model: {model}\n')
    print('Training CNN Model:\n')

    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        loss = train(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        acc = evaluate(model, test_loader, device)
        print(f"[Epoch {epoch}/{num_epochs}] Loss: {loss:.4f} | Test Acc: {acc:.2f}%")

    # Save model
    torch.save(model.state_dict(), "../results/mnist_cnn_manual.pth")

#     args = parse_args()
#     # choose device having GPU
#     device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
#     # create an unique run id
#     run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
#     # create a log directory to store training metrics
#     writer = SummaryWriter(log_dir=f'runs/mnist_{run_id}')
#     train_loader, val_loader = get_dataloaders(args.batch_size, augment=args.augment, num_workers=args.workers)

#     model = ComplexCNN(num_classes=10, dropout=args.dropout).to(device)

if __name__ == '__main__':
    main()