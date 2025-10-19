import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

import torch.nn as nn
import torch.nn.functional as F

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

class CNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5):
        # super(CNN, self).__init__()
        super().__init__()
        # Layer 1:
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Layer 2:
        


def main():
    train_images = load_mnist_images('train-images.idx3-ubytes')
    train_labels = load_mnist_labels('train-labels.idx1-ubytes')
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