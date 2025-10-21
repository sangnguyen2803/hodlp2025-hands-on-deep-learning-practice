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
            # The Conv2d layer is a 2D convolution
            # 1. A filter or convolutional kernel
            # Another matrix which will drag across the image
            # ! PyTorch initializes these kernels/filter weights randomly
            # 1.1. Dimension of kernel matrix
            # number of input x kernel_size x kernel_size

            # 1.2. Convolution process
            # For each filter (total filters = total number of output):
            # It slides over the whole image
            # At each spatial position:
            # Multiplies the input x kernel_size x kernel_size patch of pixels by the input x kernel_size x kernel_size weights of the filter
            # Sums all those products + bias
            # Writes that value to the output feature map
            # => This gives one 2D feature map per filter.
            # After processing all filters → you get output channels,
            # i.e. a feature map stack shaped: [batch_size, 32, H, W]

            # 2. Channels/feature maps
            # If the image is gray-scale -> input channel will be 1
            # If the image is rgb -> input channel will be 3
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            
            # ReLU is a non-linear activation
            # it outputs:
            # - 0 for negative inputs
            # - x itself for positive inputs
            # 1. Introducing non-linearity
            # Without an activation function like ReLU => all layers in a NN collapse into a single linear transformation
            # => By inserting ReLU, we break that linearity because it changes the relationship between two layers (the input and the output)
            # => The network can now learn complex, non-linear functions - like curves, edges, shapes, etc.
            
            # 2. Avoiding the "vanishing" gradient problem
            # Other activation functions (like sigmoid or tanh) squash outputs into really small range:
            # 0->1 for sigmoid
            # -1->1 for tanh
            # This causes the derivatives to be very small
            # => kills gradients during backpropagation -> slow/no learning

            # 3. Encourage sparsity in feature maps
            #  It outputs 0 for half the inputs, it creates sparse activations (some neurons completely silent).
            # => helps networks focus only on useful patterns and often improves generalization
            nn.ReLU(),

            # 1. Downsampling / spatial reduction
            # Max Pooling (2D) reduces wifth and height (of each feature map) x2
            # => affects the resolution of detected patterns
            # => Smaller feature maps => fewer parameters => faster computation

            # 2. Two meanings of “feature”
            # When people say “downsampling reduces features”, they are usually talking about spatial features — i.e. the width × height of the feature map
            # => Downsampling affects the W, H not the channels (feature maps)
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            # After the convolution + pooling layers, your tensor might look like this:
            # [batch_size, 32, 16, 16]
            # nn.Flatten() simply reshapes it into a long 1D vector per image:
            # This flattening prepares the data for fully connected (Linear) layers, which expect 1D inputs.
            nn.Flatten(),

            # This is a fully connected (dense) layer.
            # Each of the 8192 input features connects to all 128 output neurons.
            # This layer combines all spatial and channel features learned by the CNN into global patterns that help with classification.
            # Answer this problem:
            # “Given all features detected so far, which combinations seem important for recognizing my classes?”

            nn.Linear(32 * 16 * 16, 128),  # since image 64x64 → after 2 pools → 16x16
            nn.ReLU(),

            # This is the final classification layer.
            # Takes the 128 learned features and maps them to however many output classes you have.
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

# the most common loss for multi-class classification problems
# It combines two steps in one:
# 1. Softmax → converts your model’s raw outputs (“logits”) into probabilities
# 2. Negative log likelihood (NLL) → compares those probabilities to the true label
# → Bigger penalty for being wrong.
criterion = nn.CrossEntropyLoss()

# The optimizer updates your model’s weights based on the gradients computed from the loss.
# Adam is like an improved version of SGD (Stochastic Gradient Descent).
# Instead of using the same step size for all weights,
# it keeps a running average of past gradients and adapts each weight’s learning rate individually.
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 8. Training loop (supervised)

# Each iteration:
# 1. Model makes a prediction
# 2. Loss measures how wrong it was
# 3. Backprop computes how much each weight contributed to that error
# 4. Adam updates the weights to reduce future errors
# And that’s how CNN learns
# From random filters → to edge detectors → to meaningful shape detectors → to confident class predictions
for epoch in range(50):  # run 3 epochs for demo
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:  # <-- labels are used here => supervised
        # During backpropagation
        # PyTorch computes the gradients of the loss with respect to all model parameters (weights, biases, etc.) using backpropagation
        # Those gradients are stored in each parameter’s .grad attribute.
        # => PyTorch accumulates gradients by default
        # => in other words, every time you call loss.backward(), the new gradients get added (accumulated) to the existing ones
        # if you don’t clear them, you end up mixing gradients from multiple batches
        # fix: using optimizer.zero_grad()
        # => This sets all parameter gradients to zero — so that the next .backward() starts clean.
        optimizer.zero_grad()             # Reset previous gradients
        outputs = model(images)           # Forward pass
        loss = criterion(outputs, labels) # # Compute loss
        loss.backward()                   # Backpropagate (compute gradients)
        optimizer.step()                  # Update weights using Adam
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader):.4f}")

print("✅ Training done (supervised mode).")

# Dropout
# One rrecurring issue with neural networks is their tendency to overfit to training data
# The Dropout layer is a devilishly simple way of doing this
# -> simply not train a random bunch of nodes within the network during a training cycle
# because it’s random,
# each training cycle will ignore a different selection of the input,
# => which should help generalization even further

# By default, the Dropout layers in our example CNN network are initialized with
# 0.5
# To change -> e.g. Dropout(p=0.2)

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

# Using Pretrained Models in PyTorch
# For example:
# import torchvision.models as models
# alexnet = models.alexnet(num_classes=2)
# call models.alexnet(pretrained=True) to download a pretrained
# set of weights for AlexNet, allowing you to use it immediately for classification
# with no extra training

# Examining a Model’s Structure
# print(model)

# BarchNorm (batch normalization)
# a simple layer that has one task in life:
# using two learned parameters (meaning that it will be trained along with the rest of the network)
# to try to ensure that each minibatch that goes through the
# network has a mean centered around zero with a variance of 1
# -> why?
# For smaller networks -> BatchNorm is less useful
# For larger networks -> the effect of any layer on another, say 20 layers
# down, can be vast because of repeated multiplication
# => either vanishing or exploding gradients
# => fatal to training process
# => The BatchNorm layers make sure that even if you use a model such as
# ResNet-152, the multiplications inside your network don’t get out of hand
