import os
import torch
from torchvision import transforms
from PIL import Image, ImageOps
from model import CNN
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "../results/mnist_cnn_manual.pth" 
model = CNN(num_classes=10, dropout=0.5).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(" Model loaded successfully!")


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


test_folder = "../my-digits" 

image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for img_name in image_files:
    img_path = os.path.join(test_folder, img_name)
    image = Image.open(img_path).convert('L')

    image = ImageOps.invert(image)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        probs = F.softmax(output, dim=1)[0].cpu().numpy()
        pred_label = predicted.item()

    for i, p in enumerate(probs):
        print(f"  {i}: {p*100:.2f}%")
    print(f"{img_name} Prediction: {pred_label}")