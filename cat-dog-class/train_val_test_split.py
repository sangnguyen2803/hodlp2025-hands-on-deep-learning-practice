import os
import shutil
import random

# Paths
original_train = "./source/train"  # your current train folder
output_dir = "dataset_split"  # folder to store new structure
classes = ["cats", "dogs"]

# Split ratios
val_ratio = 0.15  # 15% for validation
test_ratio = 0.15  # 15% for test
# Remaining ~70% will stay in train

# Create folder structure
for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

# Function to split images
for cls in classes:
    class_path = os.path.join(original_train, cls)
    images = os.listdir(class_path)
    random.shuffle(images)

    num_total = len(images)
    num_val = int(num_total * val_ratio)
    num_test = int(num_total * test_ratio)
    num_train = num_total - num_val - num_test

    # Split images
    train_imgs = images[:num_train]
    val_imgs = images[num_train:num_train + num_val]
    test_imgs = images[num_train + num_val:]

    # Copy files to new folders
    for img in train_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(output_dir, "train", cls, img))
    for img in val_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(output_dir, "val", cls, img))
    for img in test_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(output_dir, "test", cls, img))

    print(f"{cls}: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

print("Dataset split complete!")