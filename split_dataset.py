import os
import shutil
import random

source_dir = r"C:\Users\PRIYADARSHINI S\hand-gesture-recognition\dataset"
train_dir = os.path.join(source_dir, "train")
test_dir = os.path.join(source_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)

    if not os.path.isdir(class_path) or class_name in ["train", "test"]:
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split = int(0.8 * len(images))
    train_imgs = images[:split]
    test_imgs = images[split:]

    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    for img in train_imgs:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(train_dir, class_name, img))

    for img in test_imgs:
        shutil.copy(os.path.join(class_path, img),
                    os.path.join(test_dir, class_name, img))

print("✅ Dataset split completed!")