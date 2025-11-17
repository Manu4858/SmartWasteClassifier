import os
import random
import shutil

# Set your dataset directory
data_dir = "data/train"
val_dir = "data/val"
test_dir = "data/test"

# Create val and test directories if not exist
for folder in [val_dir, test_dir]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Class folders: recyclable / non_recyclable (ignore .DS_Store)
classes = [c for c in os.listdir(data_dir) if not c.startswith('.')]

for cls in classes:
    cls_train_path = os.path.join(data_dir, cls)

    # Get list of images (ignore hidden files)
    images = [img for img in os.listdir(cls_train_path) if not img.startswith('.')]

    random.shuffle(images)

    total = len(images)
    val_count = int(total * 0.15)
    test_count = int(total * 0.15)

    # Paths for val and test folders
    cls_val_path = os.path.join(val_dir, cls)
    cls_test_path = os.path.join(test_dir, cls)

    os.makedirs(cls_val_path, exist_ok=True)
    os.makedirs(cls_test_path, exist_ok=True)

    # Move images
    val_images = images[:val_count]
    test_images = images[val_count:val_count + test_count]

    for img in val_images:
        shutil.move(os.path.join(cls_train_path, img), os.path.join(cls_val_path, img))

    for img in test_images:
        shutil.move(os.path.join(cls_train_path, img), os.path.join(cls_test_path, img))

print("\n✅ Dataset split complete!")
print("70% → train | 15% → val | 15% → test")
