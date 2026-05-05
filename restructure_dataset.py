import os
import shutil

source_dir = r"C:\Users\PRIYADARSHINI S\Downloads\archive (2)\leapGestRecog"
target_dir = r"C:\Users\PRIYADARSHINI S\hand-gesture-recognition\dataset"

os.makedirs(target_dir, exist_ok=True)

for user_folder in os.listdir(source_dir):
    user_path = os.path.join(source_dir, user_folder)

    if not os.path.isdir(user_path):
        continue

    for gesture_folder in os.listdir(user_path):

        # Handle folder names safely
        if "_" in gesture_folder:
            gesture_name = gesture_folder.split("_", 1)[1]
        else:
            gesture_name = gesture_folder

        src_path = os.path.join(user_path, gesture_folder)
        dst_path = os.path.join(target_dir, gesture_name)

        # Skip if not a folder
        if not os.path.isdir(src_path):
            continue

        os.makedirs(dst_path, exist_ok=True)

        # ✅ Copy only files inside folder
        for img in os.listdir(src_path):
            img_path = os.path.join(src_path, img)

            if os.path.isfile(img_path):   # 🔥 THIS FIXES YOUR ERROR
                shutil.copy(img_path, os.path.join(dst_path, img))

print("✅ Dataset restructured successfully!")