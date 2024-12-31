import os
import random
import shutil


def rename_images_in_folder(folder_path):
    
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    
    file_paths = [os.path.join(folder_path, f) for f in files]

    
    shuffled_files = files.copy()
    while True:
        random.shuffle(shuffled_files)
        if all(shuffled_files[i] != files[i] for i in range(len(files))):
            break

    
    temp_files = []
    for old_name in files:
        temp_name = os.path.join(folder_path, f"temp_{random.randint(30000, 99999)}.tmp")
        temp_files.append(temp_name)
        shutil.move(os.path.join(folder_path, old_name), temp_name)

    
    for temp_name, new_name in zip(temp_files, shuffled_files):
        new_name_path = os.path.join(folder_path, new_name)
        shutil.move(temp_name, new_name_path)

    print("finish！")



folder_path = r'datasets\Dehaze\outdoor\train\clear_bak'
rename_images_in_folder(folder_path)
