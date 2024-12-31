from PIL import Image
import os


input_dir = r"datasets\Dehaze\outdoor\test\clear"
output_dir = r"datasets\Dehaze\outdoor128\test\clear"


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        try:
            
            img = Image.open(os.path.join(input_dir, filename))
            
            img_resized = img.resize((128, 128))
            
            img_resized.save(os.path.join(output_dir, filename))
            print(f"Resized and saved: {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
