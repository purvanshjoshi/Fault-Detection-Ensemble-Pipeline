from PIL import Image
import os
img_path = r"d:\BOOM BAAM\Grafestt\train_images\train_0.png"
img = Image.open(img_path)
print(f"Dimensions: {img.size}")
