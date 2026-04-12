"""
Utility script to check image dimensions.
"""
import os
from PIL import Image

IMG_PATH = r"d:\BOOM BAAM\Grafestt\train_images\train_0.png"
img = Image.open(IMG_PATH)
print(f"Dimensions: {img.size}")
