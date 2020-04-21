import os
from PIL import Image

def resize(img_path, size):
  img = Image.open(img_path)
  img = img.resize([size, size])
  img_name, ext = os.path.splitext(img_path)
  img.save(img_name + "_resized" + ext)
