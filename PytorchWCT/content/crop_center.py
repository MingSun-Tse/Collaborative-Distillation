from PIL import Image as I
import sys
import os
from utils import is_img
pjoin = os.path.join

def crop_and_save(img_path):
  img = I.open(img_path)
  w, h = img.size
  center_img = img
  if w > h:
    margin = int((w - h)/2)
    center_img = img.crop((margin, 0, margin+h, h))
  elif w < h:
    margin = int((h - w)/2)
    center_img = img.crop((0, margin, w, margin+w))
  ext = os.path.splitext(img_path)[1]
  center_img.save(img_path.replace(ext, "_center" + ext))

inDir = sys.argv[1]
[crop_and_save(pjoin(inDir, i)) for i in os.listdir(inDir) if is_img(i)]
