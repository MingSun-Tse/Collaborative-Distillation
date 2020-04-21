import torch.utils.data as data
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as transforms
import os
import numpy as np
from my_utils import is_img

class Dataset_npy(data.Dataset):
  def __init__(self, img_dir):
    self.img_list = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if i.endswith(".npy")]

  def __getitem__(self, index):
    img = np.load(self.img_list[index])
    img = Image.fromarray(img).convert("RGB")
    img = transforms.RandomCrop(256)(img)
    img = transforms.RandomHorizontalFlip()(img)
    img = transforms.ToTensor()(img)
    return img.squeeze(0), self.img_list[index]

  def __len__(self):
    return len(self.img_list)
    
class Dataset(data.Dataset):
  def __init__(self, img_dir, shorter_side, transform=None):
    self.img_list = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if is_img(i)]
    self.shorter_side = shorter_side

  def __getitem__(self, index):
    img = Image.open(self.img_list[index]).convert("RGB")
    if self.shorter_side:
      w, h = img.size
      if w < h: # resize the shorter side to `shorter_side`
        neww = self.shorter_side
        newh = int(h * neww / w)
      else:
        newh = self.shorter_side
        neww = int(w * newh / h)
      img = img.resize((neww, newh))
      img = transforms.RandomCrop(256)(img)
      img = transforms.RandomHorizontalFlip()(img)
      img = transforms.ToTensor()(img)
    return img.squeeze(0), self.img_list[index]

  def __len__(self):
    return len(self.img_list)

class TestDataset(data.Dataset):
  def __init__(self, img_dir, shorter_side, transform=None):
    self.img_list = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if is_img(i)]
    random_order = np.random.permutation(len(self.img_list))
    self.img_list = list(np.array(self.img_list)[random_order])
    self.shorter_side = shorter_side

  def __getitem__(self, index):
    img = Image.open(self.img_list[index]).convert("RGB")
    if self.shorter_side:
      w, h = img.size
      if w < h: # resize the shorter side to `shorter_side`
        neww = self.shorter_side
        newh = int(h * neww / w)
      else:
        newh = self.shorter_side
        neww = int(w * newh / h)
      img = img.resize((neww, newh))
      img = transforms.CenterCrop(256)(img)
      img = transforms.ToTensor()(img)
    return img.squeeze(0), self.img_list[index]

  def __len__(self):
    return len(self.img_list)
    
class ContentStylePair(data.Dataset):
  def __init__(self, pathC, pathS, shorter_side):
    self.imgListC = [os.path.join(pathC, i) for i in os.listdir(pathC) if is_img(i)]
    self.imgListS = [os.path.join(pathS, i) for i in os.listdir(pathS) if is_img(i)]
    self.shorter_side = shorter_side
  
  def __getitem__(self, ix):
    imgC = Image.open(self.imgListC[ix % len(self.imgListC)]).convert("RGB")
    imgS = Image.open(self.imgListS[ix % len(self.imgListS)]).convert("RGB")
    if self.shorter_side:
      # content
      w, h = imgC.size
      if w < h: # resize the shorter side to `shorter_side`
        neww = self.shorter_side
        newh = int(h * neww / w)
      else:
        newh = self.shorter_side
        neww = int(w * newh / h)
      imgC = imgC.resize((neww, newh))
      imgC = transforms.RandomCrop(256)(imgC) # the real image size for training is 256!
      imgC = transforms.RandomHorizontalFlip()(imgC)
      imgC = transforms.ToTensor()(imgC)
      # style
      w, h = imgS.size
      if w < h: # resize the shorter side to `shorter_side`
        neww = self.shorter_side
        newh = int(h * neww / w)
      else:
        newh = self.shorter_side
        neww = int(w * newh / h)
      imgS = imgS.resize((neww, newh))
      imgS = transforms.RandomCrop(256)(imgS) # the real image size for training is 256!
      imgS = transforms.RandomHorizontalFlip()(imgS)
      imgS = transforms.ToTensor()(imgS)
    return imgC.squeeze(0), imgS.squeeze(0)
    
  def __len__(self):
    return max(len(self.imgListC), len(self.imgListS))
  