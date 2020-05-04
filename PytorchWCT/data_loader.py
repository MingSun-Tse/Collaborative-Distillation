from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data as data
from os import listdir
from os.path import join
import numpy as np
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def default_loader(path):
    return Image.open(path).convert('RGB')

class Dataset(data.Dataset):
    def __init__(self, contentPath, stylePath, texturePath, c_size=0, s_size=0, picked_content_mark=".", picked_style_mark=".", synthesis=False):
      super(Dataset,self).__init__()
      self.content_size = c_size
      self.style_size = s_size
      self.synthesis = synthesis
      if synthesis:
        self.texturePath = texturePath
        self.texture_image_list = [x for x in listdir(texturePath) if is_image_file(x)]
      else:
        self.contentPath = contentPath
        self.stylePath   = stylePath
        content_imgs = [x for x in listdir(contentPath) if is_image_file(x) and picked_content_mark in x]
        style_imgs   = [x for x in listdir(stylePath)   if is_image_file(x) and picked_style_mark   in x]
        pairs = [[c, s] for c in content_imgs for s in style_imgs]
        self.content_image_list = list(np.array(pairs)[:, 0])
        self.style_image_list   = list(np.array(pairs)[:, 1])
      
      # self.normalize = transforms.Normalize(mean=[103.939,116.779,123.68],std=[1, 1, 1])
      # normalize = transforms.Normalize(mean=[123.68,103.939,116.779],std=[1, 1, 1])
      # self.prep = transforms.Compose([
                  # transforms.Scale(fineSize),
                  # transforms.ToTensor(),
                  # #transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                  # ])

    def __getitem__(self, index):
      if not self.synthesis: # style transfer
        contentImgPath = os.path.join(self.contentPath, self.content_image_list[index])
        styleImgPath = os.path.join(self.stylePath, self.style_image_list[index])
        contentImg = default_loader(contentImgPath)
        styleImg = default_loader(styleImgPath)
        if self.content_size:
          contentImg = contentImg.resize(self.content_size)
        if self.style_size:
          styleImg = styleImg.resize(self.style_size)
        contentImg = transforms.ToTensor()(contentImg)
        styleImg   = transforms.ToTensor()(styleImg)
        return contentImg.squeeze(0), styleImg.squeeze(0), \
               self.content_image_list[index].split(".")[0] + "+" + self.style_image_list[index].split(".")[0] + ".jpg"
      
      else: # texture synthesis
        textureImgPath = os.path.join(self.texturePath, self.texture_image_list[index])
        textureImg = default_loader(textureImgPath)
        if self.style_size:
          w, h = textureImg.size
          if w > h:
            neww = self.style_size
            newh = int(h * neww / w)
          else:
            newh = self.style_size
            neww = int(w * newh / h)
          textureImg = textureImg.resize((neww,newh))
        w, h = textureImg.size
        contentImg = torch.rand_like(textureImg)
        textureImg = transforms.ToTensor()(textureImg)
        return contentImg.squeeze(0), textureImg.squeeze(0), self.texture_image_list[index].split(".")[0] + ".jpg"

    def __len__(self):
        return len(self.texture_image_list) if self.synthesis else len(self.content_image_list)