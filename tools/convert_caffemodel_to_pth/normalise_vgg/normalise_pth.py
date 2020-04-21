from __future__ import print_function
import sys
import os
pjoin = os.path.join
import shutil
import time
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob 
import math
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.distributions.one_hot_categorical import OneHotCategorical
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.serialization import load_lua
# my lib
from data_loader import Dataset

def load_param_from_t7(model, in_layer_index, out_layer):
  out_layer.weight = torch.nn.Parameter(model.get(in_layer_index).weight.float())
  out_layer.bias = torch.nn.Parameter(model.get(in_layer_index).bias.float())
load_param = load_param_from_t7

class SmallEncoder4_2(nn.Module):
  def __init__(self, model=None):
    super(SmallEncoder4_2, self).__init__()
    self.vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 16, (3, 3)),
        nn.ReLU(),  # relu1-1 # 3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(16, 16, (3, 3)),
        nn.ReLU(),  # relu1-2 # 6
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(16, 32, (3, 3)),
        nn.ReLU(),  # relu2-1 # 10
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(32, 32, (3, 3)),
        nn.ReLU(),  # relu2-2 # 13
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(32, 64, (3, 3)),
        nn.ReLU(),  # relu3-1 # 17
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu3-2 # 20
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu3-3 # 23
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu3-4 # 26
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu4-1 # 30
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 512, (3, 3)),
        nn.ReLU(),  # relu4-2 # 33
    )
    if model:
      self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage)["model"])
  def forward(self, x):
    return self.vgg(x)

class SmallEncoder4_16x_plus(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallEncoder4_16x_plus, self).__init__()
    self.fixed = fixed
    self.vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 16, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(16, 16, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(16, 32, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(32, 32, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(32, 64, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
    )
    
    if model:
      assert(os.path.splitext(model)[1] in {".t7", ".pth"})
      if model.endswith(".t7"):
        t7_model = load_lua(model)
        load_param(t7_model, 0,  self.vgg[0])
        load_param(t7_model, 2,  self.vgg[2])
        load_param(t7_model, 5,  self.vgg[5])
        load_param(t7_model, 9,  self.vgg[9])
        load_param(t7_model, 12, self.vgg[12])
        load_param(t7_model, 16, self.vgg[16])
        load_param(t7_model, 19, self.vgg[19])
        load_param(t7_model, 22, self.vgg[22])
        load_param(t7_model, 25, self.vgg[25])
        load_param(t7_model, 29, self.vgg[29])
      else: # pth model with different arch design but same learnable weights 
        net = torch.load(model)
        odict_keys = list(net.keys())
        cnt = 0; i = 0
        for m in self.vgg.children():
          if isinstance(m, nn.Conv2d):
            print("layer %s is loaded with trained params" % i)
            m.weight.data.copy_(net[odict_keys[cnt]]); cnt += 1
            m.bias.data.copy_(net[odict_keys[cnt]]); cnt += 1
          i += 1
            
  def forward(self, x):
    return self.vgg(x)
    
class Encoder4(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Encoder4, self).__init__()
    self.fixed = fixed
    self.vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
    )
    
    if model:
      assert(os.path.splitext(model)[1] in {".t7", ".pth"})
      if model.endswith(".t7"):
        t7_model = load_lua(model)
        load_param(t7_model, 0,  self.vgg[0])
        load_param(t7_model, 2,  self.vgg[2])
        load_param(t7_model, 5,  self.vgg[5])
        load_param(t7_model, 9,  self.vgg[9])
        load_param(t7_model, 12, self.vgg[12])
        load_param(t7_model, 16, self.vgg[16])
        load_param(t7_model, 19, self.vgg[19])
        load_param(t7_model, 22, self.vgg[22])
        load_param(t7_model, 25, self.vgg[25])
        load_param(t7_model, 29, self.vgg[29])
      else:
        self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
      
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
          
  def forward(self, x):
    return self.vgg(x)
    
def is_image(x):
  name, ext = os.path.splitext(x)
  ext = ext.lower()
  return ext in [".jpg", ".jpeg", ".png", ".bmp"]

# -----------------------------------------------------------------------------------------
# 2019/10/06
# This file is to normalize pytorch models for NST based on Gatys normalization method.
# run script: CUDA_VISIBLE_DEVICES=5 python normalise_pth.py
# -----------------------------------------------------------------------------------------

# set up model
SE_path = "../../../Experiments/SERVER218-20190925-034318_SE/weights/20190925-034318_E20.pth" # use base model as init
SE_path = "../../../Experiments/SERVER218-20190921-123129_SE_savemodel/weights/20190921-123129_E20.pth" # not use base model as init
# SE_path = "20190925-034318_E20_normalized.pth"
net = SmallEncoder4_2(SE_path).cuda().eval()

# SE_path = "../../../Experiments/e4_ploss0.05_conv1234_QA/weights/192-20181114-0458_4SE_16x_QA_E20S10000-2.pth" # previous 16x model
# SE_path = "/home4/wanghuan/Projects/20180918_KD_for_NST/KDLowlevelVision/Bin/models/small16x_ae_base/e4_base.pth"
# net = SmallEncoder4_16x_plus(SE_path).cuda().eval()

# # check whether the normalized vgg is really normalized on each fm channel. The answer is YES!
# BE_path = "../../../PytorchWCT/models/vgg_normalised_conv4_1.t7"
# net = Encoder4(BE_path).cuda().eval()

# set up images
data_path = "/home4/wanghuan/Dataset/val2014_subset256"
dataset = Dataset(data_path, 300) # shorter side = 300
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, shuffle=False)

# before normalization
for m in net.vgg.children():
  if isinstance(m, nn.Conv2d):
    w = m.weight.data
    w = w.view(w.size(0), -1)
    print(torch.mean(w, dim=1))
print("")

layer_num = [3,6,10,13,17,20,23,26,30] # for all conv layers up to conv4_1 relu
for layer in layer_num:
  cnn = net.vgg[:layer+1]
  means = []
  for step, (x, _) in enumerate(train_loader):
    x = x.cuda()
    feat = cnn.forward(x) # [N, C, H, W]
    N, C, H, W = feat.size()
    feat = feat.view([N, C, H*W])
    mean1 = torch.mean(feat, dim=0)
    mean2 = torch.mean(mean1, dim=1)
    means.append(mean2.data.cpu().numpy()) # [[C-dim], ..., [C-dim]]
  mean_for_filter = np.mean(means, axis=0)
  global_mean = float(np.mean(mean_for_filter))
  assert(len(mean_for_filter) == C)
  print(mean_for_filter, " ", global_mean)
  for c in range(C):
    # if float(mean_for_filter[c]) == 0:
      # mean_ = global_mean
    # else:
      # mean_ = float(mean_for_filter[c])
    net.vgg[layer-1].weight[c].data /= global_mean # net params have been updated
    net.vgg[layer-1].bias[c].data /= global_mean # net params have been updated
  print("normalized layer #%d\n" % layer)

# after normalization
for m in net.vgg.children():
  if isinstance(m, nn.Conv2d):
    w = m.weight.data
    w = w.view(w.size(0), -1)
    print(torch.mean(w, dim=1))
print("")

save_model = {"model": net.state_dict()}
torch.save(save_model, "20190925-034318_E20_normalized.pth")