import numpy as np
import os
import torch.nn as nn
import torch
from torch.utils.serialization import load_lua
from utils import load_param_from_t7 as load_param
import pickle
pjoin = os.path.join

# ------------------------------------------------------------
# small decoder 16x with 1x1 auxiliary mapping to get KD supervision from BD
class SmallDecoder5_16x_aux(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallDecoder5_16x_aux, self).__init__()
    self.fixed = fixed

    self.conv51 = nn.Conv2d(128,128,3,1,0)
    self.conv44 = nn.Conv2d(128,128,3,1,0)
    self.conv43 = nn.Conv2d(128,128,3,1,0)
    self.conv42 = nn.Conv2d(128,128,3,1,0)
    self.conv41 = nn.Conv2d(128, 64,3,1,0)
    self.conv34 = nn.Conv2d( 64, 64,3,1,0, dilation=1)
    self.conv33 = nn.Conv2d( 64, 64,3,1,0, dilation=1)
    self.conv32 = nn.Conv2d( 64, 64,3,1,0, dilation=1)
    self.conv31 = nn.Conv2d( 64, 32,3,1,0, dilation=1)
    self.conv22 = nn.Conv2d( 32, 32,3,1,0, dilation=1)
    self.conv21 = nn.Conv2d( 32, 16,3,1,0, dilation=1)
    self.conv12 = nn.Conv2d( 16, 16,3,1,0, dilation=1)
    self.conv11 = nn.Conv2d( 16,  3,3,1,0, dilation=1)

    self.aux51 = nn.Conv2d(128, 512, 1, 1, 0)
    self.aux41 = nn.Conv2d( 64, 256, 1, 1, 0)
    self.aux31 = nn.Conv2d( 32, 128, 1, 1, 0)
    self.aux21 = nn.Conv2d( 16,  64, 1, 1, 0)
    
    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      weights = torch.load(model, map_location=lambda storage, location: storage)
      if "model" in weights:
        self.load_state_dict(weights["model"])
      else:
        self.load_state_dict(weights)
      print("load model '%s' successfully" % model)

    if fixed:
      for param in self.parameters():
          param.requires_grad = False
          
  def forward(self, y):
    y = self.relu(self.conv51(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv44(self.pad(y)))
    y = self.relu(self.conv43(self.pad(y)))
    y = self.relu(self.conv42(self.pad(y)))
    y = self.relu(self.conv41(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv34(self.pad(y)))
    y = self.relu(self.conv33(self.pad(y)))
    y = self.relu(self.conv32(self.pad(y)))
    y = self.relu(self.conv31(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv22(self.pad(y)))
    y = self.relu(self.conv21(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv12(self.pad(y)))
    y = self.relu(self.conv11(self.pad(y))) # self.conv11(self.pad(y))
    return y

  def forward_aux(self, x, relu=False):
    out51 = self.relu(self.conv51(self.pad(x)))
    out51 = self.unpool(out51)
    out44 = self.relu(self.conv44(self.pad(out51)))
    out43 = self.relu(self.conv43(self.pad(out44)))
    out42 = self.relu(self.conv42(self.pad(out43)))
    out41 = self.relu(self.conv41(self.pad(out42)))
    out41 = self.unpool(out41)
    out34 = self.relu(self.conv34(self.pad(out41)))
    out33 = self.relu(self.conv33(self.pad(out34)))
    out32 = self.relu(self.conv32(self.pad(out33)))
    out31 = self.relu(self.conv31(self.pad(out32)))
    out31 = self.unpool(out31)
    out22 = self.relu(self.conv22(self.pad(out31)))
    out21 = self.relu(self.conv21(self.pad(out22)))
    out21 = self.unpool(out21)
    out12 = self.relu(self.conv12(self.pad(out21)))
    out11 = self.relu(self.conv11(self.pad(out12)))

    if relu:
      out51_aux = self.relu(self.aux51(out51))
      out41_aux = self.relu(self.aux41(out41))
      out31_aux = self.relu(self.aux31(out31))
      out21_aux = self.relu(self.aux21(out21))
    else:
      out51_aux = self.aux51(out51)
      out41_aux = self.aux41(out41)
      out31_aux = self.aux31(out31)
      out21_aux = self.aux21(out21)

    return out51_aux, out41_aux, out31_aux, out21_aux, out11


class SmallDecoder4_16x_aux(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallDecoder4_16x_aux, self).__init__()
    self.fixed = fixed

    self.conv41 = nn.Conv2d(128, 64,3,1,0)
    self.conv34 = nn.Conv2d( 64, 64,3,1,0, dilation=1)
    self.conv33 = nn.Conv2d( 64, 64,3,1,0, dilation=1)
    self.conv32 = nn.Conv2d( 64, 64,3,1,0, dilation=1)
    self.conv31 = nn.Conv2d( 64, 32,3,1,0, dilation=1)
    self.conv22 = nn.Conv2d( 32, 32,3,1,0, dilation=1)
    self.conv21 = nn.Conv2d( 32, 16,3,1,0, dilation=1)
    self.conv12 = nn.Conv2d( 16, 16,3,1,0, dilation=1)
    self.conv11 = nn.Conv2d( 16,  3,3,1,0, dilation=1)

    self.aux41 = nn.Conv2d( 64, 256, 1, 1, 0)
    self.aux31 = nn.Conv2d( 32, 128, 1, 1, 0)
    self.aux21 = nn.Conv2d( 16,  64, 1, 1, 0)
    
    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      weights = torch.load(model, map_location=lambda storage, location: storage)
      if "model" in weights:
        self.load_state_dict(weights["model"])
      else:
        self.load_state_dict(weights)
      print("load model '%s' successfully" % model)

    if fixed:
      for param in self.parameters():
          param.requires_grad = False
          
  def forward(self, y):
    y = self.relu(self.conv41(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv34(self.pad(y)))
    y = self.relu(self.conv33(self.pad(y)))
    y = self.relu(self.conv32(self.pad(y)))
    y = self.relu(self.conv31(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv22(self.pad(y)))
    y = self.relu(self.conv21(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv12(self.pad(y)))
    y = self.relu(self.conv11(self.pad(y))) # self.conv11(self.pad(y))
    return y

  def forward_aux(self, x, relu=False):
    out41 = self.relu(self.conv41(self.pad(x)))
    out41 = self.unpool(out41)
    out34 = self.relu(self.conv34(self.pad(out41)))
    out33 = self.relu(self.conv33(self.pad(out34)))
    out32 = self.relu(self.conv32(self.pad(out33)))
    out31 = self.relu(self.conv31(self.pad(out32)))
    out31 = self.unpool(out31)
    out22 = self.relu(self.conv22(self.pad(out31)))
    out21 = self.relu(self.conv21(self.pad(out22)))
    out21 = self.unpool(out21)
    out12 = self.relu(self.conv12(self.pad(out21)))
    out11 = self.relu(self.conv11(self.pad(out12)))

    if relu:
      out41_aux = self.relu(self.aux41(out41))
      out31_aux = self.relu(self.aux31(out31))
      out21_aux = self.relu(self.aux21(out21))
    else:
      out41_aux = self.aux41(out41)
      out31_aux = self.aux31(out31)
      out21_aux = self.aux21(out21)

    return out41_aux, out31_aux, out21_aux, out11


class SmallDecoder3_16x_aux(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallDecoder3_16x_aux, self).__init__()
    self.fixed = fixed

    self.conv31 = nn.Conv2d( 64, 32,3,1,0, dilation=1)
    self.conv22 = nn.Conv2d( 32, 32,3,1,0, dilation=1)
    self.conv21 = nn.Conv2d( 32, 16,3,1,0, dilation=1)
    self.conv12 = nn.Conv2d( 16, 16,3,1,0, dilation=1)
    self.conv11 = nn.Conv2d( 16,  3,3,1,0, dilation=1)

    self.aux31 = nn.Conv2d( 32, 128, 1, 1, 0)
    self.aux21 = nn.Conv2d( 16,  64, 1, 1, 0)
    
    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      weights = torch.load(model, map_location=lambda storage, location: storage)
      if "model" in weights:
        self.load_state_dict(weights["model"])
      else:
        self.load_state_dict(weights)
      print("load model '%s' successfully" % model)

    if fixed:
      for param in self.parameters():
          param.requires_grad = False
          
  def forward(self, y):
    y = self.relu(self.conv31(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv22(self.pad(y)))
    y = self.relu(self.conv21(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv12(self.pad(y)))
    y = self.relu(self.conv11(self.pad(y))) # self.conv11(self.pad(y))
    return y

  def forward_aux(self, x, relu=False):
    out31 = self.relu(self.conv31(self.pad(x)))
    out31 = self.unpool(out31)
    out22 = self.relu(self.conv22(self.pad(out31)))
    out21 = self.relu(self.conv21(self.pad(out22)))
    out21 = self.unpool(out21)
    out12 = self.relu(self.conv12(self.pad(out21)))
    out11 = self.relu(self.conv11(self.pad(out12)))

    if relu:
      out31_aux = self.relu(self.aux31(out31))
      out21_aux = self.relu(self.aux21(out21))
    else:
      out31_aux = self.aux31(out31)
      out21_aux = self.aux21(out21)

    return out31_aux, out21_aux, out11


class SmallDecoder2_16x_aux(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallDecoder2_16x_aux, self).__init__()
    self.fixed = fixed

    self.conv21 = nn.Conv2d( 32, 16,3,1,0, dilation=1)
    self.conv12 = nn.Conv2d( 16, 16,3,1,0, dilation=1)
    self.conv11 = nn.Conv2d( 16,  3,3,1,0, dilation=1)

    self.aux21 = nn.Conv2d( 16,  64, 1, 1, 0)
    
    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      weights = torch.load(model, map_location=lambda storage, location: storage)
      if "model" in weights:
        self.load_state_dict(weights["model"])
      else:
        self.load_state_dict(weights)
      print("load model '%s' successfully" % model)

    if fixed:
      for param in self.parameters():
          param.requires_grad = False
          
  def forward(self, y):
    y = self.relu(self.conv21(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv12(self.pad(y)))
    y = self.relu(self.conv11(self.pad(y))) # self.conv11(self.pad(y))
    return y

  def forward_aux(self, x, relu=False):
    out21 = self.relu(self.conv21(self.pad(x)))
    out21 = self.unpool(out21)
    out12 = self.relu(self.conv12(self.pad(out21)))
    out11 = self.relu(self.conv11(self.pad(out12)))

    if relu:
      out21_aux = self.relu(self.aux21(out21))
    else:
      out21_aux = self.aux21(out21)

    return out21_aux, out11

class SmallDecoder1_16x_aux(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallDecoder1_16x_aux, self).__init__()
    self.fixed = fixed

    self.conv11 = nn.Conv2d( 24,  3,3,1,0, dilation=1)
    
    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      weights = torch.load(model, map_location=lambda storage, location: storage)
      if "model" in weights:
        self.load_state_dict(weights["model"])
      else:
        self.load_state_dict(weights)
      print("load model '%s' successfully" % model)

    if fixed:
      for param in self.parameters():
          param.requires_grad = False
          
  def forward(self, y):
    y = self.relu(self.conv11(self.pad(y)))
    return y

  def forward_aux(self, x, relu=False):
    out11 = self.relu(self.conv11(self.pad(x)))
    return out11, out11