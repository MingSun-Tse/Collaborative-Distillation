import numpy as np
import os
import torch.nn as nn
import torch
# from torch.utils.serialization import load_lua
from utils import load_param_from_t7 as load_param
from model.model_kd2sd import SmallDecoder1_16x_aux, SmallDecoder2_16x_aux, SmallDecoder3_16x_aux, SmallDecoder4_16x_aux, SmallDecoder5_16x_aux
import pickle
pjoin = os.path.join

# calculate style distances in CVPR paper
# since 5-stage style distances are shown separately, there is no need to normalize it by num_channel.
# ref https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
def gram_matrix(input):
  a, b, c, d = input.size() # [N, C, H, W]
  batch_feat = input.view(a, b, c*d) # [N, C, HW]
  batch_gram = torch.stack([torch.mm(feat, feat.t()) for feat in batch_feat])
  batch_gram = batch_gram.div(a*b*c*d) 
  return batch_gram # shape: [N, C, C]

# ref: AdaIN impel. (https://github.com/naoto0804/pytorch-AdaIN/blob/master/function.py)
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)
    
# calculate average style distance, which needs normalization by num_channel.
def gram_matrix_ave(input):
  a, b, c, d = input.size()
  batch_feat = input.view(a, b, c*d)
  batch_gram = torch.stack([torch.mm(feat, feat.t()).div(b*c*d) for feat in batch_feat])
  return batch_gram # shape: [batch_size, channel, channel]

# Load param from model1 to model2
# For each layer of model2, if model1 has the same layer, then copy the params.
def load_param2(model1_path, model2):
  dict_param1 = torch.load(model1_path) # model1_path: .pth model path
  dict_param2 = model2.state_dict()
  for name2 in dict_param2:
    if name2 in dict_param1:
      # print("tensor '%s' found in both models, so copy it from model 1 to model 2" % name2)
      dict_param2[name2].data.copy_(dict_param1[name2].data)
  model2.load_state_dict(dict_param2)
  return model2
  
# -----------------------------------------------
class SmallDecoder1_16x(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallDecoder1_16x, self).__init__()
    self.fixed = fixed

    self.conv11 = nn.Conv2d(24,3,3,1,0, dilation=1)
    self.relu = nn.ReLU(inplace=True)
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

  def forward_pwct(self, input):
    out11 = self.conv11(self.pad(input))
    return out11
    
class SmallDecoder2_16x(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallDecoder2_16x, self).__init__()
    self.fixed = fixed

    self.conv21 = nn.Conv2d( 32, 16,3,1,0)
    self.conv12 = nn.Conv2d( 16, 16,3,1,0, dilation=1)
    self.conv11 = nn.Conv2d( 16,  3,3,1,0, dilation=1)

    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.unpool_pwct = nn.MaxUnpool2d(kernel_size=2, stride=2)
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
    y = self.relu(self.conv11(self.pad(y)))
    return y

  def forward_pwct(self, x, pool1_idx=None, pool1_size=None, pool2_idx=None, pool2_size=None, pool3_idx=None, pool3_size=None):
    out21 = self.relu(self.conv21(self.pad(x)))
    out21 = self.unpool_pwct(out21, pool1_idx, output_size=pool1_size)
    out12 = self.relu(self.conv12(self.pad(out21)))
    out11 = self.conv11(self.pad(out12))
    return out11
    
class SmallDecoder3_16x(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallDecoder3_16x, self).__init__()
    self.fixed = fixed

    self.conv31 = nn.Conv2d( 64, 32,3,1,0)
    self.conv22 = nn.Conv2d( 32, 32,3,1,0)
    self.conv21 = nn.Conv2d( 32, 16,3,1,0)
    self.conv12 = nn.Conv2d( 16, 16,3,1,0, dilation=1)
    self.conv11 = nn.Conv2d( 16,  3,3,1,0, dilation=1)

    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.unpool_pwct = nn.MaxUnpool2d(kernel_size=2, stride=2)
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
    y = self.relu(self.conv11(self.pad(y)))
    return y
    
  def forward_pwct(self, x, pool1_idx=None, pool1_size=None, pool2_idx=None, pool2_size=None, pool3_idx=None, pool3_size=None):
    out31 = self.relu(self.conv31(self.pad(x)))
    out31 = self.unpool_pwct(out31, pool2_idx, output_size=pool2_size)
    out22 = self.relu(self.conv22(self.pad(out31)))
    out21 = self.relu(self.conv21(self.pad(out22)))
    out21 = self.unpool_pwct(out21, pool1_idx, output_size=pool1_size)
    out12 = self.relu(self.conv12(self.pad(out21)))
    out11 = self.conv11(self.pad(out12))
    return out11
    
class SmallDecoder4_16x(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallDecoder4_16x, self).__init__()
    self.fixed = fixed

    self.conv41 = nn.Conv2d(128, 64,3,1,0)
    self.conv34 = nn.Conv2d( 64, 64,3,1,0)
    self.conv33 = nn.Conv2d( 64, 64,3,1,0)
    self.conv32 = nn.Conv2d( 64, 64,3,1,0)
    self.conv31 = nn.Conv2d( 64, 32,3,1,0)
    self.conv22 = nn.Conv2d( 32, 32,3,1,0, dilation=1)
    self.conv21 = nn.Conv2d( 32, 16,3,1,0, dilation=1)
    self.conv12 = nn.Conv2d( 16, 16,3,1,0, dilation=1)
    self.conv11 = nn.Conv2d( 16,  3,3,1,0, dilation=1)

    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.unpool_pwct = nn.MaxUnpool2d(kernel_size=2, stride=2)
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
    y = self.relu(self.conv11(self.pad(y)))
    return y
    
  def forward_pwct(self, x, pool1_idx=None, pool1_size=None, pool2_idx=None, pool2_size=None, pool3_idx=None, pool3_size=None):
    out41 = self.relu(self.conv41(self.pad(x)))
    out41 = self.unpool_pwct(out41, pool3_idx, output_size=pool3_size)
    out34 = self.relu(self.conv34(self.pad(out41)))
    out33 = self.relu(self.conv33(self.pad(out34)))
    out32 = self.relu(self.conv32(self.pad(out33)))
    out31 = self.relu(self.conv31(self.pad(out32)))
    out31 = self.unpool_pwct(out31, pool2_idx, output_size=pool2_size)
    out22 = self.relu(self.conv22(self.pad(out31)))
    out21 = self.relu(self.conv21(self.pad(out22)))
    out21 = self.unpool_pwct(out21, pool1_idx, output_size=pool1_size)
    out12 = self.relu(self.conv12(self.pad(out21)))
    out11 = self.conv11(self.pad(out12))
    return out11
    
class SmallDecoder5_16x(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallDecoder5_16x, self).__init__()
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
    
  def forward_branch(self, input):
    out51 = self.relu(self.conv51(self.pad(input)))
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
    return out11

# bridge the dimension mismatch using a 1x1 linear layer
class SmallEncoder1_16x_aux(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallEncoder1_16x_aux, self).__init__()
    self.fixed = fixed
    
    self.conv0 = nn.Conv2d(3,3,1,1,0)
    self.conv0.requires_grad = False
    self.conv11     = nn.Conv2d(  3, 24, 3, 1, 0, dilation=1)
    self.conv11_aux = nn.Conv2d( 24, 64, 1, 1, 0)
    
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
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
  
  # "forward" only outputs the final output
  # "forward_branch" outputs all the middle branch ouputs
  # "forward_aux" outputs all the middle auxiliary mapping layers
  def forward(self, y):
    y = self.conv0(y)
    y = self.relu(self.conv11(self.pad(y)))
    return y
    
  def forward_branch(self, input):
    out0  = self.conv0(input)
    out11 = self.relu(self.conv11(self.pad(out0)))
    return out11,
  
  def forward_aux(self, input, relu=True):
    out0  = self.conv0(input)
    out11 = self.relu(self.conv11(self.pad(out0)))
    if relu:
      out11_aux = self.relu(self.conv11_aux(out11))
    else:
      out11_aux = self.conv11_aux(out11)
    return out11_aux,
    
  def forward_aux2(self, input):
    out0  = self.conv0(input)
    out11 = self.relu(self.conv11(self.pad(out0)))
    out11_aux = self.relu(self.conv11_aux(out11))
    return out11_aux, out11 # used for feature loss and style loss


class SmallEncoder2_16x_aux(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallEncoder2_16x_aux, self).__init__()
    self.fixed = fixed

    self.conv0 = nn.Conv2d(3,3,1,1,0)
    self.conv0.requires_grad = False
    self.conv11     = nn.Conv2d(  3, 16,3,1,0, dilation=1)
    self.conv12     = nn.Conv2d( 16, 16,3,1,0, dilation=1)
    self.conv21     = nn.Conv2d( 16, 32,3,1,0)
    
    self.conv11_aux = nn.Conv2d( 16, 64,1,1,0)
    self.conv21_aux = nn.Conv2d( 32,128,1,1,0)
    
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)
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
    y = self.conv0(y)
    y = self.relu(self.conv11(self.pad(y)))
    y = self.relu(self.conv12(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv21(self.pad(y)))
    return y
    
  def forward_branch(self, input):
    out0  = self.conv0(input)
    out11 = self.relu(self.conv11(self.pad(out0)))
    out12 = self.relu(self.conv12(self.pad(out11)))
    out12 = self.pool(out12)
    out21 = self.relu(self.conv21(self.pad(out12)))
    return out11, out21
  
  def forward_aux(self, input, relu=True):
    out0  = self.conv0(input)
    out11 = self.relu(self.conv11(self.pad(out0)))
    out12 = self.relu(self.conv12(self.pad(out11)))
    out12 = self.pool(out12)
    out21 = self.relu(self.conv21(self.pad(out12)))
    if relu:
      out11_aux = self.relu(self.conv11_aux(out11))
      out21_aux = self.relu(self.conv21_aux(out21))
    else:
      out11_aux = self.conv11_aux(out11)
      out21_aux = self.conv21_aux(out21)
    return out11_aux, out21_aux
    
  def forward_aux2(self, input):
    out0  = self.conv0(input)
    out11 = self.relu(self.conv11(self.pad(out0)))
    out12 = self.relu(self.conv12(self.pad(out11)))
    out12 = self.pool(out12)
    out21 = self.relu(self.conv21(self.pad(out12)))
    out11_aux = self.relu(self.conv11_aux(out11))
    out21_aux = self.relu(self.conv21_aux(out21))
    return out11_aux, out21_aux, out21 # used for feature loss and style loss
    
  def forward_pwct(self, input): # for function in photo WCT
    out0  = self.conv0(input)
    out11 = self.relu(self.conv11(self.pad(out0)))
    out12 = self.relu(self.conv12(self.pad(out11)))
    pool12, out12_ix = self.pool2(out12)
    out21 = self.relu(self.conv21(self.pad(pool12)))
    return out21, out12_ix, out12.size()
    
class SmallEncoder3_16x_aux(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallEncoder3_16x_aux, self).__init__()
    self.fixed = fixed

    self.conv0 = nn.Conv2d(3,3,1,1,0)
    self.conv0.requires_grad = False
    self.conv11     = nn.Conv2d(  3, 16,3,1,0, dilation=1)
    self.conv12     = nn.Conv2d( 16, 16,3,1,0, dilation=1)
    self.conv21     = nn.Conv2d( 16, 32,3,1,0)
    self.conv22     = nn.Conv2d( 32, 32,3,1,0)
    self.conv31     = nn.Conv2d( 32, 64,3,1,0)
    
    self.conv11_aux = nn.Conv2d( 16, 64,1,1,0)
    self.conv21_aux = nn.Conv2d( 32,128,1,1,0)
    self.conv31_aux = nn.Conv2d( 64,256,1,1,0)
    
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)
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
    y = self.conv0(y)
    y = self.relu(self.conv11(self.pad(y)))
    y = self.relu(self.conv12(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv21(self.pad(y)))
    y = self.relu(self.conv22(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv31(self.pad(y)))
    return y
    
  def forward_branch(self, input):
    out0  = self.conv0(input)
    out11 = self.relu(self.conv11(self.pad(out0)))
    out12 = self.relu(self.conv12(self.pad(out11)))
    out12 = self.pool(out12)
    out21 = self.relu(self.conv21(self.pad(out12)))
    out22 = self.relu(self.conv22(self.pad(out21)))
    out22 = self.pool(out22)
    out31 = self.relu(self.conv31(self.pad(out22)))
    return out11, out21, out31
  
  def forward_aux(self, input, relu=True):
    out0  = self.conv0(input)
    out11 = self.relu(self.conv11(self.pad(out0)))
    out12 = self.relu(self.conv12(self.pad(out11)))
    out12 = self.pool(out12)
    out21 = self.relu(self.conv21(self.pad(out12)))
    out22 = self.relu(self.conv22(self.pad(out21)))
    out22 = self.pool(out22)
    out31 = self.relu(self.conv31(self.pad(out22)))
    if relu:
      out11_aux = self.relu(self.conv11_aux(out11))
      out21_aux = self.relu(self.conv21_aux(out21))
      out31_aux = self.relu(self.conv31_aux(out31))
    else:
      out11_aux = self.conv11_aux(out11)
      out21_aux = self.conv21_aux(out21)
      out31_aux = self.conv31_aux(out31)
    return out11_aux, out21_aux, out31_aux

  def forward_aux2(self, input):
    out0  = self.conv0(input)
    out11 = self.relu(self.conv11(self.pad(out0)))
    out12 = self.relu(self.conv12(self.pad(out11)))
    out12 = self.pool(out12)
    out21 = self.relu(self.conv21(self.pad(out12)))
    out22 = self.relu(self.conv22(self.pad(out21)))
    out22 = self.pool(out22)
    out31 = self.relu(self.conv31(self.pad(out22)))
    out11_aux = self.relu(self.conv11_aux(out11))
    out21_aux = self.relu(self.conv21_aux(out21))
    out31_aux = self.relu(self.conv31_aux(out31))
    return out11_aux, out21_aux, out31_aux, out31 # used for feature loss and style loss
    
  def forward_pwct(self, input): # for function in photo WCT
    out0  = self.conv0(input)
    out11 = self.relu(self.conv11(self.pad(out0)))
    out12 = self.relu(self.conv12(self.pad(out11)))
    pool12, out12_ix = self.pool2(out12)
    out21 = self.relu(self.conv21(self.pad(pool12)))
    out22 = self.relu(self.conv22(self.pad(out21)))
    pool22, out22_ix = self.pool2(out22)
    out31 = self.relu(self.conv31(self.pad(pool22)))
    return out31, out12_ix, out12.size(), out22_ix, out22.size()

class SmallEncoder4_16x_aux(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallEncoder4_16x_aux, self).__init__()
    self.fixed = fixed

    self.conv0 = nn.Conv2d(3,3,1,1,0)
    self.conv0.requires_grad = False
    self.conv11     = nn.Conv2d(  3, 16,3,1,0, dilation=1)
    self.conv12     = nn.Conv2d( 16, 16,3,1,0, dilation=1)
    self.conv21     = nn.Conv2d( 16, 32,3,1,0, dilation=1)
    self.conv22     = nn.Conv2d( 32, 32,3,1,0, dilation=1)
    self.conv31     = nn.Conv2d( 32, 64,3,1,0)
    self.conv32     = nn.Conv2d( 64, 64,3,1,0)
    self.conv33     = nn.Conv2d( 64, 64,3,1,0)
    self.conv34     = nn.Conv2d( 64, 64,3,1,0)
    self.conv41     = nn.Conv2d( 64,128,3,1,0)
    
    self.conv11_aux = nn.Conv2d( 16, 64,1,1,0)
    self.conv21_aux = nn.Conv2d( 32,128,1,1,0)
    self.conv31_aux = nn.Conv2d( 64,256,1,1,0)
    self.conv41_aux = nn.Conv2d(128,512,1,1,0)
    
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)
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
    y = self.conv0(y)
    y = self.relu(self.conv11(self.pad(y)))
    y = self.relu(self.conv12(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv21(self.pad(y)))
    y = self.relu(self.conv22(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv31(self.pad(y)))
    y = self.relu(self.conv32(self.pad(y)))
    y = self.relu(self.conv33(self.pad(y)))
    y = self.relu(self.conv34(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv41(self.pad(y)))
    return y
    
  def forward_branch(self, input):
    out0  = self.conv0(input)
    out11 = self.relu(self.conv11(self.pad(out0)))
    out12 = self.relu(self.conv12(self.pad(out11)))
    out12 = self.pool(out12)
    out21 = self.relu(self.conv21(self.pad(out12)))
    out22 = self.relu(self.conv22(self.pad(out21)))
    out22 = self.pool(out22)
    out31 = self.relu(self.conv31(self.pad(out22)))
    out32 = self.relu(self.conv32(self.pad(out31)))
    out33 = self.relu(self.conv33(self.pad(out32)))
    out34 = self.relu(self.conv34(self.pad(out33)))
    out34 = self.pool(out34)
    out41 = self.relu(self.conv41(self.pad(out34)))
    return out11, out21, out31, out41

  def forward_pwct(self, input): # for function in photo WCT
    out0  = self.conv0(input)
    out11 = self.relu(self.conv11(self.pad(out0)))
    out12 = self.relu(self.conv12(self.pad(out11)))
    pool12, out12_ix = self.pool2(out12)
    out21 = self.relu(self.conv21(self.pad(pool12)))
    out22 = self.relu(self.conv22(self.pad(out21)))
    pool22, out22_ix = self.pool2(out22)
    out31 = self.relu(self.conv31(self.pad(pool22)))
    out32 = self.relu(self.conv32(self.pad(out31)))
    out33 = self.relu(self.conv33(self.pad(out32)))
    out34 = self.relu(self.conv34(self.pad(out33)))
    pool34, out34_ix = self.pool2(out34)
    out41 = self.relu(self.conv41(self.pad(pool34)))
    return out41, out12_ix, out12.size(), out22_ix, out22.size(), out34_ix, out34.size()
    
  def forward_aux(self, input, relu=True):
    out0  = self.conv0(input)
    out11 = self.relu(self.conv11(self.pad(out0)))
    out12 = self.relu(self.conv12(self.pad(out11)))
    out12 = self.pool(out12)
    out21 = self.relu(self.conv21(self.pad(out12)))
    out22 = self.relu(self.conv22(self.pad(out21)))
    out22 = self.pool(out22)
    out31 = self.relu(self.conv31(self.pad(out22)))
    out32 = self.relu(self.conv32(self.pad(out31)))
    out33 = self.relu(self.conv33(self.pad(out32)))
    out34 = self.relu(self.conv34(self.pad(out33)))
    out34 = self.pool(out34)
    out41 = self.relu(self.conv41(self.pad(out34)))
    if relu:
      out11_aux = self.relu(self.conv11_aux(out11))
      out21_aux = self.relu(self.conv21_aux(out21))
      out31_aux = self.relu(self.conv31_aux(out31))
      out41_aux = self.relu(self.conv41_aux(out41))
    else:
      out11_aux = self.conv11_aux(out11)
      out21_aux = self.conv21_aux(out21)
      out31_aux = self.conv31_aux(out31)
      out41_aux = self.conv41_aux(out41)
    return out11_aux, out21_aux, out31_aux, out41_aux

  def forward_aux2(self, input):
    out0  = self.conv0(input)
    out11 = self.relu(self.conv11(self.pad(out0)))
    out12 = self.relu(self.conv12(self.pad(out11)))
    out12 = self.pool(out12)
    out21 = self.relu(self.conv21(self.pad(out12)))
    out22 = self.relu(self.conv22(self.pad(out21)))
    out22 = self.pool(out22)
    out31 = self.relu(self.conv31(self.pad(out22)))
    out32 = self.relu(self.conv32(self.pad(out31)))
    out33 = self.relu(self.conv33(self.pad(out32)))
    out34 = self.relu(self.conv34(self.pad(out33)))
    out34 = self.pool(out34)
    out41 = self.relu(self.conv41(self.pad(out34)))
    out11_aux = self.relu(self.conv11_aux(out11))
    out21_aux = self.relu(self.conv21_aux(out21))
    out31_aux = self.relu(self.conv31_aux(out31))
    out41_aux = self.relu(self.conv41_aux(out41))
    return out11_aux, out21_aux, out31_aux, out41_aux, out41 # used for feature loss and style loss

class SmallEncoder5_16x_aux(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallEncoder5_16x_aux, self).__init__()
    self.fixed = fixed

    self.conv0 = nn.Conv2d(3,3,1,1,0)
    self.conv0.requires_grad = False
    self.conv11     = nn.Conv2d(  3, 16,3,1,0, dilation=1)
    self.conv12     = nn.Conv2d( 16, 16,3,1,0, dilation=1)
    self.conv21     = nn.Conv2d( 16, 32,3,1,0, dilation=1)
    self.conv22     = nn.Conv2d( 32, 32,3,1,0, dilation=1)
    self.conv31     = nn.Conv2d( 32, 64,3,1,0, dilation=1)
    self.conv32     = nn.Conv2d( 64, 64,3,1,0, dilation=1)
    self.conv33     = nn.Conv2d( 64, 64,3,1,0, dilation=1)
    self.conv34     = nn.Conv2d( 64, 64,3,1,0, dilation=1)
    self.conv41     = nn.Conv2d( 64,128,3,1,0)
    self.conv42     = nn.Conv2d(128,128,3,1,0)
    self.conv43     = nn.Conv2d(128,128,3,1,0)
    self.conv44     = nn.Conv2d(128,128,3,1,0)
    self.conv51     = nn.Conv2d(128,128,3,1,0)
    self.conv11_aux = nn.Conv2d( 16, 64,1,1,0)
    self.conv21_aux = nn.Conv2d( 32,128,1,1,0)
    self.conv31_aux = nn.Conv2d( 64,256,1,1,0)
    self.conv41_aux = nn.Conv2d(128,512,1,1,0)
    self.conv51_aux = nn.Conv2d(128,512,1,1,0)
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
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
    y = self.conv0(y)
    y = self.relu(self.conv11(self.pad(y)))
    y = self.relu(self.conv12(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv21(self.pad(y)))
    y = self.relu(self.conv22(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv31(self.pad(y)))
    y = self.relu(self.conv32(self.pad(y)))
    y = self.relu(self.conv33(self.pad(y)))
    y = self.relu(self.conv34(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv41(self.pad(y)))
    y = self.relu(self.conv42(self.pad(y)))
    y = self.relu(self.conv43(self.pad(y)))
    y = self.relu(self.conv44(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv51(self.pad(y)))
    return y
    
  def forward_branch(self, input):
    out0  = self.conv0(input)
    out11 = self.relu(self.conv11(self.pad(out0)))
    out12 = self.relu(self.conv12(self.pad(out11)))
    out12 = self.pool(out12)
    out21 = self.relu(self.conv21(self.pad(out12)))
    out22 = self.relu(self.conv22(self.pad(out21)))
    out22 = self.pool(out22)
    out31 = self.relu(self.conv31(self.pad(out22)))
    out32 = self.relu(self.conv32(self.pad(out31)))
    out33 = self.relu(self.conv33(self.pad(out32)))
    out34 = self.relu(self.conv34(self.pad(out33)))
    out34 = self.pool(out34)
    out41 = self.relu(self.conv41(self.pad(out34)))
    out42 = self.relu(self.conv42(self.pad(out41)))
    out43 = self.relu(self.conv43(self.pad(out42)))
    out44 = self.relu(self.conv44(self.pad(out43)))
    out44 = self.pool(out44)
    out51 = self.relu(self.conv51(self.pad(out44)))
    return out11, out21, out31, out41, out51
  
  def forward_aux(self, input, relu=True):
    out0  = self.conv0(input)
    out11 = self.relu(self.conv11(self.pad(out0)))
    out12 = self.relu(self.conv12(self.pad(out11)))
    out12 = self.pool(out12)
    out21 = self.relu(self.conv21(self.pad(out12)))
    out22 = self.relu(self.conv22(self.pad(out21)))
    out22 = self.pool(out22)
    out31 = self.relu(self.conv31(self.pad(out22)))
    out32 = self.relu(self.conv32(self.pad(out31)))
    out33 = self.relu(self.conv33(self.pad(out32)))
    out34 = self.relu(self.conv34(self.pad(out33)))
    out34 = self.pool(out34)
    out41 = self.relu(self.conv41(self.pad(out34)))
    out42 = self.relu(self.conv42(self.pad(out41)))
    out43 = self.relu(self.conv43(self.pad(out42)))
    out44 = self.relu(self.conv44(self.pad(out43)))
    out44 = self.pool(out44)
    out51 = self.relu(self.conv51(self.pad(out44)))
    if relu:
      out11_aux = self.relu(self.conv11_aux(out11))
      out21_aux = self.relu(self.conv21_aux(out21))
      out31_aux = self.relu(self.conv31_aux(out31))
      out41_aux = self.relu(self.conv41_aux(out41))
      out51_aux = self.relu(self.conv51_aux(out51))
    else:
      out11_aux = self.conv11_aux(out11)
      out21_aux = self.conv21_aux(out21)
      out31_aux = self.conv31_aux(out31)
      out41_aux = self.conv41_aux(out41)
      out51_aux = self.conv51_aux(out51)
    return out11_aux, out21_aux, out31_aux, out41_aux, out51_aux

  def forward_aux2(self, input):
    out0  = self.conv0(input)
    out11 = self.relu(self.conv11(self.pad(out0)))
    out12 = self.relu(self.conv12(self.pad(out11)))
    out12 = self.pool(out12)
    out21 = self.relu(self.conv21(self.pad(out12)))
    out22 = self.relu(self.conv22(self.pad(out21)))
    out22 = self.pool(out22)
    out31 = self.relu(self.conv31(self.pad(out22)))
    out32 = self.relu(self.conv32(self.pad(out31)))
    out33 = self.relu(self.conv33(self.pad(out32)))
    out34 = self.relu(self.conv34(self.pad(out33)))
    out34 = self.pool(out34)
    out41 = self.relu(self.conv41(self.pad(out34)))
    out42 = self.relu(self.conv42(self.pad(out41)))
    out43 = self.relu(self.conv43(self.pad(out42)))
    out44 = self.relu(self.conv44(self.pad(out43)))
    out44 = self.pool(out44)
    out51 = self.relu(self.conv51(self.pad(out44)))
    out11_aux = self.relu(self.conv11_aux(out11))
    out21_aux = self.relu(self.conv21_aux(out21))
    out31_aux = self.relu(self.conv31_aux(out31))
    out41_aux = self.relu(self.conv41_aux(out41))
    out51_aux = self.relu(self.conv51_aux(out51))
    return out11_aux, out21_aux, out31_aux, out41_aux, out51_aux, out51 # output out51

  def forward_aux3(self, input, relu=False):
    out0  = self.conv0(input)
    out11 = self.relu(self.conv11(self.pad(out0)))
    out12 = self.relu(self.conv12(self.pad(out11)))
    out12 = self.pool(out12)
    out21 = self.relu(self.conv21(self.pad(out12)))
    out22 = self.relu(self.conv22(self.pad(out21)))
    out22 = self.pool(out22)
    out31 = self.relu(self.conv31(self.pad(out22)))
    out32 = self.relu(self.conv32(self.pad(out31)))
    out33 = self.relu(self.conv33(self.pad(out32)))
    out34 = self.relu(self.conv34(self.pad(out33)))
    out34 = self.pool(out34)
    out41 = self.relu(self.conv41(self.pad(out34)))
    out42 = self.relu(self.conv42(self.pad(out41)))
    out43 = self.relu(self.conv43(self.pad(out42)))
    out44 = self.relu(self.conv44(self.pad(out43)))
    out44 = self.pool(out44)
    out51 = self.relu(self.conv51(self.pad(out44)))
    if relu:
      out51_aux = self.relu(self.conv51_aux(out51))
    else:
      out51_aux = self.conv51_aux(out51)
    return out11, out21, out31, out41, out51, out51_aux