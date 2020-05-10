import numpy as np
import os
import torch.nn as nn
import torch
from torch.utils.serialization import load_lua
from utils import load_param_from_t7 as load_param
pjoin = os.path.join

# Original VGG19
# Encoder1/Decoder1
class Encoder1(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Encoder1, self).__init__()
    self.fixed = fixed

    self.conv0  = nn.Conv2d(  3,  3,1,1,0)
    self.conv11 = nn.Conv2d(  3, 64,3,1,0, dilation=1)
    
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      assert(os.path.splitext(model)[1] in {".t7", ".pth"})
      if model.endswith(".t7"):
        t7_model = load_lua(model)
        load_param(t7_model, 0,  self.conv0)
        load_param(t7_model, 2,  self.conv11)
      else:
        self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
      
    if fixed:
      for param in self.parameters():
          param.requires_grad = False

  def forward(self, input):
    y = self.conv0(input)
    y = self.relu(self.conv11(self.pad(y)))
    return y
  def forward_branch(self, input):
    out0  = self.conv0(input)
    out11 = self.relu(self.conv11(self.pad(out0)))
    return out11,

class Decoder1(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Decoder1, self).__init__()
    self.fixed = fixed
    
    self.conv11 = nn.Conv2d( 64,  3,3,1,0, dilation=1)
    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.pad = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      assert(os.path.splitext(model)[1] in {".t7", ".pth"})
      if model.endswith(".t7"):
        t7_model = load_lua(model)
        load_param(t7_model, 1, self.conv11)
      else:
        self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
    
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
  def forward(self, input):
    y = self.relu(self.conv11(self.pad(input)))
    return y
  
  def forward_branch(self, input):
    out11 = self.relu(self.conv11(self.pad(input)))
    return out11,

# Encoder2/Decoder2
class Encoder2(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Encoder2, self).__init__()
    self.fixed = fixed

    self.conv0  = nn.Conv2d(  3,  3,1,1,0)
    self.conv11 = nn.Conv2d(  3, 64,3,1,0, dilation=1)
    self.conv12 = nn.Conv2d( 64, 64,3,1,0, dilation=1)
    self.conv21 = nn.Conv2d( 64,128,3,1,0)
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      assert(os.path.splitext(model)[1] in {".t7", ".pth"})
      if model.endswith(".t7"):
        t7_model = load_lua(model)
        load_param(t7_model, 0,  self.conv0)
        load_param(t7_model, 2,  self.conv11)
        load_param(t7_model, 5,  self.conv12)
        load_param(t7_model, 9,  self.conv21)
      else:
        self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
      
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
          
  def forward(self, input):
    y = self.conv0(input)
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

class Decoder2(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Decoder2, self).__init__()
    self.fixed = fixed
    
    self.conv21 = nn.Conv2d(128, 64,3,1,0)
    self.conv12 = nn.Conv2d( 64, 64,3,1,0, dilation=1)
    self.conv11 = nn.Conv2d( 64,  3,3,1,0, dilation=1)

    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.pad = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      assert(os.path.splitext(model)[1] in {".t7", ".pth"})
      if model.endswith(".t7"):
        t7_model = load_lua(model)
        load_param(t7_model, 1, self.conv21)
        load_param(t7_model, 5, self.conv12)
        load_param(t7_model, 8, self.conv11)
      else:
        self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
    
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
     
  def forward(self, input):
    y = self.relu(self.conv21(self.pad(input)))
    y = self.unpool(y)
    y = self.relu(self.conv12(self.pad(y)))
    y = self.relu(self.conv11(self.pad(y)))
    return y
  def forward_branch(self, input):
    out21 = self.relu(self.conv21(self.pad(input)))
    out21 = self.unpool(out21)
    out12 = self.relu(self.conv12(self.pad(out21)))
    out11 = self.relu(self.conv11(self.pad(out12)))
    return out21, out11
    
# Encoder3/Decoder3
class Encoder3(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Encoder3, self).__init__()
    self.fixed = fixed

    self.conv0  = nn.Conv2d(  3,  3,1,1,0)
    self.conv11 = nn.Conv2d(  3, 64,3,1,0) # conv1_1
    self.conv12 = nn.Conv2d( 64, 64,3,1,0) # conv1_2
    self.conv21 = nn.Conv2d( 64,128,3,1,0) # conv2_1
    self.conv22 = nn.Conv2d(128,128,3,1,0) # conv2_2
    self.conv31 = nn.Conv2d(128,256,3,1,0) # conv3_1
    
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      assert(os.path.splitext(model)[1] in {".t7", ".pth"})
      if model.endswith(".t7"):
        t7_model = load_lua(model)
        load_param(t7_model, 0,  self.conv0)
        load_param(t7_model, 2,  self.conv11)
        load_param(t7_model, 5,  self.conv12)
        load_param(t7_model, 9,  self.conv21)
        load_param(t7_model, 12, self.conv22)
        load_param(t7_model, 16, self.conv31)
      else:
        self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
      
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
          
  def forward(self, input):
    y = self.conv0(input)
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

class Decoder3(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Decoder3, self).__init__()
    self.fixed = fixed
    
    self.conv31 = nn.Conv2d(256,128,3,1,0)
    self.conv22 = nn.Conv2d(128,128,3,1,0)
    self.conv21 = nn.Conv2d(128, 64,3,1,0)
    self.conv12 = nn.Conv2d( 64, 64,3,1,0)
    self.conv11 = nn.Conv2d( 64,  3,3,1,0)
    
    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.pad = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      assert(os.path.splitext(model)[1] in {".t7", ".pth"})
      if model.endswith(".t7"):
        t7_model = load_lua(model)
        load_param(t7_model,  1, self.conv31)
        load_param(t7_model,  5, self.conv22)
        load_param(t7_model,  8, self.conv21)
        load_param(t7_model, 12, self.conv12)
        load_param(t7_model, 15, self.conv11)
      else:
        self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
    
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
     
  def forward(self, input):
    y = self.relu(self.conv31(self.pad(input)))
    y = self.unpool(y)
    y = self.relu(self.conv22(self.pad(y)))
    y = self.relu(self.conv21(self.pad(y)))
    y = self.unpool(y)
    y = self.relu(self.conv12(self.pad(y)))
    y = self.relu(self.conv11(self.pad(y)))
    return y
  def forward_branch(self, input):
    out31 = self.relu(self.conv31(self.pad(input)))
    out31 = self.unpool(out31)
    out22 = self.relu(self.conv22(self.pad(out31)))
    out21 = self.relu(self.conv21(self.pad(out22)))
    out21 = self.unpool(out21)
    out12 = self.relu(self.conv12(self.pad(out21)))
    out11 = self.relu(self.conv11(self.pad(out12)))
    return out31, out21, out11

# Encoder4/Decoder4
class Encoder4(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Encoder4, self).__init__()
    self.fixed = fixed

    self.conv0  = nn.Conv2d(  3,  3,1,1,0)
    self.conv11 = nn.Conv2d(  3, 64,3,1,0) # conv1_1
    self.conv12 = nn.Conv2d( 64, 64,3,1,0) # conv1_2
    self.conv21 = nn.Conv2d( 64,128,3,1,0) # conv2_1
    self.conv22 = nn.Conv2d(128,128,3,1,0) # conv2_2
    self.conv31 = nn.Conv2d(128,256,3,1,0) # conv3_1
    self.conv32 = nn.Conv2d(256,256,3,1,0) # conv3_2
    self.conv33 = nn.Conv2d(256,256,3,1,0) # conv3_3
    self.conv34 = nn.Conv2d(256,256,3,1,0) # conv3_4
    self.conv41 = nn.Conv2d(256,512,3,1,0) # conv4_1
    
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      assert(os.path.splitext(model)[1] in {".t7", ".pth"})
      if model.endswith(".t7"):
        t7_model = load_lua(model)
        load_param(t7_model, 0,  self.conv0)
        load_param(t7_model, 2,  self.conv11)
        load_param(t7_model, 5,  self.conv12)
        load_param(t7_model, 9,  self.conv21)
        load_param(t7_model, 12, self.conv22)
        load_param(t7_model, 16, self.conv31)
        load_param(t7_model, 19, self.conv32)
        load_param(t7_model, 22, self.conv33)
        load_param(t7_model, 25, self.conv34)
        load_param(t7_model, 29, self.conv41)
      else:
        self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
      
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
          
  def forward(self, input):
    y = self.conv0(input)
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
    
class Decoder4(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Decoder4, self).__init__()
    self.fixed = fixed

    self.conv41 = nn.Conv2d(512,256,3,1,0)
    self.conv34 = nn.Conv2d(256,256,3,1,0)
    self.conv33 = nn.Conv2d(256,256,3,1,0)
    self.conv32 = nn.Conv2d(256,256,3,1,0)
    self.conv31 = nn.Conv2d(256,128,3,1,0)
    self.conv22 = nn.Conv2d(128,128,3,1,0)
    self.conv21 = nn.Conv2d(128, 64,3,1,0)
    self.conv12 = nn.Conv2d( 64, 64,3,1,0)
    self.conv11 = nn.Conv2d( 64,  3,3,1,0)
    
    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.pad = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      assert(os.path.splitext(model)[1] in {".t7", ".pth"})
      if model.endswith(".t7"):
        t7_model = load_lua(model)
        load_param(t7_model,  1, self.conv41)
        load_param(t7_model,  5, self.conv34)
        load_param(t7_model,  8, self.conv33)
        load_param(t7_model, 11, self.conv32)
        load_param(t7_model, 14, self.conv31)
        load_param(t7_model, 18, self.conv22)
        load_param(t7_model, 21, self.conv21)
        load_param(t7_model, 25, self.conv12)
        load_param(t7_model, 28, self.conv11)
      else:
        self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
    
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
     
  def forward(self, input):
    y = self.relu(self.conv41(self.pad(input)))
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
    
  def forward_norule(self, input):
    y = self.relu(self.conv41(self.pad(input)))
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
    y = self.conv11(self.pad(y))
    return y
  
  def forward_branch(self, input):
    out41 = self.relu(self.conv41(self.pad(input)))
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
    return out41, out31, out21, out11
    
# Encoder5/Decoder5
class Encoder5(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Encoder5, self).__init__()
    self.fixed = fixed

    self.conv0  = nn.Conv2d(  3,  3,1,1,0)
    self.conv0.weight = nn.Parameter(torch.from_numpy(np.array(
                                    [[[[0]],[[0]],[[255]]],
                                     [[[0]],[[255]],[[0]]],
                                     [[[255]],[[0]],[[0]]]])).float())
    self.conv0.bias = nn.Parameter(torch.from_numpy(np.array(
                                    [-103.939,-116.779,-123.68])).float())
    self.conv11 = nn.Conv2d(  3, 64,3,1,0) # conv1_1
    self.conv12 = nn.Conv2d( 64, 64,3,1,0) # conv1_2
    self.conv21 = nn.Conv2d( 64,128,3,1,0) # conv2_1
    self.conv22 = nn.Conv2d(128,128,3,1,0) # conv2_2
    self.conv31 = nn.Conv2d(128,256,3,1,0) # conv3_1
    self.conv32 = nn.Conv2d(256,256,3,1,0) # conv3_2
    self.conv33 = nn.Conv2d(256,256,3,1,0) # conv3_3
    self.conv34 = nn.Conv2d(256,256,3,1,0) # conv3_4
    self.conv41 = nn.Conv2d(256,512,3,1,0) # conv4_1
    self.conv42 = nn.Conv2d(512,512,3,1,0) # conv4_2
    self.conv43 = nn.Conv2d(512,512,3,1,0) # conv4_3
    self.conv44 = nn.Conv2d(512,512,3,1,0) # conv4_4
    self.conv51 = nn.Conv2d(512,512,3,1,0) # conv5_1
    
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      assert(os.path.splitext(model)[1] in {".t7", ".pth"})
      if model.endswith(".t7"):
        t7_model = load_lua(model)
        ## load from my normalised t7 model, which has no conv0 layer.
        # load_param(t7_model, 0,  self.conv11)
        # load_param(t7_model, 2,  self.conv12)
        # load_param(t7_model, 5,  self.conv21)
        # load_param(t7_model, 7,  self.conv22)
        # load_param(t7_model, 10, self.conv31)
        # load_param(t7_model, 12, self.conv32)
        # load_param(t7_model, 14, self.conv33)
        # load_param(t7_model, 16, self.conv34)
        # load_param(t7_model, 19, self.conv41)
        # load_param(t7_model, 21, self.conv42)
        # load_param(t7_model, 23, self.conv43)
        # load_param(t7_model, 25, self.conv44)
        # load_param(t7_model, 28, self.conv51)
        # load_param(t7_model, 30, self.conv52)
        load_param(t7_model, 0,  self.conv0)
        load_param(t7_model, 2,  self.conv11)
        load_param(t7_model, 5,  self.conv12)
        load_param(t7_model, 9,  self.conv21)
        load_param(t7_model, 12, self.conv22)
        load_param(t7_model, 16, self.conv31)
        load_param(t7_model, 19, self.conv32)
        load_param(t7_model, 22, self.conv33)
        load_param(t7_model, 25, self.conv34)
        load_param(t7_model, 29, self.conv41)
        load_param(t7_model, 32, self.conv42)
        load_param(t7_model, 35, self.conv43)
        load_param(t7_model, 38, self.conv44)
        load_param(t7_model, 42, self.conv51)
      else:
        self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
      
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
          
  def forward(self, input):
    y = self.conv0(input)
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

class Decoder5(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Decoder5, self).__init__()
    self.fixed = fixed
    
    self.conv51 = nn.Conv2d(512,512,3,1,0)
    self.conv44 = nn.Conv2d(512,512,3,1,0)
    self.conv43 = nn.Conv2d(512,512,3,1,0)
    self.conv42 = nn.Conv2d(512,512,3,1,0)
    self.conv41 = nn.Conv2d(512,256,3,1,0)
    self.conv34 = nn.Conv2d(256,256,3,1,0)
    self.conv33 = nn.Conv2d(256,256,3,1,0)
    self.conv32 = nn.Conv2d(256,256,3,1,0)
    self.conv31 = nn.Conv2d(256,128,3,1,0)
    self.conv22 = nn.Conv2d(128,128,3,1,0)
    self.conv21 = nn.Conv2d(128, 64,3,1,0)
    self.conv12 = nn.Conv2d( 64, 64,3,1,0)
    self.conv11 = nn.Conv2d( 64,  3,3,1,0)

    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.pad = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      assert(os.path.splitext(model)[1] in {".t7", ".pth"})
      if model.endswith(".t7"):
        t7_model = load_lua(model)
        load_param(t7_model, 1,  self.conv51)
        load_param(t7_model, 5,  self.conv44)
        load_param(t7_model, 8,  self.conv43)
        load_param(t7_model, 11, self.conv42)
        load_param(t7_model, 14, self.conv41)
        load_param(t7_model, 18, self.conv34)
        load_param(t7_model, 21, self.conv33)
        load_param(t7_model, 24, self.conv32)
        load_param(t7_model, 27, self.conv31)
        load_param(t7_model, 31, self.conv22)
        load_param(t7_model, 34, self.conv21)
        load_param(t7_model, 38, self.conv12)
        load_param(t7_model, 41, self.conv11)
      else:
        self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
    
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
     
  def forward(self, input):
    y = self.relu(self.conv51(self.pad(input)))
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
    y = self.relu(self.conv11(self.pad(y)))
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
    return out51, out41, out31, out21, out11