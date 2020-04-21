import numpy as np
import os
import torch.nn as nn
import torch
from torch.utils.serialization import load_lua
from my_utils import load_param_from_t7 as load_param
from model_kd2sd import SmallDecoder1_16x_aux, SmallDecoder2_16x_aux, SmallDecoder3_16x_aux, SmallDecoder4_16x_aux, SmallDecoder5_16x_aux
from model_adain import vgg, decoder
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
  
# ---------------------------------------------------
# Original VGG19 Encoder/Decoder
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
        print("Given torch model, saving pytorch model")
        torch.save(self.state_dict(), os.path.splitext(model)[0] + ".pth")
        print("Saving done")
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

class Encoder4_2(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Encoder4_2, self).__init__()
    self.vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1 # 3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2 # 6
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1 # 10
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2 # 13
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1 # 17
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2 # 20
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3 # 23
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4 # 26
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1 # 30
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-2 # 33
    )
    if model:
      assert(os.path.splitext(model)[1] in {".t7", ".pth", ".pth.tar"})
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
        load_param(t7_model, 32, self.vgg[32])
      else: # pth model
        net = torch.load(model)
        odict_keys = list(net.keys())
        cnt = 0; i = 0
        for m in self.vgg.children():
          if isinstance(m, nn.Conv2d):
            print("layer %s is loaded with trained params" % i)
            m.weight.data.copy_(net[odict_keys[cnt]]); cnt += 1
            m.bias.data.copy_(net[odict_keys[cnt]]); cnt += 1
          i += 1
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
          
  def forward(self, x):
    return self.vgg(x)
    
  def forward_branch(self, x):
    y_relu1_1 = self.vgg[  : 4](x)
    y_relu2_1 = self.vgg[ 4:11](y_relu1_1)
    y_relu3_1 = self.vgg[11:18](y_relu2_1)
    y_relu4_1 = self.vgg[18:31](y_relu3_1)
    y_relu4_2 = self.vgg[31:  ](y_relu4_1)
    return y_relu1_1, y_relu2_1, y_relu3_1, y_relu4_1, y_relu4_2

class Decoder4_2(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Decoder4_2, self).__init__()
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
      assert(os.path.splitext(model)[1] in {".t7", ".pth", ".tar"})
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
     
  def forward(self, y):
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

class SmallEncoder4_2(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallEncoder4_2, self).__init__()
    self.vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)), # 0
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
    # self.updim11 = nn.Conv2d( 16,  64, (1, 1))
    # self.updim21 = nn.Conv2d( 32, 128, (1, 1))
    # self.updim31 = nn.Conv2d( 64, 256, (1, 1))
    # self.updim41 = nn.Conv2d(128, 512, (1, 1))
    self.relu = nn.ReLU()
    
    if model:
      state_dict = torch.load(model, map_location=lambda storage, location: storage)
      if "model" in state_dict: #and self.hasattr("vgg"):
        self.load_state_dict(state_dict["model"])
      else: # Init the model with previous base model params
        keys = list(state_dict.keys())
        ix = 0
        for m in self.vgg.children():
          if isinstance(m, nn.Conv2d):
            if m.weight.data.size(0) == 512: continue # conv4_2, do not init it with base model params because the dimension does not match
            w_name = keys[ix]; ix += 1
            b_name = keys[ix]; ix += 1
            m.weight.data.copy_(state_dict[w_name])
            m.bias.data.copy_(state_dict[b_name])
        self.vgg[0].weight.requires_grad = False # conv0 does not need update
        self.vgg[0].bias.requires_grad = False # conv0 does not need update
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
  def forward(self, x):
    return self.vgg(x)
  def forward_branch(self, x):
    y_relu1_1 = self.vgg[  : 4](x)
    y_relu2_1 = self.vgg[ 4:11](y_relu1_1)
    y_relu3_1 = self.vgg[11:18](y_relu2_1)
    y_relu4_1 = self.vgg[18:31](y_relu3_1)
    y_relu4_2 = self.vgg[31:  ](y_relu4_1)
    return y_relu1_1, y_relu2_1, y_relu3_1, y_relu4_1, y_relu4_2
  def forward_aux(self, x, relu=True): # increase the dimension
    if relu:
      y_relu1_1 = self.vgg[  : 4](x);         y_relu1_1_updim = self.relu(self.updim11(y_relu1_1))
      y_relu2_1 = self.vgg[ 4:11](y_relu1_1); y_relu2_1_updim = self.relu(self.updim21(y_relu2_1))
      y_relu3_1 = self.vgg[11:18](y_relu2_1); y_relu3_1_updim = self.relu(self.updim31(y_relu3_1))
      y_relu4_1 = self.vgg[18:31](y_relu3_1); y_relu4_1_updim = self.relu(self.updim41(y_relu4_1))
      y_relu4_2 = self.vgg[31:  ](y_relu4_1);
    else:
      y_relu1_1 = self.vgg[  : 4](x);         y_relu1_1_updim = self.updim11(y_relu1_1)
      y_relu2_1 = self.vgg[ 4:11](y_relu1_1); y_relu2_1_updim = self.updim21(y_relu2_1)
      y_relu3_1 = self.vgg[11:18](y_relu2_1); y_relu3_1_updim = self.updim31(y_relu3_1)
      y_relu4_1 = self.vgg[18:31](y_relu3_1); y_relu4_1_updim = self.updim41(y_relu4_1)
      y_relu4_2 = self.vgg[31:  ](y_relu4_1);
    return y_relu1_1_updim, y_relu2_1_updim, y_relu3_1_updim, y_relu4_1_updim, y_relu4_2

class SmallEncoder4_2_4x(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallEncoder4_2_4x, self).__init__()
    self.vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)), # 0
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 32, (3, 3)),
        nn.ReLU(),  # relu1-1 # 3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(32, 32, (3, 3)),
        nn.ReLU(),  # relu1-2 # 6
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(32, 64, (3, 3)),
        nn.ReLU(),  # relu2-1 # 10
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu2-2 # 13
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu3-1 # 17
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu3-2 # 20
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu3-3 # 23
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu3-4 # 26
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu4-1 # 30
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-2 # 33
    )
    self.updim11 = nn.Conv2d( 32,  64, (1, 1))
    self.updim21 = nn.Conv2d( 64, 128, (1, 1))
    self.updim31 = nn.Conv2d(128, 256, (1, 1))
    self.updim41 = nn.Conv2d(256, 512, (1, 1))
    self.relu = nn.ReLU()
    
    if model:
      state_dict = torch.load(model, map_location=lambda storage, location: storage)
      if "model" in state_dict: #and self.hasattr("vgg"):
        self.load_state_dict(state_dict["model"])
      else: # Init the model with previous base model params
        keys = list(state_dict.keys())
        ix = 0
        for m in self.vgg.children():
          if isinstance(m, nn.Conv2d):
            if m.weight.data.size(0) == 512: continue # conv4_2, do not init it with base model params because the dimension does not match
            w_name = keys[ix]; ix += 1
            b_name = keys[ix]; ix += 1
            m.weight.data.copy_(state_dict[w_name])
            m.bias.data.copy_(state_dict[b_name])
        self.vgg[0].weight.requires_grad = False # conv0 does not need update
        self.vgg[0].bias.requires_grad = False # conv0 does not need update
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
  def forward(self, x):
    return self.vgg(x)
  def forward_branch(self, x):
    y_relu1_1 = self.vgg[  : 4](x)
    y_relu2_1 = self.vgg[ 4:11](y_relu1_1)
    y_relu3_1 = self.vgg[11:18](y_relu2_1)
    y_relu4_1 = self.vgg[18:31](y_relu3_1)
    y_relu4_2 = self.vgg[31:  ](y_relu4_1)
    return y_relu1_1, y_relu2_1, y_relu3_1, y_relu4_1, y_relu4_2
  def forward_aux(self, x, relu=True): # increase the dimension
    if relu:
      y_relu1_1 = self.vgg[  : 4](x);         y_relu1_1_updim = self.relu(self.updim11(y_relu1_1))
      y_relu2_1 = self.vgg[ 4:11](y_relu1_1); y_relu2_1_updim = self.relu(self.updim21(y_relu2_1))
      y_relu3_1 = self.vgg[11:18](y_relu2_1); y_relu3_1_updim = self.relu(self.updim31(y_relu3_1))
      y_relu4_1 = self.vgg[18:31](y_relu3_1); y_relu4_1_updim = self.relu(self.updim41(y_relu4_1))
      y_relu4_2 = self.vgg[31:  ](y_relu4_1);
    else:
      y_relu1_1 = self.vgg[  : 4](x);         y_relu1_1_updim = self.updim11(y_relu1_1)
      y_relu2_1 = self.vgg[ 4:11](y_relu1_1); y_relu2_1_updim = self.updim21(y_relu2_1)
      y_relu3_1 = self.vgg[11:18](y_relu2_1); y_relu3_1_updim = self.updim31(y_relu3_1)
      y_relu4_1 = self.vgg[18:31](y_relu3_1); y_relu4_1_updim = self.updim41(y_relu4_1)
      y_relu4_2 = self.vgg[31:  ](y_relu4_1);
    return y_relu1_1_updim, y_relu2_1_updim, y_relu3_1_updim, y_relu4_1_updim, y_relu4_2
    
SmallEncoder4_2_16x = SmallEncoder4_2

class SmallEncoder4_2_64x(nn.Module):
  def __init__(self, model=None):
    super(SmallEncoder4_2_64x, self).__init__()
    self.vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3,  8, (3, 3)),
        nn.ReLU(),  # relu1-1 # 3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d( 8,  8, (3, 3)),
        nn.ReLU(),  # relu1-2 # 6
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d( 8, 16, (3, 3)),
        nn.ReLU(),  # relu2-1 # 10
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(16, 16, (3, 3)),
        nn.ReLU(),  # relu2-2 # 13
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(16, 32, (3, 3)),
        nn.ReLU(),  # relu3-1 # 17
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(32, 32, (3, 3)),
        nn.ReLU(),  # relu3-2 # 20
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(32, 32, (3, 3)),
        nn.ReLU(),  # relu3-3 # 23
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(32, 32, (3, 3)),
        nn.ReLU(),  # relu3-4 # 26
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(32, 64, (3, 3)),
        nn.ReLU(),  # relu4-1 # 30
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 512, (3, 3)),
        nn.ReLU(),  # relu4-2 # 33
    )
    if model:
      state_dict = torch.load(model, map_location=lambda storage, location: storage)
      if "model" in state_dict:
        self.load_state_dict(state_dict["model"])
      else:
        self.load_state_dict(state_dict)
  def forward(self, x):
    return self.vgg(x)
    
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
        # print("Given torch model, saving pytorch model")
        # torch.save(self.state_dict(), os.path.splitext(model)[0] + ".pth")
        # print("Saving done")
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
  
  # To get the style distance, save the gram matrix here
  def forward_style_similarity(self, y, save_format, outf="./"):
    y = self.conv0(y)
    y = self.relu(self.conv11(self.pad(y))); np.save(pjoin(outf, save_format % 1), gram_matrix(y)[0].cpu().data.numpy())
    y = self.relu(self.conv12(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv21(self.pad(y))); np.save(pjoin(outf, save_format % 2), gram_matrix(y)[0].cpu().data.numpy())
    y = self.relu(self.conv22(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv31(self.pad(y))); np.save(pjoin(outf, save_format % 3), gram_matrix(y)[0].cpu().data.numpy())
    y = self.relu(self.conv32(self.pad(y)))
    y = self.relu(self.conv33(self.pad(y)))
    y = self.relu(self.conv34(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv41(self.pad(y))); np.save(pjoin(outf, save_format % 4), gram_matrix(y)[0].cpu().data.numpy())
    y = self.relu(self.conv42(self.pad(y)))
    y = self.relu(self.conv43(self.pad(y)))
    y = self.relu(self.conv44(self.pad(y)))
    y = self.pool(y)
    y = self.relu(self.conv51(self.pad(y))); np.save(pjoin(outf, save_format % 5), gram_matrix(y)[0].cpu().data.numpy())
    return
    
class Encoder5_Gatys(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Encoder5_Gatys, self).__init__()
    self.fixed = fixed

    self.conv0  = nn.Conv2d(  3,  3,1,1,0)
    self.conv0.weight = nn.Parameter(torch.from_numpy(np.array(
                                    [[[[0]],[[0]],[[255]]],
                                     [[[0]],[[255]],[[0]]],
                                     [[[255]],[[0]],[[0]]]])).float())
    self.conv0.bias = nn.Parameter(torch.from_numpy(np.array(
                                    [-103.939,-116.779,-123.68])).float())
    self.pad11  = nn.ReflectionPad2d((1,1,1,1))
    self.conv11 = nn.Conv2d(  3, 64,3,1,0)
    self.relu11 = nn.ReLU(inplace=True)
    
    self.pad12  = nn.ReflectionPad2d((1,1,1,1))
    self.conv12 = nn.Conv2d( 64, 64,3,1,0)
    self.relu12 = nn.ReLU(inplace=True)
    
    self.pool12 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    
    self.pad21  = nn.ReflectionPad2d((1,1,1,1))
    self.conv21 = nn.Conv2d( 64,128,3,1,0)
    self.relu21 = nn.ReLU(inplace=True)
    
    self.pad22  = nn.ReflectionPad2d((1,1,1,1))
    self.conv22 = nn.Conv2d(128,128,3,1,0)
    self.relu22 = nn.ReLU(inplace=True)
    
    self.pool22 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    
    self.pad31  = nn.ReflectionPad2d((1,1,1,1))
    self.conv31 = nn.Conv2d(128,256,3,1,0)
    self.relu31 = nn.ReLU(inplace=True)
    
    self.pad32  = nn.ReflectionPad2d((1,1,1,1))
    self.conv32 = nn.Conv2d(256,256,3,1,0)
    self.relu32 = nn.ReLU(inplace=True)
    
    self.pad33  = nn.ReflectionPad2d((1,1,1,1))
    self.conv33 = nn.Conv2d(256,256,3,1,0)
    self.relu33 = nn.ReLU(inplace=True)
    
    self.pad34  = nn.ReflectionPad2d((1,1,1,1))
    self.conv34 = nn.Conv2d(256,256,3,1,0)
    self.relu34 = nn.ReLU(inplace=True)
    
    self.pool34 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    
    self.pad41  = nn.ReflectionPad2d((1,1,1,1))
    self.conv41 = nn.Conv2d(256,512,3,1,0)
    self.relu41 = nn.ReLU(inplace=True)
    
    self.pad42  = nn.ReflectionPad2d((1,1,1,1))
    self.conv42 = nn.Conv2d(512,512,3,1,0)
    self.relu42 = nn.ReLU(inplace=True)
    
    self.pad43  = nn.ReflectionPad2d((1,1,1,1))
    self.conv43 = nn.Conv2d(512,512,3,1,0)
    self.relu43 = nn.ReLU(inplace=True)
    
    self.pad44  = nn.ReflectionPad2d((1,1,1,1))
    self.conv44 = nn.Conv2d(512,512,3,1,0)
    self.relu44 = nn.ReLU(inplace=True)
    
    self.pool44 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    
    self.pad51  = nn.ReflectionPad2d((1,1,1,1))
    self.conv51 = nn.Conv2d(512,512,3,1,0)
    self.relu51 = nn.ReLU(inplace=True)
    
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
    y = self.relu11(self.conv11(self.pad11(y)))
    y = self.relu12(self.conv12(self.pad12(y)))
    y = self.pool12(y)
    y = self.relu21(self.conv21(self.pad21(y)))
    y = self.relu22(self.conv22(self.pad22(y)))
    y = self.pool22(y)
    y = self.relu31(self.conv31(self.pad31(y)))
    y = self.relu32(self.conv32(self.pad32(y)))
    y = self.relu33(self.conv33(self.pad33(y)))
    y = self.relu34(self.conv34(self.pad34(y)))
    y = self.pool34(y)
    y = self.relu41(self.conv41(self.pad41(y)))
    y = self.relu42(self.conv42(self.pad42(y)))
    y = self.relu43(self.conv43(self.pad43(y)))
    y = self.relu44(self.conv44(self.pad44(y)))
    y = self.pool44(y)
    y = self.relu51(self.conv51(self.pad51(y)))
    return y
    
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
        print("Given torch model, saving pytorch model")
        torch.save(self.state_dict(), os.path.splitext(model)[0] + ".pth")
        print("Saving done")
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

# For idea of learning transformation and decoder in one network
class TransformDecoder5(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(TransformDecoder5, self).__init__()
    self.fixed = fixed
    
    self.conv51 = nn.Conv2d(1024,512,3,1,0)
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
        print("Given torch model, saving pytorch model")
        torch.save(self.state_dict(), os.path.splitext(model)[0] + ".pth")
        print("Saving done")
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
    
# ---------------------------------------------------
# plus model VGG19 ("plus" means when we want to reduce the filters of layer x, we regress layer x+1 to avoid the dimension mismatch in layer x)
# Encoder/Decoder2
class Encoder2_plus(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Encoder2_plus, self).__init__()
    self.fixed = fixed

    self.conv0  = nn.Conv2d(  3,  3,1,1,0)
    self.conv0.weight = nn.Parameter(torch.from_numpy(np.array(
                                    [[[[0]],[[0]],[[255]]],
                                     [[[0]],[[255]],[[0]]],
                                     [[[255]],[[0]],[[0]]]])).float())
    self.conv0.bias = nn.Parameter(torch.from_numpy(np.array(
                                    [-103.939,-116.779,-123.68])).float())
    self.conv11 = nn.Conv2d(  3, 64,3,1,0)
    self.conv12 = nn.Conv2d( 64, 64,3,1,0)
    self.conv21 = nn.Conv2d( 64,128,3,1,0)
    self.conv22 = nn.Conv2d(128,128,3,1,0)
    
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      assert(os.path.splitext(model)[1] in {".t7", ".pth"})
      if model.endswith(".t7"):
        t7_model = load_lua(model)
        print(t7_model)
        load_param(t7_model, 0,  self.conv11)
        load_param(t7_model, 2,  self.conv12)
        load_param(t7_model, 5,  self.conv21)
        load_param(t7_model, 7, self.conv22)
        print("Given torch model, saving pytorch model")
        torch.save(self.state_dict(), os.path.splitext(model)[0] + ".pth")
        print("Saving done")
      else:
        self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
      
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
          
  def forward(self, input):
    out0  = self.conv0(input)
    out11 = self.relu(self.conv11(self.pad(out0)))
    out12 = self.relu(self.conv12(self.pad(out11)))
    out12 = self.pool(out12)
    out21 = self.relu(self.conv21(self.pad(out12)))
    out22 = self.relu(self.conv22(self.pad(out21)))
    return out11, out22

class Decoder2_plus(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Decoder2_plus, self).__init__()
    self.fixed = fixed
    
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
        load_param(t7_model, 1,  self.conv51) #TODO
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
        print("Given torch model, saving pytorch model")
        torch.save(self.state_dict(), os.path.splitext(model)[0] + ".pth")
        print("Saving done")
      else:
        self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
    
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
     
  def forward(self, input):
    out22 = self.relu(self.conv22(self.pad(input)))
    out21 = self.relu(self.conv21(self.pad(out22)))
    out21 = self.unpool(out21)
    out12 = self.relu(self.conv12(self.pad(out21)))
    out11 = self.relu(self.conv11(self.pad(out12)))
    return out11
    
# Encoder/Decoder3
class Encoder3_plus(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Encoder3_plus, self).__init__()
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
    
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      assert(os.path.splitext(model)[1] in {".t7", ".pth"})
      if model.endswith(".t7"):
        t7_model = load_lua(model)
        print(t7_model)
        load_param(t7_model, 0,  self.conv11)
        load_param(t7_model, 2,  self.conv12)
        load_param(t7_model, 5,  self.conv21)
        load_param(t7_model, 7, self.conv22)
        load_param(t7_model, 10, self.conv31)
        load_param(t7_model, 12, self.conv32)
        print("Given torch model, saving pytorch model")
        torch.save(self.state_dict(), os.path.splitext(model)[0] + ".pth")
        print("Saving done")
      else:
        self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
      
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
          
  def forward(self, input):
    out0  = self.conv0(input)
    out11 = self.relu(self.conv11(self.pad(out0)))
    out12 = self.relu(self.conv12(self.pad(out11)))
    out12 = self.pool(out12)
    out21 = self.relu(self.conv21(self.pad(out12)))
    out22 = self.relu(self.conv22(self.pad(out21)))
    out22 = self.pool(out22)
    out31 = self.relu(self.conv31(self.pad(out22)))
    out32 = self.relu(self.conv32(self.pad(out31)))
    return out11, out21, out32

class Decoder3_plus(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Decoder3_plus, self).__init__()
    self.fixed = fixed
    
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
        load_param(t7_model, 1,  self.conv51) #TODO
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
        print("Given torch model, saving pytorch model")
        torch.save(self.state_dict(), os.path.splitext(model)[0] + ".pth")
        print("Saving done")
      else:
        self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
    
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
     
  def forward(self, input):
    out32 = self.relu(self.conv32(self.pad(input)))
    out31 = self.relu(self.conv31(self.pad(out32)))
    out31 = self.unpool(out31)
    out22 = self.relu(self.conv22(self.pad(out31)))
    out21 = self.relu(self.conv21(self.pad(out22)))
    out21 = self.unpool(out21)
    out12 = self.relu(self.conv12(self.pad(out21)))
    out11 = self.relu(self.conv11(self.pad(out12)))
    return out11
    
# Encoder/Decoder4
class Encoder4_plus(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Encoder4_plus, self).__init__()
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
    
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      assert(os.path.splitext(model)[1] in {".t7", ".pth"})
      if model.endswith(".t7"):
        t7_model = load_lua(model)
        print(t7_model)
        load_param(t7_model, 0,  self.conv11)
        load_param(t7_model, 2,  self.conv12)
        load_param(t7_model, 5,  self.conv21)
        load_param(t7_model, 7, self.conv22)
        load_param(t7_model, 10, self.conv31)
        load_param(t7_model, 12, self.conv32)
        load_param(t7_model, 14, self.conv33)
        load_param(t7_model, 16, self.conv34)
        load_param(t7_model, 19, self.conv41)
        load_param(t7_model, 21, self.conv42)
        print("Given torch model, saving pytorch model")
        torch.save(self.state_dict(), os.path.splitext(model)[0] + ".pth")
        print("Saving done")
      else:
        self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
      
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
          
  def forward(self, input):
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
    return out11, out21, out31, out42

class Decoder4_plus(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Decoder4_plus, self).__init__()
    self.fixed = fixed
    
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
        load_param(t7_model, 1,  self.conv51) #TODO
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
        print("Given torch model, saving pytorch model")
        torch.save(self.state_dict(), os.path.splitext(model)[0] + ".pth")
        print("Saving done")
      else:
        self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
    
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
     
  def forward(self, input):
    out42 = self.relu(self.conv42(self.pad(input)))
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
    
# Encoder/Decoder5
class Encoder5_plus(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Encoder5_plus, self).__init__()
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
    self.conv52 = nn.Conv2d(512,512,3,1,0) # conv5_2
    
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      assert(os.path.splitext(model)[1] in {".t7", ".pth"})
      if model.endswith(".t7"):
        t7_model = load_lua(model)
        print(t7_model)
        load_param(t7_model, 0,  self.conv11)
        load_param(t7_model, 2,  self.conv12)
        load_param(t7_model, 5,  self.conv21)
        load_param(t7_model, 7,  self.conv22)
        load_param(t7_model, 10, self.conv31)
        load_param(t7_model, 12, self.conv32)
        load_param(t7_model, 14, self.conv33)
        load_param(t7_model, 16, self.conv34)
        load_param(t7_model, 19, self.conv41)
        load_param(t7_model, 21, self.conv42)
        load_param(t7_model, 23, self.conv43)
        load_param(t7_model, 25, self.conv44)
        load_param(t7_model, 28, self.conv51)
        load_param(t7_model, 30, self.conv52)
      else:
        self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
      
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
    y = self.relu(self.conv51(self.pad(y)))
    return y
    
class Decoder5_plus(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(Decoder5_plus, self).__init__()
    self.fixed = fixed
    
    self.conv52 = nn.Conv2d(512,512,3,1,0)
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
        load_param(t7_model, 1,  self.conv51) #TODO
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
        print("Given torch model, saving pytorch model")
        torch.save(self.state_dict(), os.path.splitext(model)[0] + ".pth")
        print("Saving done")
      else:
        self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
    
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
     
  def forward(self,input):
    out52 = self.relu(self.conv52(self.pad(input)))
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

# ------------------------------------------------------------
# 4x small decoder 
class SmallDecoder4_4x(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallDecoder4_4x, self).__init__()
    self.fixed = fixed

    self.conv41 = nn.Conv2d(256, 128,3,1,0)
    self.conv34 = nn.Conv2d(128, 128,3,1,0)
    self.conv33 = nn.Conv2d(128, 128,3,1,0)
    self.conv32 = nn.Conv2d(128, 128,3,1,0)
    self.conv31 = nn.Conv2d(128, 64,3,1,0)
    self.conv22 = nn.Conv2d( 64, 64,3,1,0, dilation=1)
    self.conv21 = nn.Conv2d( 64, 32,3,1,0, dilation=1)
    self.conv12 = nn.Conv2d( 32, 32,3,1,0, dilation=1)
    self.conv11 = nn.Conv2d( 32,  3,3,1,0, dilation=1)

    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.unpool_pwct = nn.MaxUnpool2d(kernel_size=2, stride=2)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      state_dict = torch.load(model, map_location=lambda storage, location: storage)
      if "model" in state_dict:
        self.load_state_dict(state_dict["model"])
      else:
        self.load_state_dict(state_dict)

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
# ------------------------------------------------------------xxx
# bridge the dimension mismatch using a 1x1 mapping layer (so-called "auxiliary mapping" in our paper)
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
  # "foward_aux" outputs all the middle auxiliary mapping layers
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

# ------------------------------------------------------------xxx
# for adain 64x
class SmallEncoder4_64x_aux(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallEncoder4_64x_aux, self).__init__()
    self.fixed = fixed

    self.conv0 = nn.Conv2d(3,3,1,1,0)
    self.conv0.requires_grad = False
    self.conv11     = nn.Conv2d( 3,   8,3,1,0, dilation=1)
    self.conv12     = nn.Conv2d( 8,   8,3,1,0, dilation=1)
    self.conv21     = nn.Conv2d( 8,  16,3,1,0, dilation=1)
    self.conv22     = nn.Conv2d( 16, 16,3,1,0, dilation=1)
    self.conv31     = nn.Conv2d( 16, 32,3,1,0)
    self.conv32     = nn.Conv2d( 32, 32,3,1,0)
    self.conv33     = nn.Conv2d( 32, 32,3,1,0)
    self.conv34     = nn.Conv2d( 32, 32,3,1,0)
    self.conv41     = nn.Conv2d( 32, 64,3,1,0)
    
    self.conv11_aux = nn.Conv2d(  8, 64,1,1,0)
    self.conv21_aux = nn.Conv2d( 16,128,1,1,0)
    self.conv31_aux = nn.Conv2d( 32,256,1,1,0)
    self.conv41_aux = nn.Conv2d( 64,512,1,1,0)
    
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      state_dict = torch.load(model, map_location=lambda storage, location: storage)
      if "model" in state_dict:
        self.load_state_dict(state_dict["model"])
      else:
        self.load_state_dict(state_dict)
    
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

# ------------------------------------------------------------xxx
# 2019-11-13: to train a BD4 for use in AdaIN 
class SmallEncoder4_FP16x_aux(nn.Module):
  def __init__(self, model=None):
    super(SmallEncoder4_FP16x_aux, self).__init__()
    self.vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)), # 0
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 16, (3, 3)), # 2
        nn.ReLU(),  # relu1-1 # 3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(16, 16, (3, 3)), # 5
        nn.ReLU(),  # relu1-2 # 6
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(16, 32, (3, 3)), # 9
        nn.ReLU(),  # relu2-1 # 10
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(32, 32, (3, 3)), # 12
        nn.ReLU(),  # relu2-2 # 13
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(32, 64, (3, 3)), # 16
        nn.ReLU(),  # relu3-1 # 17
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)), # 19
        nn.ReLU(),  # relu3-2 # 20
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)), # 22
        nn.ReLU(),  # relu3-3 # 23
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)), # 25
        nn.ReLU(),  # relu3-4 # 26
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)), # 29
        nn.ReLU(),  # relu4-1 # 30
        nn.Conv2d(128,512, (1, 1)), # 31, up dimension
    )
    
    # load weights
    self.vgg[0].weight = nn.Parameter(torch.from_numpy(np.array(
                                    [[[[0]],[[0]],[[255]]],
                                     [[[0]],[[255]],[[0]]],
                                     [[[255]],[[0]],[[0]]]])).float())
    self.vgg[0].bias = nn.Parameter(torch.from_numpy(np.array(
                                    [-103.939,-116.779,-123.68])).float())
    assert(os.path.splitext(model)[1] in {".t7", ".pth"})
    if model.endswith(".t7"):
      t7_model = load_lua(model)
      load_param(t7_model, 0,  self.vgg[2])
      load_param(t7_model, 2,  self.vgg[5])
      load_param(t7_model, 5,  self.vgg[9])
      load_param(t7_model, 7,  self.vgg[12])
      load_param(t7_model, 10, self.vgg[16])
      load_param(t7_model, 12, self.vgg[19])
      load_param(t7_model, 14, self.vgg[22])
      load_param(t7_model, 16, self.vgg[25])
      load_param(t7_model, 19, self.vgg[29])
      torch.save(self.state_dict(), os.path.splitext(model)[0] + "_FP16x_4E_for_adain.pth")
    else:
      self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
    
    for param in self.parameters():
      if len(param.size()) == 4 and param.size(1) == 512:
        param.requires_grad = True # only update the 1x1 conv (weight)
      elif len(param.size()) == 1 and param.size(0) == 512:
        param.requires_grad = True # only update the 1x1 conv (bias)
      else:
        param.requires_grad = False
  
  def forward(self, y):
    return self.vgg(y)
# ------------------------------------------------------------xxx
# Encoder5, directly learn the transform
class SmallEncoder5_16x(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallEncoder5_16x, self).__init__()
    self.fixed = fixed
    self.conv11     = nn.Conv2d(  6, 16,3,1,0, dilation=1)
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
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    self.pad  = nn.ReflectionPad2d((1,1,1,1))
    
    if model:
      assert(os.path.splitext(model)[1] in {".t7", ".pth"})
      if model.endswith(".t7"):
        t7_model = load_lua(model)
        load_param(t7_model, 0,  self.conv11)
        load_param(t7_model, 2,  self.conv12)
        load_param(t7_model, 5,  self.conv21)
        load_param(t7_model, 7,  self.conv22)
        load_param(t7_model, 10, self.conv31)
        load_param(t7_model, 12, self.conv32)
        load_param(t7_model, 14, self.conv33)
        load_param(t7_model, 16, self.conv34)
        load_param(t7_model, 19, self.conv41)
        load_param(t7_model, 21, self.conv42)
        load_param(t7_model, 23, self.conv43)
        load_param(t7_model, 25, self.conv44)
        load_param(t7_model, 28, self.conv51)
      else:
        self.load_state_dict(torch.load(model, map_location=lambda storage, location: storage))
    if fixed:
      for param in self.parameters():
          param.requires_grad = False

  def forward(self, y):
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

class SmallEncoder5_16x_aux_Gatys(nn.Module):
  def __init__(self, model=None, fixed=False, dilation=1):
    super(SmallEncoder5_16x_aux_Gatys, self).__init__()
    self.fixed = fixed

    self.conv0  = nn.Conv2d(  3,  3,1,1,0)
    self.conv0.weight = nn.Parameter(torch.from_numpy(np.array(
                                    [[[[0]],[[0]],[[255]]],
                                     [[[0]],[[255]],[[0]]],
                                     [[[255]],[[0]],[[0]]]])).float())
    self.conv0.bias = nn.Parameter(torch.from_numpy(np.array(
                                    [-103.939,-116.779,-123.68])).float())
    self.pad11  = nn.ReflectionPad2d((1,1,1,1))
    self.conv11 = nn.Conv2d(  3, 16,3,1,0, dilation=dilation)
    self.relu11 = nn.ReLU(inplace=True)
    
    self.pad12  = nn.ReflectionPad2d((1,1,1,1))
    self.conv12 = nn.Conv2d( 16, 16,3,1,0, dilation=dilation)
    self.relu12 = nn.ReLU(inplace=True)
    
    self.pool12 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    
    self.pad21  = nn.ReflectionPad2d((1,1,1,1))
    self.conv21 = nn.Conv2d( 16,32,3,1,0, dilation=dilation)
    self.relu21 = nn.ReLU(inplace=True)
    
    self.pad22  = nn.ReflectionPad2d((1,1,1,1))
    self.conv22 = nn.Conv2d(32,32,3,1,0)
    self.relu22 = nn.ReLU(inplace=True)
    
    self.pool22 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    
    self.pad31  = nn.ReflectionPad2d((1,1,1,1))
    self.conv31 = nn.Conv2d(32,64,3,1,0, dilation=dilation)
    self.relu31 = nn.ReLU(inplace=True)
    
    self.pad32  = nn.ReflectionPad2d((1,1,1,1))
    self.conv32 = nn.Conv2d(64,64,3,1,0, dilation=dilation)
    self.relu32 = nn.ReLU(inplace=True)
    
    self.pad33  = nn.ReflectionPad2d((1,1,1,1))
    self.conv33 = nn.Conv2d(64,64,3,1,0, dilation=dilation)
    self.relu33 = nn.ReLU(inplace=True)
    
    self.pad34  = nn.ReflectionPad2d((1,1,1,1))
    self.conv34 = nn.Conv2d(64,64,3,1,0, dilation=dilation)
    self.relu34 = nn.ReLU(inplace=True)
    
    self.pool34 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    
    self.pad41  = nn.ReflectionPad2d((1,1,1,1))
    self.conv41 = nn.Conv2d(64,128,3,1,0, dilation=dilation)
    self.relu41 = nn.ReLU(inplace=True)
    
    self.pad42  = nn.ReflectionPad2d((1,1,1,1))
    self.conv42 = nn.Conv2d(128,128,3,1,0, dilation=dilation)
    self.relu42 = nn.ReLU(inplace=True)
    
    self.pad43  = nn.ReflectionPad2d((1,1,1,1))
    self.conv43 = nn.Conv2d(128,128,3,1,0, dilation=dilation)
    self.relu43 = nn.ReLU(inplace=True)
    
    self.pad44  = nn.ReflectionPad2d((1,1,1,1))
    self.conv44 = nn.Conv2d(128,128,3,1,0, dilation=dilation)
    self.relu44 = nn.ReLU(inplace=True)
    
    self.pool44 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=False)
    
    self.pad51  = nn.ReflectionPad2d((1,1,1,1))
    self.conv51 = nn.Conv2d(128,128,3,1,0, dilation=dilation)
    self.relu51 = nn.ReLU(inplace=True)
    
    if model:
      load_param2(model, self)
    if fixed:
      for param in self.parameters():
          param.requires_grad = False

  def forward(self, input):
    y = self.conv0(input)
    y = self.relu11(self.conv11(self.pad11(y)))
    y = self.relu12(self.conv12(self.pad12(y)))
    y = self.pool12(y)
    y = self.relu21(self.conv21(self.pad21(y)))
    y = self.relu22(self.conv22(self.pad22(y)))
    y = self.pool22(y)
    y = self.relu31(self.conv31(self.pad31(y)))
    y = self.relu32(self.conv32(self.pad32(y)))
    y = self.relu33(self.conv33(self.pad33(y)))
    y = self.relu34(self.conv34(self.pad34(y)))
    y = self.pool34(y)
    y = self.relu41(self.conv41(self.pad41(y)))
    y = self.relu42(self.conv42(self.pad42(y)))
    y = self.relu43(self.conv43(self.pad43(y)))
    y = self.relu44(self.conv44(self.pad44(y)))
    y = self.pool44(y)
    y = self.relu51(self.conv51(self.pad51(y)))
    return y
    
# --------------------------------------------------------
# original VGG19 Autoencoders
class Autoencoder1_BD(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder1_BD, self).__init__()
    self.encoder1 = Encoder1(e1, fixed=True)
    self.decoder  = Decoder1(d)
    self.encoder2 = self.encoder1
    
  def forward(self, input):
    feats   = self.encoder1.forward_branch(input)
    decoded = self.decoder(feats)
    feats2  = self.encoder2.forward_branch(decoded)
    return feats, decoded, feats2

class Autoencoder2_BD(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder2_BD, self).__init__()
    self.encoder1 = Encoder2(e1, fixed=True)
    self.decoder  = Decoder2(d)
    self.encoder2 = self.encoder1
    
  def forward(self, input):
    feats   = self.encoder1.forward_branch(input)
    decoded = self.decoder(feats[-1])
    feats2  = self.encoder2.forward_branch(decoded)
    return feats, decoded, feats2
    
class Autoencoder3_BD(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder3_BD, self).__init__()
    self.encoder1 = Encoder3(e1, fixed=True)
    self.decoder  = Decoder3(d)
    self.encoder2 = self.encoder1
    
  def forward(self, input):
    feats   = self.encoder1.forward_branch(input)
    decoded = self.decoder(feats[-1])
    feats2  = self.encoder2.forward_branch(decoded)
    return feats, decoded, feats2
    
class Autoencoder4_BD(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder4_BD, self).__init__()
    self.encoder1 = Encoder4(e1, fixed=True)
    self.decoder  = Decoder4(d)
    self.encoder2 = self.encoder1
    
  def forward(self, input):
    feats   = self.encoder1.forward_branch(input)
    decoded = self.decoder(feats[-1])
    feats2  = self.encoder2.forward_branch(decoded)
    return feats, decoded, feats2

class Autoencoder5_BD(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder5_BD, self).__init__()
    self.encoder1 = Encoder5(e1, fixed=True)
    self.decoder  = Decoder5(d)
    self.encoder2 = self.encoder1
    
  def forward(self, input):
    feats   = self.encoder1.forward_branch(input)
    decoded = self.decoder(feats[-1])
    feats2  = self.encoder2.forward_branch(decoded)
    return feats, decoded, feats2

# -------------------------------------------------
# plus autoencoders
class Autoencoder2_BD_plus(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder2_BD_plus, self).__init__()
    self.encoder1 = Encoder2_plus(e1, fixed=True)
    self.decoder  = Decoder2_plus(d)
    self.encoder2 = self.encoder1
    
  def forward(self, input):
    feats   = self.encoder1(input)
    decoded = self.decoder(feats[-1])
    feats2  = self.encoder2(decoded)
    return feats, decoded, feats2
    
class Autoencoder3_BD_plus(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder3_BD_plus, self).__init__()
    self.encoder1 = Encoder3_plus(e1, fixed=True)
    self.decoder  = Decoder3_plus(d)
    self.encoder2 = self.encoder1
    
  def forward(self, input):
    feats   = self.encoder1(input)
    decoded = self.decoder(feats[-1])
    feats2  = self.encoder2(decoded)
    return feats, decoded, feats2
    
class Autoencoder4_BD_plus(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder4_BD_plus, self).__init__()
    self.encoder1 = Encoder4_plus(e1, fixed=True)
    self.decoder  = Decoder4_plus(d)
    self.encoder2 = self.encoder1
    
  def forward(self, input):
    feats   = self.encoder1(input)
    decoded = self.decoder(feats[-1])
    feats2  = self.encoder2(decoded)
    return feats, decoded, feats2
    
class Autoencoder5_BD_plus(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder5_BD_plus, self).__init__()
    self.encoder1 = Encoder5_plus(e1, fixed=True)
    self.decoder  = Decoder5_plus(d)
    self.encoder2 = self.encoder1
    
  def forward(self, input):
    feats   = self.encoder1(input)
    decoded = self.decoder(feats[-1])
    feats2  = self.encoder2(decoded)
    return feats, decoded, feats2

# ---------------------------------------------
# small 16x autoencoders
class Autoencoder5_SE_16x(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder5_SE_16x, self).__init__()
    self.encoder1 = Encoder5(e1, fixed=True) 
    self.decoder  = Decoder5(d, fixed=True)
    self.encoder2 = SmallEncoder5_16x_aux(e2)
    
  def forward(self, input):
    feats   = self.encoder1.forward_branch(input)
    feats2  = self.encoder2.forward_aux(input)
    decoded = self.decoder(feats2[-1])
    return feats, decoded, feats2

class Autoencoder5_SD_16x(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder5_SD_16x, self).__init__()
    self.encoder1 = SmallEncoder5_16x_aux(e1, fixed=True)
    self.decoder  = SmallDecoder5_16x(d)
    self.encoder2 = self.encoder1
    
  def forward(self, input):
    feats   = self.encoder1.forward_branch(input)
    decoded = self.decoder(feats[-1])
    feats2  = self.encoder2.forward_branch(decoded)
    return feats, decoded, feats2

class Autoencoder4_SE_16x(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder4_SE_16x, self).__init__()
    self.encoder1 = Encoder4(e1, fixed=True) 
    self.decoder  = Decoder4(d, fixed=True)
    self.encoder2 = SmallEncoder4_16x_aux(e2)
    
  def forward(self, input):
    feats   = self.encoder1.forward_branch(input)
    feats2  = self.encoder2.forward_aux(input)
    decoded = self.decoder(feats2[-1])
    return feats, decoded, feats2
    
class Autoencoder4_SD_16x(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder4_SD_16x, self).__init__()
    self.encoder1 = SmallEncoder4_16x_aux(e1, fixed=True)
    self.decoder  = SmallDecoder4_16x(d)
    self.encoder2 = self.encoder1
    
  def forward(self, input):
    feats   = self.encoder1.forward_branch(input)
    decoded = self.decoder(feats[-1])
    feats2  = self.encoder2.forward_branch(decoded)
    return feats, decoded, feats2
    
class Autoencoder3_SE_16x(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder3_SE_16x, self).__init__()
    self.encoder1 = Encoder3(e1, fixed=True) 
    self.decoder  = Decoder3(d, fixed=True)
    self.encoder2 = SmallEncoder3_16x_aux(e2)
    
  def forward(self, input):
    feats   = self.encoder1.forward_branch(input)
    feats2  = self.encoder2.forward_aux(input)
    decoded = self.decoder(feats2[-1])
    return feats, decoded, feats2
    
class Autoencoder3_SD_16x(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder3_SD_16x, self).__init__()
    self.encoder1 = SmallEncoder3_16x_aux(e1, fixed=True)
    self.decoder  = SmallDecoder3_16x(d)
    self.encoder2 = self.encoder1
    
  def forward(self, input):
    feats   = self.encoder1.forward_branch(input)
    decoded = self.decoder(feats[-1])
    feats2  = self.encoder2.forward_branch(decoded)
    return feats, decoded, feats2
    
class Autoencoder2_SE_16x(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder2_SE_16x, self).__init__()
    self.encoder1 = Encoder2(e1, fixed=True) 
    self.decoder  = Decoder2(d, fixed=True)
    self.encoder2 = SmallEncoder2_16x_aux(e2)
    
  def forward(self, input):
    feats   = self.encoder1.forward_branch(input)
    feats2  = self.encoder2.forward_aux(input)
    decoded = self.decoder(feats2[-1])
    return feats, decoded, feats2
    
class Autoencoder2_SD_16x(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder2_SD_16x, self).__init__()
    self.encoder1 = SmallEncoder2_16x_aux(e1, fixed=True)
    self.decoder  = SmallDecoder2_16x(d)
    self.encoder2 = self.encoder1
  def forward(self, input):
    feats   = self.encoder1.forward_branch(input)
    decoded = self.decoder(feats[-1])
    feats2  = self.encoder2.forward_branch(decoded)
    return feats, decoded, feats2

class Autoencoder1_SE_16x(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder1_SE_16x, self).__init__()
    self.encoder1 = Encoder1(e1, fixed=True) 
    self.decoder  = Decoder1(d, fixed=True)
    self.encoder2 = SmallEncoder1_16x_aux(e2)
    
  def forward(self, input):
    feats   = self.encoder1.forward_branch(input)
    feats2  = self.encoder2.forward_aux(input)
    decoded = self.decoder(feats2)
    return feats, decoded, feats2
    
class Autoencoder1_SD_16x(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder1_SD_16x, self).__init__()
    self.encoder1 = SmallEncoder1_16x_aux(e1, fixed=True)
    self.decoder  = SmallDecoder1_16x(d)
    self.encoder2 = self.encoder1
  def forward(self, input):
    feats   = self.encoder1(input)
    decoded = self.decoder(feats)
    feats2  = self.encoder2(decoded)
    return feats, decoded, feats2

# --------------------------------------------------------
# small 16x autoencoders -- train from scratch on COCO
class Autoencoder5_16x_fromscratch(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder5_16x_fromscratch, self).__init__()
    self.encoder1 = SmallEncoder5_16x_aux(e1) # not fixed
    self.decoder  = SmallDecoder5_16x(d) # not fixed
    self.encoder2 = self.encoder1
    
  def forward(self, input):
    feats   = self.encoder1.forward_branch(input)
    decoded = self.decoder(feats[-1])
    feats2  = self.encoder2.forward_branch(decoded)
    return feats, decoded, feats2

class Autoencoder4_16x_fromscratch(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder4_16x_fromscratch, self).__init__()
    self.encoder1 = SmallEncoder4_16x_aux(e1) # not fixed
    self.decoder  = SmallDecoder4_16x(d) # not fixed
    self.encoder2 = self.encoder1
    
  def forward(self, input):
    feats   = self.encoder1.forward_branch(input)
    decoded = self.decoder(feats[-1])
    feats2  = self.encoder2.forward_branch(decoded)
    return feats, decoded, feats2

class Autoencoder3_16x_fromscratch(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder3_16x_fromscratch, self).__init__()
    self.encoder1 = SmallEncoder3_16x_aux(e1) # not fixed
    self.decoder  = SmallDecoder3_16x(d) # not fixed
    self.encoder2 = self.encoder1
    
  def forward(self, input):
    feats   = self.encoder1.forward_branch(input)
    decoded = self.decoder(feats[-1])
    feats2  = self.encoder2.forward_branch(decoded)
    return feats, decoded, feats2
    
class Autoencoder2_16x_fromscratch(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder2_16x_fromscratch, self).__init__()
    self.encoder1 = SmallEncoder2_16x_aux(e1) # not fixed
    self.decoder  = SmallDecoder2_16x(d) # not fixed
    self.encoder2 = self.encoder1
    
  def forward(self, input):
    feats   = self.encoder1.forward_branch(input)
    decoded = self.decoder(feats[-1])
    feats2  = self.encoder2.forward_branch(decoded)
    return feats, decoded, feats2
 
class Autoencoder1_16x_fromscratch(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder1_16x_fromscratch, self).__init__()
    self.encoder1 = SmallEncoder1_16x_aux(e1) # not fixed
    self.decoder  = SmallDecoder1_16x(d) # not fixed
    self.encoder2 = self.encoder1
    
  def forward(self, input):
    feats   = self.encoder1.forward_branch(input)
    decoded = self.decoder(feats)
    feats2  = self.encoder2.forward_branch(decoded)
    return feats, decoded, feats2
    
# --------------------------------------------------------
# FP16x Autoencoder
class Autoencoder5_SD_FP16x(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder5_SD_FP16x, self).__init__()
    self.encoder1 = SmallEncoder5_FP16x(e1, fixed=True) 
    self.decoder  = SmallDecoder5_FP16x(d)
    self.encoder2 = self.encoder1
    
  def forward(self, input):
    feats   = self.encoder1.forward_branch(input)
    decoded = self.decoder(feats[-1])
    feats2  = self.encoder2.forward_branch(decoded)
    return feats, decoded, feats2
    
class Autoencoder4_SD_FP16x(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder4_SD_FP16x, self).__init__()
    self.encoder1 = SmallEncoder4_FP16x(e1, fixed=True) 
    self.decoder  = SmallDecoder4_FP16x(d)
    self.encoder2 = self.encoder1
    
  def forward(self, input):
    feats   = self.encoder1.forward_branch(input)
    decoded = self.decoder(feats[-1])
    feats2  = self.encoder2.forward_branch(decoded)
    return feats, decoded, feats2
    
class Autoencoder3_SD_FP16x(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder3_SD_FP16x, self).__init__()
    self.encoder1 = SmallEncoder3_FP16x(e1, fixed=True) 
    self.decoder  = SmallDecoder3_FP16x(d)
    self.encoder2 = self.encoder1
    
  def forward(self, input):
    feats   = self.encoder1.forward_branch(input)
    decoded = self.decoder(feats[-1])
    feats2  = self.encoder2.forward_branch(decoded)
    return feats, decoded, feats2
    
class Autoencoder2_SD_FP16x(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder2_SD_FP16x, self).__init__()
    self.encoder1 = SmallEncoder2_FP16x(e1, fixed=True) 
    self.decoder  = SmallDecoder2_FP16x(d)
    self.encoder2 = self.encoder1
    
  def forward(self, input):
    feats   = self.encoder1.forward_branch(input)
    decoded = self.decoder(feats[-1])
    feats2  = self.encoder2.forward_branch(decoded)
    return feats, decoded, feats2
    
class Autoencoder1_SD_FP16x(nn.Module):
  def __init__(self, e1, d, e2):
    super(Autoencoder1_SD_FP16x, self).__init__()
    self.encoder1 = SmallEncoder1_FP16x(e1, fixed=True) 
    self.decoder  = SmallDecoder1_FP16x(d)
    self.encoder2 = self.encoder1
    
  def forward(self, input):
    feats   = self.encoder1.forward_branch(input)
    decoded = self.decoder(feats)
    feats2  = self.encoder2.forward_branch(decoded)
    return feats, decoded, feats2

# --------------------------------------------------------
EigenValueThre = 1e-5 # the eigen value below this threshold will be discarded
def wct(cF_, sF_): # cF_ and sF_ have 3 dimensions
    C, W,  H  = cF_.size(0), cF_.size(1), cF_.size(2)
    _, W1, H1 = sF_.size(0), sF_.size(1), sF_.size(2)
    cF = cF_.view(C, -1)
    sF = sF_.view(C, -1)

    # ---------------------------------
    # svd for content feature
    cFSize = cF.size() # size: [c, hw]
    c_mean = torch.mean(cF, 1).unsqueeze(1).expand_as(cF)
    cF = cF - c_mean
    contentConv = torch.mm(cF, cF.t()).div(cFSize[1] - 1) + torch.eye(cFSize[0]).float().cuda()
    c_u, c_e, c_v = torch.svd(contentConv, some=False)
    k_c = cFSize[0]
    for i in range(cFSize[0]):
        if c_e[i] < EigenValueThre:
            k_c = i
            break
    
    # ---------------------------------
    # svd for style feature
    sFSize = sF.size()
    s_mean = torch.mean(sF, 1)
    sF = sF - s_mean.unsqueeze(1).expand_as(sF)
    styleConv = torch.mm(sF,sF.t()).div(sFSize[1] - 1)
    s_u, s_e, s_v = torch.svd(styleConv, some=False);
    k_s = sFSize[0]
    for i in range(sFSize[0]):
        if s_e[i] < EigenValueThre:
            k_s = i
            break
    
    # whitening
    c_d = (c_e[0:k_c]).pow(-0.5)
    step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
    step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
    whiten_cF = torch.mm(step2, cF)
    
    # coloring
    s_d = (s_e[0:k_s]).pow(0.5)
    targetFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t())), whiten_cF)
    targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
    targetFeature = targetFeature.view_as(cF_)
    return targetFeature

# Finetune the small encoder obtained from previous method
class Autoencoder5_Stage2(nn.Module):
  def __init__(self, be, se, d):
    super(Autoencoder5_Stage2, self).__init__()
    self.be = Encoder5(be, fixed=True)
    self.se = SmallEncoder5_16x_aux(se)
    self.d  = SmallDecoder5_16x(d, fixed=True)
  
  def forward(self, batchC, batchS):
    batchCF = self.SE(batchC)
    batchSF = self.SE(batchS)
    batchCSF = torch.zeros_like(batchCF, requires_grad=True).cuda()
    bs = batchC.size()[0]
    for i in range(bs):
      CSF = wct(batchCF[i], batchSF[i])
      batchCSF.data[i].copy_(CSF)
    # pair = torch.cat((batchC, batchS), dim=1) # [N, 6, 256, 256]
    # batchCSF = self.SE(pair) # [N, 128, 16, 16]
    batchCS = self.d(batchCSF) # invert feature to images [N, 3, 256, 256]
    f1_cs, f2_cs, f3_cs, f4_cs, _ = self.BE.forward_branch(batchCS)
    f1_s,  f2_s,  f3_s,  f4_s,  _ = self.BE.forward_branch(batchS)
    f1_c,  f2_c,  f3_c,  f4_c,  _ = self.BE.forward_branch(batchC)
    style_loss = 0
    for i in range(1, 5):
      f_cs = eval("f%d_cs" % i)
      f_s  = eval("f%d_s"  % i)
      f_c  = eval("f%d_c"  % i)
      style_loss += nn.MSELoss()(gram_matrix(f_cs), gram_matrix(f_s.data))
    content_loss = nn.MSELoss()(f4_cs, f4_c.data)
    return content_loss, style_loss, batchCS

# 2019/09/16 exp
class TrainSE_With_AdaINDecoder(nn.Module):
  def __init__(self, args):
    super(TrainSE_With_AdaINDecoder, self).__init__()
    self.BE = Encoder4_2(args.BE, fixed=True)
    self.d  = Decoder4_2(args.Dec, fixed=True)
    self.SE = eval("SmallEncoder4_2_%dx" % args.speedup)(args.SE, fixed=False)
    self.args = args
    # self.SE = Encoder4_2() # 2019/09/17 exp
    # self.SE = Encoder4_2(be, fixed=False) # 2019/09/18 exp
    
  def forward(self, c, s, iter):
    cF_BE = self.BE.forward_branch(c)
    sF_BE = self.BE.forward_branch(s) # BE forward, multi outputs: relu1_1, 2_1, 3_1, 4_1, 4_2
    cF_SE = self.SE.forward_aux(c, self.args.updim_relu)
    sF_SE = self.SE.forward_aux(s, self.args.updim_relu) # SE forward, multi outputs: [relu1_1, 2_1, 3_1, 4_1] -- updim, 4_2
    sd = self.d(adaptive_instance_normalization(cF_SE[-1], sF_SE[-1])) # get stylized image from feature Conv4_2
    
    # for log
    sd_BE = 0
    if iter % self.args.save_interval == 0:
      stylized_cF_BE = adaptive_instance_normalization(cF_BE[-1], sF_BE[-1])
      sd_BE = self.d(stylized_cF_BE) # stylized image using BE's feature
    
    # (loss 1) BE -> SE knowledge transfer loss. For Conv4_2, they have the same feature dimension
    feat_loss = 0
    for i in range(len(cF_BE)):
      feat_loss += nn.MSELoss()(cF_SE[i], cF_BE[i].data)
      feat_loss += nn.MSELoss()(sF_SE[i], sF_BE[i].data)
    
    # (loss 2) eval the quality of stylized image
    sdF = self.BE.forward_branch(sd)
    content_loss = nn.MSELoss()(sdF[-2], cF_BE[-2].data) # using the feat of Conv4_1 to calculate content loss
    # content_loss = nn.MSELoss()(sdF[-2], stylized_cF_BE.data) # AdaIN's impel
    style_loss = 0
    for i in range(len(cF_BE) - 1): # do not count Conv4_2, so minus 1
      # (a) use gram to measure style
      # style_loss += nn.MSELoss()(gram_matrix(sdF[i]), gram_matrix(sF_BE[i].data))
      # (b) use AdaIN's mean and variance to measure style
      mean_sd, std_sd = calc_mean_std(sdF[i])
      mean_s,  std_s  = calc_mean_std(sF_BE[i])
      style_loss += nn.MSELoss()(mean_sd, mean_s.data) + nn.MSELoss()(std_sd, std_s.data) # use mean and std as AdaIN
    return feat_loss, content_loss, style_loss, sd, sd_BE

# 2019-11-09: Use non-relu auxiliary mapping to compress WCT models
class TrainSE_With_WCTDecoder(nn.Module):
  def __init__(self, args):
    super(TrainSE_With_WCTDecoder, self).__init__()
    self.BE = eval("Encoder%d" % args.stage)(args.BE, fixed=True)
    self.BD = eval("Decoder%d" % args.stage)(args.Dec, fixed=True)
    self.SE = eval("SmallEncoder%d_%dx_aux" % (args.stage, args.speedup))(args.SE, fixed=False)
    self.args = args
    
  def forward(self, c, iter):
    cF_BE = self.BE.forward_branch(c) # BE forward, multi outputs: relu1_1, 2_1, 3_1, 4_1, 5_1
    cF_SE = self.SE.forward_aux(c, self.args.updim_relu) # SE forward, multi outputs: [relu1_1, 2_1, 3_1, 4_1, 5_1]
    rec = self.d(cF_SE[-1])
    
    # for log
    sd_BE = 0
    if iter % self.args.save_interval == 0:
      rec_BE = self.BD(cF_BE[-1])
    
    # (loss 1) BE -> SE knowledge transfer loss
    feat_loss = 0
    for i in range(len(cF_BE)):
      feat_loss += nn.MSELoss()(cF_SE[i], cF_BE[i].data)
    
    # (loss 2, 3) eval the quality of reconstructed image, pixel and perceptual loss
    rec_pixl_loss = nn.MSELoss()(rec, c.data)
    recF_BE = self.BE.forward_branch(rec)
    rec_perc_loss = 0
    for i in range(len(recF_BE)):
      rec_perc_loss += nn.MSELoss()(recF_BE[i], cF_BE[i].data)
    return feat_loss, rec_pixl_loss, rec_perc_loss, rec, c

class TrainSD_With_WCTSE(nn.Module):
  def __init__(self, args):
    super(TrainSD_With_WCTSE, self).__init__()
    self.BE = eval("Encoder%d" % args.stage)(args.BE, fixed=True) 
    self.SE = eval("SmallEncoder%d_%dx_aux" % (args.stage, args.speedup))(args.SE, fixed=True)
    self.SD = eval("SmallDecoder%d_%dx" % (args.stage, args.speedup))(args.SD, fixed=False)
    self.args = args
    
  def forward(self, c, iter):
    rec = self.SD(self.SE(c))
    # loss (1) pixel loss
    rec_pixl_loss = nn.MSELoss()(rec, c.data)
    
    # loss (2) perceptual loss
    recF_BE = self.BE.forward_branch(rec)
    cF_BE = self.BE.forward_branch(c)
    rec_perc_loss = 0
    for i in range(len(recF_BE)):
      rec_perc_loss += nn.MSELoss()(recF_BE[i], cF_BE[i].data)
      
    return rec_pixl_loss, rec_perc_loss, rec

# CVPR-20 Rebuttal
class TrainSD_With_FPSE(nn.Module):
  def __init__(self, args):
    super(TrainSD_With_FPSE, self).__init__()
    self.SE = eval("SmallEncoder%d_FP%dx" % (args.stage, args.speedup))(None, fixed=True) # load SE weights outside
    self.SD = eval("SmallDecoder%d_FP%dx" % (args.stage, args.speedup))(args.SD, fixed=False)
    self.BE = self.SE # as before, the evaluation network uses the SE
    self.args = args
    
  def forward(self, c, iter):
    rec = self.SD(self.SE(c))
    # loss (1) pixel loss
    rec_pixl_loss = nn.MSELoss()(rec, c.data)
    
    # loss (2) perceptual loss
    recF_BE = self.BE.forward_branch(rec)
    cF_BE = self.BE.forward_branch(c)
    rec_perc_loss = 0
    for i in range(len(recF_BE)):
      rec_perc_loss += nn.MSELoss()(recF_BE[i], cF_BE[i].data)
    return rec_pixl_loss, rec_perc_loss, rec

# CVPR-20 Rebuttal
class TrainBD(nn.Module):
  def __init__(self, args):
    super(TrainBD, self).__init__()
    self.SE = eval("Encoder%d" % args.stage)(None, fixed=True) # load SE weights outside
    self.SD = eval("Decoder%d" % args.stage)(args.SD, fixed=False)
    self.BE = self.SE # as before, the evaluation network uses the SE
    self.args = args
    
  def forward(self, c, iter):
    rec = self.SD(self.SE(c))
    # loss (1) pixel loss
    rec_pixl_loss = nn.MSELoss()(rec, c.data)
    
    # loss (2) perceptual loss
    recF_BE = self.BE.forward_branch(rec)
    cF_BE = self.BE.forward_branch(c)
    rec_perc_loss = 0
    for i in range(len(recF_BE)):
      rec_perc_loss += nn.MSELoss()(recF_BE[i], cF_BE[i].data)
    return rec_pixl_loss, rec_perc_loss, rec

# CVPR-20 Rebuttal. Apply KD to SD
class TrainSD_With_WCTSE_KDtoSD(nn.Module):
  def __init__(self, args):
    super(TrainSD_With_WCTSE_KDtoSD, self).__init__()
    self.BE = eval("Encoder%d" % args.stage)(args.BE, fixed=True)
    self.BD = eval("Decoder%d" % args.stage)(None, fixed=True)
    self.SE = eval("SmallEncoder%d_%dx_aux" % (args.stage, args.speedup))(None, fixed=True)
    self.SD = eval("SmallDecoder%d_%dx_aux" % (args.stage, args.speedup))(args.SD, fixed=False)
    self.args = args
    
  def forward(self, c, iter):
    feats_BE = self.BE.forward_branch(c) # for perceptual loss

    *_, feat_SE_aux, feat_SE = self.SE.forward_aux2(c) # output the last up-size feature and normal-size feature
    feats_BD = self.BD.forward_branch(feat_SE_aux)
    feats_SD = self.SD.forward_aux(feat_SE, relu=self.args.updim_relu)
    rec = feats_SD[-1]

    # loss (1) pixel loss
    rec_pixl_loss = nn.MSELoss()(rec, c.data)
    
    # loss (2) perceptual loss
    rec_feats_BE = self.BE.forward_branch(rec)
    rec_perc_loss = 0
    for i in range(len(rec_feats_BE)):
      rec_perc_loss += nn.MSELoss()(rec_feats_BE[i], feats_BE[i].data)
    
    # loss (3) kd feature loss
    kd_feat_loss = 0
    for i in range(len(feats_BD)):
      kd_feat_loss += nn.MSELoss()(feats_SD[i], feats_BD[i].data)

    return rec_pixl_loss, rec_perc_loss, kd_feat_loss, rec


# here we use the BE BD trained by the author of pytorch adain
class TrainSE_With_AdaINDecoder2(nn.Module):
  def __init__(self, args):
    super(TrainSE_With_AdaINDecoder2, self).__init__()
    # set up BE
    self.BE = vgg
    self.BE.load_state_dict(torch.load(args.BE))
    self.BE = self.BE[:31]
    for p in self.BE.parameters():
      p.requires_grad = False
    self.BE1 = self.BE[  : 4]
    self.BE2 = self.BE[ 4:11]
    self.BE3 = self.BE[11:18]
    self.BE4 = self.BE[18:31]
    
    # set up BD
    self.d = decoder
    self.d.load_state_dict(torch.load(args.Dec))
    for p in self.d.parameters():
      p.requires_grad = False
    
    # set up SE
    self.SE = eval("SmallEncoder4_%dx_aux" % args.speedup)()
    self.args = args
    
  def forward(self, c, s, iter):
    cF_BE1 = self.BE1(c)
    cF_BE2 = self.BE2(cF_BE1)
    cF_BE3 = self.BE3(cF_BE2)
    cF_BE4 = self.BE4(cF_BE3)
    cF_BE = (cF_BE1, cF_BE2, cF_BE3, cF_BE4)
    
    sF_BE1 = self.BE1(s)
    sF_BE2 = self.BE2(sF_BE1)
    sF_BE3 = self.BE3(sF_BE2)
    sF_BE4 = self.BE4(sF_BE3)
    sF_BE = (sF_BE1, sF_BE2, sF_BE3, sF_BE4)
    
    cF_SE = self.SE.forward_aux(c, self.args.updim_relu)
    sF_SE = self.SE.forward_aux(s, self.args.updim_relu)
    sd = self.d(adaptive_instance_normalization(cF_SE[-1], sF_SE[-1]))
    
    # for log
    sd_BE = 0
    if iter % self.args.save_interval == 0:
      stylized_cF_BE = adaptive_instance_normalization(cF_BE[-1], sF_BE[-1])
      sd_BE = self.d(stylized_cF_BE) # stylized image using BE's feature
    
    # (loss 1) BE -> SE knowledge transfer loss
    feat_loss = 0
    for i in range(len(cF_BE)):
      feat_loss += nn.MSELoss()(cF_SE[i], cF_BE[i].data)
      feat_loss += nn.MSELoss()(sF_SE[i], sF_BE[i].data)
    
    # (loss 2) eval the quality of stylized image
    sdF_BE1 = self.BE1(sd)
    sdF_BE2 = self.BE2(sdF_BE1)
    sdF_BE3 = self.BE3(sdF_BE2)
    sdF_BE4 = self.BE4(sdF_BE3)
    sdF = (sdF_BE1, sdF_BE2, sdF_BE3, sdF_BE4)
    
    content_loss = nn.MSELoss()(sdF[-1], cF_BE[-1].data) # using the feat of Conv4_1 to calculate content loss
    # content_loss = nn.MSELoss()(sdF[-1], stylized_cF_BE.data) # AdaIN's impel
    style_loss = 0
    for i in range(len(cF_BE)):
      # (a) use gram to measure style
      # style_loss += nn.MSELoss()(gram_matrix(sdF[i]), gram_matrix(sF_BE[i].data))
      # (b) use AdaIN's mean and variance to measure style
      mean_sd, std_sd = calc_mean_std(sdF[i])
      mean_s,  std_s  = calc_mean_std(sF_BE[i])
      style_loss += nn.MSELoss()(mean_sd, mean_s.data) + nn.MSELoss()(std_sd, std_s.data) # use mean and std as AdaIN
    return feat_loss, content_loss, style_loss, sd, sd_BE

class Train_SE_SD_Jointly_WCT(nn.Module):
  '''
    Train SE SD jointly for WCT. Not using BE's feat to guide SE.
  '''
  def __init__(self, args):
    super(Train_SE_SD_Jointly_WCT, self).__init__()
    self.BE = eval("Encoder%d" % args.stage)(args.BE, fixed=True) 
    self.SE = eval("SmallEncoder%d_%dx_aux" % (args.stage, args.speedup))(args.SE, fixed=False)
    self.SD = eval("SmallDecoder%d_%dx" % (args.stage, args.speedup))(fixed=False)
    self.args = args
    
  def forward(self, c, iter):
    rec = self.SD(self.SE(c))
    # loss (1) pixel loss
    rec_pixl_loss = nn.MSELoss()(rec, c.data)
    
    # loss (2) perceptual loss
    recF_BE = self.BE.forward_branch(rec)
    cF_BE = self.BE.forward_branch(c)
    rec_perc_loss = 0
    for i in range(len(recF_BE)):
      rec_perc_loss += nn.MSELoss()(recF_BE[i], cF_BE[i].data)
      
    return rec_pixl_loss, rec_perc_loss, rec
    
class Train_SE_SD_Jointly_WCT_KD(nn.Module):
  '''
    Train SE SD jointly for WCT. Using BE's feat to guide SE.
  '''
  def __init__(self, args):
    super(Train_SE_SD_Jointly_WCT_KD, self).__init__()
    self.BE = eval("Encoder%d" % args.stage)(args.BE, fixed=True)
    self.SE = eval("SmallEncoder%d_%dx_aux" % (args.stage, args.speedup))(args.SE, fixed=False)
    self.SD = eval("SmallDecoder%d_%dx" % (args.stage, args.speedup))(fixed=False)
    self.args = args
    
  def forward(self, c, iter):
    cF_BE = self.BE.forward_branch(c) # BE forward, multi outputs: relu1_1, 2_1, 3_1, 4_1, 5_1
    cF_SE = self.SE.forward_aux2(c) # SE forward, multi outputs: [relu1_1, 2_1, 3_1, 4_1, 5_1]
    rec   = self.SD(cF_SE[-1])
    
    # (loss 1) BE -> SE knowledge transfer loss
    feat_loss = 0
    for i in range(len(cF_BE)):
      feat_loss += nn.MSELoss()(cF_SE[i], cF_BE[i].data)
    
    # (loss 2, 3) eval the quality of reconstructed image, pixel and perceptual loss
    rec_pixl_loss = nn.MSELoss()(rec, c.data)
    recF_BE = self.BE.forward_branch(rec)
    rec_perc_loss = 0
    for i in range(len(recF_BE)):
      rec_perc_loss += nn.MSELoss()(recF_BE[i], cF_BE[i].data)
    return feat_loss, rec_pixl_loss, rec_perc_loss, rec, c

# --------------------------------------------------------
# Autoencoder summary
# QA means Quality Assurance, i.e., using the decoder regularizer. Without QA means without that regularizer.
Autoencoders = {
# train my decoders
"1BD":     Autoencoder1_BD,
"2BD":     Autoencoder2_BD,
"3BD":     Autoencoder3_BD,
"4BD":     Autoencoder4_BD,
"5BD":     Autoencoder5_BD,

# 16x smaller, train small encoder
"1SE_16x": Autoencoder1_SE_16x,
"2SE_16x": Autoencoder2_SE_16x,
"3SE_16x": Autoencoder3_SE_16x,
"4SE_16x": Autoencoder4_SE_16x,
"5SE_16x": Autoencoder5_SE_16x,
"1SE_16x_QA": Autoencoder1_SE_16x,
"2SE_16x_QA": Autoencoder2_SE_16x,
"3SE_16x_QA": Autoencoder3_SE_16x,
"4SE_16x_QA": Autoencoder4_SE_16x,
"5SE_16x_QA": Autoencoder5_SE_16x,

# 16x smaller, train small decoder
"1SD_16x": Autoencoder1_SD_16x,
"2SD_16x": Autoencoder2_SD_16x,
"3SD_16x": Autoencoder3_SD_16x,
"4SD_16x": Autoencoder4_SD_16x,
"5SD_16x": Autoencoder5_SD_16x,

# from scratch 16x model
"5SD_16x_fromscratch": Autoencoder5_16x_fromscratch,
"4SD_16x_fromscratch": Autoencoder4_16x_fromscratch,
"3SD_16x_fromscratch": Autoencoder3_16x_fromscratch,
"2SD_16x_fromscratch": Autoencoder2_16x_fromscratch,
"1SD_16x_fromscratch": Autoencoder1_16x_fromscratch,

# FP 16x model
"5SD_FP16x": Autoencoder5_SD_FP16x,
"4SD_FP16x": Autoencoder4_SD_FP16x,
"3SD_FP16x": Autoencoder3_SD_FP16x,
"2SD_FP16x": Autoencoder2_SD_FP16x,
"1SD_FP16x": Autoencoder1_SD_FP16x,
}