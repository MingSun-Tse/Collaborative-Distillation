import torch
import torch.nn as nn
import time
import sys
import os
import numpy as np

'''
    This file is to test practical speedup on AdaIN for CVPR-20 Rebuttal.
    No dependency, all is here.
    Usage: !CUDA_VISIBLE_DEVICES=1 python <this_file> 3000 10
2020-02-05 Log:
    processing time of original model: 1.8775s
    processing time of compress model: 0.2556s
    processing time of original model: 1.8440s
    processing time of compress model: 0.2560s
    processing time of original model: 1.8448s
    processing time of compress model: 0.2559s
    processing time of original model: 1.8492s
    processing time of compress model: 0.2561s
    processing time of original model: 1.8456s
    processing time of compress model: 0.2558s
    processing time of original model: 1.8462s
    processing time of compress model: 0.2572s
    processing time of original model: 1.8502s
    processing time of compress model: 0.2584s
    processing time of original model: 1.8541s
    processing time of compress model: 0.2571s
    processing time of original model: 1.8599s
    processing time of compress model: 0.2576s
    processing time of original model: 1.8598s
    processing time of compress model: 0.2575s
    summary: mean run time of original model: 1.8531s, compress model: 0.2567s
'''


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

########################################
# set up dummy input
img_size = int(sys.argv[1])
num_run = int(sys.argv[2])

C = torch.randn([1, 3, img_size, img_size]).cuda()
S = torch.randn([1, 3, img_size, img_size]).cuda()

time_original = []
time_compress = []

@torch.no_grad()
def run():
    ####### original model
    enc = Encoder4().cuda()
    dec = Decoder4().cuda()
    t1 = time.time()
    cF = enc(C); torch.cuda.empty_cache() # same as the test in WCT, use empty_cache()
    sF = enc(S); torch.cuda.empty_cache() 
    _ = dec(adaptive_instance_normalization(cF, sF)); torch.cuda.empty_cache()
    t = time.time() - t1
    time_original.append(t)
    print("processing time of original model: %.4fs" % t)


    ####### compressed model
    small_enc = SmallEncoder4_16x_aux().cuda()
    small_dec = SmallDecoder4_16x().cuda()
    t1 = time.time()
    cF = small_enc(C); torch.cuda.empty_cache()
    sF = small_enc(S); torch.cuda.empty_cache()
    _ = small_dec(adaptive_instance_normalization(cF, sF)); torch.cuda.empty_cache()
    t = time.time() - t1
    time_compress.append(t)
    print("processing time of compress model: %.4fs" % t)

for _ in range(num_run):
    run()
print("summary: mean run time of original model: %.4fs, compress model: %.4fs" % (np.mean(time_original), np.mean(time_compress)))
########################################