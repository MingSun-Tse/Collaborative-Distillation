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
# torch
import torch
import torch.nn as nn
from torch.utils.serialization import load_lua
import torch.utils.data as Data
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
# my libs
sys.path.insert(0, './')
from data_loader import Dataset, is_img
from model.model_original import Encoder5, Encoder4, Encoder3, Encoder2, Encoder1 # original big encoder
from model.model_original import Decoder5, Decoder4, Decoder3, Decoder2, Decoder1 # original big decoder
from model.model_cd import SmallEncoder5_16x_aux, SmallEncoder4_16x_aux, SmallEncoder3_16x_aux, SmallEncoder2_16x_aux, SmallEncoder1_16x_aux # small encoder
from model.model_cd import SmallDecoder5_16x, SmallDecoder4_16x, SmallDecoder3_16x, SmallDecoder2_16x, SmallDecoder1_16x # small decoder

def filter_prune(weight, num_keeped_row, use_channel=False):
  if not use_channel:
    weight_ = weight.reshape(weight.shape[0], -1)
    abs_row = np.abs(weight_).sum(1)
    order = np.argsort(abs_row)[-num_keeped_row:]
  else:
    abs_channel = np.abs(weight).sum(0).sum(2).sum(1)
    order = np.argsort(abs_channel)[-num_keeped_row:]
  return order

# change layer name from Yijun's model to new name
name_changer_encoder = {
"conv1": "conv0",
"conv2": "conv11",
"conv3": "conv12",
"conv4": "conv21",
"conv5": "conv22",
"conv6": "conv31",
"conv7": "conv32",
"conv8": "conv33",
"conv9": "conv34",
"conv10": "conv41",
"conv11": "conv42",
"conv12": "conv43",
"conv13": "conv44",
"conv14": "conv51",
}

name_changer_decoder = {
"conv15": "conv51",
"conv16": "conv44",
"conv17": "conv43",
"conv18": "conv42",
"conv19": "conv41",
"conv20": "conv34",
"conv21": "conv33",
"conv22": "conv32",
"conv23": "conv31",
"conv24": "conv22",
"conv25": "conv21",
"conv26": "conv12",
"conv27": "conv11",
}

if __name__ == "__main__":
  # Passed-in params
  parser = argparse.ArgumentParser(description="Prune")
  parser.add_argument('-m', '--model', type=str, default='trained_models/original_wct_models/vgg_normalised_conv5_1.t7', help='path of the weights of big model')
  parser.add_argument('-o', '--output', type=str)
  parser.add_argument('--unpruned_model_arch', type=str, default='Encoder5', choices=['Encoder5', 'Encoder4', 'Encoder3', 'Encoder2', 'Encoder1'])
  parser.add_argument('--pruned_model_arch', type=str, default='SmallEncoder5_16x_aux', choices=['SmallEncoder5_16x_aux', 'SmallEncoder4_16x_aux', 'SmallEncoder3_16x_aux', 'SmallEncoder2_16x_aux', 'SmallEncoder1_16x_aux'])
  args = parser.parse_args()

  # set up model
  model = eval(args.unpruned_model_arch)(args.model)
  small_model = eval(args.pruned_model_arch)()

  # get params (weights)
  params, params_s = model.state_dict(), small_model.state_dict()
  keeped_row_index = {}
  layers = []
  for tensor_name, tensor in params.items():
    if "bn" in tensor_name: 
      print("\n====> encounter a bn tensor (%s), continue" % tensor_name)
      continue
    
    layer_name = tensor_name.split(".")[0]
    small_tensor_name = tensor_name
    print("\n====> now processing tensor '%s', shape = %s vs. %s" % (tensor_name, str(tensor.shape), str(params_s[small_tensor_name].shape)))
    # shape is like: torch.Size([64, 128, 3, 3]), the **first** is number of filters.
    
    if len(tensor.shape) == 4: # weights
      last_layer_name = layers[-1] if len(layers) else "None" # the first layer
      layers.append(layer_name) # keep layer name for index
      # get keeped_row_index
      if tensor.shape[0] != params_s[small_tensor_name].shape[0]: # filter number
        keeped_row_index[layer_name] = filter_prune(tensor, params_s[small_tensor_name].shape[0]) # prune by L1-norm
      else:
        keeped_row_index[layer_name] = range(tensor.shape[0])
      print("keeped row index:", keeped_row_index[layer_name])

      # squeeze the remaining weights into the small tensor
      if last_layer_name != "None": # not the first conv layer
        params_s[small_tensor_name] = tensor[keeped_row_index[layer_name]][:, keeped_row_index[last_layer_name], :, :]
      else:
        # for encoder pruning, the first conv layers have the same input dimension, usually 3, i.e., the channel number of image
        if tensor.shape[1] == params_s[small_tensor_name].shape[1]: # column number is the same
          params_s[small_tensor_name] = tensor[keeped_row_index[layer_name]]
        # for decoder pruning, input channels are not the same
        else:
          keeped_col_index = filter_prune(tensor, params_s[small_tensor_name].shape[1], True)
          print(keeped_col_index)
          params_s[small_tensor_name] = tensor[keeped_row_index[layer_name]][:, keeped_col_index, :, :]
        
    else: # biases
      params_s[small_tensor_name] = tensor[keeped_row_index[layer_name]]
      
  torch.save(params_s, args.output)


'''Usage:
At the code root path, run: "python tools/prune.py --output Encoder5_pruned.pth"
For other models, change "--unpruned_model_arch, --pruned_model_arch, --model, --output" accordingly.
'''