import sys
import numpy as np
import os
import torch.nn as nn
import torch
from torch.utils.serialization import load_lua
pjoin = os.path.join
from model_MobileNet import Encoder5, Encoder4, Encoder3, Encoder2, Encoder1


tensor_map_5 = {
"conv11":"0.0", "bn11":"0.1",
"conv21":"1.0", "bn21":"1.1", "conv22":"1.3", "bn22":"1.4", 
"conv31":"2.0", "bn31":"2.1", "conv32":"2.3", "bn32":"2.4", 
"conv41":"3.0", "bn41":"3.1", "conv42":"3.3", "bn42":"3.4", 
"conv51":"4.0", "bn51":"4.1", "conv52":"4.3", "bn52":"4.4", 
"conv61":"5.0", "bn61":"5.1", "conv62":"5.3", "bn62":"5.4", 
"conv71":"6.0", "bn71":"6.1", "conv72":"6.3", "bn72":"6.4", 
"conv81":"7.0", "bn81":"7.1", "conv82":"7.3", "bn82":"7.4", 
"conv91":"8.0", "bn91":"8.1",
}

tensor_map_4 = {
"conv11":"0.0", "bn11":"0.1",
"conv21":"1.0", "bn21":"1.1", "conv22":"1.3", "bn22":"1.4", 
"conv31":"2.0", "bn31":"2.1", "conv32":"2.3", "bn32":"2.4", 
"conv41":"3.0", "bn41":"3.1", "conv42":"3.3", "bn42":"3.4", 
"conv51":"4.0", "bn51":"4.1", "conv52":"4.3", "bn52":"4.4", 
"conv61":"5.0", "bn61":"5.1", "conv62":"5.3", "bn62":"5.4", 
"conv71":"6.0", "bn71":"6.1",
}

tensor_map_3 = {
"conv11":"0.0", "bn11":"0.1",
"conv21":"1.0", "bn21":"1.1", "conv22":"1.3", "bn22":"1.4", 
"conv31":"2.0", "bn31":"2.1", "conv32":"2.3", "bn32":"2.4", 
"conv41":"3.0", "bn41":"3.1", "conv42":"3.3", "bn42":"3.4", 
"conv51":"4.0", "bn51":"4.1",
}

tensor_map_2 = {
"conv11":"0.0", "bn11":"0.1",
"conv21":"1.0", "bn21":"1.1", "conv22":"1.3", "bn22":"1.4", 
"conv31":"2.0", "bn31":"2.1",
}

tensor_map_1 = {
"conv11":"0.0", "bn11":"0.1",
}

# load original model
original_model = sys.argv[1] ## param 1 is the original mobilenet path
model = torch.load(original_model)["state_dict"]

# my model
Encoders = [Encoder1, Encoder2, Encoder3, Encoder4, Encoder5]
for i in range(1, 6):
  print("=====> processing encoder %s" % i)
  dict_params = Encoders[i-1]().state_dict()
  tensor_map = eval("tensor_map_" + str(i))
  for tensor in tensor_map:
    print("processing tensor " + tensor)
    if "conv" in tensor:
      dict_params[tensor + ".weight"].data.copy_(model["module.model." + tensor_map[tensor] + ".weight"])
    else:
      dict_params[tensor + ".weight"].data.copy_(model["module.model." + tensor_map[tensor] + ".weight"])
      dict_params[tensor + ".bias"].data.copy_(model["module.model." + tensor_map[tensor] + ".bias"])
      dict_params[tensor + ".running_mean"].data.copy_(model["module.model." + tensor_map[tensor] + ".running_mean"])
      dict_params[tensor + ".running_var"].data.copy_(model["module.model." + tensor_map[tensor] + ".running_var"])
  torch.save(dict_params, original_model.replace(".pth", "_my_e%s.pth" % i))




