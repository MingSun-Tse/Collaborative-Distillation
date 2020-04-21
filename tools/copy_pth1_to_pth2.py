import os
import time
import sys
import torch
from model import SmallEncoder5_16x_plus
from model import SmallEncoder5_16x_plus_Gatys

# take model1's params to model2
# for each layer of model2, if model1 has the same layer, then copy the params.
def cut_pth(model1, model2):
  params1 = model1.named_parameters()
  params2 = model2.named_parameters()
  
  dict_params1 = dict(params1)
  dict_params2 = dict(params2)
  print("\n", dict_params1.keys())
  print("\n", dict_params2.keys())
  
  for name2 in dict_params2:
    if name2 in dict_params1:
      print("tensor '%s' found in both models, so copy it from model 1 to model 2" % name2)
      dict_params2[name2].data.copy_(dict_params1[name2].data)
  model2.load_state_dict(dict_params2)
  torch.save(model2.state_dict(), "model2.pth")

if __name__ == "__main__":
  e5_16x   = SmallEncoder5_16x_plus("models/5SE_16x_QA_E20S0.pth") # my model
  e5_16x_2 = SmallEncoder5_16x_plus_Gatys() # my model but re-formated
  cut_pth(e5_16x, e5_16x_2)