from __future__ import print_function
import torch
import sys
from PIL import Image
import torchvision.utils as vutils
import torchvision.transforms as transforms
import os
import numpy as np
from my_utils import is_img
from model import SmallEncoder2_16x_aux, SmallDecoder2_16x

#########################################################
# demonstrate collaboration 2019-11-12
# E1:     relu conv2_1 16x D1: E1's decoder
# E2: non-relu conv2_1 16x D2: E2's decoder
E1_model = "../PytorchWCT/models/16x_models/2SE_16x_QA_E19S6000.pth"
D1_model = "../PytorchWCT/models/16x_models/2SD_16x_QA_E25S0.pth"
E2_model = "Experiments/SERVER218-20191110-132713_wct_se_stage2_16x/weights/20191110-132713_E20.pth"
D2_model = "Experiments/SERVER218-20191111-091932_sd_16x_wct_stage2/weights/20191111-091932_E20.pth"

if sys.argv[1] == "E1D1":
  E = SmallEncoder2_16x_aux(E1_model).cuda()
  D = SmallDecoder2_16x(D1_model).cuda()
  
if sys.argv[1] == "E2D2":
  E = SmallEncoder2_16x_aux(E2_model).cuda()
  D = SmallDecoder2_16x(D2_model).cuda()
  
if sys.argv[1] == "E1D2":
  E = SmallEncoder2_16x_aux(E1_model).cuda()
  D = SmallDecoder2_16x(D2_model).cuda()
  
if sys.argv[1] == "E2D1":
  E = SmallEncoder2_16x_aux(E2_model).cuda()
  D = SmallDecoder2_16x(D1_model).cuda()
#########################################################

img = Image.open("../PytorchWCT/images/content/in2.jpg").convert("RGB")
img = transforms.ToTensor()(img).cuda().unsqueeze(0)
rec = D(E(img))
vutils.save_image(rec.data.cpu(), "%s.jpg" % sys.argv[1])

