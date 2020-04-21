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
from data_loader import Dataset, Dataset_npy
from data_loader import TestDataset, is_img, ContentStylePair
from model import Autoencoder5_Stage2, LearnTranformAndDecoder
from model import TrainSE_With_AdaINDecoder, TrainSE_With_AdaINDecoder2, TrainSE_With_WCTDecoder # collaborative distillation
from model import Train_SE_SD_Jointly_WCT_KD
from my_utils import LogPrint, set_up_dir, get_CodeID, LogHub

'''
This file is to improve the method based on the idea "involving NST into the compression process directly".
So I test it first on AdaIN.
'''

def gram(input):
  a, b, c, d = input.size()
  batch_feat = input.view(a, b, c*d)
  batch_gram = torch.stack([torch.mm(feat, feat.t()).div(c*d) for feat in batch_feat])
  return batch_gram # shape: [batch_size, channel, channel]
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_data', type=str, help='the directory of train images', default="../../../Dataset/COCO/train2014")
  parser.add_argument('--content_train', type=str, default="../../../Dataset/COCO/train2014/")
  parser.add_argument('--content_train_npy', type=str, default="../../../Dataset/COCO/train2014_npy/")
  parser.add_argument('--style_train', type=str, default="../../../Dataset/WikiArt/train")
  parser.add_argument('--style_train_npy', type=str, default="../../../Dataset/WikiArt/train_npy")
  parser.add_argument('--BE', type=str, default="../PytorchWCT/models/vgg_normalised_conv5_1.t7")
  parser.add_argument('--SE', type=str, default="")
  parser.add_argument('-D', '--Dec', type=str, default="")
  parser.add_argument('--pretrained_init', action="store_true")
  parser.add_argument('--shorter_side', type=int, help='the shorter side of resized image', default=300)
  parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=16)
  parser.add_argument('--lr', type=float, help='learning rate', default=1e-4) 
  parser.add_argument('-r', '--resume', type=str, default="")
  parser.add_argument('--lw_style', type=float, default=10) # Default value refers to AdaIN
  parser.add_argument('--lw_content', type=float, default=1) # Default value refers to AdaIN
  parser.add_argument('--lw_feat', type=float, default=10)
  parser.add_argument('--lw_pixl', type=float, default=10)
  parser.add_argument('--lw_perc', type=float, default=10)
  parser.add_argument('--save_interval', type=int, default=100)
  parser.add_argument('--show_interval', type=int, default=10)
  parser.add_argument('--epoch', type=int, default=20)
  parser.add_argument('--CodeID', type=str, default="")
  parser.add_argument('-p', '--project_name', type=str, default="")
  parser.add_argument('--speedup', type=int, default=16)
  parser.add_argument('--debug', action="store_true")
  parser.add_argument('--updim_relu', action="store_true")
  parser.add_argument('--mode', type=str)
  parser.add_argument('--stage', type=int)
  args = parser.parse_args()

  # set up log dirs
  TimeID, ExpID, rec_img_path, weights_path, log = set_up_dir(args.project_name, args.resume, args.debug)
  logprint = LogPrint(log, ExpID)
  args.ExpID = ExpID
  args.CodeID = get_CodeID()
  logtmp = "{"
  for k in sorted(args.__dict__):
    logtmp += '"%s": %s, ' % (k, args.__dict__[k])
  logtmp = logtmp[:-2] + "}"
  logprint(logtmp)
  loghub = LogHub()

  # Set up model, data loader, optim
  if args.mode == "adain":
    args.Dec = "../PytorchAdaIN/experiments_Decoder4_2/decoder_iter_160000.pth.tar"
    dataset = ContentStylePair(args.content_train, args.style_train, args.shorter_side)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    net = TrainSE_With_AdaINDecoder(args).cuda()
  if args.mode == "adain2":
    args.Dec = "../PytorchAdaIN/decoder_iter_160000.pth.tar"
    args.BE  = "../PytorchAdaIN/models/vgg_normalised.pth"
    dataset = ContentStylePair(args.content_train, args.style_train, args.shorter_side)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    net = TrainSE_With_AdaINDecoder2(args).cuda()
  elif args.mode == "wct":
    args.Dec = "models/my_decoder/%dBD_E30S0.pth" % args.stage
    if args.pretrained_init:
      args.SE = "models/small16x_ae_base/e%d_base.pth" % args.stage
    # dataset = Dataset(args.content_train, args.shorter_side) # use raw image
    dataset = Dataset_npy(args.content_train_npy) # use npy-format image
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    net = TrainSE_With_WCTDecoder(args).cuda()
  elif args.mode == "joint_se_sd_wct_kd":
    args.BE = "../PytorchWCT/models/vgg_normalised_conv%d_1.t7" % args.stage
    if args.pretrained_init:
      args.SE = "models/small16x_ae_base/e%d_base.pth" % args.stage
    dataset = Dataset_npy(args.content_train_npy)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    net = Train_SE_SD_Jointly_WCT_KD(args).cuda()
  optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
  
  # Train
  t1 = time.time()
  num_step_per_epoch = len(train_loader)
  for epoch in range(1, args.epoch+1):
    for step, (c, s) in enumerate(train_loader): # for WCT, s is not style image but content image path
      if "adain" in args.mode:
        c = c.cuda(); s = s.cuda()
        feat_loss, content_loss, style_loss, stylized_img, stylized_img_baseline = net(c, s, step)
        loss = feat_loss * args.lw_feat + content_loss * args.lw_content +  style_loss * args.lw_style
        loghub.update("feat (*%s)" % args.lw_feat, feat_loss.item())
        loghub.update("cont (*%s)" % args.lw_content, content_loss.item())
        loghub.update("styl (*%s)" % args.lw_style, style_loss.item())
        
      elif args.mode in ["wct", "joint_se_sd_wct_kd"]:
        c = c.cuda()
        feat_loss, rec_pixel_loss, rec_perc_loss, rec, _ = net(c, step)
        loss = feat_loss * args.lw_feat + rec_pixel_loss * args.lw_pixl + rec_perc_loss * args.lw_perc
        loghub.update("feat (*%s)" % args.lw_feat, feat_loss.item())
        loghub.update("pixl (*%s)" % args.lw_pixl, rec_pixel_loss.item())
        loghub.update("perc (*%s)" % args.lw_perc, rec_perc_loss.item())
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
      if step % args.show_interval == 0:
        speed = (time.time() - t1) / args.show_interval
        logtmp = "E%dS%d " % (epoch, step) + loghub.format() + " (%.2f s/step)" % speed
        logprint(logtmp)
        t1 = time.time()
      
      # save image samples
      if step % args.save_interval == 0:
        out_img_path = pjoin(rec_img_path, "%s_E%sS%s.jpg" % (TimeID, epoch, step))
        if "adain" in args.mode:
          save_img = torch.cat([c, s, stylized_img, stylized_img_baseline], dim=0)
        elif args.mode in ["wct", "joint_se_sd_wct_kd"]:
          save_img = torch.cat([c, rec], dim=0)
        vutils.save_image(save_img, out_img_path, nrow=args.batch_size)
      
      # save model
      if step == num_step_per_epoch - 1:
        file = {"epoch": epoch, "model": net.SE.state_dict()}
        if args.mode == "joint_se_sd_wct_kd":
          model = {}
          model["SE"] = net.SE.state_dict()
          model["SD"] = net.SD.state_dict()
          file = {"epoch": epoch, "model": model}
        torch.save(file, pjoin(weights_path, "%s.pth" % ExpID))