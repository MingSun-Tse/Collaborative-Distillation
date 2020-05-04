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
from data_loader import Dataset, ContentStylePair
from model.model import TrainSE_With_WCTDecoder, TrainSD_With_WCTSE
from utils import LogPrint, set_up_dir, get_CodeID, LogHub, check_path

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--content_train', type=str, default="data/COCO/train2014/")
  parser.add_argument('--style_train', type=str, default="data/WikiArt/train")
  parser.add_argument('--pretrained_init', action="store_true", help="if use L1-pruned models for SE initialization")
  parser.add_argument('--shorter_side', type=int, help='the shorter side of resized image', default=300)
  parser.add_argument('-b', '--batch_size', type=int, default=16)
  parser.add_argument('--lr', type=float, help='learning rate', default=1e-4) 
  parser.add_argument('--resume', type=str, default="")
  # --- model path
  parser.add_argument('--BE', type=str, default="", help="big encoder path")
  parser.add_argument('--BD', type=str, default="", help="big decoder path")
  parser.add_argument('--SE', type=str, default="", help="small encoder path")
  # --- loss weight
  parser.add_argument('--lw_style', type=float, default=10)
  parser.add_argument('--lw_content', type=float, default=1)
  parser.add_argument('--lw_feat', type=float, default=10)
  parser.add_argument('--lw_pixl', type=float, default=1)
  parser.add_argument('--lw_perc', type=float, default=1)
  # ---
  parser.add_argument('--save_interval', type=int, default=100)
  parser.add_argument('--print_interval', type=int, default=10)
  parser.add_argument('--epoch', type=int, default=20)
  parser.add_argument('-p', '--project_name', type=str, default="")
  parser.add_argument('--speedup', type=int, default=16)
  parser.add_argument('--debug', action="store_true", help="if debug, log will be printed to screen rather than saved")
  parser.add_argument('--screen', action="store_true", help="if print log to screen")
  parser.add_argument('--updim_relu', action="store_true", help="if use relu for the 1x1 conv")
  parser.add_argument('--mode', type=str, choices=['wct_se', 'wct_sd'])
  parser.add_argument('--stage', type=int, choices=[1,2,3,4,5])
  args = parser.parse_args()

  # set up log dirs
  TimeID, ExpID, rec_img_path, weights_path, log = set_up_dir(args.project_name, args.resume, args.debug)
  logprint = LogPrint(log, ExpID, args.screen)
  args.ExpID = ExpID
  args.CodeID = get_CodeID()
  loghub = LogHub()

  # Set up model, data, optimizer
  if args.mode == "wct_se":
    args.BE = "trained_models/original_wct_models/vgg_normalised_conv%d_1.t7" % args.stage
    args.BD = "trained_models/our_BD/%dBD_E30S0.pth" % args.stage
    if args.pretrained_init:
      args.SE = "trained_models/small16x_ae_base/e%d_base.pth" % args.stage
    net = TrainSE_With_WCTDecoder(args).cuda()
    dataset = Dataset(args.content_train, args.shorter_side)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
  
  elif args.mode == "wct_sd":
    args.BE = "trained_models/original_wct_models/vgg_normalised_conv%d_1.t7" % args.stage
    if args.pretrained_init:
        args.SD = "trained_models/small16x_ae_base/d%d_base.pth" % args.stage
    net = TrainSD_With_WCTSE(args).cuda()
    SE_path = check_path(args.SE)
    net.SE.load_state_dict(torch.load(SE_path)["model"])
    dataset = Dataset(args.content_train, args.shorter_side)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

  optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
  
  # Train
  logtmp = "{"
  for k in sorted(args.__dict__):
    logtmp += '"%s": %s, ' % (k, args.__dict__[k])
  logtmp = logtmp[:-2] + "}"
  logprint(logtmp) # print options to log for later check
  t1 = time.time()
  num_step_per_epoch = len(train_loader)
  for epoch in range(1, args.epoch+1):
    for step, (c, s) in enumerate(train_loader): # for WCT, s is not style image but the content image path
      if args.mode == "wct_se":
        c = c.cuda()
        feat_loss, rec_pixl_loss, rec_perc_loss, rec, _ = net(c, step)
        loss = feat_loss * args.lw_feat + rec_pixl_loss * args.lw_pixl + rec_perc_loss * args.lw_perc
        loghub.update("feat (*%s)" % args.lw_feat, feat_loss.item())
        loghub.update("pixl (*%s)" % args.lw_pixl, rec_pixl_loss.item())
        loghub.update("perc (*%s)" % args.lw_perc, rec_perc_loss.item())
      
      elif args.mode == "wct_sd":
        c = c.cuda()
        rec_pixl_loss, rec_perc_loss, rec = net(c, step)
        loss = rec_pixl_loss * args.lw_pixl + rec_perc_loss * args.lw_perc
        loghub.update("pixl (*%s)" % args.lw_pixl, rec_pixl_loss.item())
        loghub.update("perc (*%s)" % args.lw_perc, rec_perc_loss.item())
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
      if step % args.print_interval == 0:
        speed = (time.time() - t1) / args.print_interval
        logtmp = "E%dS%d " % (epoch, step) + loghub.format() + " (%.2f s/step)" % speed
        logprint(logtmp)
        t1 = time.time()
      
      # save image samples
      if step % args.save_interval == 0:
        out_img_path = pjoin(rec_img_path, "%s_E%sS%s.jpg" % (TimeID, epoch, step))
        if "wct" in args.mode:
          save_img = torch.cat([c, rec], dim=0)
        vutils.save_image(save_img, out_img_path, nrow=args.batch_size)
      
      # save model
      if step == num_step_per_epoch - 1:
        if "se" in args.mode:
          f = {"epoch": epoch, "model": net.SE.state_dict()}
        elif "sd" in args.mode:
          f = {"epoch": epoch, "model": net.SD.state_dict()}
        torch.save(f, pjoin(weights_path, "%s.pth" % ExpID))