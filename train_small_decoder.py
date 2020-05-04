from __future__ import print_function
from utils import LogPrint, set_up_dir, get_CodeID, LogHub, smart_load
from model import TrainSE_With_AdaINDecoder, TrainSE_With_WCTDecoder, TrainSD_With_WCTSE, TrainSD_With_FPSE, Train_SE_SD_Jointly_WCT
from model import TrainSD_With_WCTSE_KDtoSD, TrainBD
from data_loader import TestDataset, is_img
from data_loader import Dataset, Dataset_npy
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data as Data
from torch.utils.serialization import load_lua
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import numpy as np
import argparse
import time
import shutil
import sys
import os
import glob
import pickle
pjoin = os.path.join

'''
This file is to improve the method based on the idea "involving NST into the compression process directly".
So I test it first on AdaIN.
'''

def gram(input):
    a, b, c, d = input.size()
    batch_feat = input.view(a, b, c*d)
    batch_gram = torch.stack(
        [torch.mm(feat, feat.t()).div(c*d) for feat in batch_feat])
    return batch_gram  # shape: [batch_size, channel, channel]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, help='the directory of train images',
                        default="../../../Dataset/COCO/train2014")
    parser.add_argument('--content_train', type=str,
                        default="../../../Dataset/COCO/train2014/")
    parser.add_argument('--content_train_npy', type=str,
                        default="../../../Dataset/COCO/train2014_npy/")
    parser.add_argument('--style_train', type=str,
                        default="../../../Dataset/WikiArt/train")
    parser.add_argument('--style_train_npy', type=str,
                        default="../../../Dataset/WikiArt/train_npy")
    parser.add_argument(
        '--BE', type=str, default="../PytorchWCT/models/vgg_normalised_conv5_1.t7")
    parser.add_argument('--SE', type=str, default="")
    parser.add_argument('--SD', type=str, default="")
    parser.add_argument('-D', '--Dec', type=str, default="")
    parser.add_argument('--pretrained_init', action="store_true")
    parser.add_argument('--shorter_side', type=int,
                        help='the shorter side of resized image', default=300)
    parser.add_argument('-b', '--batch_size', type=int,
                        help='batch size', default=16)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
    parser.add_argument('-r', '--resume', type=str, default="")
    # Default value refers to AdaIN
    parser.add_argument('--lw_style', type=float, default=10)
    # Default value refers to AdaIN
    parser.add_argument('--lw_content', type=float, default=1)
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
    TimeID, ExpID, rec_img_path, weights_path, log = set_up_dir(
        args.project_name, args.resume, args.debug)
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
    # dataset = Dataset(args.content_train, args.shorter_side) # use raw image
    dataset = Dataset_npy(args.content_train_npy)  # use npy-format image
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True)
    if args.mode == "adain":
        raise NotImplementedError
    
    elif args.mode == "wct":
        args.BE = "../PytorchWCT/models/vgg_normalised_conv%d_1.t7" % args.stage
        if args.pretrained_init:
            args.SD = "models/small16x_ae_base/d%d_base.pth" % args.stage
        net = TrainSD_With_WCTSE(args).cuda()
        net.SE.load_state_dict(torch.load(args.SE)["model"])
    
    elif args.mode == "wct_new_normalized_vgg":
        args.BE = ""
        net = TrainBD(args).cuda()
        model = "models/normalise_original_mine/weights.pkl"
        with open(model, 'rb') as f:
            weights = pickle.load(f, encoding='latin1')  # to load pickle2 file
            for k, v in weights.items():
                print("loading pkl weights for layer %s" % k)  # example: conv4_2_weight, conv4_2_bias
                layer_name, w_or_b = k[:7], k[7:]
                layer = layer_name[:5] + layer_name[6]
                if hasattr(net.SE, layer):
                    layer_tensor = eval("net.SE.%s" % layer)
                    v = torch.from_numpy(v)
                    if "weight" in w_or_b:
                        layer_tensor.weight.data.copy_(v)
                    else:
                        layer_tensor.bias.data.copy_(v)
        torch.save(net.SE.state_dict(), "models/normalise_original_mine/SE%d.pth" % args.stage)
        logprint("given pkl model, save pth model done")

    elif args.mode == "joint_se_sd_wct":
        args.BE = "../PytorchWCT/models/vgg_normalised_conv%d_1.t7" % args.stage
        if args.pretrained_init:
            args.SE = "models/small16x_ae_base/e%d_base.pth" % args.stage
        net = Train_SE_SD_Jointly_WCT(args).cuda()
    
    elif args.mode == "fp":
        net = TrainSD_With_FPSE(args).cuda()
        model = "models/normalise_fp16x_2/weights.pkl"
        with open(model, 'rb') as f:
            weights = pickle.load(f, encoding='latin1')  # to load pickle2 file
            for k, v in weights.items():
                print("loading pkl weights for layer %s" % k)  # example: conv4_2_weight, conv4_2_bias
                layer_name, w_or_b = k[:7], k[7:]
                layer = layer_name[:5] + layer_name[6]
                if hasattr(net.SE, layer):
                    layer_tensor = eval("net.SE.%s" % layer)
                    v = torch.from_numpy(v)
                    if "weight" in w_or_b:
                        layer_tensor.weight.data.copy_(v)
                    else:
                        layer_tensor.bias.data.copy_(v)
        torch.save(net.SE.state_dict(), "models/normalise_fp16x_2/SE%d.pth" % args.stage)
        logprint("given pkl model, save pth model done")
    
    elif args.mode == "16x_kd2sd":
        args.BE = "models/WCT_models/vgg_normalised_conv%d_1.t7" % args.stage
        SE_path = "models/wct_se_16x_new/%dSE.pth" % args.stage
        net = TrainSD_With_WCTSE_KDtoSD(args).cuda()
        net.SE.load_state_dict(smart_load(SE_path))
        BD_path = "models/my_BD/%dBD_E30S0.pth" % args.stage
        net.BD.load_state_dict(smart_load(BD_path))
    optimizer=torch.optim.Adam(net.parameters(), lr = args.lr)

    # Train
    t1=time.time()
    num_step_per_epoch=len(train_loader)
    for epoch in range(1, args.epoch + 1):
        # for WCT, s is not style image but content image path
        for step, (c, s) in enumerate(train_loader):
            if args.mode == "adain":
                raise NotImplementedError

            elif args.mode in ["wct", "wct_new_normalized_vgg", "joint_se_sd_wct", "fp"]:
                c = c.cuda()
                rec_pixl_loss, rec_perc_loss, rec = net(c, step)
                loss = rec_pixl_loss * args.lw_pixl + rec_perc_loss * args.lw_perc
                loghub.update("pixl (*%s)" %
                              args.lw_pixl, rec_pixl_loss.item())
                loghub.update("perc (*%s)" %
                              args.lw_perc, rec_perc_loss.item())

            elif args.mode in ["16x_kd2sd"]:
                c = c.cuda()
                rec_pixl_loss, rec_perc_loss, kd_feat_loss, rec = net(c, step)
                loss = rec_pixl_loss * args.lw_pixl + rec_perc_loss * args.lw_perc + kd_feat_loss * args.lw_feat
                loghub.update("pixl (*%s)" %
                              args.lw_pixl, rec_pixl_loss.item())
                loghub.update("perc (*%s)" %
                              args.lw_perc, rec_perc_loss.item())
                loghub.update("feat (*%s)" %
                              args.lw_feat, kd_feat_loss.item())
            
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
                out_img_path = pjoin(
                    rec_img_path, "%s_E%sS%s.jpg" % (TimeID, epoch, step))
                if args.mode == "adain":
                    raise NotImplementedError
                elif args.mode in ["wct", "wct_new_normalized_vgg", "joint_se_sd_wct", "fp", "16x_kd2sd"]:
                    save_img = torch.cat([c, rec], dim=0)
                vutils.save_image(save_img, out_img_path, nrow=args.batch_size)

            # save model
            if step == num_step_per_epoch - 1:
                file = {"epoch": epoch, "model": net.SD.state_dict()}
                if args.mode == "joint_se_sd_wct":
                    model = {}
                    model["SE"] = net.SE.state_dict()
                    model["SD"] = net.SD.state_dict()
                    file = {"epoch": epoch, "model": model}
                torch.save(file, pjoin(weights_path, "%s.pth" % ExpID))