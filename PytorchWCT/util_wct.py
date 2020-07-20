import torch
import torch.nn as nn
# from torch.utils.serialization import load_lua
import torchvision.transforms as transforms
import argparse
import time
import os
from PIL import Image
from numpy import linalg
import numpy as np

# original model (unpruned vgg-19)
# from model.model_original import Encoder1, Encoder2, Encoder3, Encoder4, Encoder5
# from model.model_original import Decoder1, Decoder2, Decoder3, Decoder4, Decoder5

# 16x model
from model.model_cd import SmallEncoder1_16x_aux, SmallEncoder2_16x_aux, SmallEncoder3_16x_aux, SmallEncoder4_16x_aux, SmallEncoder5_16x_aux
from model.model_cd import SmallDecoder1_16x,     SmallDecoder2_16x,     SmallDecoder3_16x,     SmallDecoder4_16x,     SmallDecoder5_16x

# 16x model kd2sd (apply kd to the small decoder)
from model.model_kd2sd import SmallDecoder1_16x_aux, SmallDecoder2_16x_aux, SmallDecoder3_16x_aux, SmallDecoder4_16x_aux, SmallDecoder5_16x_aux

##########################
## some ratios to explore how the number of eigenvalues influences the stylized quality
EigenValueThre = 1e-100 # the eigenvalues below this threshold will be discarded
NumEigenValue  = 30 # the number of kept eigenvalues
RatEigenValue  = 0.25 # ratio of kept eigenvalues
##########################

class WCT(nn.Module):
    def __init__(self, args):
        super(WCT, self).__init__()
        self.args = args

        # load pre-trained models
        # if args.mode == None or args.mode == "original":
        #     self.e1 = Encoder1(args.e1); self.d1 = Decoder1(args.d1)
        #     self.e2 = Encoder2(args.e2); self.d2 = Decoder2(args.d2)
        #     self.e3 = Encoder3(args.e3); self.d3 = Decoder3(args.d3)
        #     self.e4 = Encoder4(args.e4); self.d4 = Decoder4(args.d4)
        #     self.e5 = Encoder5(args.e5); self.d5 = Decoder5(args.d5)

        if args.mode == "16x":
            self.e5 = SmallEncoder5_16x_aux(args.e5); self.d5 = SmallDecoder5_16x(args.d5)
            self.e4 = SmallEncoder4_16x_aux(args.e4); self.d4 = SmallDecoder4_16x(args.d4)
            self.e3 = SmallEncoder3_16x_aux(args.e3); self.d3 = SmallDecoder3_16x(args.d3)
            self.e2 = SmallEncoder2_16x_aux(args.e2); self.d2 = SmallDecoder2_16x(args.d2)
            self.e1 = SmallEncoder1_16x_aux(args.e1); self.d1 = SmallDecoder1_16x(args.d1)
          
        elif args.mode == "16x_kd2sd":
            self.e5 = SmallEncoder5_16x_aux(args.e5); self.d5 = SmallDecoder5_16x_aux(args.d5)
            self.e4 = SmallEncoder4_16x_aux(args.e4); self.d4 = SmallDecoder4_16x_aux(args.d4)
            self.e3 = SmallEncoder3_16x_aux(args.e3); self.d3 = SmallDecoder3_16x_aux(args.d3)
            self.e2 = SmallEncoder2_16x_aux(args.e2); self.d2 = SmallDecoder2_16x_aux(args.d2)
            self.e1 = SmallEncoder1_16x_aux(args.e1); self.d1 = SmallDecoder1_16x_aux(args.d1)
        
        else:
            print("Wrong mode. Please check.")
            exit(1)
    
    # WCT with torch matrix multiplication
    def whiten_and_color_torch(self, cF, sF):
        # print("*" * 30 + " whiten_and_color begin")
        
        # ---------------------------------
        # svd for content feature
        cFSize = cF.size() # size: c * hw
        c_mean = torch.mean(cF, 1).unsqueeze(1).expand_as(cF)
        cF = cF - c_mean
        contentConv = torch.mm(cF, cF.t()).div(cFSize[1] - 1) # + torch.eye(cFSize[0]).double()
        # print("-" * 10 + " contentConv torch:")
        # print("convariance matrix abs sum:", np.abs(contentConv).sum().data.cpu().numpy()) # checked, same
        
        c_u, c_e, c_v = torch.svd(contentConv, some=False)
        # print("-" * 10 + " content svd torch:")
        # print("size of U, E, V:", c_u.size(), c_e.size(), c_v.size())
        # print("eigen values:", c_e.data.cpu().numpy())
        # print(np.abs(c_u).sum().data.cpu().numpy()) # checked, different
        # print(np.abs(c_e).sum().data.cpu().numpy())
        # print(np.abs(c_v).sum().data.cpu().numpy())
        
        k_c = cFSize[0]
        for i in range(cFSize[0]):
            if c_e[i] < EigenValueThre:
                k_c = i
                break
        # k_c = NumEigenValue
        # k_c = int(cFSize[0] * RatEigenValue)
        # print("k_c = %s\n" % k_c)
        
        # ---------------------------------
        # svd for style feature
        sFSize = sF.size()
        s_mean = torch.mean(sF, 1)
        sF = sF - s_mean.unsqueeze(1).expand_as(sF)
        styleConv = torch.mm(sF,sF.t()).div(sFSize[1] - 1)
        # print("-" * 10 + " styleConv torch:")
        # print("convariance matrix abs sum:", np.abs(styleConv).sum().data.cpu().numpy()) # checked, same
        
        s_u, s_e, s_v = torch.svd(styleConv, some=False);
        # print("-" * 5 + " style svd torch:")
        # print("size of U, E, V:", s_u.shape, s_e.shape, s_v.shape)
        # print("eigen values:", s_e.data.cpu().numpy())
        # print(np.abs(s_u).sum().data.cpu().numpy())
        # print(np.abs(s_e).sum().data.cpu().numpy())
        # print(np.abs(s_v).sum().data.cpu().numpy())
        
        k_s = sFSize[0]
        for i in range(sFSize[0]):
            if s_e[i] < EigenValueThre:
                k_s = i
                break
        # k_s = NumEigenValue
        # k_s = int(sFSize[0] * RatEigenValue)
        # print("k_s = %s\n" % k_s)
        
        c_d = (c_e[0:k_c]).pow(-0.5)
        step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
        step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
        whiten_cF = torch.mm(step2, cF)
        # print("-" * 5 + " whiten_cF torch")
        # print("whitened content abs sum:", np.abs(whiten_cF).sum().data.cpu().numpy()) # checked, same
        
        s_d = (s_e[0:k_s]).pow(0.5)
        targetFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t())), whiten_cF)
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
        # print("-" * 5 + " targetFeature torch")
        # print("stylized feature abs sum:", np.abs(targetFeature).sum().data.cpu().numpy()) # checked, different
        
        # print("*" * 30 + " whiten_and_color done\n\n")
        return targetFeature
    
    # WCT with numpy matrix multiplication
    def whiten_and_color_np(self, cF, sF):
        # print("*" * 30 + " whiten_and_color begin")
        
        # ---------------------------------
        # svd for content feature
        cF = cF.data.cpu().numpy()
        cFSize = cF.shape
        c_mean = np.repeat(np.mean(cF, 1), cFSize[1], axis=0).reshape(cFSize)
        cF = cF - c_mean
        contentConv = np.divide(np.matmul(cF, np.transpose(cF)), cFSize[1] - 1) + np.eye(cFSize[0])
        # print("-" * 5 + " contentConv np")
        # print(np.abs(contentConv).sum()) # checked, same
        
        c_u, c_e, c_v = linalg.svd(contentConv)
        c_v = np.transpose(c_v)
        # print("-" * 5 + " content svd np")
        # print(c_u.shape, c_e.shape, c_v.shape)
        # print(np.abs(c_u).sum())
        # print(np.abs(c_e).sum())
        # print(np.abs(c_v).sum())
        
        k_c = cFSize[0]
        for i in range(cFSize[0]):
            if c_e[i] < EigenValueThre:
                k_c = i
                break
        # print("k_c = %s\n" % k_c)
        
        # ---------------------------------
        # svd for style feature
        sF = sF.data.cpu().numpy()
        sFSize = sF.shape
        s_mean = np.mean(sF, 1)
        sF = sF - np.repeat(s_mean, sFSize[1], axis=0).reshape(sFSize)
        styleConv = np.divide(np.matmul(sF, np.transpose(sF)), sFSize[1] - 1)
        # print("-" * 5 + " styleConv np")
        # print(np.abs(styleConv).sum()) # checked, same
        
        s_u, s_e, s_v = linalg.svd(styleConv)
        s_v = np.transpose(s_v)
        # print("-" * 5 + " style svd np")
        # print(s_u.shape, s_e.shape, s_v.shape)
        # print(np.abs(s_u).sum())
        # print(np.abs(s_e).sum())
        # print(np.abs(s_v).sum())
        
        k_s = sFSize[0]
        for i in range(sFSize[0]):
            if s_e[i] < EigenValueThre:
                k_s = i
                break
        # print("k_s = %s\n" % k_s)
        
        c_d = pow(c_e[0:k_c], -0.5)
        step1 = np.matmul(c_v[:, 0:k_c], np.diag(c_d))
        step2 = np.matmul(step1, (np.transpose(c_v[:, 0:k_c])))
        whiten_cF = np.matmul(step2, cF)
        # print("*" * 30 + " whiten_cF np")
        # print(np.abs(whiten_cF).sum()) # checked, same

        
        s_d = pow(s_e[0:k_s], 0.5)
        targetFeature = np.matmul(np.matmul(np.matmul(s_v[:, 0:k_s], np.diag(s_d)), np.transpose(s_v[:, 0:k_s])), whiten_cF)
        targetFeature = targetFeature + np.repeat(s_mean, cFSize[1], axis=0).reshape(cFSize)
        # print("-" * 5 + " targetFeature np")
        # print(np.abs(targetFeature).sum()) # checked, different
        
        # print("*" * 30 + " whiten_and_color done\n")
        return torch.from_numpy(targetFeature)
    
    def whiten_and_color(self, cF, sF):
          if self.args.numpy:
            return self.whiten_and_color_np(cF, sF)
          else:
            return self.whiten_and_color_torch(cF, sF)

    def transform(self,cF,sF,csF,alpha):
        cF = cF.double()
        sF = sF.double()
        C, W,  H  = cF.size(0), cF.size(1), cF.size(2)
        _, W1, H1 = sF.size(0), sF.size(1), sF.size(2)
        cFView = cF.view(C, -1)
        sFView = sF.view(C, -1)
        targetFeature = self.whiten_and_color(cFView, sFView)
        targetFeature = targetFeature.view_as(cF)
        ccsF = alpha * targetFeature + (1.0 - alpha) * cF
        ccsF = ccsF.float().unsqueeze(0)
        with torch.no_grad():
            csF.resize_(ccsF.size()).copy_(ccsF)
            torch.cuda.empty_cache()
        return csF
