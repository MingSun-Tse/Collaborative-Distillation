import numpy as np
import os
import torch.nn as nn
import torch
from torch.utils.serialization import load_lua
from my_utils import load_param_from_t7 as load_param
from my_utils import smart_load
import pickle
pjoin = os.path.join

# ---------------------------------------------------
# FP 16x VGG19 model
# Encoder5/Decoder5


class SmallEncoder5_FP16x(nn.Module):
    def __init__(self, model=None, fixed=False):
        super(SmallEncoder5_FP16x, self).__init__()
        self.fixed = fixed

        self.conv0 = nn.Conv2d(3,  3, 1, 1, 0)
        self.conv0.weight = nn.Parameter(torch.from_numpy(np.array(
            [[[[0]], [[0]], [[255]]],
             [[[0]], [[255]], [[0]]],
             [[[255]], [[0]], [[0]]]])).float())
        self.conv0.bias = nn.Parameter(torch.from_numpy(np.array(
            [-103.939, -116.779, -123.68])).float())
        self.conv11 = nn.Conv2d(3, 16, 3, 1, 0, dilation=1)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 0, dilation=1)
        self.conv21 = nn.Conv2d(16, 32, 3, 1, 0, dilation=1)
        self.conv22 = nn.Conv2d(32, 32, 3, 1, 0, dilation=1)
        self.conv31 = nn.Conv2d(32, 64, 3, 1, 0, dilation=1)
        self.conv32 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv33 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv34 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv41 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv42 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv43 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv44 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv51 = nn.Conv2d(128, 128, 3, 1, 0)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))

        if model:
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
                print("Given torch model, saving pytorch model")
                torch.save(self.state_dict(), os.path.splitext(
                    model)[0] + "_FP16x_5E.pth")
                print("Saving done")
            else:
                self.load_state_dict(smart_load(model))

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
        out0 = self.conv0(input)
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


class SmallEncoder5_FP16x_Gatys(nn.Module):
    def __init__(self, model=None, fixed=False):
        super(SmallEncoder5_FP16x_Gatys, self).__init__()
        self.fixed = fixed

        self.conv0 = nn.Conv2d(3,  3, 1, 1, 0)
        self.conv0.weight = nn.Parameter(torch.from_numpy(np.array(
            [[[[0]], [[0]], [[255]]],
             [[[0]], [[255]], [[0]]],
             [[[255]], [[0]], [[0]]]])).float())
        self.conv0.bias = nn.Parameter(torch.from_numpy(np.array(
            [-103.939, -116.779, -123.68])).float())
        self.pad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(3, 16, 3, 1, 0)
        self.relu11 = nn.ReLU(inplace=True)

        self.pad12 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 0)
        self.relu12 = nn.ReLU(inplace=True)

        self.pool12 = nn.MaxPool2d(
            kernel_size=2, stride=2, return_indices=False)

        self.pad21 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv21 = nn.Conv2d(16, 32, 3, 1, 0)
        self.relu21 = nn.ReLU(inplace=True)

        self.pad22 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv22 = nn.Conv2d(32, 32, 3, 1, 0)
        self.relu22 = nn.ReLU(inplace=True)

        self.pool22 = nn.MaxPool2d(
            kernel_size=2, stride=2, return_indices=False)

        self.pad31 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv31 = nn.Conv2d(32, 64, 3, 1, 0)
        self.relu31 = nn.ReLU(inplace=True)

        self.pad32 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv32 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu32 = nn.ReLU(inplace=True)

        self.pad33 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv33 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu33 = nn.ReLU(inplace=True)

        self.pad34 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv34 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu34 = nn.ReLU(inplace=True)

        self.pool34 = nn.MaxPool2d(
            kernel_size=2, stride=2, return_indices=False)

        self.pad41 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv41 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu41 = nn.ReLU(inplace=True)

        self.pad42 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv42 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu42 = nn.ReLU(inplace=True)

        self.pad43 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv43 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu43 = nn.ReLU(inplace=True)

        self.pad44 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv44 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu44 = nn.ReLU(inplace=True)

        self.pool44 = nn.MaxPool2d(
            kernel_size=2, stride=2, return_indices=False)

        self.pad51 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv51 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu51 = nn.ReLU(inplace=True)

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
                # print("Given torch model, saving pytorch model")
                # torch.save(self.state_dict(), os.path.splitext(model)[0] + "_FP16x_5E.pth")
                # print("Saving done")
            else:
                self.load_state_dict(smart_load(model))
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


class SmallDecoder5_FP16x(nn.Module):
    def __init__(self, model=None, fixed=False):
        super(SmallDecoder5_FP16x, self).__init__()
        self.fixed = fixed

        self.conv51 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv44 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv43 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv42 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv41 = nn.Conv2d(128, 64, 3, 1, 0)
        self.conv34 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv33 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv32 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv31 = nn.Conv2d(64, 32, 3, 1, 0, dilation=1)
        self.conv22 = nn.Conv2d(32, 32, 3, 1, 0, dilation=1)
        self.conv21 = nn.Conv2d(32, 16, 3, 1, 0, dilation=1)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 0, dilation=1)
        self.conv11 = nn.Conv2d(16,  3, 3, 1, 0, dilation=1)

        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))

        if model:
            self.load_state_dict(smart_load(model))

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

# Encoder4/Decoder4
class SmallEncoder4_FP16x(nn.Module):
    def __init__(self, model=None, fixed=False):
        super(SmallEncoder4_FP16x, self).__init__()
        self.fixed = fixed

        self.conv0 = nn.Conv2d(3,  3, 1, 1, 0)
        self.conv0.weight = nn.Parameter(torch.from_numpy(np.array(
            [[[[0]], [[0]], [[255]]],
             [[[0]], [[255]], [[0]]],
             [[[255]], [[0]], [[0]]]])).float())
        self.conv0.bias = nn.Parameter(torch.from_numpy(np.array(
            [-103.939, -116.779, -123.68])).float())
        self.conv11 = nn.Conv2d(3, 16, 3, 1, 0, dilation=1)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 0, dilation=1)
        self.conv21 = nn.Conv2d(16, 32, 3, 1, 0, dilation=1)
        self.conv22 = nn.Conv2d(32, 32, 3, 1, 0, dilation=1)
        self.conv31 = nn.Conv2d(32, 64, 3, 1, 0, dilation=1)
        self.conv32 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv33 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv34 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv41 = nn.Conv2d(64, 128, 3, 1, 0)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))

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
                print("Given torch model, saving pytorch model")
                torch.save(self.state_dict(), os.path.splitext(
                    model)[0] + "_FP16x_4E.pth")
                print("Saving done")
            else:
                self.load_state_dict(smart_load(model))
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
        out0 = self.conv0(input)
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


class SmallDecoder4_FP16x(nn.Module):
    def __init__(self, model=None, fixed=False):
        super(SmallDecoder4_FP16x, self).__init__()
        self.fixed = fixed

        self.conv41 = nn.Conv2d(128, 64, 3, 1, 0)
        self.conv34 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv33 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv32 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv31 = nn.Conv2d(64, 32, 3, 1, 0, dilation=1)
        self.conv22 = nn.Conv2d(32, 32, 3, 1, 0, dilation=1)
        self.conv21 = nn.Conv2d(32, 16, 3, 1, 0, dilation=1)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 0, dilation=1)
        self.conv11 = nn.Conv2d(16,  3, 3, 1, 0, dilation=1)

        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))

        if model:
            self.load_state_dict(smart_load(model))

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

# Encoder3/Decoder3
class SmallEncoder3_FP16x(nn.Module):
    def __init__(self, model=None, fixed=False):
        super(SmallEncoder3_FP16x, self).__init__()
        self.fixed = fixed

        self.conv0 = nn.Conv2d(3,  3, 1, 1, 0)
        self.conv0.weight = nn.Parameter(torch.from_numpy(np.array(
            [[[[0]], [[0]], [[255]]],
             [[[0]], [[255]], [[0]]],
             [[[255]], [[0]], [[0]]]])).float())
        self.conv0.bias = nn.Parameter(torch.from_numpy(np.array(
            [-103.939, -116.779, -123.68])).float())
        self.conv11 = nn.Conv2d(3, 16, 3, 1, 0, dilation=1)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 0, dilation=1)
        self.conv21 = nn.Conv2d(16, 32, 3, 1, 0, dilation=1)
        self.conv22 = nn.Conv2d(32, 32, 3, 1, 0, dilation=1)
        self.conv31 = nn.Conv2d(32, 64, 3, 1, 0, dilation=1)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))

        if model:
            assert(os.path.splitext(model)[1] in {".t7", ".pth"})
            if model.endswith(".t7"):
                t7_model = load_lua(model)
                load_param(t7_model, 0,  self.conv11)
                load_param(t7_model, 2,  self.conv12)
                load_param(t7_model, 5,  self.conv21)
                load_param(t7_model, 7,  self.conv22)
                load_param(t7_model, 10, self.conv31)
                print("Given torch model, saving pytorch model")
                torch.save(self.state_dict(), os.path.splitext(
                    model)[0] + "_FP16x_3E.pth")
                print("Saving done")
            else:
                self.load_state_dict(smart_load(model))
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
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        out12 = self.relu(self.conv12(self.pad(out11)))
        out12 = self.pool(out12)
        out21 = self.relu(self.conv21(self.pad(out12)))
        out22 = self.relu(self.conv22(self.pad(out21)))
        out22 = self.pool(out22)
        out31 = self.relu(self.conv31(self.pad(out22)))
        return out11, out21, out31


class SmallDecoder3_FP16x(nn.Module):
    def __init__(self, model=None, fixed=False):
        super(SmallDecoder3_FP16x, self).__init__()
        self.fixed = fixed

        self.conv31 = nn.Conv2d(64, 32, 3, 1, 0, dilation=1)
        self.conv22 = nn.Conv2d(32, 32, 3, 1, 0, dilation=1)
        self.conv21 = nn.Conv2d(32, 16, 3, 1, 0, dilation=1)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 0, dilation=1)
        self.conv11 = nn.Conv2d(16,  3, 3, 1, 0, dilation=1)

        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))

        if model:
            self.load_state_dict(smart_load(model))
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

# Encoder2/Decoder2
class SmallEncoder2_FP16x(nn.Module):
    def __init__(self, model=None, fixed=False):
        super(SmallEncoder2_FP16x, self).__init__()
        self.fixed = fixed

        self.conv0 = nn.Conv2d(3,  3, 1, 1, 0)
        self.conv0.weight = nn.Parameter(torch.from_numpy(np.array(
            [[[[0]], [[0]], [[255]]],
             [[[0]], [[255]], [[0]]],
             [[[255]], [[0]], [[0]]]])).float())
        self.conv0.bias = nn.Parameter(torch.from_numpy(np.array(
            [-103.939, -116.779, -123.68])).float())
        self.conv11 = nn.Conv2d(3, 16, 3, 1, 0, dilation=1)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 0, dilation=1)
        self.conv21 = nn.Conv2d(16, 32, 3, 1, 0, dilation=1)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))

        if model:
            assert(os.path.splitext(model)[1] in {".t7", ".pth"})
            if model.endswith(".t7"):
                t7_model = load_lua(model)
                load_param(t7_model, 0,  self.conv11)
                load_param(t7_model, 2,  self.conv12)
                load_param(t7_model, 5,  self.conv21)
                print("Given torch model, saving pytorch model")
                torch.save(self.state_dict(), os.path.splitext(
                    model)[0] + "_FP16x_2E.pth")
                print("Saving done")
            else:
                self.load_state_dict(smart_load(model))
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
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        out12 = self.relu(self.conv12(self.pad(out11)))
        out12 = self.pool(out12)
        out21 = self.relu(self.conv21(self.pad(out12)))
        return out11, out21


class SmallDecoder2_FP16x(nn.Module):
    def __init__(self, model=None, fixed=False):
        super(SmallDecoder2_FP16x, self).__init__()
        self.fixed = fixed

        self.conv21 = nn.Conv2d(32, 16, 3, 1, 0, dilation=1)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 0, dilation=1)
        self.conv11 = nn.Conv2d(16,  3, 3, 1, 0, dilation=1)

        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))

        if model:
            self.load_state_dict(smart_load(model))
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, y):
        y = self.relu(self.conv21(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv12(self.pad(y)))
        y = self.relu(self.conv11(self.pad(y)))
        return y

# Encoder1/Decoder1


class SmallEncoder1_FP16x(nn.Module):
    def __init__(self, model=None, fixed=False):
        super(SmallEncoder1_FP16x, self).__init__()
        self.fixed = fixed

        self.conv0 = nn.Conv2d(3,  3, 1, 1, 0)
        self.conv0.weight = nn.Parameter(torch.from_numpy(np.array(
            [[[[0]], [[0]], [[255]]],
             [[[0]], [[255]], [[0]]],
             [[[255]], [[0]], [[0]]]])).float())
        self.conv0.bias = nn.Parameter(torch.from_numpy(np.array(
            [-103.939, -116.779, -123.68])).float())
        self.conv11 = nn.Conv2d(3, 16, 3, 1, 0, dilation=1)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))

        if model:
            assert(os.path.splitext(model)[1] in {".t7", ".pth"})
            if model.endswith(".t7"):
                t7_model = load_lua(model)
                load_param(t7_model, 0,  self.conv11)
                print("Given torch model, saving pytorch model")
                torch.save(self.state_dict(), os.path.splitext(
                    model)[0] + "_FP16x_1E.pth")
                print("Saving done")
            else:
                self.load_state_dict(smart_load(model))
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, y):
        y = self.conv0(y)
        y = self.relu(self.conv11(self.pad(y)))
        return y

    def forward_branch(self, input):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        return out11


class SmallDecoder1_FP16x(nn.Module):
    def __init__(self, model=None, fixed=False):
        super(SmallDecoder1_FP16x, self).__init__()
        self.fixed = fixed

        self.conv11 = nn.Conv2d(16,  3, 3, 1, 0, dilation=1)

        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))

        if model:
            self.load_state_dict(smart_load(model))
        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, y):
        y = self.relu(self.conv11(self.pad(y)))
        return y
