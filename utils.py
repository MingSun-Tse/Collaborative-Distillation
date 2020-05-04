import matplotlib; matplotlib.use("Agg")
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import shutil
import time
import sys
import collections
pjoin = os.path.join

class LogPrint():
    def __init__(self, file, ExpID, print_to_screen):
        self.file = file
        self.ExpID = ExpID
        self.print_to_screen = print_to_screen
    def __call__(self, some_str):
        sstr = "[%s %s %s " % (self.ExpID[-6:], os.getpid(), time.strftime("%Y/%m/%d-%H:%M:%S]")) + str(some_str)
        print(sstr, file=self.file, flush=True)
        if self.print_to_screen:
            print(sstr)

def check_path(x):
    if x:
        complete_path = glob.glob(x)
        assert(len(complete_path) == 1), "The provided path points to more than 1 entity. Please check."
        x = complete_path[0]
    return x

def my_makedirs(d):
    if not os.path.exists(d):
        os.makedirs(d)

def set_up_dir(project_name, resume, debug):
    TimeID = time.strftime("%Y%m%d-%H%M%S")
    if "SERVER" in os.environ.keys():
        ExpID = "SERVER" + os.environ["SERVER"] + "-" + TimeID
    else:
        ExpID = TimeID
    
    project_path = "Debug_Dir" if debug else pjoin("Experiments", ExpID + "_" + project_name)
    rec_img_path = pjoin(project_path, "reconstructed_images")
    weights_path = pjoin(project_path, "weights")
    my_makedirs(rec_img_path)
    my_makedirs(weights_path)
    log_path = pjoin(weights_path, "log_" + ExpID + ".txt")
    log = open(log_path, "w+")
    print(" ".join(["CUDA_VISIBLE_DEVICES=0 python", *sys.argv]),
          file=log, flush=True)  # save the script
    return TimeID, ExpID, rec_img_path, weights_path, log

def get_CodeID():
    script = "git log --pretty=oneline >> wh_CodeID_file.tmp"
    os.system(script)
    x = open("wh_CodeID_file.tmp").readline()
    os.remove("wh_CodeID_file.tmp")
    return x[:8]

def is_img(x):
    return any(x.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_param_from_t7(model, in_layer_index, out_layer):
    out_layer.weight = torch.nn.Parameter(
        model.get(in_layer_index).weight.float())
    out_layer.bias = torch.nn.Parameter(model.get(in_layer_index).bias.float())

class LogHub(object):
    def __init__(self, momentum=0):
        self.losses = {}
        self.momentum = momentum

    def update(self, name, value):
        if name not in self.losses:
            self.losses[name] = value
        else:
            self.losses[name] = self.losses[name] * \
                self.momentum + value * (1 - self.momentum)

    def format(self):
        keys = self.losses.keys()
        keys = sorted(keys)
        logtmp = ""
        for k in keys:
            logtmp += "%s: %.3f | " % (k, self.losses[k])
        return logtmp[:-3]


def smart_load(model_path):
    sth = torch.load(model_path, map_location=lambda storage, location: storage)
    if isinstance(sth, collections.OrderedDict): # state_dict
        return sth
    elif isinstance(sth, dict): # dict which has a value of state_dict
        for k, v in sth.items():
            if isinstance(v, collections.OrderedDict):
                return v
        print("smart load failed, please manually check the given model")
