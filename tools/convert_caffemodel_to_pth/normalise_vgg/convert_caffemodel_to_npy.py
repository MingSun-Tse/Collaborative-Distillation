import time
import sys
import os
import argparse
import numpy as np
pjoin = os.path.join
HOME = os.environ["HOME"]
CAFFE_ROOT = pjoin(HOME, "Caffe/Caffe_default")
sys.path.insert(0, pjoin(CAFFE_ROOT, "python"))
# import lmdb
import caffe
import pickle

model = sys.argv[1]
weights =sys.argv[2]
out = {}
net = caffe.Net(model, weights, caffe.TEST)
for name, param in net.params.iteritems():
    out[name + "_weight"] = param[0].data
    out[name + "_bais"] = param[1].data
    print("processing layer %s done" % name)

with open("weights.pkl", 'wb') as f:
    pickle.dump(out, f)



