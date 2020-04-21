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
from caffe.proto import caffe_pb2

''' Usage:
  1. set up caffe and pycaffe. set up the pathes above right.
  2. set up the ImageNet val subset dataset (1000 jpeg images).
  3. run "python normalise_caffe.py -p normalize_vgg --model vgg19_16x_deploy.prototxt --weights retrain_iter_40000.caffemodel"
'''

def is_img(x):
    return any(x.endswith(extension) for extension in [".png", ".jpg", ".jpeg", "JPEG"])


if __name__ == "__main__":
    # Passed-in params
    parser = argparse.ArgumentParser(description="Normalise")
    parser.add_argument('-d', '--val_data', type=str, help='the directory of val images', default="/home/wanghuan/Dataset/ImageNet_Dataset/val_subset_1000")
    parser.add_argument('-l', '--val_lmdb', type=str, default="")
    parser.add_argument('-g', '--gpu', type=int)
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-w', '--weights', type=str)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('--scale_factor_dir', type=str,
                        help="the directory of scale_factor npy for each layer", default=None)
    # not implemented for using lmdb
    parser.add_argument('--use_lmdb', type=int, default=False)
    parser.add_argument('-p', '--project', type=str,
                        help="project name to save results")
    args = parser.parse_args()

    # make dir
    if not os.path.exists(args.project):
        os.makedirs(args.project)

    # Prepare net
    if args.gpu:
        caffe.set_device(args.gpu)
        caffe.set_mode_gpu()
    net = caffe.Net(args.model, args.weights, caffe.TEST)
    mu = np.load(
        pjoin(CAFFE_ROOT, 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))
    # average over pixels to obtain the mean (BGR) pixel values
    mu = mu.mean(1).mean(1)
    print('mean-subtracted values:', zip('BGR', mu))
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # move image channels to outermost dimension
    transformer.set_transpose('data', (2, 0, 1))
    # subtract the dataset-mean value in each channel
    transformer.set_mean('data', mu)
    # rescale from [0, 1] to [0, 255]
    transformer.set_raw_scale('data', 255)
    # swap channels from RGB to BGR
    transformer.set_channel_swap('data', (2, 1, 0))
    net.blobs['data'].reshape(1, 3, 224, 224)

    if args.scale_factor_dir:  # use given scale factors
        for layer_name, param in net.params.iteritems():
            if len(param[0].data.shape) != 4:
                continue
            scale_factor = np.load(
                pjoin(args.scale_factor_dir, layer_name + "_scale_factor.npy"))
            print(scale_factor)
            # Scale the weights and biases
            net.params[layer_name][0].data[:] = net.params[layer_name][0].data[:] * scale_factor
            net.params[layer_name][1].data[:] = net.params[layer_name][1].data[:] * scale_factor
    else:
        # way 1: load image
        if not args.use_lmdb:
            imgs = [pjoin(args.val_data, i)
                    for i in os.listdir(args.val_data) if is_img(i)]
            num_img = len(imgs)
            print("number of image: %s" % num_img)
            for layer_name, param in net.params.iteritems():
                if len(param[0].data.shape) != 4:
                    continue  # only Conv layers are normalised
                filter_ix = 0  # TODO
                feat = 0
                cnt = 0
                for img_path in imgs:
                    time_id = time.strftime(
                        "[%s" % os.getpid() + "-%Y/%m/%d-%H:%M] ")
                    print(time_id + "%s-%s-current processing image '%s'" %
                          (layer_name, cnt, img_path))
                    img = caffe.io.load_image(img_path)
                    transformed_image = transformer.preprocess('data', img)
                    net.blobs["data"].data[...] = transformed_image
                    net.forward()
                    feat += np.average(net.blobs[layer_name].data[0]) # Why 0? batch size = 1
                    cnt += 1
                scale_factor = 1.0 / (feat / num_img)
                np.save(pjoin(args.project, "%s_scale_factor.npy" %
                              layer_name), scale_factor)

                # Scale the weights and biases
                net.params[layer_name][0].data[:] = net.params[layer_name][0].data[:] * scale_factor
                net.params[layer_name][1].data[:] = net.params[layer_name][1].data[:] * scale_factor

        # way 2: load lmdb -- not implemented
        else:
            lmdb_env = lmdb.open(args.val_lmdb)
            lmdb_txn = lmdb_env.begin()
            lmdb_cursor = lmdb_txn.cursor()
            datum = caffe_pb2.Datum()
            for key, value in lmdb_cursor:
                datum.ParseFromString(value)
                label = datum.label
                data = caffe.io.datum_to_array(datum)
                print(data.shape)
                # CxHxW to HxWxC in cv2
                image = np.transpose(data, (1, 2, 0))
                cv2.imshow('cv2', image)
                cv2.waitKey(1)
                print('{},{}'.format(key, label))
    net.save(pjoin(args.project, "normalised.caffemodel"))
