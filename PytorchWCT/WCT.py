import os
import sys
import torch
import argparse
from PIL import Image
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.datasets as datasets
from data_loader import Dataset
from util_wct import WCT
import scipy.misc
import time
import glob

parser = argparse.ArgumentParser(description='WCT Pytorch')
parser.add_argument('--UHD_contentPath', type=str, default='content/UHD_content')
parser.add_argument('--UHD_stylePath', type=str, default='style/UHD_style')
parser.add_argument('--contentPath', type=str, default='content')
parser.add_argument('--stylePath', type=str, default='style')
parser.add_argument('--texturePath', type=str, default='style/texture')
parser.add_argument('--outf', type=str, default='stylized_results', help='folder to output images')
parser.add_argument('--picked_content_mark', type=str, default=".")
parser.add_argument('--picked_style_mark', type=str, default=".")
parser.add_argument('--mode', type=str, help="to choose different trained models", default=None, choices=['original', '16x', 'fp16x', '16x_kd2sd'])
parser.add_argument('--UHD', action='store_true', help="if use the UHD images")
parser.add_argument('--synthesis', action="store_true", help="for style synthesis")
parser.add_argument('--fineSize', type=int, default=0, help='resize both content and style to fineSize x fineSize, leave it to 0 if not resize')
parser.add_argument('--alpha', type=float,default=1, help='hyperparameter to blend wct feature and content feature')
parser.add_argument('--log_mark', type=str, default=time.strftime("%Y%m%d-%H%M") )
parser.add_argument('--num_run', type=int, default=1, help="you can run WCT for multiple times")
parser.add_argument('--debug', action="store_true")
parser.add_argument('--numpy', action="store_true", help="if use numpy for stylization rather than torch")
args = parser.parse_args()

if args.mode == "original" or args.mode == None:
  args.e5 = '../trained_models/original_wct_models/vgg_normalised_conv5_1.t7'
  args.e4 = '../trained_models/original_wct_models/vgg_normalised_conv4_1.t7'
  args.e3 = '../trained_models/original_wct_models/vgg_normalised_conv3_1.t7'
  args.e2 = '../trained_models/original_wct_models/vgg_normalised_conv2_1.t7'
  args.e1 = '../trained_models/original_wct_models/vgg_normalised_conv1_1.t7'
  args.d5 = '../trained_models/original_wct_models/feature_invertor_conv5_1.t7'
  args.d4 = '../trained_models/original_wct_models/feature_invertor_conv4_1.t7'
  args.d3 = '../trained_models/original_wct_models/feature_invertor_conv3_1.t7'
  args.d2 = '../trained_models/original_wct_models/feature_invertor_conv2_1.t7'
  args.d1 = '../trained_models/original_wct_models/feature_invertor_conv1_1.t7'

elif args.mode == "16x":
  args.e5 = '../trained_models/wct_se_16x_new/5SE.pth'
  args.e4 = '../trained_models/wct_se_16x_new/4SE.pth'
  args.e3 = '../trained_models/wct_se_16x_new/3SE.pth'
  args.e2 = '../trained_models/wct_se_16x_new/2SE.pth'
  args.e1 = '../trained_models/wct_se_16x_new/1SE.pth'
  args.d5 = '../trained_models/wct_se_16x_new_sd/5SD.pth'
  args.d4 = '../trained_models/wct_se_16x_new_sd/4SD.pth'
  args.d3 = '../trained_models/wct_se_16x_new_sd/3SD.pth'
  args.d2 = '../trained_models/wct_se_16x_new_sd/2SD.pth'
  args.d1 = '../trained_models/wct_se_16x_new_sd/1SD.pth'

elif args.mode == "16x_kd2sd":
  args.e5 = '../trained_models/wct_se_16x_new/5SE.pth'
  args.e4 = '../trained_models/wct_se_16x_new/4SE.pth'
  args.e3 = '../trained_models/wct_se_16x_new/3SE.pth'
  args.e2 = '../trained_models/wct_se_16x_new/2SE.pth'
  args.e1 = '../trained_models/wct_se_16x_new/1SE.pth'
  args.d5 = '../trained_models/wct_se_16x_new_sd_kd2sd/5SD.pth'
  args.d4 = '../trained_models/wct_se_16x_new_sd_kd2sd/4SD.pth'
  args.d3 = '../trained_models/wct_se_16x_new_sd_kd2sd/3SD.pth'
  args.d2 = '../trained_models/wct_se_16x_new_sd_kd2sd/2SD.pth'
  args.d1 = '../trained_models/wct_se_16x_new_sd_kd2sd/1SD.pth'
  # args.d5 = '../trained_models/wct_se_16x_new_sd_kd2sd_lwfeat0.0001/5SD.pth'
  # args.d4 = '../trained_models/wct_se_16x_new_sd_kd2sd_lwfeat0.0001/4SD.pth'
  # args.d3 = '../trained_models/wct_se_16x_new_sd_kd2sd_lwfeat0.0001/3SD.pth'
  # args.d2 = '../trained_models/wct_se_16x_new_sd_kd2sd_lwfeat0.0001/2SD.pth'
  # args.d1 = '../trained_models/wct_se_16x_new_sd_kd2sd_lwfeat0.0001/1SD.pth'

# Set up log
class LogPrinter():
  def __init__(self, debug, f):
    self.log = sys.stdout if debug else open(f, "a+")
  def __call__(self, sth):
    print(str(sth), file=self.log, flush=True)
log = os.path.join(args.outf, "log_%s_%s.txt" % (args.log_mark, args.mode))
logprinter = LogPrinter(args.debug, log)
logprinter(args._get_kwargs())

# Set up data
contentPath = args.UHD_contentPath if args.UHD else args.contentPath
stylePath = args.UHD_stylePath if args.UHD else args.stylePath
dataset = Dataset(contentPath, stylePath, args.texturePath, args.fineSize, args.picked_content_mark, args.picked_style_mark, args.synthesis)
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=1,
                                     shuffle=False)

# Set up model and transform
wct = WCT(args).cuda()
@torch.no_grad()
def styleTransfer(encoder, decoder, contentImg, styleImg, csF):
    sF  = encoder(styleImg); torch.cuda.empty_cache() # empty cache to save memory
    cF  = encoder(contentImg); torch.cuda.empty_cache()
    sF  = sF.data.cpu().squeeze(0) # note: svd runs on CPU
    cF  = cF.data.cpu().squeeze(0)
    csF = wct.transform(cF, sF, csF, args.alpha)
    Img = decoder(csF); torch.cuda.empty_cache()
    return Img

# Run
avgTime = 0
csF = torch.Tensor().cuda()
logprinter("Number of content-style pairs: %s" % len(loader))
for i, (cImg, sImg, imname) in enumerate(loader):
    imname = imname[0]
    logprinter('\n' + '*' * 30 + ' #%s: Transferring "%s"' % (i, imname))
    cImg = cImg.cuda()
    sImg = sImg.cuda()
    
    start_time = time.time()
    # WCT Style Transfer
    for k in range(args.num_run):
      logprinter("Processing stage 5"); cImg = styleTransfer(wct.e5, wct.d5, cImg, sImg, csF)
      logprinter("Processing stage 4"); cImg = styleTransfer(wct.e4, wct.d4, cImg, sImg, csF)
      logprinter("Processing stage 3"); cImg = styleTransfer(wct.e3, wct.d3, cImg, sImg, csF)
      logprinter("Processing stage 2"); cImg = styleTransfer(wct.e2, wct.d2, cImg, sImg, csF)
      logprinter("Processing stage 1"); cImg = styleTransfer(wct.e1, wct.d1, cImg, sImg, csF)
    
    out_path = os.path.join(args.outf, "%s_mode=%s_alpha=%s_%s" % (args.log_mark, args.mode, args.alpha, imname))
    vutils.save_image(cImg.data.cpu(), out_path)
    end_time = time.time()
    avgTime += (end_time - start_time)
    logprinter('Elapsed time is: %.4f seconds' % (end_time - start_time))

logprinter('Processed %d images. Average processing time per pair is: %.4f seconds' % (i + 1, avgTime / (i + 1)))
# logprinter("Max GPU memory allocated: %.4f GB" % (torch.cuda.max_memory_allocated(device=0) / 1024. / 1024. / 1024.))
# logprinter("Total memory of the current GPU: %.4f GB" % (torch.cuda.get_device_properties(device=0).total_memory / 1024. / 1024 / 1024))