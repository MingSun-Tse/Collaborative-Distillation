import os
import sys
import torch
import argparse
from PIL import Image
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.datasets as datasets
from Loader import Dataset
from util import WCT
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
parser.add_argument('--mode', type=str, help="to choose different trained models")
parser.add_argument('--UHD', action='store_true', help="if use the UHD images")
parser.add_argument('--synthesis', action="store_true", help="for style synthesis")
parser.add_argument('--fineSize', type=int, default=0, help='resize image to fineSize x fineSize, leave it to 0 if not resize')
parser.add_argument('--alpha', type=float,default=1, help='hyperparameter to blend wct feature and content feature')
parser.add_argument('--log_mark', type=str, default=time.strftime("%Y%m%d-%H%M") )
parser.add_argument('--num_run', type=int, default=1, help="you can run WCT for multiple times")
parser.add_argument('--debug', action="store_true")
parser.add_argument('--numpy', action="store_true", help="if use numpy for stylization rather than torch")

### original WCT models
parser.add_argument('--vgg1', default='../trained_models/WCT_models/vgg_normalised_conv1_1.t7', help='Path to the VGG conv1_1')
parser.add_argument('--vgg2', default='../trained_models/WCT_models/vgg_normalised_conv2_1.t7', help='Path to the VGG conv2_1')
parser.add_argument('--vgg3', default='../trained_models/WCT_models/vgg_normalised_conv3_1.t7', help='Path to the VGG conv3_1')
parser.add_argument('--vgg4', default='../trained_models/WCT_models/vgg_normalised_conv4_1.t7', help='Path to the VGG conv4_1')
parser.add_argument('--vgg5', default='../trained_models/WCT_models/vgg_normalised_conv5_1.t7', help='Path to the VGG conv5_1')
parser.add_argument('--decoder5', default='../trained_models/WCT_models/feature_invertor_conv5_1.t7', help='Path to the decoder5')
parser.add_argument('--decoder4', default='../trained_models/WCT_models/feature_invertor_conv4_1.t7', help='Path to the decoder4')
parser.add_argument('--decoder3', default='../trained_models/WCT_models/feature_invertor_conv3_1.t7', help='Path to the decoder3')
parser.add_argument('--decoder2', default='../trained_models/WCT_models/feature_invertor_conv2_1.t7', help='Path to the decoder2')
parser.add_argument('--decoder1', default='../trained_models/WCT_models/feature_invertor_conv1_1.t7', help='Path to the decoder1')

### 16x model (KD2SD, which will degrade the stylization quality)
# parser.add_argument('--e5', default='../trained_models/wct_se_16x_new/5SE.pth')
# parser.add_argument('--e4', default='../trained_models/wct_se_16x_new/4SE.pth')
# parser.add_argument('--e3', default='../trained_models/wct_se_16x_new/3SE.pth')
# parser.add_argument('--e2', default='../trained_models/wct_se_16x_new/2SE.pth')
# parser.add_argument('--e1', default='../trained_models/wct_se_16x_new/1SE.pth')
# # parser.add_argument('--d5', default='../trained_models/wct_se_16x_new_sd_kd2sd/5SD.pth')
# # parser.add_argument('--d4', default='../trained_models/wct_se_16x_new_sd_kd2sd/4SD.pth')
# # parser.add_argument('--d3', default='../trained_models/wct_se_16x_new_sd_kd2sd/3SD.pth')
# # parser.add_argument('--d2', default='../trained_models/wct_se_16x_new_sd_kd2sd/2SD.pth')
# # parser.add_argument('--d1', default='../trained_models/wct_se_16x_new_sd_kd2sd/1SD.pth')
# parser.add_argument('--d5', default='../trained_models/wct_se_16x_new_sd_kd2sd_lwfeat0.0001/5SD.pth')
# parser.add_argument('--d4', default='../trained_models/wct_se_16x_new_sd_kd2sd_lwfeat0.0001/4SD.pth')
# parser.add_argument('--d3', default='../trained_models/wct_se_16x_new_sd_kd2sd_lwfeat0.0001/3SD.pth')
# parser.add_argument('--d2', default='../trained_models/wct_se_16x_new_sd_kd2sd_lwfeat0.0001/2SD.pth')
# parser.add_argument('--d1', default='../trained_models/wct_se_16x_new_sd_kd2sd_lwfeat0.0001/1SD.pth')

### 16x model -- UHD model, Conv1_1 pruned, test new pruned models (non-relu experiments)
parser.add_argument('--e5', default='../trained_models/wct_se_16x_new/5SE.pth')
parser.add_argument('--e4', default='../trained_models/wct_se_16x_new/4SE.pth')
parser.add_argument('--e3', default='../trained_models/wct_se_16x_new/3SE.pth')
parser.add_argument('--e2', default='../trained_models/wct_se_16x_new/2SE.pth')
parser.add_argument('--e1', default='../trained_models/wct_se_16x_new/1SE.pth')
parser.add_argument('--d5', default='../trained_models/wct_se_16x_new_sd/5SD.pth')
parser.add_argument('--d4', default='../trained_models/wct_se_16x_new_sd/4SD.pth')
parser.add_argument('--d3', default='../trained_models/wct_se_16x_new_sd/3SD.pth')
parser.add_argument('--d2', default='../trained_models/wct_se_16x_new_sd/2SD.pth')
parser.add_argument('--d1', default='../trained_models/wct_se_16x_new_sd/1SD.pth')
args = parser.parse_args()

# Set up log
class LogPrinter():
  def __init__(self, debug, f):
    self.log = sys.stdout if debug else open(f, "a+")
  def __call__(self, sth):
    print(str(sth), file=self.log, flush=True)

log = os.path.join(args.outf, "log_%s_%s.txt" % (args.log_mark, args.mode))
logprinter = LogPrinter(args.debug, log)
logprint(args._get_kwargs())

# Set up data
contentPath = args.UHD_contentPath if args.UHD else args.contentPath
stylePath   = args.UHD_stylePath   if args.UHD else args.stylePath
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
logprint("Number of content-style pairs: %s" % len(loader))
for i, (cImg, sImg, imname) in enumerate(loader):
    imname = imname[0]
    logprint('\n' + '*' * 30 + '#%s: Transferring "%s"' % (i, imname))
    cImg = cImg.cuda()
    sImg = sImg.cuda()
    
    start_time = time.time()
    # WCT Style Transfer
    for k in range(args.num_run):
      logprint("stage 5:"); cImg = styleTransfer(wct.e5, wct.d5, cImg, sImg, csF)
      logprint("stage 4:"); cImg = styleTransfer(wct.e4, wct.d4, cImg, sImg, csF)
      logprint("stage 3:"); cImg = styleTransfer(wct.e3, wct.d3, cImg, sImg, csF)
      logprint("stage 2:"); cImg = styleTransfer(wct.e2, wct.d2, cImg, sImg, csF)
      logprint("stage 1:"); cImg = styleTransfer(wct.e1, wct.d1, cImg, sImg, csF)
    vutils.save_image(cImg.data.cpu(), os.path.join(args.outf, args.log_mark + "_alpha=" + str(args.alpha) + "_" + imname))
    end_time = time.time()
    logprint('Elapsed time is: %.4f seconds' % (end_time - start_time))
    avgTime += (end_time - start_time)

logprint('Processed %d images. Average time is: %.4f seconds' % ((i+1), avgTime/(i+1)))
logprint("Max GPU memory allocated: %.4f GB" % (torch.cuda.max_memory_allocated(device=0) / 1024. / 1024. / 1024.))
logprint("Total memory of the current GPU: %.4f GB" % (torch.cuda.get_device_properties(device=0).total_memory / 1024. / 1024 / 1024))
