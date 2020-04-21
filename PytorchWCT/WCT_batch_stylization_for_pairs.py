import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1] # The args MUST has an option "--gpu".
import torch
import argparse
from PIL import Image
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.datasets as datasets
from Loader import Dataset
from util import *
import scipy.misc
from torch.utils.serialization import load_lua
import time
from utils import logprint
import glob
pjoin = os.path.join

parser = argparse.ArgumentParser(description='WCT Pytorch For pairs')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--UHD', action='store_true')
parser.add_argument('--outf', help='folder to output images')
parser.add_argument('--inf',  help='folder to input images')
parser.add_argument('--gpu', type=int, default=0, help="which gpu to run on. default is 0")
parser.add_argument('--log_mark', type=str, default=time.strftime("%Y%m%d-%H%M") )
parser.add_argument('--debug', action="store_true")
parser.add_argument('--mode', type=str, default="original")
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--numpy', action="store_true")
parser.add_argument('--img_resize', type=int, default=0)
parser.add_argument('--e5')
parser.add_argument('--e4')
parser.add_argument('--e3')
parser.add_argument('--e2')
parser.add_argument('--e1')
parser.add_argument('--d5')
parser.add_argument('--d4')
parser.add_argument('--d3')
parser.add_argument('--d2')
parser.add_argument('--d1')
args = parser.parse_args()
if not args.outf:
  args.outf = args.inf

if args.mode == "mobile":
  args.e1 = glob.glob('../Bin/models/mobilenet_sgd_rmsprop_69.526_my_e1.pth')[0]
  args.e2 = glob.glob('../Bin/models/mobilenet_sgd_rmsprop_69.526_my_e2.pth')[0]
  args.e3 = glob.glob('../Bin/models/mobilenet_sgd_rmsprop_69.526_my_e3.pth')[0]
  args.e4 = glob.glob('../Bin/models/mobilenet_sgd_rmsprop_69.526_my_e4.pth')[0]
  args.e5 = glob.glob('../Bin/models/mobilenet_sgd_rmsprop_69.526_my_e5.pth')[0]
  args.d1 = glob.glob('../Experiments/MobileNet/d1/weights/*1BD_E20S10000*.pth')[0]
  args.d2 = glob.glob('../Experiments/MobileNet/d2/weights/*2BD_E20S10000*.pth')[0]
  args.d3 = glob.glob('../Experiments/MobileNet/d3/weights/*3BD_E20S10000*.pth')[0]
  args.d4 = glob.glob('../Experiments/MobileNet/d4/weights/*4BD_E20S10000*.pth')[0]
  args.d5 = glob.glob('../Experiments/MobileNet/d5/weights/*5BD_E20S10000*.pth')[0]

if args.mode == "scratch":
  args.e1 = glob.glob('../Experiments/*Scratch*/ae1/w*/*1SD*E0S5000-1.pth')[0]
  args.e2 = glob.glob('../Experiments/*Scratch*/ae2/w*/*2SD*E0S5000-1.pth')[0]
  args.e3 = glob.glob('../Experiments/*Scratch*/ae3/w*/*3SD*E0S5000-1.pth')[0]
  args.e4 = glob.glob('../Experiments/*Scratch*/ae4/w*/*4SD*E0S5000-1.pth')[0]
  args.e5 = glob.glob('../Experiments/*Scratch*/ae5/w*/*5SD*E0S5000-1.pth')[0]
  args.d1 = glob.glob('../Experiments/*Scratch*/ae1/w*/*1SD*E0S5000-3.pth')[0]
  args.d2 = glob.glob('../Experiments/*Scratch*/ae2/w*/*2SD*E0S5000-3.pth')[0]
  args.d3 = glob.glob('../Experiments/*Scratch*/ae3/w*/*3SD*E0S5000-3.pth')[0]
  args.d4 = glob.glob('../Experiments/*Scratch*/ae4/w*/*4SD*E0S5000-3.pth')[0]
  args.d5 = glob.glob('../Experiments/*Scratch*/ae5/w*/*5SD*E0S5000-3.pth')[0]

try:
  os.makedirs(args.outf)
except OSError:
  pass

logprint(args.log_mark)
log = open(pjoin(args.inf, "log_%s.txt" % args.log_mark), "a+") if not args.debug else sys.stdout
logprint(args._get_kwargs(), log)

wct = WCT(args)
@torch.no_grad()
def styleTransfer(encoder, decoder, contentImg, styleImg, csF):
    sF  = encoder(styleImg); torch.cuda.empty_cache()
    cF  = encoder(contentImg); torch.cuda.empty_cache()
    sF  = sF.data.cpu().squeeze(0)
    cF  = cF.data.cpu().squeeze(0)
    csF = wct.transform(cF, sF, csF, args.alpha)
    Img = decoder(csF); torch.cuda.empty_cache()
    return Img

avgTime = 0
csF = torch.Tensor()
csF = csF.cuda()
wct.cuda()

for pair in range(1, 21):
  start_time = time.time()
  logprint('=======> Processing pair %s' % pair, log)
  cImg = glob.glob(pjoin(args.inf, "pair%s_content_resized*" % pair))[0]
  sImg = glob.glob(pjoin(args.inf, "pair%s_style*" % pair))[0]
  cImg = Image.open(cImg).convert("RGB")
  sImg = Image.open(sImg).convert("RGB")
  if args.img_resize:
    cImg = cImg.resize([args.img_resize, args.img_resize])
    sImg = sImg.resize([args.img_resize, args.img_resize])
  cImg = transforms.ToTensor()(cImg).cuda().unsqueeze(0)
  sImg = transforms.ToTensor()(sImg).cuda().unsqueeze(0)

  # Stylization
  cImg = styleTransfer(wct.e5, wct.d5, cImg, sImg, csF)
  cImg = styleTransfer(wct.e4, wct.d4, cImg, sImg, csF)
  cImg = styleTransfer(wct.e3, wct.d3, cImg, sImg, csF)
  cImg = styleTransfer(wct.e2, wct.d2, cImg, sImg, csF)
  cImg = styleTransfer(wct.e1, wct.d1, cImg, sImg, csF)
  vutils.save_image(cImg.data.cpu().float(), pjoin(args.outf, "pair{}_stylized_{}.jpg".format(pair, args.mode)))

  end_time = time.time()
  logprint('Elapsed time is: %f' % (end_time - start_time), log)
  avgTime += (end_time - start_time)

log.close()
