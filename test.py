import model
import sys
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch
import os
pjoin = os.path.join
'''
    This is the image reconstruction for previous test images.
'''

SE_path  = sys.argv[1]
SD_path  = sys.argv[2]
dir_path = "../TestData"
rec_img_path = "test_output"

SE = model.SmallEncoder5_16x_aux()
SD = model.SmallDecoder5_16x()
SE.load_state_dict(torch.load(SE_path)["model"]["SE"])
SD.load_state_dict(torch.load(SD_path)["model"]["SD"])

test_imgs = [pjoin(dir_path, i) for i in os.listdir(dir_path)]
for img_path in test_imgs:
    try:
        img = Image.open(img_path).convert("RGB")
    except:
        continue
    img = transforms.ToTensor()(img).unsqueeze(0).cuda()
    rec = SD(SE(img))
    out_img_path = pjoin(rec_img_path,
                        "test_data_%s_rec.jpg" % os.path.splitext(os.path.basename(img_path))[0])
    vutils.save_image(rec.data.cpu().float(), out_img_path)