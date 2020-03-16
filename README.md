# Collaborative-Distillation
Official PyTorch code for our CVPR-20 paper "Collaborative Distillation for Ultra-Resolution Universal Style Transfer". TLDR: We propose a new knowledge distillation method to reduce CNN filters, and thus realize ultra-resolution style transfer of universal styles.

# Environment


# Test


# Train


# A



## Universal Style Transfer

This is the Pytorch implementation of [Universal Style Transfer via Feature Transforms](https://arxiv.org/pdf/1705.08086.pdf).

Official Torch implementation can be found [here](https://github.com/Yijunmaverick/UniversalStyleTransfer) and Tensorflow implementation can be found [here](https://github.com/eridgd/WCT-TF).

## Prerequisites
- Python 3.5
- [Pytorch 0.4.1](http://pytorch.org/)
- [torchvision](https://github.com/pytorch/vision)
- Pretrained encoder and decoder [models](https://drive.google.com/file/d/1REga1z1rKezQtBebIZ86_iNR-mxum-KB/view?usp=sharing) for image reconstruction only (download and uncompress them under `models/`)
- For ultra-resolution style transfer, we use the slimmed VGG-19. The models are already in the `models/16x_models`.
- CUDA + CuDNN

## Prepare images
1. low-resolution images

Simply put content and image pairs in `images/content` and `images/style` respectively.

2. ultra-resolution images

Simply put content and image pairs in `images/UHD_content` and `images/UHD_style` respectively.


## Style Transfer
Explanation of some important args:

`-m` or `--mode`
- if not provided (default `None`), use the original VGG-19 models (which can test images in up to about 3000x3000 pixels).
- if `--mode 16x`, use the slimmed 16x models (which can test ultra-resolution images).

`--debug`
- if not provided, i.e., not debugging, then the log will be printed into a txt file in `samples`.
- if `--debug`, then the log will be printed on screen.

Style transfer on low-resolution images.
```
python WCT.py --cuda --gpu 0 --debug # use original VGG-19
python WCT.py --cuda --gpu 0 --debug  -m 16x # use slimmed VGG-19
```

Style transfer on ultra-resolution images. This will generate all the combination pairs with content in `UHD_content` and style in `UHD_style`. Say, if there are 5 contents and 4 styles, there will be 5x4=20 pairs of stylized images.
```
python WCT.py --cuda --gpu 0 --UHD  --debug  -m 16x
```

If you only want to test specific one content or style, use the `picked_content_mark` and `picked_style_mark`. This will filter out all the images only containing the field (`picked_content_mark`, `picked_style_mark`) in their names.
```
python WCT.py --cuda --gpu 0 --UHD  --debug  -m 16x  --picked_content_mark "green_park"  --picked_style_mark "Vincent"
```

Note: The default GPU id is 0, if you want to change the GPU id, change this line `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` in `WCT.py`.

## Results
<img src="images/UHD_style/Vincent_2K.png" width="400" hspace="10">

<img src="images/UHD_content/green_park-wallpaper-3840x2160.jpg" width="400" hspace="10">

<img src="samples/20181122-1715_1_green_park-wallpaper-3840x2160+Vincent_2K.jpg" width="400" hspace="10">

### Acknowledgments
- Many thanks to the author of WCT Yijun Li for his kind help. 
- This code is initially based on Xueting Li's [implementation](https://github.com/sunshineatnoon/PytorchWCT). Many thanks to her!

### Reference
