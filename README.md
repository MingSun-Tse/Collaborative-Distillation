The code is being prepared, which will be updated soon. (04/11/2020)

# Collaborative-Distillation
Official PyTorch code for our CVPR-20 paper "Collaborative Distillation for Ultra-Resolution Universal Style Transfer". 
> TL'DR: We propose a new knowledge distillation method to reduce CNN filters, realizing the ultra-resolution universal style transfer on a single 12GB GPU.
<center><img src="UHD_stylized.jpg" width="1000" hspace="10"></center>

## Environment
- python==3.6.9
- pytorch==0.4.1
- torchvision==0.2.1
- CUDA + cuDNN

## Test (style transfer)
Step 1: Prepare images

Step 2: Prepare models
- Download our trained unpruned encoder and decoder [models](https://drive.google.com/file/d/1REga1z1rKezQtBebIZ86_iNR-mxum-KB/view?usp=sharing) for image reconstruction only (download and uncompress them under `models/`)
- For ultra-resolution style transfer, we use the pruned VGG-19. The models are already in the `PytorchWCT/models/16x_models`.

Step 3: Stylization
```
python WCT.py --cuda --gpu 0 --debug # use original VGG-19
python WCT.py --cuda --gpu 0 --debug  -m 16x # use slimmed VGG-19
```

## Train (model compressiom)
> TODO

## Results
<img src="PytorchWCT/style/UHD_style/Vincent_2K.png" width="400" hspace="10">

<img src="PytorchWCT/content/UHD_content/green_park-wallpaper-3840x2160.jpg" width="400" hspace="10">

<img src="stylized_results/20181122-1715_1_green_park-wallpaper-3840x2160+Vincent_2K.jpg" width="400" hspace="10">

[//]: # More results can be found in this folder.

Image copyrights: We use the UHD images from [this wallpaper website](http://wallpaperswide.com/). All copyrights are attributed to them.

### Acknowledgments
In this code we refers to the following implementations: [PytorchWCT](https://github.com/sunshineatnoon/PytorchWCT), [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN), [AdaIN-style](https://github.com/xunhuang1995/AdaIN-style). Great thanks to them!

### Reference
Please cite this in your publications if the code helps your research:

    @inproceedings{wang2020collaborative,
      Author = {Wang, Huan and Li, Yijun and Wang, Yuehai and Hu, Haoji and Yang, Ming-Hsuan},
      Title = {Collaborative Distillation for Ultra-Resolution Universal Style Transfer},
      Booktitle = {CVPR},
      Year = {2020}
    }