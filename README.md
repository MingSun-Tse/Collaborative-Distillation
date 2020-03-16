# Collaborative-Distillation
Official PyTorch code for our CVPR-20 paper "Collaborative Distillation for Ultra-Resolution Universal Style Transfer". 
> TL'DR: We propose a new knowledge distillation method to reduce CNN filters, realizing the ultra-resolution universal style transfer on a single 12GB GPU.
<center><img src="UHD_stylized.jpg" width="1000" hspace="10"></center>


## Environment
- python==3.5
- pytorch==0.4.1
- torchvision
- CUDA + CuDNN

## Test (style transfer)
Step 1: Prepare images

Step 2: Prepare models
- Download our trained encoder and decoder [models](https://drive.google.com/file/d/1REga1z1rKezQtBebIZ86_iNR-mxum-KB/view?usp=sharing) for image reconstruction only (download and uncompress them under `models/`)
- For ultra-resolution style transfer, we use the slimmed VGG-19. The models are already in the `models/16x_models`.

Step 3: Stylization
```
python WCT.py --cuda --gpu 0 --debug # use original VGG-19
python WCT.py --cuda --gpu 0 --debug  -m 16x # use slimmed VGG-19
```

## Train (model compressiom)
> TODO

## Results
<img src="style/UHD/Vincent_2K.png" width="400" hspace="10">

<img src="content/UHD/green_park-wallpaper-3840x2160.jpg" width="400" hspace="10">

<img src="stylized_results/20181122-1715_1_green_park-wallpaper-3840x2160+Vincent_2K.jpg" width="400" hspace="10">

More results can be found in our supplementary material.

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
