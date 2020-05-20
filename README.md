# ImageEnhancement
Various models for handling underexposure, overexposure, super-resolution, shadow removal, etc.

This repo serves as implementations of various image enhancement models meant to
run on Herbie, an autonomous robot project advised by Dr. Seng within Cal Poly's
CSC/CPE Department.

## Dependencies
- Python 3.6
- TensorFlow
- OpenCV
- Pillow
- scikit-image

## Models
Implementations of the following models are provided:
- CycleGAN by Zhu et al.: [Paper](https://arxiv.org/pdf/1703.10593.pdf) | [Code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- Mask-ShadowGAN by Hu et al.: [Paper](https://arxiv.org/pdf/1903.10683.pdf) | [Code](https://github.com/xw-hu/Mask-ShadowGAN)
- EnligthenGAN by Jiang et al. (in progress): [Paper](https://arxiv.org/pdf/1906.06972.pdf) | [Code](https://github.com/TAMU-VITA/EnlightenGAN)
- DeShadowNet by Liangqiong et al. (in progress): [Paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qu_DeshadowNet_A_Multi-Context_CVPR_2017_paper.pdf) | [Code](https://github.com/Liangqiong/DeShadowNet) 

## Datasets
- Download a CycleGAN dataset using:
   ```
   bash ./download_cyclegan_dataset.sh [apple2orange|summer2winter_yosemite|horse2zebra|monet2photo|cezanne2photo|ukiyoe2photo|vangogh2photo|maps|cityscapes|facades|iphone2dslr_flower|ae_photos]
   ```
- Download the Unpaired Shadow Removal (USR) dataset for shadow removal from: [USR Dataset](https://drive.google.com/file/d/1PPAX0W4eyfn1cUrb2aBefnbrmhB1htoJ/view)
- Download the ISTD dataset for shadow removal from: [ISTD Dataset](https://drive.google.com/file/d/1I0qw-65KBA6np8vIZzO6oeiOvcDBttAY/view)

## Updating PYTHONPATH
To ensure all modules in repo can be found you must update your **PYTHONPATH** environment variable:
```
export PYTHONPATH=$PYTHONPATH:/path/to/ImageEnhancement
```

## Training
#### CycleGAN
The CycleGAN model takes approximately 20 hours to train to completion using a Tesla V100 GPU. To train run:
```
python train/cyclegan_train.py --dirA /path/to/dataA  --dirB /path/to/dataB --batch_size 1 --lr 0.0002 --layer_norm_type instance --weight_init_gain 0.02
```

#### Mask-ShadowGAN
The Mask-ShadowGAN model takes approximately 24 hours to train to completion using a Tesla V100 GPU. To train run:
```
python train/maskshadowgan_train.py --dirA /path/to/shadow_data --dirB /path/to/shadow_free_data --batch_size 1 --lr 0.0002 --layer_norm_type instance --weight_init_gain 0.02 --lambda_ident 0.5
```

#### EnlightenGAN
To be added...

#### DeShadowNet
To be added...

To continue training from a saved checkpoint, add the following argument to the end of the command line arguments passed into the training script you are running:
```
--load_model /checkpoint_dir (e.g. /20022019-0801)
```

## Testing
#### CycleGAN
To test the CycleGAN model run:
```
python test/cyclegan_test.py --dir /path/to/dataA --batch_size 1 --layer_norm_type instance --load_model /checkpoint_dir --sample_directory /path/to/save/samples/to
```

#### Mask-ShadowGAN
To test the Mask-ShadowGAN model run:
```
python test/maskshadowgan_test.py --dir /path/to/shadow_data --batch_size 1 --layer_norm_type instance --load_model /checkpoint_dir --sample_directory /path/to/save/samples/to
```

#### EnlightenGAN
To be added...

#### DeShadowNet
To be added...
