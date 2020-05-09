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

## Models
Implementations of the following models are provided:
- [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf)
- [Mask-ShadowGAN](https://arxiv.org/pdf/1903.10683.pdf) (in progress)
- [EnlightenGAN](https://arxiv.org/pdf/1906.06972.pdf) (in progress)
- [DeshadowNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qu_DeshadowNet_A_Multi-Context_CVPR_2017_paper.pdf) (in progress)

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
To be added...

#### EnlightenGAN
To be added...

#### DeshadowNet
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
To be added...

#### EnlightenGAN
To be added...

#### DeshadowNet
To be added...
