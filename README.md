# 107-th place solution for Kaggle TGS Salt Identification Challenge

At first glance, this challenge looked like a binary image segmentation task. 
So at the beginning, I experimented with many model architectures and hyperparameters. 
Ran hundreds of experiments, thanks for 4x1080Ti rig. Final solution included three U-Net like models with a few bells & whistles. They were ensembled with models of my teammate for final submission.

## Lessons learned

1. Be patient. Let it train for night, instead of stopping training after 10 epochs if it seems not converging. Don't meditate on those curves. Have a walk!
1. Use heavier encoders. Top-performers reported on using Senet154 as encoder. Funny, I had this encoder, but have not even tried it.
1. Don't rely single-fold validation results. Always do cross-validation. 
1. Keep in mind number of submits. In a late game I was unable to merge with another team due to exceeding number of submits.
1. Understand where and why your model fails the most. In this challenge, it was a key to understand and use the fact that there were no solid masks in trainset.
1. Don't be too lazy. 'Assemble mosaic' was in my roadmap since the first week of competition. Shame for me, I didn't use it at all.


## Model zoo

### DPN encoder with U-Net like decoder

Dual-path encoder with U-Net decoder implementation borrowed from https://github.com/selimsef/dsb2018_topcoders/tree/master/albu/src.

### WiderResNet encoder with U-Net like decoder and Object Context

1. WiderResNet38 encoder
1. U-Net like decoder (Double [conv3x3 + bn + relu])
1. OCNet in the central bottleneck. OCNet dilation factors were [2,4,6]
1. SCSE blocks in decoder
1. Input tensor was 3-channel image [I, Y, I * Y]
1. In addition to mask output, model was also predicting salt presence
1. There was additional loss for regularization attached to conv3 output of the encoder

### WiderResNet encoder with U-Net like decoder and hypercolumn

1. WiderResNet38 encoder
1. U-Net like decoder (Double [conv3x3 + bn + relu])
1. SCSE blocks in decoder
1. Input tensor was 3-channel image [I, Y, I * Y]
1. In addition to mask output, model was also predicting salt presence

## Dataset & Folds

Trainset was split into 5 folds based on salt area. 
I used image size of 128x128 (resized with Lancsoz). 
I experimented with padding, but did not notice any improvement. 
Also tried 224 patches with padding, but I didn't run full validation on all folds and abandoned it.

## Training

There were a 3 losses:

1. Mask loss (BCE/Jaccard/Lovasz) with weight 1.0
1. Classification loss (BCE) with weight 0.5
1. Auxilarity loss (BCE) with weight 0.1

Training was done in 3 stages:

1. Warmup train for 50 epochs with BCE loss for mask with Adam.
1. Main train for 250 epochs with BCE+Jaccard loss for mask with Adam.
1. Fine-tune with 5 cycles of cosine annealing and restart.

## Inference

For prediction, I used horisontal flip TTA. 
Predictions after 5 annealing cycles and main train phase were averaged. In total, there were 60 predictions per model (6 weights x 5 folds x 2 TTA).
Masks were conditionaly zeroed if classifier predicted empty mask. Non-zero masks were postprocessed with `binary_fill_holes` method.

## What didn't work at all

1. CRF postprocessing
1. Geometric / Harmonic mean at ensembling
1. Mixup augmentation
1. Xception encoder
1. DeepLab-like models
1. Resnet34 :)


## What did work
1. Threshold tuning
1. Regularization with auxilarity loss
1. Predicting salt/not-salt

## Tried but not properly tested
1. Stochastic weight averaging
1. GANs for data augmentation

# Disclaimer

Repository will be not maintained. You can use it at own risk.
