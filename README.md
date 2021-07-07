# CVASV-Speaker-Verification-Training-
This is the training section for the CVASV command system. To use the voice activation and speaker verification system, you can directly download: https://github.com/yunfei96/Voice-Activation-and-Speaker-Verification-Command-System.git


This is the training section for the speaker verification part of the CVASV command system. contain mobilenetV1 and VGG-M trainning code. 

You need to have the VoxCeleb1 dataset for training

This contains data preparation,change the path and you are good to train. 
training_mbn.py: main training file for mobilenet v1.
training_vgg.py: main training file for vgg.
val_roc.py: ROC curve to verify speaker verification performemce

others files are helper functions

The trainning setting is using Titian X GPU, E5-1650, 32g RAM, windows 10,took around 20 hrs to get to 80% classificattion accuracy. 

Lib setting
anaconda: numpy, tf
