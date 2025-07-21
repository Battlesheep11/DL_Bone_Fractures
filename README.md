This project focuses on designing and assessing deep learning models for the automatic classification of wrist X-ray images. The task is a binary classification: determining whether a wrist is fractured or not fractured based on radiographic data.

The dataset used for this project was obtained from GRAZPEDWRI-DX - link: 
https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193
a publicly available pediatric wrist X-ray dataset. After downloading, we performed preprocessing and restructuring of the data into a clean directory format suitable for training and evaluation.

Two model architectures were implemented and compared:

A Custom Convolutional Neural Network (CustomCNN) built from scratch.
A ResNet50 model leveraging transfer learning with pretrained ImageNet weights.
project workflow included:

Dataset preparation and augmentation
Building and training models
Evaluating on a validation set
Visualizing training progress and confusion matrices
Comparing results and drawing key conclusions
