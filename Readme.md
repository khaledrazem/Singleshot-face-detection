################### Instructions ####################

0- download pretrained model from the following link.
https://uniofnottm-my.sharepoint.com/:f:/g/personal/ecyka1_nottingham_ac_uk/EiJJNN_CY2JGmLhPy4Xe4zMBbDtWcP34r6E0ZM1zMx0qwQ?e=VrFeJh

1- Open evaluation script.

2- set the path for training set and testing set in the first two lines.

3- run the script to view results of template matching and pretrained model.

4-to train a model from scrach open siadraft, change the training and testing path, and run.


################### Scripts ####################

siadraft: main script for trianing the model, contains network structure, training loop, and testing method.

modelloss: triplet loss implementation.

evaluation: testing script between provided template matching method and new CNN method.

getsiamesebatch: compliles batch of (anchor,similar,dissimilar) images for training.

getsimilarpair: returns augmented image from a given image.

getdissimilarpair: returns two unique images, given a list of images.

predictsiamese and forwardsiamese: returns network prediction for given data, one updates the network and one is used for evaluation.

methodtests: script used to test certain methods and debugging.

cnntestvisual: script used to retirive visualisation of cnn activation.
