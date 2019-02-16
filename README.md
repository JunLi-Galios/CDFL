# Segmentation-Transformer-Networks-for-Weakly-Supervised-Action-Segmentation

Pytorch implementation for the paper "Segmentation Transformer Networks for Weakly Supervised ActionSegmentation".

## Abstract
This paper is about weakly supervised action segmentation by labeling video frames with action classes. Weak supervision means that in training we have access only to a temporal ordering of actions in the video, but their start and end frames are unknown. To address the fundamental challenge that multiple distinct action segmentations have the same temporal ordering of actions, we explicitly represent all candidate segmentations of a video as paths in a graph whose nodes are segmentation cuts and edges are segments between two cuts. Inference is formulated as an efficient search for the minimum-energy path in the segmentation graph. In a training segmentation graph, some paths are valid as they satisfy the ground-truth ordering of actions, and other paths are invalid. We consider three novel formulations of loss aimed at maximizing the decision margin between all valid paths and invalid paths. This differs from training in prior work that typically seeks to minimize loss incurred on a single video segmentation. Our evaluation on action segmentation and alignment gives superior results to those of the state of the art on the benchmark datasets.

## Requirements
Python3.x with the libraries numpy and pytorch (version 0.4.1)

## Train
Run `python train.py`

## Inference
Run `python inference.py`
