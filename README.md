# Weakly Supervised Energy-based Learning for Action Segmentation

Pytorch implementation for our ICCV 2019 Oral Paper "Weakly Supervised Energy-based Learning for Action Segmentation".

## Abstract
This paper is about labeling video frames with action classes under weak supervision in training, where we have access only to a temporal ordering of actions, but their start and end frames in training videos are unknown. Following prior work, we use an HMM grounded on a Gated Recurrent Unit (GRU) for frame labeling. Our key contribution is a new formulation of constrained discriminative forward loss (CDFL) for training the HMM and GRU under weak supervision. Unlike prior work where the loss is typically estimated on a single, inferred video segmentation, we specify the CDFL to discriminate between the energy of all valid and invalid frame labelings of a training video. A valid frame labeling satisfies the ground-truth temporal ordering of actions, whereas an invalid one violates the ground truth. We specify an efficient recursive algorithm for computing the CDFL in terms of the logadd function of the segmentation energy. Our evaluation on action segmentation and alignment gives superior results to those of the state of the art on the benchmark Breakfast Action, Hollywood  Extended, and 50Salads datasets.

## Requirements
Python3.x with the libraries numpy and pytorch (version 0.4.1)

## Train
Run `python train.py`

## Inference
Run `python inference.py`

## Reference
https://github.com/alexanderrichard/NeuralNetwork-Viterbi
