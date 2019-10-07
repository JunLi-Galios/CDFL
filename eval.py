#!/usr/bin/python

import argparse
import glob
import re


def recog_file(filename, ground_truth_path):

    # read ground truth
    gt_file = ground_truth_path + re.sub('.*/','/',filename) + '.txt'
    with open(gt_file, 'r') as f:
        ground_truth = f.read().split('\n')[0:-1]
        f.close()
    # read recognized sequence
    with open(filename, 'r') as f:
        recognized = f.read().split('\n')[5].split() # framelevel recognition is in 6-th line of file
        f.close()

    n_frame_errors = 0
    for i in range(len(recognized)):
        if not recognized[i] == ground_truth[i]:
            n_frame_errors += 1

    return n_frame_errors, len(recognized)


### MAIN #######################################################################

### arguments ###
### --recog_dir: the directory where the recognition files from inferency.py are placed
### --ground_truth_dir: the directory where the framelevel ground truth can be found
parser = argparse.ArgumentParser()
parser.add_argument('--recog_dir', default='results')
parser.add_argument('--ground_truth_dir', default='data/groundTruth')
args = parser.parse_args()

filelist = glob.glob(args.recog_dir + '/P*')

print('Evaluate %d video files...' % len(filelist))

n_frames = 0
n_errors = 0
# loop over all recognition files and evaluate the frame error
for filename in filelist:
    errors, frames = recog_file(filename, args.ground_truth_dir)
    n_errors += errors
    n_frames += frames

# print frame accuracy (1.0 - frame error rate)
print('frame accuracy: %f' % (1.0 - float(n_errors) / n_frames))
