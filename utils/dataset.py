#!/usr/bin/python3.7

import numpy as np
import random

# self.features[video]: the feature array of the given video (dimension x frames)
# self.transcrip[video]: the transcript (as label indices) for each video
# self.input_dimension: dimension of video features
# self.n_classes: number of classes
class Dataset(object):

    def __init__(self, base_path, video_list, label2index, shuffle = False):
        self.features = dict()
        self.transcript = dict()
        self.shuffle = shuffle
        self.idx = 0
        # read features for each video
        for video in video_list:
            # video features
            self.features[video] = np.load(base_path + '/features/' + video + '.npy')
            # transcript
            with open(base_path + '/transcripts/' + video + '.txt') as f:
                self.transcript[video] = [ label2index[line] for line in f.read().split('\n')[0:-1] ]
        # selectors for random shuffling
        self.selectors = list(self.features.keys())
        # print(self.selectors)
        if self.shuffle:
            random.shuffle(self.selectors)
        # set input dimension and number of classes
        self.input_dimension = list(self.features.values())[0].shape[0]
        self.n_classes = len(label2index)

    def videos(self):
        return list(self.features.keys())

    def __len__(self):
        return len(self.features)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx == len(self):
            self.idx = 0
            if self.shuffle:
                random.shuffle(self.selectors)
            raise StopIteration
        else:
            video = self.selectors[self.idx]
            self.idx += 1
            return self.features[video], self.transcript[video]

    def get(self):
        try:
            return next(self)
        except StopIteration:
            return self.get()

