#!/usr/bin/python3.7

import numpy as np
import multiprocessing as mp
import queue
from utils.dataset import Dataset
from utils.network import Forwarder
from utils.grammar import PathGrammar
from utils.length_model import PoissonModel
from utils.viterbi import Viterbi


### helper function for parallelized Viterbi decoding ##########################
def decode(queue, log_probs, decoder, index2label):
    while not queue.empty():
        try:
            video = queue.get(timeout = 3)
            score, labels, segments = decoder.decode( log_probs[video] )
            # save result
            with open('results/' + video, 'w') as f:
                f.write( '### Recognized sequence: ###\n' )
                f.write( ' '.join( [index2label[s.label] for s in segments] ) + '\n' )
                f.write( '### Score: ###\n' + str(score) + '\n')
                f.write( '### Frame level recognition: ###\n')
                f.write( ' '.join( [index2label[l] for l in labels] ) + '\n' )
        except queue.Empty:
            pass
        
def stn_decode(queue, log_probs, decoder, index2label, window, step):
    while not queue.empty():
        try:
            video = queue.get(timeout = 3)
            score, labels, segments = decoder.decode( log_probs[video])
#             cum_segments = np.array([segment.length for segment in segments])
#             cum_segments = np.cumsum(cum_segments)
#             print('segments', cum_segments)
#             print('labels', len(labels))
#             labels = np.array(labels)
            trancript = [s.label for s in segments]
#             print('trancript', trancript)
            stn_score, stn_labels, stn_segments = decoder.stn_decode( log_probs[video], segments, trancript, window, step)
#             stn_labels2 =  [s.label for s in stn_segments]
#             print('stn_labels2', stn_labels2)
#             print('stn_labels', len(stn_labels))
#             cum_segments = np.array([stn_segment.length for stn_segment in stn_segments])
#             cum_segments = np.cumsum(cum_segments)
#             print('stn_segments', cum_segments)
            # save result
            with open('results/' + video, 'w') as f:
                f.write( '### Recognized sequence: ###\n' )
                f.write( ' '.join( [index2label[s.label] for s in stn_segments] ) + '\n' )
                f.write( '### Score: ###\n' + str(stn_score) + '\n')
                f.write( '### Frame level recognition: ###\n')
                f.write( ' '.join( [index2label[l] for l in stn_labels] ) + '\n' )
        except queue.Empty:
            pass


### read label2index mapping and index2label mapping ###########################
label2index = dict()
index2label = dict()
with open('/scratch/liju2/nn_viterbi/data/mapping.txt', 'r') as f:
    content = f.read().split('\n')[0:-1]
    for line in content:
        label2index[line.split()[1]] = int(line.split()[0])
        index2label[int(line.split()[0])] = line.split()[1]

### read test data #############################################################
with open('/scratch/liju2/nn_viterbi/data/split1.test', 'r') as f:
    video_list = f.read().split('\n')[0:-1]
dataset = Dataset('/scratch/liju2/nn_viterbi/data', video_list, label2index, shuffle = False)

# load prior, length model, grammar, and network
load_iteration = 100000
log_prior = np.log( np.loadtxt('results/prior.iter-' + str(load_iteration) + '.txt') )
grammar = PathGrammar('results/grammar.txt', label2index)
length_model = PoissonModel('results/lengths.iter-' + str(load_iteration) + '.txt', max_length = 2000)
forwarder = Forwarder(dataset.input_dimension, dataset.n_classes)
forwarder.load_model('results/network.iter-' + str(load_iteration) + '.net')
window = 10
step = 5

# parallelization
n_threads = 4

# Viterbi decoder
viterbi_decoder = Viterbi(grammar, length_model, frame_sampling = 30)
# forward each video
log_probs = dict()
queue = mp.Queue()
for i, data in enumerate(dataset):
    sequence, _ = data
    video = list(dataset.features.keys())[i]
    queue.put(video)
    log_probs[video] = forwarder.forward(sequence).data.cpu().numpy() - log_prior
    log_probs[video] = log_probs[video] - np.max(log_probs[video])
# Viterbi decoding
procs = []
for i in range(n_threads):
    p = mp.Process(target = stn_decode, args = (queue, log_probs, viterbi_decoder, index2label, window, step) )
#     p = mp.Process(target = decode, args = (queue, log_probs, viterbi_decoder, index2label) )
    procs.append(p)
    p.start()
for p in procs:
    p.join()

