#!/usr/bin/python3.7

import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from .grammar import SingleTranscriptGrammar
from .length_model import PoissonModel


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.tower_stage = TowerModel(num_layers, num_f_maps, dim, num_classes)
        self.single_stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes, 3))
                                     for s in range(num_stages-1)])

    def forward(self, x, mask):
        middle_out, out = self.tower_stage(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.single_stages:
            middle_out, out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return middle_out, outputs


class TowerModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(TowerModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes, 3)
        self.stage2 = SingleStageModel(num_layers, num_f_maps, dim, num_classes, 5)

    def forward(self, x, mask):
        out1, final_out1 = self.stage1(x, mask)
        out2, final_out2 = self.stage2(x, mask)

        return out1 + out2, final_out1 + final_out2


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, kernel_size):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps, kernel_size))
                                     for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        final_out = self.conv_out(out) * mask[:, 0:1, :]
        return out, final_out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, kernel_size):
        super(DilatedResidualLayer, self).__init__()
        padding = int(dilation + dilation * (kernel_size - 3) / 2)
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


# buffer for old sequences (robustness enhancement: old frames are sampled from the buffer during training)
class Buffer(object):

    def __init__(self, buffer_size, n_classes):
        self.features = []
        self.transcript = []
        self.framelabels = []
        self.instance_counts = []
        self.label_counts = []
        self.buffer_size = buffer_size
        self.n_classes = n_classes
        self.next_position = 0
        self.frame_selectors = []

    def add_sequence(self, features, transcript, framelabels):
        if len(self.features) < self.buffer_size:
            # sequence data 
            self.features.append(features)
            self.transcript.append(transcript)
            self.framelabels.append(framelabels)
            # statistics for prior and mean lengths
            self.instance_counts.append( np.array( [ sum(np.array(transcript) == c) for c in range(self.n_classes) ] ) )
            self.label_counts.append( np.array( [ sum(np.array(framelabels) == c) for c in range(self.n_classes) ] ) )
            self.next_position = (self.next_position + 1) % self.buffer_size
        else:
            # sequence data
            self.features[self.next_position] = features
            self.transcript[self.next_position] = transcript
            self.framelabels[self.next_position] = framelabels
            # statistics for prior and mean lengths
            self.instance_counts[self.next_position] = np.array( [ sum(np.array(transcript) == c) for c in range(self.n_classes) ] )
            self.label_counts[self.next_position] = np.array( [ sum(np.array(framelabels) == c) for c in range(self.n_classes) ] )
            self.next_position = (self.next_position + 1) % self.buffer_size
        # update frame selectors
        self.frame_selectors = []
        for seq_idx in range(len(self.features)):
            self.frame_selectors += [ (seq_idx, frame) for frame in range(self.features[seq_idx].shape[1]) ]

    def random(self):
        return random.choice(self.frame_selectors) # return sequence_idx and frame_idx within the sequence

    def n_frames(self):
        return len(self.frame_selectors)


# wrapper class to provide torch tensors for the network
class DataWrapper(torch.utils.data.Dataset):

    # for each frame in the sequence, create a subsequence of length window_size
    def __init__(self, sequence, window_size = 21):
        self.features = []
        self.labels = []
        # ensure window_size is odd
        if window_size % 2 == 0:
            window_size += 1
        self.window_size = window_size
        # extract temporal window around each frame of the sequence
        for frame in range(sequence.shape[1]):
            left, right = max(0, frame - window_size // 2), min(sequence.shape[1], frame + 1 + window_size // 2)
            tmp = np.zeros((sequence.shape[0], window_size), dtype=np.float32 )
            tmp[:, window_size // 2 - (frame - left) : window_size // 2 + (right - frame)] = sequence[:, left : right]
            self.features.append(np.transpose(tmp))
            self.labels.append(-1) # dummy label, will be updated after Viterbi decoding

    # add a sampled (windowed frame, label) pair to the data wrapper (include buffered data during training)
    # @sequence the sequence from which the frame is sampled
    # @label the Viterbi decoding label for the frame at frame_idx
    # @frame_idx the index of the frame to sample
    def add_buffered_frame(self, sequence, label, frame_idx):
        left, right = max(0, frame_idx - self.window_size // 2), min(sequence.shape[1], frame_idx + 1 + self.window_size // 2)
        tmp = np.zeros((sequence.shape[0], self.window_size), dtype=np.float32 )
        tmp[:, self.window_size // 2 - (frame_idx - left) : self.window_size // 2 + (right - frame_idx)] = sequence[:, left : right]
        self.features.append(np.transpose(tmp))
        self.labels.append(label)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        assert idx < len(self)
        features = torch.from_numpy( self.features[idx] )
        labels = torch.from_numpy( np.array([self.labels[idx]], dtype=np.int64) )
        return features, labels


class Net(nn.Module):

    def __init__(self, input_dim, hidden_size, n_classes):
        super(Net, self).__init__()
        self.n_classes = n_classes
        self.gru = nn.GRU(input_dim, hidden_size, 1, bidirectional = False, batch_first = True)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        dummy, output = self.gru(x)
        output = self.fc(output)
        output = nn.functional.log_softmax(output, dim=2) # tensor is of shape (batch_size, 1, features)
        return output


class Forwarder(object):

    def __init__(self, num_blocks, num_layers, num_f_maps, input_dimension, n_classes):
        self.n_classes = n_classes
        hidden_size = 64
#         self.net = Net(input_dimension, hidden_size, n_classes)
        self.net = MultiStageModel(num_blocks, num_layers, num_f_maps, input_dimension, n_classes)
        self.net.cuda()

    def _forward(self, data_wrapper, batch_size = 512):
        dataloader = torch.utils.data.DataLoader(data_wrapper, batch_size = batch_size, shuffle = False)
        # output probability container
        log_probs_list = []
#         offset = 0
        # forward all frames
        for data in dataloader:
            input, _ = data
            input = input.cuda()
            output = self.net(input)[0,:,:]
#             print('output', output.size())
            log_probs_list.append(output)
#             offset += output.shape[1]
        log_probs = torch.cat(log_probs_list, dim=0)
        return log_probs

    def forward(self, sequence, batch_size = 512):
#         data_wrapper = DataWrapper(sequence, window_size = 21)
        print('MS-TCN sequence', sequence.size())
        out = self.net(sequence)
        print('MS-TCN out', out.size())
        return out

    def load_model(self, model_file):
        self.net.cpu()
        self.net.load_state_dict( torch.load(model_file) )
        self.net.cuda()


class Trainer(Forwarder):

    def __init__(self, decoder, num_blocks, num_layers, num_f_maps, input_dimension, n_classes, buffer_size, buffered_frame_ratio = 25):
        super(Trainer, self).__init__(input_dimension, n_classes)
        self.buffer = Buffer(buffer_size, n_classes)
        self.decoder = decoder
        self.buffered_frame_ratio = buffered_frame_ratio
        self.criterion = nn.NLLLoss()
        self.prior = np.ones((self.n_classes), dtype=np.float32) / self.n_classes
        self.mean_lengths = np.ones((self.n_classes), dtype=np.float32)


    def update_mean_lengths(self):
        self.mean_lengths = np.zeros( (self.n_classes), dtype=np.float32 )
        for label_count in self.buffer.label_counts:
            self.mean_lengths += label_count
        instances = np.zeros((self.n_classes), dtype=np.float32)
        for instance_count in self.buffer.instance_counts:
            instances += instance_count
        # compute mean lengths (backup to average length for unseen classes)
        self.mean_lengths = np.array( [ self.mean_lengths[i] / instances[i] if instances[i] > 0 else sum(self.mean_lengths) / sum(instances) for i in range(self.n_classes) ] )


    def update_prior(self):
        # count labels
        self.prior = np.zeros((self.n_classes), dtype=np.float32)
        for label_count in self.buffer.label_counts:
            self.prior += label_count
        self.prior = self.prior / np.sum(self.prior)
        # backup to uniform probability for unseen classes
        n_unseen = sum(self.prior == 0)
        self.prior = self.prior * (1.0 - float(n_unseen) / self.n_classes)
        self.prior = np.array( [ self.prior[i] if self.prior[i] > 0 else 1.0 / self.n_classes for i in range(self.n_classes) ] )


    def train(self, sequence, transcript, batch_size = 512, learning_rate = 0.1, window = 20, step = 5):
        #print('--------------------new video-----------------')
#         data_wrapper = DataWrapper(sequence, window_size = 21)
        # forwarding and Viterbi decoding
        log_probs_origin = self.forward(sequence)
        log_probs = log_probs_origin.data.cpu().numpy() - np.log(self.prior)
        log_probs = log_probs - np.max(log_probs)
        # define transcript grammar and updated length model
        self.decoder.grammar = SingleTranscriptGrammar(transcript, self.n_classes)
        self.decoder.length_model = PoissonModel(self.mean_lengths)
        # decoding
        score, labels, segments = self.decoder.decode(log_probs)

        video_length = log_probs_origin.shape[0]
        optimizer = optim.SGD(self.net.parameters(), lr = learning_rate / 512)
        optimizer.zero_grad()
        penalty = -log_probs_origin
        loss1 = self.decoder.forward_score(penalty, segments, transcript, window, step)
        loss2 = self.decoder.incremental_forward_score(penalty, segments, transcript, window, step)
        loss = loss1 - loss2
        loss.backward()
        optimizer.step()

        # add sequence to buffer
        self.buffer.add_sequence(sequence, transcript, labels)
        # update prior and mean length
        self.update_prior()
        self.update_mean_lengths()
        return loss1 / video_length, loss2 / video_length


    def save_model(self, network_file, length_file, prior_file):
        self.net.cpu()
        torch.save(self.net.state_dict(), network_file)
        self.net.cuda()
        np.savetxt(length_file, self.mean_lengths)
        np.savetxt(prior_file, self.prior)
        

