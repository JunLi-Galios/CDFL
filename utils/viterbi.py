#!/usr/bin/python3.7

import numpy as np
from .grammar import PathGrammar
from .length_model import PoissonModel
import glob
import re
import torch

# Viterbi decoding
class Viterbi(object):

    ### helper structure ###
    class TracebackNode(object):
        def __init__(self, label, predecessor, length = 0, boundary = False):
            self.label = label
            self.length = length
            self.predecessor = predecessor
            self.boundary = boundary

    ### helper structure ###
    class HypDict(dict):
        class Hypothesis(object):
            def __init__(self, score, traceback):
                self.score = score
                self.traceback = traceback

        def update(self, key, score, traceback, logadd=False):
            if (not key in self):
                self[key] = self.Hypothesis(score, traceback)
            else:
                if logadd:
                    self[key].score = self.logadd(score, self[key].score)
                elif self[key].score <= score:
                    self[key] = self.Hypothesis(score, traceback)

        def logadd(self, a, b):
#             print('a', a)
#             print('b', b)
            if a <= b:
                result = - torch.log(1 + torch.exp(a - b)) + a
            else:
                result = - torch.log(1 + torch.exp(b - a)) + b
            return result

    # @grammar: the grammar to use, must inherit from class Grammar
    # @length_model: the length model to use, must inherit from class LengthModel
    # @frame_sampling: generate hypotheses every frame_sampling frames
    # @max_hypotheses: maximal number of hypotheses. Smaller values result in stronger pruning
    def __init__(self, grammar, length_model, frame_sampling = 1, max_hypotheses = np.inf):
        self.grammar = grammar
        self.length_model = length_model
        self.frame_sampling = frame_sampling
        self.max_hypotheses = max_hypotheses

    # Viterbi decoding of a sequence
    # @log_frame_probs: logarithmized frame probabilities
    #                   (usually log(network_output) - log(prior) - max_val, where max_val ensures negativity of all log scores)
    # @return: the score of the best sequence,
    #          the corresponding framewise labels (len(labels) = len(sequence))
    #          and the inferred segments in the form (label, length)
    def decode(self, log_frame_probs):
        assert log_frame_probs.shape[1] == self.grammar.n_classes()
        frame_scores = np.cumsum(log_frame_probs, axis=0) # cumulative frame scores allow for quick lookup if frame_sampling > 1
        # create initial hypotheses
        hyps = self.init_decoding(frame_scores)
        # decode each following time step
        for t in range(2 * self.frame_sampling - 1, frame_scores.shape[0], self.frame_sampling):
            hyps = self.decode_frame(t, hyps, frame_scores)
            self.prune(hyps)
        # transition to end symbol
        final_hyp = self.finalize_decoding(hyps)
        labels, segments = self.traceback(final_hyp, frame_scores.shape[0])
        return final_hyp.score, labels, segments


    ### helper functions ###
    def frame_score(self, frame_scores, t, label):
        if t >= self.frame_sampling:
            return frame_scores[t, label] - frame_scores[t - self.frame_sampling, label]
        else:
            return frame_scores[t, label]

    def prune(self, hyps):
        if len(hyps) > self.max_hypotheses:
            tmp = sorted( [ (hyps[key].score, key) for key in hyps ] )
            del_keys = [ x[1] for x in tmp[0 : -self.max_hypotheses] ]
            for key in del_keys:
                del hyps[key]

    def init_decoding(self, frame_scores):
        hyps = self.HypDict()
        context = (self.grammar.start_symbol(),)
        for label in self.grammar.possible_successors(context):
            key = context + (label, self.frame_sampling)
            score = self.grammar.score(context, label) + self.frame_score(frame_scores, self.frame_sampling - 1, label)
            hyps.update(key, score, self.TracebackNode(label, None, boundary = True))
        return hyps

    def decode_frame(self, t, old_hyp, frame_scores):
        new_hyp = self.HypDict()
        for key, hyp in list(old_hyp.items()):
            context, label, length = key[0:-2], key[-2], key[-1]
            # stay in the same label...
            new_key = context + (label, min(length + self.frame_sampling, self.length_model.max_length()))
            score = hyp.score + self.frame_score(frame_scores, t, label)
            new_hyp.update(new_key, score, self.TracebackNode(label, hyp.traceback, boundary = False))
            # ... or go to the next label
            context = context + (label,)
            for new_label in self.grammar.possible_successors(context):
                if new_label == self.grammar.end_symbol():
                    continue
                new_key = context + (new_label, self.frame_sampling)
                score = hyp.score + self.frame_score(frame_scores, t, label) + self.length_model.score(length, label) + self.grammar.score(context, new_label)
                new_hyp.update(new_key, score, self.TracebackNode(new_label, hyp.traceback, boundary = True))
        # return new hypotheses
        return new_hyp

    def finalize_decoding(self, old_hyp):
        final_hyp = self.HypDict.Hypothesis(-np.inf, None)
        for key, hyp in list(old_hyp.items()):
            context, label, length = key[0:-2], key[-2], key[-1]
            context = context + (label,)
            score = hyp.score + self.length_model.score(length, label) + self.grammar.score(context, self.grammar.end_symbol())
            if score >= final_hyp.score:
                final_hyp.score, final_hyp.traceback = score, hyp.traceback
        # return final hypothesis
        return final_hyp

    def traceback(self, hyp, n_frames):
        class Segment(object):
            def __init__(self, label):
                self.label, self.length = label, 0
        traceback = hyp.traceback
        labels = []
        segments = [Segment(traceback.label)]
        while not traceback == None:
            segments[-1].length += self.frame_sampling
            labels += [traceback.label] * self.frame_sampling
            if traceback.boundary and not traceback.predecessor == None:
                segments.append( Segment(traceback.predecessor.label) )
            traceback = traceback.predecessor
        segments[-1].length += n_frames - len(labels) # append length of missing frames
        labels += [hyp.traceback.label] * (n_frames - len(labels)) # append labels for missing frames
        return list(reversed(labels)), list(reversed(segments))


    def init_forward(self):
        hyps = self.HypDict()
        context = (self.grammar.start_symbol(),)
        key = context + (-1, )
        hyps.update(key, 0, None)
        return hyps
    
    def init_discriminative_forward(self):
        hyps = self.HypDict()
        key = -1
        hyps.update(key, 0, None)
        return hyps

    def forward_frame(self, t, old_hyp, frame_scores, next_label):
        new_hyp = self.HypDict()
        new_key = None
        for key, hyp in list(old_hyp.items()):
            context, idx = key[0:-1], key[-1]
            if t <= idx:
                continue

            # ... go to the next label
            if idx < 0:
                segment_score = frame_scores[t, next_label]
            else:
                segment_score = frame_scores[t, next_label] - frame_scores[idx, next_label]
            score = hyp.score + segment_score           
            context = context + (next_label,)
            new_key = context + (t,)
            new_hyp.update(new_key, score, None, logadd=True)
        return new_hyp, new_key
    
    def discriminative_forward_frame(self, t, old_hyp, frame_scores):
        new_hyp = self.HypDict()
        new_key = t
        for key, hyp in list(old_hyp.items()):
            idx = key
            if t <= idx:
                continue

            # ... go to the next label
            for label in range(self.grammar.n_classes()):                
                if idx < 0:
                    segment_score = frame_scores[t, label]
                else:
                    segment_score = frame_scores[t, label] - frame_scores[idx, label]
                score = hyp.score + segment_score
                new_hyp.update(new_key, score, None, logadd=True)  
        return new_hyp, new_key

    def forward_score(self, log_frame_probs, segments, transcript, window, step):
        assert log_frame_probs.shape[1] == self.grammar.n_classes()
        frame_scores = torch.cumsum(log_frame_probs, dim=0)  # cumulative frame scores
        hyps = self.init_forward()
        
        length = frame_scores.shape[0]
        cum_segments = np.array([segment.length for segment in segments])
        cum_segments = np.cumsum(cum_segments)

        for i in range(len(cum_segments) - 1):
            new_hyps = self.HypDict()
            for t in range(cum_segments[i] - window // 2 - 1, cum_segments[i] + window // 2, step):
                t = min(t, length - 1)
                hyp, _ = self.forward_frame(t, hyps, frame_scores, transcript[i])
                new_hyps = self.HypDict(list(new_hyps.items()) + list(hyp.items()))        
            hyps = new_hyps
        # transition to end symbol
        final_hyp, final_key = self.forward_frame(length - 1, hyps, frame_scores, transcript[-1])

        return final_hyp[final_key].score

    def discriminative_forward_score(self, log_frame_probs, segments, window, step):
        assert log_frame_probs.shape[1] == self.grammar.n_classes()
        frame_scores = torch.cumsum(log_frame_probs, dim=0)  # cumulative frame scores
        hyps = self.init_discriminative_forward()
        
        length = frame_scores.shape[0]
        cum_segments = np.array([segment.length for segment in segments])
        cum_segments = np.cumsum(cum_segments)

        for i in range(len(cum_segments) - 1):
            new_hyps = self.HypDict()
            for t in range(cum_segments[i] - window // 2 - 1, cum_segments[i] + window // 2, step):
                t = min(t, length - 1)
                hyp, _ = self.discriminative_forward_frame(t, hyps, frame_scores)
                new_hyps = self.HypDict(list(new_hyps.items()) + list(hyp.items()))        
            hyps = new_hyps
        # transition to end symbol
        final_hyp, final_key = self.discriminative_forward_frame(length - 1, hyps, frame_scores)

        return final_hyp[final_key].score


    def init_incremental_forward(self):
        hyps = self.HypDict()
        key = -1
        hyps.update(key, torch.Tensor([0.]).cuda(), None)
        return hyps
    
    
    def incremental_forward_frame(self, t, old_hyp, frame_scores, next_label):
        new_hyp = self.HypDict()
        new_key = t
        for key, hyp in list(old_hyp.items()):
            idx = key
            if t <= idx:
                continue

            # ... go to the next label
            if idx < 0:
                segment_score = frame_scores[t, :]
            else:
                segment_score = frame_scores[t, :] - frame_scores[idx, :]
            for label in range(self.grammar.n_classes()):                
                if segment_score[label] < segment_score[next_label]:
                    score = hyp.score + segment_score[label]
                else:
                    score = hyp.score
                new_hyp.update(new_key, score, None, logadd=True)  
        return new_hyp

    
    def incremental_forward_score(self, log_frame_probs, segments, transcript, window, step):
        assert log_frame_probs.shape[1] == self.grammar.n_classes()
        frame_scores = torch.cumsum(log_frame_probs, dim=0)  # cumulative frame scores
        hyps = self.init_incremental_forward()
        
        length = frame_scores.shape[0]
        cum_segments = np.array([segment.length for segment in segments])
        cum_segments = np.cumsum(cum_segments)

        for i in range(len(cum_segments) - 1):
            new_hyps = self.HypDict()
            for t in range(cum_segments[i] - window // 2 - 1, cum_segments[i] + window // 2, step):
                if t >= length:
                    t = (cum_segments[i] + cum_segments[i + 1]) / 2
                hyp = self.incremental_forward_frame(t, hyps, frame_scores, transcript[i])
                new_hyps = self.HypDict(list(new_hyps.items()) + list(hyp.items()))
            hyps = new_hyps
        # transition to end symbol
        final_hyp = self.incremental_forward_frame(length - 1, hyps, frame_scores, transcript[-1])
        final_key = length - 1

        return final_hyp[final_key].score


    def stn_decode(self, log_frame_probs, segments, trancript, window, step):
        assert log_frame_probs.shape[1] == self.grammar.n_classes()
        frame_scores = np.cumsum(log_frame_probs, axis=0) # cumulative frame scores
        hyps = self.init_stn_decoding()
        
        length = frame_scores.shape[0]
        cum_segments = np.array([segment.length for segment in segments])
        cum_segments = np.cumsum(cum_segments)

        for i in range(len(cum_segments) - 1):
            new_hyps = self.HypDict()
            for t in range(cum_segments[i] - window // 2 - 1, cum_segments[i] + window // 2, step):
                t = min(t, length - 1)
                hyp = self.stn_decode_frame(t, hyps, frame_scores, trancript[i])
                new_hyps = self.HypDict(list(new_hyps.items()) + list(hyp.items()))        
            hyps = new_hyps
        final_hyp, final_score = self.stn_finalize_decoding(length - 1, hyps, frame_scores, trancript[-1])
        
        stn_labels, stn_segments = self.stn_traceback(final_hyp, frame_scores.shape[0])

        return final_score, stn_labels, stn_segments
    
    
    def init_stn_decoding(self):
        hyps = self.HypDict()
        context = (self.grammar.start_symbol(),)
        idx = -1
        key = context + (idx, )
        hyps.update(key, 0, self.TracebackNode(self.grammar.start_symbol(), None, 0, boundary = True))
        return hyps

    def stn_decode_frame(self, t, old_hyp, frame_scores, next_label):
        new_hyp = self.HypDict()
        for key, hyp in list(old_hyp.items()):
            context, idx = key[0:-1], key[-1]
            if t <= idx:
                continue

            # ... go to the next label
            if idx < 0:
                segment_score = frame_scores[t, next_label]
            else:
                segment_score = frame_scores[t, next_label] - frame_scores[idx, next_label]
            length = t - idx
            score = hyp.score + segment_score + self.length_model.score(length, next_label) + self.grammar.score(context, next_label)
            context = context + (next_label,)
            new_key = context + (t,)
            new_hyp.update(new_key, score, self.TracebackNode(next_label, hyp.traceback, length, boundary = True), logadd=False)
        return new_hyp
    

    def stn_finalize_decoding(self, t, old_hyp, frame_scores, next_label):
        final_hyp = self.HypDict.Hypothesis(-np.inf, None)
        for key, hyp in list(old_hyp.items()):
            context, idx = key[0:-1], key[-1]
            if idx < 0:
                segment_score = frame_scores[t, next_label]
            else:
                segment_score = frame_scores[t, next_label] - frame_scores[idx, next_label]
            context = context + (next_label,)
            length = t - idx
            score = hyp.score + segment_score + self.length_model.score(length, next_label) + self.grammar.score(context, self.grammar.end_symbol())
            if score >= final_hyp.score:
                final_hyp.score, final_hyp.traceback = score, self.TracebackNode(next_label, hyp.traceback, length, boundary = True)
        return final_hyp, final_hyp.score

    def stn_traceback(self, hyp, n_frames):
        class Segment(object):
            def __init__(self, label):
                self.label, self.length = label, 0
        traceback = hyp.traceback
        labels = []
        segments = []
        while not traceback.predecessor == None:
            segments.append(Segment(traceback.label))
            segments[-1].length = traceback.length
            labels += [traceback.label] * traceback.length
            traceback = traceback.predecessor
        return list(reversed(labels)), list(reversed(segments))




