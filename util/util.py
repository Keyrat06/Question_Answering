import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import Counter
from io import open
import numpy as np
import random
import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print '%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000)
        return result
    return timed


# This class offers the self.raw_corpus, the word to id map self.vocab_map, and the self.mapped_corpus.
# You might also find vocab useful
@timeit
class data_loader():
    def __init__(self, data_file="../data/text_tokenized.txt", unk="<unk>", \
                 padding="<padding>", start="<s>", end="</s>", cut_off=2):
        self.raw_corpus = {}
        with open(data_file) as data:
            for line in data:
                id, title, body = line.split("\t")
                if len(title) == 0:
                    continue
                title = title.strip().split()
                body = body.strip().split()
                self.raw_corpus[id] = (title, body)
        self.vocab = Counter(w for id, pair in self.raw_corpus.iteritems() \
                                   for x in pair for w in x)
        self.unk = unk
        self.padding = padding
        self.start = start
        self.end = end
        self.vocab[unk] = cut_off + 1
        self.vocab[padding] = cut_off + 1
        self.vocab[start] = cut_off + 1
        self.vocab[end] = cut_off + 1
        self.vocab_map = {}
        for word in self.vocab:
            if self.vocab[word] <= cut_off: word = unk
            if word not in self.vocab_map:
                self.vocab_map[word] = len(self.vocab_map)

        self.mapped_corpus = {}
        for id in self.raw_corpus:
            mapped_title = [self.vocab_map.get(word, self.vocab_map[unk]) for word in self.raw_corpus[id][0]]
            mapped_body = [self.vocab_map.get(word, self.vocab_map[unk]) for word in self.raw_corpus[id][1]]
            self.mapped_corpus[id] = (mapped_title, mapped_body)
        self.num_tokens = len(self.vocab_map)


    def read_annotations(self, path, K_neg=10, prune_pos_cnt=10):
        lst = [ ]
        with open(path) as fin:
            for line in fin:
                parts = line.split("\t")
                pid, pos, neg = parts[:3]
                pos = pos.split()
                neg = neg.split()
                if len(pos) == 0 or (len(pos) > prune_pos_cnt and prune_pos_cnt != -1): continue
                if K_neg != -1:
                    random.shuffle(neg)
                    neg = neg[:K_neg]
                s = set()
                qids = [ ]
                qlabels = [ ]
                for q in neg:
                    if q not in s:
                        qids.append(q)
                        qlabels.append(0 if q not in pos else 1)
                        s.add(q)
                for q in pos:
                    if q not in s:
                        qids.append(q)
                        qlabels.append(1)
                        s.add(q)
                lst.append((pid, qids, qlabels))
        return lst

    def create_batches(self, data, batch_size):
        perm = range(len(data))
        random.shuffle(perm)
        N = len(data)
        pid2id = {}
        titles = [ ]
        bodies = [ ]
        triples = [ ]
        batches = [ ]
        cnt = 0
        for u in xrange(N):
            i = perm[u]
            pid, qids, qlabels = data[i]
            if pid not in self.mapped_corpus: continue
            cnt += 1
            for id in [pid] + qids:
                if id not in pid2id:
                    if id not in self.mapped_corpus: continue
                    pid2id[id] = len(titles)
                    t, b = self.mapped_corpus[id]
                    titles.append(t)
                    bodies.append(b)
            pid = pid2id[pid]
            pos = [ pid2id[q] for q, l in zip(qids, qlabels) if l == 1 and q in pid2id ]
            neg = [ pid2id[q] for q, l in zip(qids, qlabels) if l == 0 and q in pid2id ]
            triples += [ [pid,x]+neg for x in pos ]
            if cnt == batch_size or u == N-1:
                titles, bodies = self.create_one_batch(titles, bodies)
                triples = self.create_hinge_batch(triples)
                batches.append((titles, bodies, triples))
                titles = [ ]
                bodies = [ ]
                triples = [ ]
                pid2id = {}
                cnt = 0
        return batches

    def create_eval_batches(self, data):
        lst = [ ]
        for pid, qids, qlabels in data:
            titles = [ ]
            bodies = [ ]
            for id in [pid]+qids:
                t, b = self.mapped_corpus[id]
                titles.append(t)
                bodies.append(b)
            titles, bodies = self.create_one_batch(titles, bodies)
            lst.append((titles, bodies, np.array(qlabels, dtype="int32")))
        return lst

    def create_one_batch(self, titles, bodies):
        padding_id = self.vocab_map[self.padding]
        max_title_len = max(1, max(len(x) for x in titles))
        max_body_len = max(1, max(len(x) for x in bodies))
        titles = np.column_stack([ np.pad(x,(max_title_len-len(x),0),'constant', \
                                          constant_values=padding_id) for x in titles])
        bodies = np.column_stack([ np.pad(x,(max_body_len-len(x),0),'constant', \
                                          constant_values=padding_id) for x in bodies])
        return titles, bodies

    def create_hinge_batch(self, triples):
        max_len = max(len(x) for x in triples)
        triples = np.vstack([ np.pad(x,(0,max_len-len(x)),'edge')
                              for x in triples ]).astype('int32')
        return triples


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, padding_id):
        super(Encoder, self).__init__()
        self.padding_id = padding_id
        self.embedding = nn.Embedding(input_size, embedding_dim)

    def forward(self, input):
        if torch.cuda.is_available():
            return self.embedding(Variable(torch.from_numpy(input)).cuda())
        else:
            self.embedding(Variable(torch.from_numpy(input)))


class pre_trained_Encoder():

    @timeit
    def __init__(self, padding_id, data_loader, emb_path, cuda):
        self.cuda = cuda
        self.padding_id = padding_id
        self.embedding_size = 200
        self.embeddings = self.load_embedding_iterator(emb_path, data_loader)

    def load_embedding_iterator(self, path, data_loader):
        embs = np.zeros((len(data_loader.vocab_map), self.embedding_size))
        with open(path) as fin:
            for line in fin:
                line = line.strip()
                if line:
                    parts = line.split()
                    word = parts[0]
                    vals = np.array([ float(x) for x in parts[1:] ])
                    if word in data_loader.vocab_map:
                        embs[data_loader.vocab_map[word]] = vals
        return embs

    def __call__(self, *input, **kwargs):
        return self.embed_batch(*input, **kwargs)

    def embed_batch(self, input):
        output = np.zeros((len(input), len(input[0]), self.embedding_size))
        for i in xrange(len(input)):
            for j in xrange(len(input[i])):
                output[i, j] = self.embeddings[input[i, j]]
        if self.cuda:
            return Variable(torch.from_numpy(output), requires_grad=False).float().cuda()
        else:
            return Variable(torch.from_numpy(output), requires_grad=False).float()



class CNN(nn.Module):
    def __init__(self, embedding_dim, output_size, conv_width):
        super(CNN, self).__init__()
        self.conv_width = conv_width
        self.output_size = output_size
        self.conv = nn.Conv1d(embedding_dim, output_size, conv_width, stride=1, padding=(conv_width-1)/2, dilation=1, groups=1, bias=True)

    def forward(self, input):
        conv = F.tanh(self.conv(input))
        return conv


class Evaluation:
    def __init__(self, data):
        self.data = data

    def Precision(self, precision_at):
        scores = []
        for item in self.data:
            temp = item[:precision_at]
            if any(val==1 for val in item):
                scores.append(sum([1 if val==1 else 0 for val in temp])*1.0 / len(temp) if len(temp) > 0 else 0.0)
        return sum(scores)/len(scores) if len(scores) > 0 else 0.0

    def MAP(self):
        scores = []
        missing_MAP = 0
        for item in self.data:
            temp = []
            count = 0.0
            for i,val in enumerate(item):
                if val == 1:
                    count += 1.0
                    temp.append(count/(i+1))
            if len(temp) > 0:
                scores.append(sum(temp) / len(temp))
            else:
                missing_MAP += 1
        return sum(scores)/len(scores) if len(scores) > 0 else 0.0

    def MRR(self):
        scores = []
        for item in self.data:
            for i,val in enumerate(item):
                if val == 1:
                    scores.append(1.0/(i+1))
                    break
        return sum(scores)/len(scores) if len(scores) > 0 else 0.0


def evaluate(data, score_func, encoder, CNN, cuda):
    cs = nn.CosineSimilarity(dim=1)
    res = [ ]
    for idts, idbs, labels in data:
        scores = score_func(idts, idbs, encoder, CNN, cs, cuda)
        assert len(scores) == len(labels)
        ranks = (-scores).argsort()
        ranked_labels = labels[ranks]
        res.append(ranked_labels)
    e = Evaluation(res)
    MAP = e.MAP()*100
    MRR = e.MRR()*100
    P1 = e.Precision(1)*100
    P5 = e.Precision(5)*100
    return MAP, MRR, P1, P5
