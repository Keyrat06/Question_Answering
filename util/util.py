import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from io import open
import numpy as np
import random
import time
from tqdm import trange
from collections import Counter, defaultdict
import math
import numbers


class Meter(object):
    def reset(self):
        pass

    def add(self):
        pass

    def value(self):
        pass


class AUCMeter(Meter):
    """
    The AUCMeter measures the area under the receiver-operating characteristic
    (ROC) curve for binary classification problems. The area under the curve (AUC)
    can be interpreted as the probability that, given a randomly selected positive
    example and a randomly selected negative example, the positive example is
    assigned a higher score by the classification model than the negative example.

    The AUCMeter is designed to operate on one-dimensional Tensors `output`
    and `target`, where (1) the `output` contains model output scores that ought to
    be higher when the model is more convinced that the example should be positively
    labeled, and smaller when the model believes the example should be negatively
    labeled (for instance, the output of a signoid function); and (2) the `target`
    contains only values 0 (for negative examples) and 1 (for positive examples).
    """
    def __init__(self):
        super(AUCMeter, self).__init__()
        self.reset()

    def reset(self):
        self.scores = torch.DoubleTensor(torch.DoubleStorage()).numpy()
        self.targets = torch.LongTensor(torch.LongStorage()).numpy()

    def add(self, output, target):
        if torch.is_tensor(output):
            output = output.cpu().squeeze().numpy()
        if torch.is_tensor(target):
            target = target.cpu().squeeze().numpy()
        elif isinstance(target, numbers.Number):
            target = np.asarray([target])
        assert np.ndim(output) == 1, \
            'wrong output size (1D expected)'
        assert np.ndim(target) == 1, \
            'wrong target size (1D expected)'
        assert output.shape[0] == target.shape[0], \
            'number of outputs and targets does not match'
        assert np.all(np.add(np.equal(target, 1), np.equal(target, 0))), \
            'targets should be binary (0, 1)'

        self.scores = np.append(self.scores, output)
        self.targets = np.append(self.targets, target)
        self.sortind = None


    def value(self, max_fpr=1.0):
        assert max_fpr > 0

        # case when number of elements added are 0
        if self.scores.shape[0] == 0:
            return 0.5

        # sorting the arrays
        if self.sortind is None:
            scores, sortind = torch.sort(torch.from_numpy(self.scores), dim=0, descending=True)
            scores = scores.numpy()
            self.sortind = sortind.numpy()
        else:
            scores, sortind = self.scores, self.sortind

        # creating the roc curve
        tpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)
        fpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)

        for i in range(1, scores.size + 1):
            if self.targets[sortind[i - 1]] == 1:
                tpr[i] = tpr[i - 1] + 1
                fpr[i] = fpr[i - 1]
            else:
                tpr[i] = tpr[i - 1]
                fpr[i] = fpr[i - 1] + 1

        tpr /= (self.targets.sum() * 1.0)
        fpr /= ((self.targets - 1.0).sum() * -1.0)

        for n in range(1, scores.size + 1):
            if fpr[n] >= max_fpr:
                break

        # calculating area under curve using trapezoidal rule
        #n = tpr.shape[0]
        h = fpr[1:n] - fpr[0:n - 1]
        sum_h = np.zeros(fpr.shape)
        sum_h[0:n - 1] = h
        sum_h[1:n] += h
        area = (sum_h * tpr).sum() / 2.0

        return area / max_fpr

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
# @timeit
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
        self.cut_off = cut_off
        self.unk = unk
        self.padding = padding
        self.start = start
        self.end = end
        self.vocab[self.unk] = self.cut_off + 1
        self.vocab[self.padding] = self.cut_off + 1
        self.vocab[self.start] = self.cut_off + 1
        self.vocab[self.end] = self.cut_off + 1
        self.vocab_map = {}
        for word in self.vocab:
            if self.vocab[word] <= self.cut_off: word = self.unk
            if word not in self.vocab_map:
                self.vocab_map[word] = len(self.vocab_map)

        self.mapped_corpus = {}
        for id in self.raw_corpus:
            mapped_title = [self.vocab_map.get(word, self.vocab_map[self.unk]) for word in self.raw_corpus[id][0]]
            mapped_body = [self.vocab_map.get(word, self.vocab_map[self.unk]) for word in self.raw_corpus[id][1]]
            self.mapped_corpus[id] = (mapped_title, mapped_body)
        self.num_tokens = len(self.vocab_map)

    def read_new_corpus(self, data_file):
        self.second_raw_corpus = {}
        with open(data_file) as data:
            for line in data:
                id, title, body = line.split("\t")
                if len(title) == 0:
                    continue
                title = title.strip().split()
                body = body.strip().split()
                self.second_raw_corpus[id] = (title, body)

        vocab_2 = Counter(w for id, pair in self.second_raw_corpus.iteritems() \
                             for x in pair for w in x)

        self.vocab += vocab_2

        for word in vocab_2:
            if self.vocab[word] <= self.cut_off: word = self.unk
            if word not in self.vocab_map:
                self.vocab_map[word] = len(self.vocab_map)

        self.mapped_corpus_2 = {}
        for id in self.second_raw_corpus:
            mapped_title = [self.vocab_map.get(word, self.vocab_map[self.unk]) for word in self.second_raw_corpus[id][0]]
            mapped_body = [self.vocab_map.get(word, self.vocab_map[self.unk]) for word in self.second_raw_corpus[id][1]]
            self.mapped_corpus_2[id] = (mapped_title, mapped_body)
        self.num_tokens = len(self.vocab_map)

    @staticmethod
    def read_annotations(path, K_neg=10, prune_pos_cnt=3):
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


    def create_advisarial_data(self, N):
        ids = self.mapped_corpus_2.keys()
        ids = np.random.choice(ids, N, replace=False)
        titles = []
        bodies = []
        for id in ids:
            t, b = self.mapped_corpus_2[id]
            titles.append(t)
            bodies.append(b)
        titles, bodies = self.create_one_batch(titles, bodies)
        return titles, bodies



    def create_eval_batches(self, data, first_corpus=True):
        lst = [ ]
        for pid, qids, qlabels in data:
            titles = [ ]
            bodies = [ ]
            for id in [pid]+qids:
                if first_corpus:
                    t, b = self.mapped_corpus[id]
                else:
                    t, b = self.mapped_corpus_2[id]
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

def read_annotations_2(pos_path, neg_path, K_neg=10, prune_pos_cnt=3):
    q_2_pos = defaultdict(list)
    q_2_neg = defaultdict(list)
    with open(pos_path) as fin:
        for line in fin:
            parts = line.strip("\n").split(" ")
            if len(parts) != 2:
                continue
            q_2_pos[parts[0]] = q_2_pos[parts[0]] + [parts[1]]

    with open(neg_path) as fin:
        for line in fin:
            parts = line.strip("\n").split(" ")
            if len(parts) != 2:
                continue
            q_2_neg[parts[0]] = q_2_neg[parts[0]] + [parts[1]]

    lst = []
    for id in q_2_pos:
        neg = q_2_neg[id]
        if K_neg != -1:
            random.shuffle(neg)
            neg = neg[:K_neg]
        pos = q_2_pos[id]
        if len(pos) > prune_pos_cnt and prune_pos_cnt != -1: continue

        s = set()
        qids = [ ]
        qlabels = [ ]
        for q in pos:
            if q not in s:
                qids.append(q)
                qlabels.append(0 if q not in pos else 1)
                s.add(q)
        for q in neg:
            if q not in s:
                qids.append(q)
                qlabels.append(0 if q not in pos else 1)
                s.add(q)
        lst.append((id, qids, np.array(qlabels, dtype="int32")))
    return lst


class Encoder(nn.Module):
    def __init__(self, padding_id, data_loader, emb_path, cuda, embedding_size = 200):
        super(Encoder, self).__init__()
        self.data_loader = data_loader
        self.padding_id = padding_id
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(len(data_loader.vocab_map), self.embedding_size)
        self.cuda_is = cuda
        self.embedding.weight = nn.Parameter(torch.from_numpy(self.load_embedding_iterator(emb_path, self.data_loader)).float())
        self.embedding.weight.requires_grad = False


    def load_embedding_iterator(self, path, data_loader):
        embs = np.zeros((len(data_loader.vocab_map), self.embedding_size))
        with open(path) as fin:
            for line in fin:
                line = line.strip()
                if line:
                    parts = line.split(' ')
                    word = parts[0]
                    vals = np.array([ float(x) for x in parts[1:] ])
                    if word in data_loader.vocab_map:
                        embs[data_loader.vocab_map[word]] = vals
        return embs

    def forward(self, input):
        if self.cuda_is:
            return self.embedding(Variable(torch.from_numpy(input)).cuda())
        else:
            self.embedding(Variable(torch.from_numpy(input)))

@timeit
class pre_trained_Encoder():
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


def CNN_forward(idts, idbs, encoder, CNN, cuda=False):
    xt = encoder(idts)
    xb = encoder(idbs)
    xt = xt.permute(1, 2, 0)
    xb = xb.permute(1, 2, 0)
    ot = CNN(xt)
    ob = CNN(xb)
    ot = average_without_padding(ot, idts, encoder.padding_id, cuda)
    ob = average_without_padding(ob, idbs, encoder.padding_id, cuda)
    out = (ot+ob)*0.5
    return out


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, depth=1):
        super(LSTM, self).__init__()
        assert output_size % 2 == 0
        self.output_size = output_size
        self.input_size = input_size

        # The LSTM takes word encodings as inputs, and outputs hidden states with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.input_size, self.output_size//2, depth, bidirectional=True)


    def forward(self, input):
        out = self.lstm(input)[0]
        return out

def LSTM_forward(idts, idbs, encoder, model, cuda=False):
    xt = encoder(idts)
    xb = encoder(idbs)
    ot = model(xt)
    ob = model(xb)
    ot = ot.permute(1,2,0)
    ob = ob.permute(1,2,0)
    ot = average_without_padding(ot, idts, encoder.padding_id, cuda)
    ob = average_without_padding(ob, idbs, encoder.padding_id, cuda)
    out = (ot+ob)*0.5
    return out

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Classifier, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, 1)
        # self.softmax = nn.Sig()

    def forward(self, x):
        hidden = self.i2h(x)
        hidden = F.tanh(hidden)
        output = self.h2o(hidden)
        # output = F.sigmoid(output)
        return output


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


def normalize_2d(x, eps=1e-8):
    l2 = torch.norm(x, p=2, dim=1,  keepdim=True)
    return x/(l2+eps)


def average_without_padding(x, ids, padding_id, cuda=False, eps=1e-8):
    if cuda:
        mask = Variable(torch.from_numpy(np.not_equal(ids, padding_id).astype(int)[:,:,np.newaxis])).float().cuda().permute(1, 2, 0).expand_as(x)
    else:
        mask = Variable(torch.from_numpy(np.not_equal(ids, padding_id).astype(int)[:,:,np.newaxis])).float().permute(1, 2, 0).expand_as(x)
    s = torch.sum(x*mask, dim=2) / (torch.sum(mask, dim=2)+eps)
    return s


def get_loss(out, idps, model, cs, cuda):
    if cuda:
        out = out[torch.LongTensor(idps.ravel().astype(int)).cuda()]
    else:
        out = out[torch.LongTensor(idps.ravel().astype(int))]
    out = out.view((idps.shape[0], idps.shape[1], model.output_size))
    scores = cs(out[:, 0, :].unsqueeze(1).expand_as(out[:, 1:, :],), out[:, 1:, :]).view(-1, len(out[0])-1)
    pos_scores = scores[:, 0]
    neg_scores = torch.max(scores[:, 1:], dim=1)[0]
    diff = neg_scores - pos_scores + 1.0
    loss = torch.mean(diff)
    return loss


def score(idts, idbs, forward, encoder, model, cs, cuda):
    out = forward(idts, idbs, encoder, model, cuda)
    scores = cs(out[0].unsqueeze(0).expand_as(out[1:],), out[1:])
    scores = scores.cpu().data.numpy()
    return scores


def evaluate(data, score_func, encoder, model, cuda, forward):
    cs = nn.CosineSimilarity(dim=1)
    res = [ ]
    for idts, idbs, labels in data:
        scores = score_func(idts, idbs, forward, encoder, model, cs, cuda)
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

def evaluate_AUC(data, score_func, encoder, model, cuda, forward):
    cs = nn.CosineSimilarity(dim=1)
    AUC = AUCMeter()
    AUC.reset()
    for idts, idbs, labels in data:
        scores = score_func(idts, idbs, forward, encoder, model, cs, cuda)
        assert len(scores) == len(labels)
        AUC.add(scores, labels)
    return AUC.value(0.05)


def evaluate_BM25_AUC(data, model):
    AUC = AUCMeter()
    AUC.reset()
    for question, possibilities, labels in data:
        labels = np.array(labels)
        scores = model.BM25Score(question, possibilities)
        assert len(scores) == len(labels)
        AUC.add(scores, labels)
    return AUC.value(0.05)

def evaluate_TFIDF_AUC(data, model):
    AUC = AUCMeter()
    AUC.reset()
    for question, possibilities, labels in data:
        labels = np.array(labels)
        scores = model.TFIDFScore(question, possibilities)
        assert len(scores) == len(labels)
        AUC.add(scores, labels)
    return AUC.value(0.05)

def train(encoder, model, num_epoch, data_loader, train_data, dev_data, test_data, batch_size, forward, pre_trained_encoder=True, cuda=False, LR=0.001):
    train_losses = []
    dev_metrics = []
    test_metrics = []

    cs = torch.nn.CosineSimilarity(dim=2)
    print("doing evaluations")
    # Say metrics as we start
    dev_metrics.append(evaluate(dev_data, score, encoder, model, cuda, forward))
    test_metrics.append(evaluate(test_data, score, encoder, model, cuda, forward))
    print "At the start of epoch"
    print "The DEV MAP is {}, MRR is {}, P1 is {}, P5 is {}".format(dev_metrics[-1][0], dev_metrics[-1][1], dev_metrics[-1][2], dev_metrics[-1][3])
    print "The TEST MAP is {}, MRR is {}, P1 is {}, P5 is {}".format(test_metrics[-1][0], test_metrics[-1][1], test_metrics[-1][2], test_metrics[-1][3])
    if not(pre_trained_encoder):
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), LR, weight_decay=0.0)

    model_optimizer = torch.optim.Adam(model.parameters(), LR, weight_decay=0.0)
    for epoch in xrange(num_epoch):
        print "Training epoch {}".format(epoch)
        train_batches = data_loader.create_batches(train_data, batch_size)
        N = len(train_batches)
        train_loss = 0.0
        t = trange(N, desc='batch_loss: ??')
        for i in t:
            #  reset gradients
            model_optimizer.zero_grad()
            if not(pre_trained_encoder):
                encoder_optimizer.zero_grad()

            #  get train batch and find current loss
            idts, idbs, idps = train_batches[i]

            try:
                out = forward(idts, idbs, encoder, model, cuda)
                loss = get_loss(out, idps, model, cs, cuda)

                l = loss.cpu().data[0]
                if l < 0 or l > 2:
                    pass

                #  back propegate and optimize
                loss.backward()
                if not(pre_trained_encoder):
                    encoder_optimizer.step()

                model_optimizer.step()

                #  update tqdm description
                t.set_description("batch_loss: {}".format(loss.cpu().data[0]))
                train_loss += loss.cpu().data[0]
            except:
                print idts, idbs, idps
                continue

        train_losses.append(train_loss)
        dev_metrics.append(evaluate(dev_data, score, encoder, model, cuda, forward))
        test_metrics.append(evaluate(test_data, score, encoder, model, cuda, forward))
        print "At end of epoch {}:".format(epoch)
        print "The train loss is {}".format(train_loss)
        print "The DEV MAP is {}, MRR is {}, P1 is {}, P5 is {}".format(dev_metrics[-1][0], dev_metrics[-1][1], dev_metrics[-1][2], dev_metrics[-1][3])
        print "The TEST MAP is {}, MRR is {}, P1 is {}, P5 is {}".format(test_metrics[-1][0], test_metrics[-1][1], test_metrics[-1][2], test_metrics[-1][3])
    return train_losses, dev_metrics, test_metrics

def train_cross(encoder, model, num_epoch, data_loader, train_data, dev_data, test_data, batch_size, forward, pre_trained_encoder=True, cuda=False, LR=0.001):
    train_losses = []
    dev_metrics = []
    test_metrics = []

    cs = torch.nn.CosineSimilarity(dim=2)
    print("doing evaluations (This takes a while :()")
    dev_metrics.append(evaluate_AUC(dev_data, score, encoder, model, cuda, forward))
    test_metrics.append(evaluate_AUC(test_data, score, encoder, model, cuda, forward))
    print("dev AUC(0.05) score : {}".format(dev_metrics[-1]))
    print("test AUC(0.05) score : {}".format(test_metrics[-1]))
    model_optimizer = torch.optim.Adam(model.parameters(), LR, weight_decay=0.0)
    for epoch in xrange(num_epoch):
        print "Training epoch {}".format(epoch)
        train_batches = data_loader.create_batches(train_data, batch_size)
        N = len(train_batches)
        train_loss = 0.0
        t = trange(N, desc='batch_loss: ??')
        for i in t:
            #  reset gradients
            model_optimizer.zero_grad()

            #  get train batch and find current loss
            idts, idbs, idps = train_batches[i]

            try:
                out = forward(idts, idbs, encoder, model, cuda)
                loss = get_loss(out, idps, model, cs, cuda)

                #  back propegate and optimize
                loss.backward()

                model_optimizer.step()

                #  update tqdm description
                t.set_description("batch_loss: {}".format(loss.cpu().data[0]))
                train_loss += loss.cpu().data[0]
            except:
                print idts, idbs, idps
                continue

        train_losses.append(train_loss)
        dev_metrics.append(evaluate_AUC(dev_data, score, encoder, model, cuda, forward))
        test_metrics.append(evaluate_AUC(test_data, score, encoder, model, cuda, forward))
        print("At end of epoch {}:".format(epoch))
        print("The train loss is {}".format(train_loss))
        print("dev AUC(0.05) score : {}".format(dev_metrics[-1]))
        print("test AUC(0.05) score : {}".format(test_metrics[-1]))
    return train_losses, dev_metrics, test_metrics


def advisarial_trainer(encoder, model, classifier, num_epoch, data_loader, train_data, dev_data, test_data, batch_size, forward, cuda, LR=0.0001, L=0.01):
    train_losses = []
    train_classic_losses = []
    train_advisarial_losses = []
    dev_metrics = []
    test_metrics = []
    cs = torch.nn.CosineSimilarity(dim=2)
    print("doing evaluations (This takes a while :()")
    dev_metrics.append(evaluate_AUC(dev_data, score, encoder, model, cuda, forward))
    test_metrics.append(evaluate_AUC(test_data, score, encoder, model, cuda, forward))
    print("dev AUC(0.05) score : {}".format(dev_metrics[-1]))
    print("test AUC(0.05) score : {}".format(test_metrics[-1]))

    criterion = nn.BCEWithLogitsLoss()
    print "L =", L
    adaptive = False
    if L == False:
        adaptive = True

    model_optimizer = torch.optim.Adam(model.parameters(), LR, weight_decay=0.0)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), -10 * LR, weight_decay=0.0)
    for epoch in xrange(num_epoch):
        print "Training epoch {}".format(epoch)
        train_batches = data_loader.create_batches(train_data, batch_size)
        N = len(train_batches)
        train_loss = 0.0
        classic_loss = 0.0
        advisarial_loss = 0.0
        t = trange(N, desc='batch_loss: ??')
        for i in t:
            try:
                model_optimizer.zero_grad()
                classifier_optimizer.zero_grad()


                idts, idbs, idps = train_batches[i]
                out = forward(idts, idbs, encoder, model, cuda)

                loss_1 = get_loss(out, idps, model, cs, cuda)

                M = len(idts[0])

                advisarial_idts, advisarial_idbs = data_loader.create_advisarial_data(M)
                advisarial_out = forward(advisarial_idts, advisarial_idbs, encoder, model, cuda)

                samples = np.random.choice(range(2*M), M, replace=False)

                if cuda:
                    Xs = torch.cat((out, advisarial_out))[torch.LongTensor(samples).cuda()]
                    Ys = torch.autograd.Variable(torch.from_numpy(np.array([j < M for j in samples], dtype=int)).float()).cuda()
                else:
                    Xs = torch.cat((out, advisarial_out))[torch.LongTensor(samples)]
                    Ys = torch.autograd.Variable(torch.from_numpy(np.array([j < M for j in samples], dtype=int)).float())

                out = classifier(Xs).view(-1)
                loss_2 = criterion(out, Ys)
                if adaptive:
                    L = float((2/(1+np.exp(-0.5*(((float(i)/N)+epoch)/num_epoch)))) - 1)
                loss = loss_1 - L * loss_2
                loss.backward()

                model_optimizer.step()
                classifier_optimizer.step()
            except:
                print ":("
            t.set_description("batch_loss: {}, qa_loss: {}, advisarial_loss: {}, L: {}".format(loss.cpu().data[0], loss_1.cpu().data[0], loss_2.cpu().data[0], L))
            train_loss += loss.cpu().data[0]
            advisarial_loss += loss_2.cpu().data[0]
            classic_loss += loss_1.cpu().data[0]

        train_losses.append(train_loss)
        train_classic_losses.append(classic_loss)
        train_advisarial_losses.append(classic_loss)
        dev_metrics.append(evaluate_AUC(dev_data, score, encoder, model, cuda, forward))
        test_metrics.append(evaluate_AUC(test_data, score, encoder, model, cuda, forward))

        print("At end of epoch {}:".format(epoch))
        print("The train loss is {}, classic loss is {}, advisarial loss is {}".format(train_loss, classic_loss, advisarial_loss))
        print("dev AUC(0.05) score : {}".format(dev_metrics[-1]))
        print("test AUC(0.05) score : {}".format(test_metrics[-1]))
    return (train_losses, classic_loss, advisarial_loss), dev_metrics, test_metrics


class BM25_TDIDF:
    def __init__(self, data_file, delimiter='\t'):
        self.doc_term_counter = defaultdict(lambda: 0)
        self.documents = dict()
        self.document_lengths = dict()
        self.total_lengths = 0
        self.N = 0
        with open(data_file) as data:
            for line in data:
                id, title, body = line.split(delimiter)
                if len(title) == 0:
                    continue
                title = title.strip().split()
                body = body.strip().split()
                text = title + body
                self.documents[id] = Counter(text)
                for term in self.documents[id]:
                    self.doc_term_counter[term] += 1
                self.document_lengths[id] = len(text)
                self.total_lengths += len(text)
                self.N += 1
        self.avg_len = self.total_lengths/float(self.N)
        self.BMIDF = dict()
        self.TFIDF = dict()
        for term in self.doc_term_counter:
            self.BMIDF[term] = math.log((self.N - self.doc_term_counter[term] + 0.5) / (self.doc_term_counter[term] + 0.5))
            self.TFIDF[term] = math.log((self.N)/(self.doc_term_counter[term]+1))+1


    def BM25Score(self, q1, q2s, k1=1.5, b=0.75):
        scores = []
        for q2 in q2s:
            doc = self.documents[q2]
            commonTerms = set(self.documents[q1]) & set(doc)
            tmp_score = []
            doc_terms_len = self.document_lengths[q2]
            for term in commonTerms:
                upper = (doc[term] * (k1+1))
                below = ((doc[term]) + k1*(1 - b + b*doc_terms_len/self.avg_len))
                tmp_score.append(self.BMIDF[term] * upper / below)
            scores.append(sum(tmp_score))
        return np.array(scores)

    def TFIDFScore(self, q1, q2s):
        scores = []
        for q2 in q2s:
            doc = self.documents[q2]
            commonTerms = set(self.documents[q1]) & set(doc)
            tmp_score = []
            doc_terms_len = self.document_lengths[q2]
            for term in commonTerms:
                tmp_score.append( self.TFIDF[term] * math.sqrt(doc[term]) * 1.0/math.sqrt(doc_terms_len))
            scores.append(sum(tmp_score))
        return np.array(scores)
