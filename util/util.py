import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from io import open
import numpy as np
import random
import time
from tqdm import trange, tqdm
from collections import Counter, defaultdict
import math



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


    def read_annotations(self, path, K_neg=10, prune_pos_cnt=3):
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
        lst.append((id, qids, qlabels))
    return lst


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

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, depth=1):
        super(LSTM, self).__init__()
        self.output_size = output_size//2
        self.input_size = input_size

        # The LSTM takes word encodings as inputs, and outputs hidden states with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.input_size, self.output_size, depth, bidirectional=True)


    def forward(self, input):
        out = self.lstm(input)[0]
        return F.tanh(out)


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
    return scores.cpu().data.numpy()


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

def evaluate_BM25(data, model):
    res = []
    for question, possibilities, labels in data:
        scores = model.BM25Score(question, possibilities)
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

def evaluate_TFIDF(data, model):
    res = []
    for question, possibilities, labels in data:
        scores = model.TFIDFScore(question, possibilities)
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


class BM25_TDIDF:
    def __init__(self, data_file, delimiter='\t'):
        self.doc_term_counter = defaultdict(lambda: 0)
        self.documents = dict()
        self.document_lengths = dict()
        self.total_lengths = 0
        self.N = 0
        with open(data_file) as data:
            for line in tqdm(data):
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
        return scores

    def TFIDFScore(self, q1, q2s):
        scores = []
        for q2 in q2s:
            doc = self.documents[q2]
            commonTerms = set(self.documents[q1]) & set(doc)
            tmp_score = []
            doc_terms_len = self.document_lengths[q2]
            for term in commonTerms:
                tmp_score.append(self.TFIDF[term] * math.sqrt(doc[term]) * 1.0/math.sqrt(doc_terms_len))
            scores.append(sum(tmp_score))
        return scores
