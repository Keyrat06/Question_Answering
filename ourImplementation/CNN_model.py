import torch
from torch.autograd import Variable
from util import util
from tqdm import tqdm

embedding_size = 50
convolution_size = 3
CNN_size = 40
batch_size = 15
num_epoch = 10
train_file = "data/train_random.txt"
corpus_file = "data/text_tokenized.txt"

data_loader = util.data_loader(corpus_file, cut_off=10)
encoder = util.Encoder(data_loader.num_tokens, embedding_size)
CNN = util.CNN(embedding_size, CNN_size, convolution_size)

train = data_loader.read_annotations(train_file)
losses = []
for epoch in xrange(num_epoch):
    print "Training batch {}".format(epoch)
    train_batches = data_loader.create_batches(train, batch_size)
    N = len(train_batches)
    train_loss = 0.0
    for i in tqdm(xrange(N)):
        idts, idbs, idps = train_batches[i]
        xt = encoder(Variable(torch.from_numpy(idts)))
        xb = encoder(Variable(torch.from_numpy(idbs)))
        xt = xt.permute(1, 2, 0)
        xb = xb.permute(1, 2, 0)
        ot = CNN(xt)
        ob = CNN(xb)
        out = (ot+ob)*0.5
        scores = torch.dot(out[1:], out[0])
        out = out[torch.LongTensor(idps.ravel().astype(int))]
        out = out.view((idps.shape[0], idps.shape[1], CNN_size))
        # num query * n_d
        query_vecs = out[:,0,:]
        # num query
        pos_scores = torch.sum(query_vecs*out[:,1,:], dim=1)
        # num query * candidate size
        neg_scores = torch.sum(query_vecs.unsqueeze(1).expand_as(out[:,2:,:],)*out[:,2:,:], dim=2)
        neg_scores = torch.max(neg_scores, dim=1)[0]

        diff = neg_scores - pos_scores + 1.0
        loss = torch.mean((diff>0).type(torch.FloatTensor)*diff)
        loss.backward()
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), 0.001, weight_decay=0.0000001)
        CNN_optimizer = torch.optim.Adam(CNN.parameters(), 0.001, weight_decay=0.0000001)
        encoder_optimizer.step()
        CNN_optimizer.step()
        print "batch_loss: {}".format(loss.cpu().data[0])
        train_loss += loss.cpu().data[0]
    print "At end of epoch {} the train loss is {}".format(epoch, train_loss)
    losses.append(train_loss)
