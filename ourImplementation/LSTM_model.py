# 6.864 - Advanced NLP, Fall 2017
# Question Retrieval

# LSTM model


#%%
import torch
from torch.autograd import Variable
from tqdm import trange
import numpy as np

import sys
sys.path.append("/home/sepiatone/at/6_864-qa")

#import os
#os.environ['PYTHONPATH'] = os.path.expanduser('~at/6_864-qa')
#from util import data_loader, evaluate
from util import util


import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word encodings as inputs, and outputs hidden states with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        #self.scores = nn.Linear(hidden_dim, output_size)
        #self.hidden = self.init_hidden()

    def init_hidden(self):
        # Initialize the hidden states
        return (Variable(torch.zeros(1, 1, self.hidden_dim)), Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, input):
        out = F.tanh(self.lstm(input))
        return out 

#%%

corpus_file = "data/text_tokenized.txt"
padding = "<padding>"
embedding_path = "data/vectors_pruned.200.txt"
input_size = 100 # Check
hidden_size = 200
output_size = 100
dev_file = "data/dev.txt"
test_file = "data/test.txt"
train_file = "data/train_random.txt"
num_epoch = 10
batch_size = 1



#%%
def main():
    # Check if a GPU is available
    cuda = torch.cuda.is_available() and True
    
    # Represent each question as a word sequence (and not as a bog of words)
    data_loader = util.data_loader(corpus_file, cut_off=2, padding=padding)
    
    dev  = data_loader.read_annotations(dev_file)
    dev_data  = data_loader.create_eval_batches(dev)
    test = data_loader.read_annotations(test_file)
    test_data = data_loader.create_eval_batches(test)
    train_data = data_loader.read_annotations(train_file)
    
    # Utilize an exisiting vector representation of the words
    pre_encoder = util.pre_trained_Encoder(data_loader.vocab_map[padding], data_loader, embedding_path, cuda)
    
    print "Embeddings done"
    
    model = LSTM(input_size, hidden_size)
    if cuda:
        model = model.cuda()

    train_losses, dev_metrics, test_metrics = train(pre_encoder, model, data_loader, num_epoch, train_data, dev_data, test_data, batch_size, True, cuda)

    torch.save(model, "LSTM.model")
    
    return train_losses, dev_metrics, test_metrics


#%%

def train(encoder, model, data_loader, num_epoch, train_data, dev_data, test_data, batch_size, pre_trained_encoder = True, cuda = False):
    train_losses = []
    dev_metrics = []
    test_metrics = []
    
    # Note - gather metrics before we start training
    dev_metrics.append(util.evaluate(dev_data, score, encoder, model, cuda))
    test_metrics.append(util.evaluate(test_data, score, encoder, model, cuda))
    print "At the start of epoch"
    print "The DEV MAP is {}, MRR is {}, P1 is {}, P5 is {}".format(dev_metrics[-1][0], dev_metrics[-1][1], dev_metrics[-1][2], dev_metrics[-1][3])
    print "The TEST MAP is {}, MRR is {}, P1 is {}, P5 is {}".format(test_metrics[-1][0], test_metrics[-1][1], test_metrics[-1][2], test_metrics[-1][3])
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), 0.001, weight_decay = 0.0)
    
    # Run for num_epoch
    for epoch in xrange(num_epoch):
        print "Training epoch {}".format(epoch)
        
        train_loss = 0.0
        
        print "Training epoch {}".format(epoch)
        train_batches = data_loader.create_batches(train_data, batch_size)
        N = len(train_batches)
        train_loss = 0.0
        t = trange(N, desc='batch_loss: ??')
        
        for i in t:
        
            # Clear the gradients
            model.zero_grad()
        
            # Clear the hidden state of the LSTM
            model.hidden = model.init_hidden()
            
            idts, idbs, idps = train_batches[i]
            
            out_ts = model(idts)
            out_bs = model(idbs)
            
            out = (out_ts + out_bs) * 0.5
            
            # Compute the loss, gradients, and update the parameters by calling optimizer.step()
            loss = get_loss(idts, idbs, idps, encoder, model, cuda) #get_loss(idts, idbs, idps, encoder, CNN, cuda)
    
            # Back propagate the losses and optimize
            loss.backward()
    
            optimizer.step()
    
            #  update tqdm description
            t.set_description("batch_loss: {}".format(loss.cpu().data[0]))
            train_loss += loss.cpu().data[0]
        
        # Gather statistics at the end of an epoch
        train_losses.append(train_loss)
        
        dev_metrics.append(util.evaluate(dev_data, score, encoder, CNN, cuda))
        test_metrics.append(util.evaluate(test_data, score, encoder, CNN, cuda))
        
        print "At end of epoch {}:".format(epoch)
        print "The training loss is {}".format(train_loss)
        print "The DEV MAP is {}, MRR is {}, P1 is {}, P5 is {}".format(dev_metrics[-1][0], dev_metrics[-1][1], dev_metrics[-1][2], dev_metrics[-1][3])
        print "The TEST MAP is {}, MRR is {}, P1 is {}, P5 is {}".format(test_metrics[-1][0], test_metrics[-1][1], test_metrics[-1][2], test_metrics[-1][3])
        

def get_loss(idts, idbs, idps, encoder, model, cuda):
    out = forward(idts, idbs, encoder, model, cuda)
    if cuda:
        out = out[torch.LongTensor(idps.ravel().astype(int)).cuda()]
    else:
        out = out[torch.LongTensor(idps.ravel().astype(int))]
    out = out.view((idps.shape[0], idps.shape[1], model.output_size))
    scores = torch.sum(out[:, 0, :].unsqueeze(1).expand_as(out[:, 1:, :],)*out[:, 1:, :], dim=2)
    pos_scores = scores[:, 0]
    neg_scores = torch.max(scores[:, 1:], dim=1)[0]
    diff = neg_scores - pos_scores + 1.0
    # skipping this since diff is alwayse > 0 since we normalize scores
    # if cuda:
    #     loss = torch.mean((diff>0).type(torch.cuda.FloatTensor)*diff)
    # else:
    #     loss = torch.mean((diff>0).type(torch.FloatTensor)*diff)
    loss = torch.mean(diff)
    
    return loss


def forward(idts, idbs, encoder, model, cuda):
    xt = encoder(idts)
    xb = encoder(idbs)
    xt = xt.permute(1, 2, 0)
    xb = xb.permute(1, 2, 0)
    ot = model(xt)
    ob = model(xb)
    ot = average_without_padding(ot, idts, encoder.padding_id, cuda)
    ob = average_without_padding(ob, idbs, encoder.padding_id, cuda)
    out = (ot+ob)*0.5
    out = normalize_2d(out)
    return out


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


def score(idts, idbs, encoder, model, cs, cuda):
    out = forward(idts, idbs, encoder, model, cuda)
    scores = torch.sum(out[0].unsqueeze(0).expand_as(out[1:],)*out[1:], dim=1)
    return scores.cpu().data.numpy()

#main()


#%%

if __name__ == "__main__":
    train_losses, dev_metrics, test_metrics = main()
    #  should graph these later
    print train_losses
    print dev_metrics
    print test_metrics    
