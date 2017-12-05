# 6.864 - Advanced NLP, Fall 2017
# Question Retrieval

# LSTM model

import torch
from tqdm import trange
# import sys
# sys.path.append("/home/sepiatone/at/6_864-qa")
from util import util

corpus_file = "data/text_tokenized.txt"
padding = "<padding>"
embedding_path = "data/vectors_pruned.200.txt"
input_size = 200
output_size = 240
dev_file = "data/dev.txt"
test_file = "data/test.txt"
train_file = "data/train_random.txt"
num_epoch = 10
batch_size = 5


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
    encoder = util.pre_trained_Encoder(data_loader.vocab_map[padding], data_loader, embedding_path, cuda)
    
    print "Embeddings done"
    
    model = util.LSTM(input_size, output_size)
    if cuda:
        model = model.cuda()


    train_losses, dev_metrics, test_metrics = train(encoder, model, num_epoch, data_loader, train_data, dev_data, test_data, batch_size, True, cuda)
    torch.save(model, "LSTM.model")
    
    return train_losses, dev_metrics, test_metrics


#%%

def train(encoder, model, num_epoch, data_loader, train_data, dev_data, test_data, batch_size, pre_trained_encoder=True, cuda=False):
    train_losses = []
    dev_metrics = []
    test_metrics = []

    cs = torch.nn.CosineSimilarity(dim=2)

    # Say metrics as we start
    dev_metrics.append(util.evaluate(dev_data, util.score, encoder, model, cuda, forward))
    test_metrics.append(util.evaluate(test_data, util.score, encoder, model, cuda, forward))
    print "At the start of epoch"
    print "The DEV MAP is {}, MRR is {}, P1 is {}, P5 is {}".format(dev_metrics[-1][0], dev_metrics[-1][1], dev_metrics[-1][2], dev_metrics[-1][3])
    print "The TEST MAP is {}, MRR is {}, P1 is {}, P5 is {}".format(test_metrics[-1][0], test_metrics[-1][1], test_metrics[-1][2], test_metrics[-1][3])
    if not(pre_trained_encoder):
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), 0.001, weight_decay=0.0)

    model_optimizer = torch.optim.Adam(model.parameters(), 0.001, weight_decay=0.0)
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

            out = forward(idts, idbs, encoder, model, cuda)
            loss = util.get_loss(out, idps, model, cs, cuda)

            #  back propegate and optimize
            loss.backward()
            if not(pre_trained_encoder):
                encoder_optimizer.step()

            model_optimizer.step()

            #  update tqdm description
            t.set_description("batch_loss: {}".format(loss.cpu().data[0]))
            train_loss += loss.cpu().data[0]


        train_losses.append(train_loss)
        dev_metrics.append(util.evaluate(dev_data, util.score, encoder, model, cuda, forward))
        test_metrics.append(util.evaluate(test_data, util.score, encoder, model, cuda, forward))
        print "At end of epoch {}:".format(epoch)
        print "The train loss is {}".format(train_loss)
        print "The DEV MAP is {}, MRR is {}, P1 is {}, P5 is {}".format(dev_metrics[-1][0], dev_metrics[-1][1], dev_metrics[-1][2], dev_metrics[-1][3])
        print "The TEST MAP is {}, MRR is {}, P1 is {}, P5 is {}".format(test_metrics[-1][0], test_metrics[-1][1], test_metrics[-1][2], test_metrics[-1][3])
    return train_losses, dev_metrics, test_metrics


def forward(idts, idbs, encoder, model, cuda=False):
    xt = encoder(idts)
    xb = encoder(idbs)
    ot = model(xt)
    ob = model(xb)
    ot = util.normalize_2d(ot)
    ob = util.normalize_2d(ob)
    out = (ot+ob)*0.5
    return out


if __name__ == "__main__":
    train_losses, dev_metrics, test_metrics = main()
    #  should graph these later
    print train_losses
    print dev_metrics
    print test_metrics    
