import torch
from torch.autograd import Variable
from util import util
from tqdm import trange
import numpy as np

def main():
    cuda = torch.cuda.is_available() and True
    embedding_size = 200
    convolution_size = 11
    CNN_size = 100
    batch_size = 5
    num_epoch = 10
    padding = "<padding>"
    train_file = "data/train_random.txt"
    dev_file = "data/dev.txt"
    test_file = "data/test.txt"
    corpus_file = "data/text_tokenized.txt"
    embedding_path = "data/vectors_pruned.200.txt"

    data_loader = util.data_loader(corpus_file, cut_off=2, padding=padding)

    encoder = util.pre_trained_Encoder(data_loader.vocab_map[padding], data_loader, embedding_path, cuda)
    print "loaded Encoder"
    CNN = util.CNN(embedding_size, CNN_size, convolution_size)
    if cuda:
        CNN = CNN.cuda()

    print "loading annotations"
    dev  = data_loader.read_annotations(dev_file, 20, 10)
    dev_data  = data_loader.create_eval_batches(dev)
    test = data_loader.read_annotations(test_file, 20, 10)
    test_data = data_loader.create_eval_batches(test)
    train_data = data_loader.read_annotations(train_file)
    print "annotations loaded"

    train_losses, dev_metrics, test_metrics = util.train(encoder, CNN, num_epoch, data_loader, train_data, dev_data, test_data, batch_size, forward, True, cuda)
    torch.save(CNN, "CNN.model")
    return train_losses, dev_metrics, test_metrics


def forward(idts, idbs, encoder, CNN, cuda=False):
    xt = encoder(idts)
    xb = encoder(idbs)
    xt = xt.permute(1, 2, 0)
    xb = xb.permute(1, 2, 0)
    ot = CNN(xt)
    ob = CNN(xb)
    ot = util.average_without_padding(ot, idts, encoder.padding_id, cuda)
    ob = util.average_without_padding(ob, idbs, encoder.padding_id, cuda)
    out = (ot+ob)*0.5
    return out



if __name__ == "__main__":
    train_losses, dev_metrics, test_metrics = main()
    #  should graph these later
    print train_losses
    print dev_metrics
    print test_metrics
