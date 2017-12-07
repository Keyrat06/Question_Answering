from util import util
import torch
from torch import nn
from tqdm import trange
import random
import numpy as np

def main():
    LSTM = True
    cuda = torch.cuda.is_available() and True
    embedding_size = 200
    convolution_size = 3
    output_size = 240
    CNN_output_size = 640
    batch_size = 5
    num_epoch = 10
    classifier_hidden_size = 20




    padding = "<padding>"
    embedding_path = "data/vectors_pruned.200.txt"
    corpus_file = "data/text_tokenized.txt"
    corpus_2 = "data/Android/corpus.txt"
    train_file = "data/train_random.txt"
    pos_dev = "data/Android/dev.pos.txt"
    neg_dev = "data/Android/dev.neg.txt"
    pos_test = "data/Android/test.pos.txt"
    neg_test = "data/Android/test.neg.txt"

    data_loader = util.data_loader(corpus_file, cut_off=2, padding=padding)
    data_loader.read_new_corpus(corpus_2)


    dev_annotations = util.read_annotations_2(pos_dev, neg_dev, 20, 5)
    test_annotations = util.read_annotations_2(pos_test, neg_test, 20, 5)
    dev_data = data_loader.create_eval_batches(dev_annotations, first_corpus=False)
    test_data = data_loader.create_eval_batches(test_annotations, first_corpus=False)

    train_data = data_loader.read_annotations(train_file)

    if LSTM:
        model = util.LSTM(embedding_size, output_size)
        forward = util.LSTM_forward
        classifier = util.Classifier(output_size, classifier_hidden_size)
    else:
        model = util.CNN(embedding_size, CNN_output_size, convolution_size)
        forward = util.CNN_forward
        classifier = util.Classifier(CNN_output_size, classifier_hidden_size)


    if cuda:
        model = model.cuda()
        classifier = classifier.cuda()

    encoder = util.pre_trained_Encoder(data_loader.vocab_map[padding], data_loader, embedding_path, False)



    train_losses, dev_metrics, test_metrics = util.advisarial_trainer(encoder, model, classifier, num_epoch, data_loader, train_data, dev_data, test_data, batch_size, forward, cuda)

    return train_losses, dev_metrics, test_metrics



if __name__ == "__main__":
    train_losses, dev_metrics, test_metrics = main()
    #  should graph these later
    print train_losses
    print dev_metrics
    print test_metrics