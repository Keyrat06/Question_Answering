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



    dev_metrics, test_metrics = advisarial_trainer(encoder, model, classifier, num_epoch, data_loader, train_data, dev_data, test_data, batch_size, forward, cuda)

    return dev_metrics, test_metrics


def advisarial_trainer(encoder, model, classifier, num_epoch, data_loader, train_data, dev_data, test_data, batch_size, forward, cuda, LR=0.001):
    train_losses = []
    dev_metrics = []
    test_metrics = []
    L = 10**-3
    cs = torch.nn.CosineSimilarity(dim=2)
    # print("doing evaluations (This takes a while :()")
    # dev_metrics.append(util.evaluate_AUC(dev_data, util.score, encoder, model, False, forward))
    # test_metrics.append(util.evaluate_AUC(test_data, util.score, encoder, model, False, forward))
    # print("dev AUC(0.05) score : {}".format(dev_metrics[-1]))
    # print("test AUC(0.05) score : {}".format(test_metrics[-1]))

    criterion = nn.BCEWithLogitsLoss()

    model_optimizer = torch.optim.Adam(model.parameters(), LR, weight_decay=0.0)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), -100 * LR, weight_decay=0.0)
    for epoch in xrange(num_epoch):
        print "Training epoch {}".format(epoch)
        train_batches = data_loader.create_batches(train_data, batch_size)
        N = len(train_batches)
        train_loss = 0.0
        t = trange(N, desc='batch_loss: ??')
        for i in t:
            model_optimizer.zero_grad()
            classifier_optimizer.zero_grad()

            idts, idbs, idps = train_batches[i]
            out = forward(idts, idbs, encoder, model, cuda)

            loss_1 = util.get_loss(out, idps, model, cs, cuda)

            M = len(idts[0])

            advisarial_idts, advisarial_idbs = data_loader.create_advisarial_data(M)
            advisarial_out = forward(advisarial_idts, advisarial_idbs, encoder, model, cuda)

            samples = np.random.choice(range(2*M), M, replace=False)

            if cuda:
                Xs = torch.cat((out, advisarial_out))[torch.LongTensor(samples).cuda()]
                Ys = torch.autograd.Variable(torch.from_numpy(np.array([i < M for i in samples], dtype=int)).float()).cuda()
            else:
                Xs = torch.cat((out, advisarial_out))[torch.LongTensor(samples)]
                Ys = torch.autograd.Variable(torch.from_numpy(np.array([i < M for i in samples], dtype=int)).float())

            out = classifier(Xs).view(-1)
            print out
            loss_2 = criterion(out, Ys)

            loss = loss_1 - L * loss_2
            loss.backward()

            print loss_2
            print loss_1

            model_optimizer.step()
            classifier_optimizer.step()
            t.set_description("batch_loss: {}, qa_loss: {}, advisarial_loss: {}".format(loss.cpu().data[0], loss_1.cpu().data[0], loss_2.cpu().data[0]))
            train_loss += loss.cpu().data[0]

        train_losses.append(train_loss)
        dev_metrics.append(util.evaluate(dev_data, util.score, encoder, model, cuda, forward))
        test_metrics.append(util.evaluate(test_data, util.score, encoder, model, cuda, forward))
        print "At end of epoch {}:".format(epoch)
        print "The train loss is {}".format(train_loss)
        print "The DEV MAP is {}, MRR is {}, P1 is {}, P5 is {}".format(dev_metrics[-1][0], dev_metrics[-1][1], dev_metrics[-1][2], dev_metrics[-1][3])
        print "The TEST MAP is {}, MRR is {}, P1 is {}, P5 is {}".format(test_metrics[-1][0], test_metrics[-1][1], test_metrics[-1][2], test_metrics[-1][3])
    return train_losses, dev_metrics, test_metrics




if __name__ == "__main__":
    train_losses, dev_metrics, test_metrics = main()
    #  should graph these later
    print train_losses
    print dev_metrics
    print test_metrics