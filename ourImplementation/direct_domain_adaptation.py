from util import util
import torch

def main():
    cuda = torch.cuda.is_available() and True
    num_epoch = 5
    batch_size = 3
    embedding_size = 300
    output_size = 200
    CNN_output_size = 600
    convolution_size = 3

    corpus_file = "data/text_tokenized.txt"
    padding = "<padding>"
    embedding_path = "data/Android/glove.txt"
    train_file = "data/train_random.txt"
    LSTM = True
    corpus_2 = "data/Android/corpus.txt"
    pos_dev = "data/Android/dev.pos.txt"
    neg_dev = "data/Android/dev.neg.txt"
    pos_test = "data/Android/test.pos.txt"
    neg_test = "data/Android/test.neg.txt"

    data_loader = util.data_loader(corpus_file, cut_off=0, padding=padding)
    data_loader.read_new_corpus(corpus_2)
    encoder = util.Encoder(data_loader.vocab_map[padding], data_loader, embedding_path, cuda, embedding_size)

    dev_annotations = util.read_annotations_2(pos_dev, neg_dev, -1, -1)
    test_annotations = util.read_annotations_2(pos_test, neg_test, -1, -1)
    dev_data = data_loader.create_eval_batches(dev_annotations, first_corpus=False)
    test_data = data_loader.create_eval_batches(test_annotations, first_corpus=False)

    train_data = data_loader.read_annotations(train_file, 20, 3)

    if LSTM:
        model = util.LSTM(embedding_size, output_size)
        forward = util.LSTM_forward
    else:
        model = util.CNN(embedding_size, CNN_output_size, convolution_size)
        forward = util.CNN_forwardd

    if cuda:
        model = model.cuda()
        encoder = encoder.cuda()

    return util.train_cross(encoder, model, num_epoch, data_loader, train_data, dev_data, test_data, batch_size, forward, pre_trained_encoder=True, cuda=cuda, LR=0.001)

if __name__ == "__main__":
    train_losses, dev_metrics, test_metrics = main()
    #  should graph these later
    print train_losses
    print dev_metrics
    print test_metrics
