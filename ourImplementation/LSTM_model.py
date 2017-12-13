import torch
from util import util


def main():
    cuda = torch.cuda.is_available() and True
    num_epoch = 2
    batch_size = 2
    input_size = 200
    output_size = 100
    LR = 0.0005
    dev_file = "data/dev.txt"
    test_file = "data/test.txt"
    train_file = "data/train_random.txt"
    corpus_file = "data/text_tokenized.txt"
    padding = "<padding>"
    embedding_path = "data/vectors_pruned.200.txt"


    print "LSTM Embedding Size: ",output_size
    print "learning rate: ", LR
    print "Batch Size: ", batch_size
    print "num epoch: ", num_epoch



    # Represent each question as a word sequence (and not as a bog of words)
    data_loader = util.data_loader(corpus_file, cut_off=1, padding=padding)

    
    dev  = data_loader.read_annotations(dev_file, 20, 10)
    dev_data  = data_loader.create_eval_batches(dev)
    test = data_loader.read_annotations(test_file, 20, 10)
    test_data = data_loader.create_eval_batches(test)
    train_data = data_loader.read_annotations(train_file, 20, 3)
    
    # Utilize an exisiting vector representation of the words
    encoder = util.Encoder(data_loader.vocab_map[padding], data_loader, embedding_path, cuda)
    
    print "Embeddings done"
    
    model = util.LSTM(input_size, output_size)
    if cuda:
        model = model.cuda()
        encoder = encoder.cuda()


    train_losses, dev_metrics, test_metrics = util.train(encoder, model, num_epoch, data_loader, train_data, dev_data, test_data, batch_size, util.LSTM_forward, True, cuda, LR)
    model = model.cpu()
    torch.save(model, "LSTM.model")
    return train_losses, dev_metrics, test_metrics


if __name__ == "__main__":
    train_losses, dev_metrics, test_metrics = main()
    #  should graph these later
    print train_losses
    print dev_metrics
    print test_metrics    
