from util import util
import torch


corpus_file = "data/text_tokenized.txt"
padding = "<padding>"
embedding_path = "data/vectors_pruned.200.txt"
LSTM = True
corpus_2 = "data/Android/corpus.txt"
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



encoder = util.pre_trained_Encoder(data_loader.vocab_map[padding], data_loader, embedding_path, False)

if LSTM:
    model_path = "LSTM.model"
    model = torch.load(model_path)
    model = model.eval()
    print("direct domain transfer using LSTM AUC(0.05) score : {}".format(util.evaluate_AUC(dev_data, util.score, encoder, model, False, util.LSTM_forward)))
else:
    model_path = "CNN.model"
    model = torch.load(model_path)
    model = model.eval()
    print("direct domain transfer using CNN AUC(0.05) score : {}".format(util.evaluate_AUC(dev_data, util.score, encoder, model, False, util.LSTM_forward)))
