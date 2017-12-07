from util import util

corpus = "data/Android/corpus.txt"
pos_dev = "data/Android/dev.pos.txt"
neg_dev = "data/Android/dev.neg.txt"
pos_test = "data/Android/test.pos.txt"
neg_test = "data/Android/test.neg.txt"

dev_annotations = util.read_annotations_2(pos_dev, neg_dev, 20, 5)
test_annotations = util.read_annotations_2(pos_test, neg_test, 20, 5)


model = util.BM25_TDIDF(corpus)


print "The BM25 DEV AUC is {}".format(util.evaluate_BM25_AUC(dev_annotations, model))
print "The TD-IDF DEV AUC is {}".format(util.evaluate_TFIDF_AUC(dev_annotations, model))


print "The BM25 TEST AUC is {}".format(util.evaluate_BM25_AUC(test_annotations, model))
print "The TD-IDF TEST AUC is {}".format(util.evaluate_TFIDF_AUC(test_annotations, model))

