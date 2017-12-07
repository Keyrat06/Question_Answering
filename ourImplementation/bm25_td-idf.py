from util import util

corpus = "data/Android/corpus.txt"
pos_dev = "data/Android/dev.pos.txt"
neg_dev = "data/Android/dev.neg.txt"
pos_test = "data/Android/test.pos.txt"
neg_test = "data/Android/test.neg.txt"

dev_annotations = util.read_annotations_2(pos_dev, neg_dev, 20, 5)
test_annotations = util.read_annotations_2(pos_test, neg_test, 20, 5)

model = util.BM25_TDIDF(corpus)

print "The BM25 DEV MAP is {0[0]}, MRR is {0[1]}, P1 is {0[2]}, P5 is {0[3]}".format(util.evaluate_BM25(dev_annotations, model))
print "The TD-IDF DEV MAP is {0[0]}, MRR is {0[1]}, P1 is {0[2]}, P5 is {0[3]}".format(util.evaluate_TFIDF(dev_annotations, model))

print "The BM25 TEST MAP is {0[0]}, MRR is {0[1]}, P1 is {0[2]}, P5 is {0[3]}".format(util.evaluate_BM25(test_annotations, model))
print "The TD-IDF TEST MAP is {0[0]}, MRR is {0[1]}, P1 is {0[2]}, P5 is {0[3]}".format(util.evaluate_TFIDF(test_annotations, model))

# from util import util
#
# corpus = "data/text_tokenized.txt"
# dev = "data/dev.txt"
# test = "data/test.txt"
#
# dev_annotations = util.data_loader.read_annotations(dev, 20, 5)
# test_annotations = util.data_loader.read_annotations(test, 20, 5)
#
# model = util.BM25_TDIDF(corpus)
#
# print "The BM25 DEV MAP is {0[0]}, MRR is {0[1]}, P1 is {0[2]}, P5 is {0[3]}".format(util.evaluate_BM25(dev_annotations, model))
# print "The TD-IDF DEV MAP is {0[0]}, MRR is {0[1]}, P1 is {0[2]}, P5 is {0[3]}".format(util.evaluate_TFIDF(dev_annotations, model))
#
# print "The BM25 TEST MAP is {0[0]}, MRR is {0[1]}, P1 is {0[2]}, P5 is {0[3]}".format(util.evaluate_BM25(test_annotations, model))
# print "The TD-IDF TEST MAP is {0[0]}, MRR is {0[1]}, P1 is {0[2]}, P5 is {0[3]}".format(util.evaluate_TFIDF(test_annotations, model))
