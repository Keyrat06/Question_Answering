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

    # encoder = util.Encoder(data_loader.num_tokens, embedding_size, data_loader.vocab_map[padding])
    # if cuda:
    #     encoder = encoder.cuda()
    pre_encoder = util.pre_trained_Encoder(data_loader.vocab_map[padding], data_loader, embedding_path, cuda)
    CNN = util.CNN(embedding_size, CNN_size, convolution_size)
    if cuda:
        CNN = CNN.cuda()

    dev  = data_loader.read_annotations(dev_file)
    dev_data  = data_loader.create_eval_batches(dev)
    test = data_loader.read_annotations(test_file)
    test_data = data_loader.create_eval_batches(test)
    train_data = data_loader.read_annotations(train_file)
    # train_losses, dev_metrics, test_metrics = train(encoder, CNN, num_epoch, data_loader, train_data, dev_data, test_data, batch_size, False, cuda)
    train_losses, dev_metrics, test_metrics = train(pre_encoder, CNN, num_epoch, data_loader, train_data, dev_data, test_data, batch_size, True, cuda)
    # torch.save(encoder, "encoder.model")
    torch.save(CNN, "CNN.model")
    return train_losses, dev_metrics, test_metrics


def train(encoder, CNN, num_epoch, data_loader, train_data, dev_data, test_data, batch_size, pre_trained_encoder=True, cuda=False):
    train_losses = []
    dev_metrics = []
    test_metrics = []

    cs = torch.nn.CosineSimilarity(dim=2)

    # Say metrics as we start
    dev_metrics.append(util.evaluate(dev_data, score, encoder, CNN, cuda))
    test_metrics.append(util.evaluate(test_data, score, encoder, CNN, cuda))
    print "At the start of epoch"
    print "The DEV MAP is {}, MRR is {}, P1 is {}, P5 is {}".format(dev_metrics[-1][0], dev_metrics[-1][1], dev_metrics[-1][2], dev_metrics[-1][3])
    print "The TEST MAP is {}, MRR is {}, P1 is {}, P5 is {}".format(test_metrics[-1][0], test_metrics[-1][1], test_metrics[-1][2], test_metrics[-1][3])
    if not(pre_trained_encoder):
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), 0.001, weight_decay=0.0)

    CNN_optimizer = torch.optim.Adam(CNN.parameters(), 0.001, weight_decay=0.0)
    for epoch in xrange(num_epoch):
        print "Training epoch {}".format(epoch)
        train_batches = data_loader.create_batches(train_data, batch_size)
        N = len(train_batches)
        train_loss = 0.0
        t = trange(N, desc='batch_loss: ??')
        for i in t:
            #  reset gradients
            CNN_optimizer.zero_grad()
            if not(pre_trained_encoder):
                encoder_optimizer.zero_grad()

            #  get train batch and find current loss
            idts, idbs, idps = train_batches[i]
            loss = get_loss(idts, idbs, idps, encoder, CNN, cs, cuda)

            #  back propegate and optimize
            loss.backward()
            if not(pre_trained_encoder):
                encoder_optimizer.step()

            CNN_optimizer.step()

            #  update tqdm description
            t.set_description("batch_loss: {}".format(loss.cpu().data[0]))
            train_loss += loss.cpu().data[0]

            if i > 10:
                break

        train_losses.append(train_loss)
        dev_metrics.append(util.evaluate(dev_data, score, encoder, CNN, cuda))
        test_metrics.append(util.evaluate(test_data, score, encoder, CNN, cuda))
        print "At end of epoch {}:".format(epoch)
        print "The train loss is {}".format(train_loss)
        print "The DEV MAP is {}, MRR is {}, P1 is {}, P5 is {}".format(dev_metrics[-1][0], dev_metrics[-1][1], dev_metrics[-1][2], dev_metrics[-1][3])
        print "The TEST MAP is {}, MRR is {}, P1 is {}, P5 is {}".format(test_metrics[-1][0], test_metrics[-1][1], test_metrics[-1][2], test_metrics[-1][3])
    return train_losses, dev_metrics, test_metrics


def forward(idts, idbs, encoder, CNN, cuda):
    xt = encoder(idts)
    xb = encoder(idbs)
    xt = xt.permute(1, 2, 0)
    xb = xb.permute(1, 2, 0)
    ot = CNN(xt)
    ob = CNN(xb)
    ot = average_without_padding(ot, idts, encoder.padding_id, cuda)
    ob = average_without_padding(ob, idbs, encoder.padding_id, cuda)
    out = (ot+ob)*0.5
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


def get_loss(idts, idbs, idps, encoder, CNN, cs, cuda):
    out = forward(idts, idbs, encoder, CNN, cuda)
    if cuda:
        out = out[torch.LongTensor(idps.ravel().astype(int)).cuda()]
    else:
        out = out[torch.LongTensor(idps.ravel().astype(int))]
    out = out.view((idps.shape[0], idps.shape[1], CNN.output_size))
    scores = cs(out[:, 0, :].unsqueeze(1).expand_as(out[:, 1:, :],), out[:, 1:, :])
    pos_scores = scores[:, 0]
    neg_scores = torch.max(scores[:, 1:], dim=1)[0]
    diff = neg_scores - pos_scores + 1.0
    loss = torch.mean(diff)
    return loss


def score(idts, idbs, encoder, CNN, cs, cuda):
    out = forward(idts, idbs, encoder, CNN, cuda)
    scores = cs(out[0].unsqueeze(0).expand_as(out[1:],), out[1:])
    return scores.cpu().data.numpy()


if __name__ == "__main__":
    train_losses, dev_metrics, test_metrics = main()
    #  should graph these later
    print train_losses
    print dev_metrics
    print test_metrics
