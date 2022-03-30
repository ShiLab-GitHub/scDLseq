import numpy as np
import torch
from torch import nn, optim
#import torch.nn.functional as F
from torchtext.legacy import data
import torchtext
#import math
import time
import tqdm
#from lsoftmax import LSoftmaxLinear
from models import BiLSTM_Attention_L, BiLSTM_Attention_L_vis
import os
import argparse

def get_text_and_label(line):
        line=line.split(" ",1)
        # obtain content and labels
        
        # print(text)
        label = line[0]
        text = line[1]
        return text, label
    
def get_dataset(corpus_path,text_field, label_field, test=False):
    fields = [('text', text_field), ('label', label_field)]
    examples = []

    if test:
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in tqdm.tqdm(f):
                text,label = get_text_and_label(line)
                examples.append(data.Example.fromlist([ text, None], fields))
    else:
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in tqdm.tqdm(f):
                text,label = get_text_and_label(line)
                examples.append(data.Example.fromlist([text, label], fields))
    return examples, fields

def mean(item: list) -> float:
    res = sum(item) / len(item) if len(item) > 0 else 0
    return res


def accuracy(pred_y, true_y):
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]
    corr = 0
    for i in range(len(pred_y)):
        if pred_y[i] == true_y[i]:
            corr += 1
    acc = corr / len(pred_y) if len(pred_y) > 0 else 0
    return acc

def binary_precision(pred_y, true_y, positive=1):
    corr = 0
    pred_corr = 0
    for i in range(len(pred_y)):
        if pred_y[i] == positive:
            pred_corr += 1
            if pred_y[i] == true_y[i]:
                corr += 1

    prec = corr / pred_corr if pred_corr > 0 else 0
    return prec


def binary_recall(pred_y, true_y, positive=1):
    
    corr = 0
    true_corr = 0
    for i in range(len(pred_y)):
        if true_y[i] == positive:
            true_corr += 1
            if pred_y[i] == true_y[i]:
                corr += 1

    rec = corr / true_corr if true_corr > 0 else 0
    return rec


def binary_f_beta(pred_y, true_y, beta=1.0, positive=1):
    precision = binary_precision(pred_y, true_y, positive)
    recall = binary_recall(pred_y, true_y, positive)
    try:
        f_b = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
    except:
        f_b = 0
    return f_b


def multi_precision(pred_y, true_y, labels):
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    precisions = [binary_precision(pred_y, true_y, label) for label in labels]
    prec = mean(precisions)
    return prec


def multi_recall(pred_y, true_y, labels):
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    recalls = [binary_recall(pred_y, true_y, label) for label in labels]
    rec = mean(recalls)
    return rec


def multi_f_beta(pred_y, true_y, labels, beta=1.0):
    
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    f_betas = [binary_f_beta(pred_y, true_y, beta, label) for label in labels]
    f_beta = mean(f_betas)
    return f_beta


def get_binary_metrics(pred_y, true_y, f_beta=1.0):
    """
    get binary classification performance
    :param pred_y:
    :param true_y:
    :param f_beta:
    :return:
    """
    acc = accuracy(pred_y, true_y)
    recall = binary_recall(pred_y, true_y)
    precision = binary_precision(pred_y, true_y)
    f_beta = binary_f_beta(pred_y, true_y, f_beta)
    return acc, recall, precision, f_beta


def get_multi_metrics(pred_y, true_y, labels, f_beta=1.0):
    """
    get multi-class annotaion performance
    :param pred_y:
    :param true_y:
    :param labels:
    :param f_beta:
    :return:
    """
    acc = accuracy(pred_y, true_y)
    recall = multi_recall(pred_y, true_y, labels)
    precision = multi_precision(pred_y, true_y, labels)
    f_beta = multi_f_beta(pred_y, true_y, labels, f_beta)
    
    return acc, recall, precision, f_beta



#training function
def train(rnn, iterator, optimizer, criteon):

    avg_loss = []
    avg_acc = []
    rnn.train()        #enter training mode

    for i, batch in enumerate(iterator):
        pred, _ = rnn(batch.text, batch.label)
        pred = pred.squeeze()
        loss = criteon(pred, batch.label)
        
        predictions = torch.argmax(pred, 1)
        acc = accuracy(predictions, true_y=batch.label)
        #acc = accuracy(prediction, batch.label).item()   #calculate accuracy of every batch
        #acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batch.label, labels = idx_list)
        
        avg_loss.append(loss.item())
        #avg_acc.append(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #avg_acc = np.array(avg_acc).mean()
    avg_loss = np.array(avg_loss).mean()
    return avg_loss, acc



def adjust_learning_rate(optimizer, epoch, args):
    # Decreasing the learning rate to the factor of 0.1 at epochs 51 and 100
    # with a batch size of 256 this would comply with changing the lr at iterations 12k and 15k
    if 25 < epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#validation function
def evaluate(rnn, iterator, criteon):

    avg_loss = []
    avg_acc = []
    avg_recall = []
    avg_prec = []
    avg_f = []
    rnn.eval()         #enter validation mode

    with torch.no_grad():
        for batch in iterator:
            pred, _ = rnn(batch.text)        #[batch, 1] -> [batch]
            
            pred = pred.squeeze()
            loss = criteon(pred, batch.label)               
            predictions = torch.argmax(pred, 1)
            acc = accuracy(predictions, true_y=batch.label)
            avg_loss.append(loss.item())
            
            
    avg_loss = np.array(avg_loss).mean()
    
    return avg_loss, acc
def main():
    parser = argparse.ArgumentParser(description='PyTorch BLSTM Cell Annotation Example')
    parser.add_argument('--traindir', type=str, metavar='T',
                        help='train data direction')
    parser.add_argument('--devdir', type=str, metavar='D',
                        help='validation data direction')
    parser.add_argument('--modeldir', type=str, metavar='M',
                        help='model direction')
    parser.add_argument('--name', type=str, metavar='N',
                        help='subject name')
    parser.add_argument('--use-gpu', default=True, metavar='G',
                        help='determine wether to use gpu or not')
    parser.add_argument('--device', type=str, default='cuda:0', metavar='D',
                        help='cuda device id')
    parser.add_argument('--margin', type=int, default=1, metavar='M',
                        help='the margin for the l-softmax formula (m=1, 2, 3, 4)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='the batch size for training process')
    parser.add_argument('--vocab-size', type=int, default=5000, metavar='N',
                        help='number of marker genes in dataset')
    parser.add_argument('--embed-size', type=int, default=400, metavar='N',
                        help='size of embedding')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='initial learning rate (default: 0.1)')
    parser.add_argument('--vis', default=False, metavar='V',
                        help='enables visualizing 2d features (default: False).')
    parser.add_argument('--load-model', default=False, metavar='L',
                        help='load existed model (default: False).')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.use_gpu else "cpu")
    tokenize = lambda x: x.split()
    
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True)
    LABEL = data.Field(sequential=False, use_vocab=True)
    
    train_examples, train_fields = get_dataset(args.traindir, TEXT, LABEL)
    dev_examples, dev_fields = get_dataset(args.devdir, TEXT, LABEL)
    
    train_data = data.Dataset(train_examples, train_fields)
    dev_data = data.Dataset(dev_examples, dev_fields)
    
    TEXT.build_vocab(train_data, max_size=args.vocab_size)
    LABEL.build_vocab(train_data)
    n_class = len(LABEL.vocab.stoi)
    if args.use_gpu:
        train_iterator, dev_iterator = data.BucketIterator.splits(
        (train_data, dev_data),
        device = device,
        batch_size=args.batch_size,
        sort = False)
    else:
        train_iterator, dev_iterator = data.BucketIterator.splits(
        (train_data, dev_data),
        device = -1,
        batch_size=args.batch_size,
        sort = False)



    if args.vis:
        rnn = BiLSTM_Attention_L_vis(len(TEXT.vocab), args.embed_size, output_size=n_class, hidden_dim=32, n_layers=1, margin=args.margin, device=device)
    else:
        rnn = BiLSTM_Attention_L(len(TEXT.vocab), args.embed_size, output_size=n_class, hidden_dim=64, n_layers=2, margin=args.margin, device=device)

    optimizer = optim.Adam(rnn.parameters(), lr=args.lr)
    criteon = nn.CrossEntropyLoss()
    rnn = rnn.cuda()
    criteon = criteon.cuda()


    #train model and show its performance
    best_valid_acc = float('-inf')
    if args.load_model:
        rnn.load_state_dict(torch.load(args.modeldir+args.name+".pt"))
    for epoch in range(150):

        start_time = time.time()
        adjust_learning_rate(optimizer, epoch, args)
        train_loss, train_acc = train(rnn, train_iterator, optimizer, criteon)
        dev_loss, dev_acc = evaluate(rnn, dev_iterator, criteon)

        end_time = time.time()

        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        #early-stopping mechanism
        if dev_acc > best_valid_acc:
            best_valid_acc = dev_acc
            es = 0
            torch.save(rnn.state_dict(), args.modeldir+args.name+".pt")
        else:
            es += 1
            print("Performance didn't improve, start early-stopping {} of 5".format(es))

            if es > 4:
                print(f'\tEarly stopping with best_acc: {best_valid_acc*100:.2f} | and val_acc for this epoch: {dev_acc*100:.2f}%')
                break

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs:.2f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {dev_loss:.3f} |  Val. Acc: {dev_acc*100:.2f}%')
    
if __name__ == '__main__':
    main()
