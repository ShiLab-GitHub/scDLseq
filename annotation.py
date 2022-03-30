import numpy as np
import torch
from torch import nn, optim
from torchtext.legacy import data
import torchtext
import time
import tqdm
from models import BiLSTM_Attention_L, BiLSTM_Attention_L_vis
import os
import argparse
import codecs

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


def test(rnn, iterator, criteon):

    #avg_loss = []
    predict = []
    attention = []
    rnn.eval()         #enter test mode

    with torch.no_grad():
        for batch in iterator:
            pred, attn = rnn(batch.text)        #[batch, 1] -> [batch]
            pred = pred.squeeze()
            loss = criteon(pred, batch.label)
            
            predictions = torch.argmax(pred, 1)
            #acc = accuracy(prediction, batch.label).item()
            #avg_loss.append(loss.item())
            for item in predictions:
                predict.append(item.item())
            for item in attn:
                attention.append(item.cpu().numpy())
    #avg_loss = np.array(avg_loss).mean()
    
    return predict,attention

def saveResult(target,path):
    with codecs.open(path,'a',encoding='utf-8') as f:
        for item in target:
            f.write(item)
            f.write(' ')
        f.close()

def main():
    parser = argparse.ArgumentParser(description='PyTorch BLSTM Cell Annotation Example')
    parser.add_argument('--refdir', type=str, metavar='R',
                        help='reference data direction')
    parser.add_argument('--testdir', type=str, metavar='T',
                        help='test data direction')
    parser.add_argument('--outdir', type=str, metavar='O',
                        help='annotation output direction')
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
    parser.add_argument('--load_model', default=False, metavar='L',
                        help='load existed model (default: False).')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.use_gpu else "cpu")
    tokenize = lambda x: x.split()
    
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True)
    LABEL = data.Field(sequential=False, use_vocab=True)
    
    test_examples, test_fields = get_dataset(args.testdir, TEXT, LABEL)
    ref_examples, ref_fields = get_dataset(args.refdir, TEXT, LABEL)
    
    test_data = data.Dataset(test_examples, test_fields)
    ref_data = data.Dataset(ref_examples, ref_fields)
    
    TEXT.build_vocab(ref_data, max_size=args.vocab_size)
    LABEL.build_vocab(ref_data)
    n_class = len(LABEL.vocab.stoi)
    if args.use_gpu:
        test_iterator = data.Iterator(test_data, batch_size=args.batch_size,
                            device=device, train=False,
                            shuffle=False, sort=False)
    else:
        test_iterator = data.Iterator(test_data, batch_size=args.batch_size,
                            device=-1, train=False,
                            shuffle=False, sort=False)
        
    if args.vis:
        rnn = BiLSTM_Attention_L_vis(len(TEXT.vocab), args.embed_size, output_size=n_class, hidden_dim=32, n_layers=1, margin=args.margin, device=device)
    else:
        rnn = BiLSTM_Attention_L(len(TEXT.vocab), args.embed_size, output_size=n_class, hidden_dim=64, n_layers=2, margin=args.margin, device=device)

    optimizer = optim.Adam(rnn.parameters(), lr=args.lr)
    criteon = nn.CrossEntropyLoss()
    rnn = rnn.cuda()
    criteon = criteon.cuda()

    rnn.load_state_dict(torch.load(args.modeldir+args.name+".pt"))

    predict,_ = test(rnn, test_iterator, criteon)
    label_dict = dict(LABEL.vocab.stoi)
    idx_list = list(label_dict.values())[:-2]
    idx_dict = dict(zip(label_dict.values(), label_dict.keys()))

    predictions = []
    for item in predict:
        predictions.append(idx_dict[item])
        
    saveResult(predictions,args.outdir+args.name+".txt")
    
if __name__ == '__main__':
    main()
