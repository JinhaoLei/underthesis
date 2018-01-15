# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import model
import torch.utils.data as Data  
import cnn_model
import os 
import gc

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
'''parser.add_argument('--train_data', type=str, default='./data/train.csv',
                    help='')
parser.add_argument('--dev_data', type=str, default='./data/dev.csv',
                    help='')
parser.add_argument('--test_data', type=str, default='./data/test.csv',
                    help='')'''
parser.add_argument('--data_path', type=str, default='/home1/ljh/data/')
parser.add_argument('--model_path', type=str, default='./model/')
parser.add_argument('--type', type=int, default=11,
                    help='')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=40,
                    help='size of word embeddings')
parser.add_argument('--hidden_size', type=int, default=80,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.007,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.5,   #??try
                    help='gradient clipping')
parser.add_argument('--max_epoch', type=int, default=30,
                    help='')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='batch size')
parser.add_argument('--e_batch_size', type=int, default=200, metavar='N',
                    help='')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
#parser.add_argument('--tied', action='store_true',
#                   help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--method', type=str,  default='cnn-lstm',
                    help='ave, sample, cnn-lstm')
parser.add_argument('--ifnormalize', type=int,  default=0,
                    help='')
parser.add_argument('--data_length', type=int,  default=500,
                    help='')
parser.add_argument('--num_channel', type=int,  default=4,
                    help='')
parser.add_argument('--c1_channel', type=int,  default=4,
                    help='')
parser.add_argument('--c2_channel', type=int,  default=4,
                    help='')
parser.add_argument('--c1_kernel', type=int,  default=2,
                    help='')
parser.add_argument('--c2_kernel', type=int,  default=2,
                    help='')
parser.add_argument('--mlp_in', type=int,  default=4 * 97,
                    help='')
parser.add_argument('--mlp_out', type=int,  default=40,
                    help='')
args = parser.parse_args()



# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

#corpus = data.Corpus(args.data)

def print_args():
    print 'data_path %s'%(args.data_path)
    print 'classification type %d'%(args.type)
    print 'model %s'%(args.model)
    print 'embedding size %d'%(args.emsize)
    print 'hidden_size %d'%(args.hidden_size)
    print 'num_layers %d'%(args.nlayers)
    print 'init_lr %.1f'%(args.lr)
    print 'clip %.1f'%(args.clip)
    print 'max_epoch %d'%(args.max_epoch)
    print 'epochs %d'%(args.epochs)
    print 'batch_size %d'%(args.batch_size)
    print 'evaluate_batch_size %d'%(args.e_batch_size)
    print 'drop_out %.1f'%(args.dropout)
    print 'method %s'%(args.method)
    print 'if_normalize %d'%(args.ifnormalize)
    print 'data_length %d'%(args.data_length)
    print 'c1_channel %d'%(args.c1_channel)
    print 'c2_channel %d'%(args.c2_channel)
    print 'c1_kernel %d'%(args.c1_kernel)
    print 'c2_kernel %d'%(args.c2_kernel)
    print 'mlp_in %d'%(args.mlp_in)
    print 'mlp_out %d'%(args.mlp_out)

print_args()

def read_data_cnn(mapfile, iftrain=True):
    data = []
    y = []
    with open(mapfile) as f:
        for n, line in enumerate(f):
            x, target = line.strip().split('\t')
            data.append(int(x))
            y.append(int(target))
    data = torch.IntTensor(data).contiguous()
    y = torch.LongTensor(y).contiguous()
    print 'cnn map data:', data.size()
    print 'cnn map data y:', y.size()
    dataset = Data.TensorDataset(data_tensor=data, target_tensor=y)  
    if iftrain:
        data_loader = Data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=iftrain, drop_last=iftrain)
    else:
        data_loader = Data.DataLoader(dataset=dataset, batch_size=args.e_batch_size, shuffle=iftrain, drop_last=iftrain)
    return data_loader               #[num_of_data, length, num_of_channel]


def read_data(xfile, yfile, iftrain=True):
    data = []
    y = []
    with open(xfile) as f:
        for n, line in enumerate(f):
            line = line.strip().split('\t')
            line = [float(line[i]) for i in range(len(line))]
            data.append(line)
            if n%10000==0:
                print 'read %s %d'%(xfile, n)

    with open(yfile) as f:
        for n, line in enumerate(f):
            y.append(int(line.strip()))
            if n%10000==0:
                print 'read %s %d'%(yfile, n)

    data = torch.Tensor(data)
    data = data.view(data.size()[0],  -1, args.num_channel * 10).contiguous()
    y = torch.LongTensor(y).contiguous()
    print data.size()
    print y.size()
    dataset = Data.TensorDataset(data_tensor=data, target_tensor=y)  
    if iftrain:
        data_loader = Data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=iftrain, drop_last=iftrain)
    else:
        data_loader = Data.DataLoader(dataset=dataset, batch_size=args.e_batch_size, shuffle=iftrain, drop_last=iftrain)
    return data_loader               #[num_of_data, length, num_of_channel]

if args.ifnormalize==1:
    data_path = args.data_path + 'window_data/'
else:
    data_path = args.data_path + 'window_data_no_normalize/'

if args.method == 'ave':
    train_data = data_path + 'train/' + 'train_ave'
    train_label = data_path + 'train/' + 'train_ave_gold'
    dev_data = data_path + 'dev/' + 'dev_ave'
    dev_label = data_path + 'dev/' + 'dev_ave_gold'
    test_data = data_path + 'test/' + 'test_ave'
    test_label = data_path + 'test/' + 'test_ave_gold'
    train_data = read_data(train_data, train_label, True)
    dev_data = read_data(dev_data, dev_label, False)
    test_data = read_data(test_data, test_label, False)
elif args.method == 'sample':
    train_data = data_path + 'train/' + 'train_sample'
    train_label = data_path + 'train/' + 'train_sample_gold'
    dev_data = data_path + 'dev/' + 'dev_sample'
    dev_label = data_path + 'dev/' + 'dev_sample_gold'
    test_data = data_path + 'test/' + 'test_sample'
    test_label = data_path + 'test/' + 'test_sample_gold'
    train_data = read_data(train_data, train_label, True)
    dev_data = read_data(dev_data, dev_label, False)
    test_data = read_data(test_data, test_label, False)
elif args.method == 'cnn-lstm':
    train_data = read_data_cnn(args.data_path + 'cnn_data/train/map.csv', True)
    dev_data = read_data_cnn(args.data_path + 'cnn_data/dev/map.csv', False)
    test_data = read_data_cnn(args.data_path + 'cnn_data/test/map.csv', False)

'''print 'train_data shape:'
print train_data.shape()
print 'dev_data_shape'
print dev_data.shape()
print 'test_data_shape'
print test_data.shape()'''
'''train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)'''

###############################################################################
# Build the model
###############################################################################

ntokens = args.type
criterion = nn.CrossEntropyLoss()
ifcnn = False
if args.method == 'cnn-lstm':
    ifcnn = True
model = model.RNNModel(ifcnn, args.c1_channel, args.c2_channel, args.c1_kernel, args.c2_kernel, 
        args.mlp_in, args.mlp_out, args.model, ntokens, args.emsize, args.hidden_size, args.nlayers, args.dropout)
if args.cuda:
    model.cuda()
    criterion.cuda()

'''def extract(data, type):
    x = []
    del x
    gc.collect()
    x = []
    start_time = time.time()
    for i in range(len(data)):
        numfile, n = data[i] / 10000, data[i] % 10000
        if type == 'train':
            adata = np.load(args.data_path + 'cnn_data/train/' + str(numfile) + '/' + str(n) + '.npy')
        elif type == 'dev':
            adata = np.load(args.data_path + 'cnn_data/dev/' + str(numfile) + '/' + str(n) + '.npy')
        elif type == 'test':
            adata = np.load(args.data_path + 'cnn_data/test/' + str(numfile) + '/' + str(n) + '.npy')
        x.append(adata)
    end_time = time.time()
    print start_time - end_time
    return torch.Tensor(x)'''

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data.cuda())
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(data_source, type):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = args.type
    correct = 0
    num = 0
    hidden = model.init_hidden(args.e_batch_size)
    confusion = torch.zeros(args.type, args.type)
    for step, (data, targets) in enumerate(data_source):
        if args.method == 'cnn-lstm':
            data = extract(data, type)
            print data.size()
            num += data.size()[0]
            data = data.view(data.size()[0] * data.size()[1], 1, data.size()[2], data.size()[3])
            print 'transformed:', data.size()
            data = Variable(data.cuda(), volatile=True)
        else:
            data = Variable(data.cuda(), volatile=True)
            num += data.size()[0]
        targets = Variable(targets.cuda(), volatile=True)
        output, hidden = model(args.e_batch_size, data, hidden)
        max_index = output.data.max(dim = 1)[1]
        for i in range(len(max_index)):
            confusion[targets.data[i]][max_index[i]] +=1
        correct += max_index.eq(targets.data.view_as(max_index)).sum()
        loss = criterion(output, targets)
        total_loss += loss.data
        hidden = repackage_hidden(hidden)
    #print confusion
    print str(correct) + '/' + str(num)
    acc = correct / float(num)
    cur_loss = total_loss[0] / (num/args.batch_size)
    return acc, cur_loss



def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = args.type
    hidden = model.init_hidden(args.batch_size)
    correct = 0
    num = 0
    confusion = torch.zeros(args.type, args.type)
    for step, (data, targets) in enumerate(train_data):
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        if args.method == 'cnn-lstm':
            data = extract(data, 'train')
            print data.size()
            num += data.size()[0]
            data = data.view(data.size()[0] * data.size()[1], 1, data.size()[2], data.size()[3])
            print 'transformed:', data.size()
            data = Variable(data.cuda())
        else:
            data = Variable(data.cuda())
            num += data.size()[0]
        targets = Variable(targets.cuda())
        
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(args.batch_size, data, hidden)
        max_index = output.data.max(dim = 1)[1]
        correct += max_index.eq(targets.data.view_as(max_index)).sum()
        optimizer = optim.Adagrad(model.parameters(), lr = lr)
        loss = criterion(output, targets)
        for i in range(len(max_index)):
            confusion[targets.data[i]][max_index[i]] +=1
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)     
        optimizer.step() # Does the update
        total_loss += loss.data
    #print confusion
    print str(correct) + '/' + str(num)
    acc = correct / float(num)
    cur_loss = total_loss[0] / (num/args.batch_size)
    return acc, cur_loss

# Loop over epochs.
lr = args.lr
best_val_acc = None
best_test = -1.0

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_acc, train_loss = train()
        val_acc, val_loss = evaluate(dev_data, 'dev')
        test_acc, _ = evaluate(test_data, 'test')
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_acc or val_acc > best_val_acc:
            #with open(args.model_path + str(val_acc) + '_' + args.save, 'wb') as f:
                #torch.save(model, f)
            best_val_acc = val_acc
            best_test = test_acc
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            if epoch > args.max_epoch:
                lr *= 0.9


        print('-' * 89)
        print('| end of epoch {:3d} | lr: {:.6f} | time: {:5.2f}s |\n| train acc {:5.4f} | train loss {:5.7f} |\n| valid acc {:5.4f} | '
                'valid loss {:5.7f} |\n| test acc {:5.4f} | best acc {:.4f}'.format(epoch, lr, (time.time() - epoch_start_time), train_acc, train_loss,
                                           val_acc, val_loss, test_acc, best_test))
        print('-' * 89)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
#with open(args.save, 'rb') as f:
#    model = torch.load(f)

