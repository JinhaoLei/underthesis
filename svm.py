# coding: utf-8
import argparse
import time
import math
import numpy as np
import os 
import gc
import random
from sklearn.svm import LinearSVC, SVC
import timefeature as tf

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data_path', type=str, default='/home1/ljh/data/newdata/')
parser.add_argument('--model_path', type=str, default='./model/')
parser.add_argument('--type', type=int, default=11,
                    help='')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
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
parser.add_argument('--method', type=str,  default='cnn',
                    help='ave, sample, cnn, single-lstm')
parser.add_argument('--ifnormalize', type=int,  default=0,
                    help='')
parser.add_argument('--data_length', type=int,  default=500,
                    help='')
parser.add_argument('--num_channel', type=int,  default=4,
                    help='')
parser.add_argument('--target_c', type=int,  default=0,
                    help='')
parser.add_argument('--top_k', type=int,  default=100,
                    help='')

args = parser.parse_args()



###############################################################################
# Load data
###############################################################################

#corpus = data.Corpus(args.data)




def read_data(xfile, yfile, iftrain=True):
    data = []
    y = []
    data = np.load(xfile) #(8200, 50, 100, 4)
    newdata = []
    #if args.num_channel ==4:
        #data = data.view(data.size()[0], 50, 100, 4)
        #np.reshape(data, (np.shape(data)[0], np.shape(data)[1] * np.shape(data)[2]))
    #elif args.num_channel ==1:
    #    pass
        #data = abs(np.fft.fft(data))
        #indexes = np.argpartition(-data, args.top_k)
        #data = indexes[:, :args.top_k] / float(np.shape(data)[1])
        #print data[:3]
    

    print 'finish reading %s'%(xfile)

    with open(yfile) as f:
        for n, line in enumerate(f):
            y.append(int(line.strip()))
  
    print 'finish reading %s'%(yfile)

    
    
    print np.shape(data)
    print np.shape(y)
    return data, y

if args.num_channel==4:
    data_path = args.data_path + 'single'
elif args.num_channel==1:
    data_path = args.data_path + 'single'


if args.ifnormalize==1:
    data_path = data_path +  '_normal/'
else:
    data_path = data_path +  '/'
   
if args.num_channel==4:
    train_data = data_path + 'svm/' + 'fourA_train.npy'
    dev_data = data_path + 'svm/' + 'fourA_dev.npy'
    test_data = data_path + 'svm/' + 'fourA_test.npy'

elif args.num_channel==1:
    train_data = data_path + 'svm/' + str(args.target_c) + '_' + 'train.npy'
    dev_data = data_path + 'svm/' + str(args.target_c) + '_' + 'dev.npy'
    test_data = data_path + 'svm/' + str(args.target_c) + '_' + 'test.npy'

train_label = args.data_path + 'train_gold.csv'
dev_label = args.data_path + 'dev_gold.csv'
test_label = args.data_path + 'test_gold.csv'
 

train_data, train_label = read_data(train_data, train_label, True)
dev_data, dev_label = read_data(dev_data, dev_label,  False)
test_data, test_label = read_data(test_data, test_label,  False)

to_choose = 5000
sids = list(range(len(train_data)))

random.shuffle(sids)
new_traindata = train_data[:]
new_trainlabel = train_label[:]
for index in range(len(sids)):
    new_traindata[index] = train_data[sids[index]]
    new_trainlabel[index] = train_label[sids[index]]

clf = LinearSVC(C=100.0, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=0.000000001,
     verbose=0)

#clf = SVC(C=1.0, kernel='rbf', degree=3, coef0=0.0, shrinking=True, 
  #  probability=False, tol=0.0001, cache_size=200, class_weight=None, 
  #  verbose=False, max_iter=-1, random_state=None)


print 'begin fitting...'
clf.fit(new_traindata[:to_choose], new_trainlabel[:to_choose])
#clf.fit(dev_data, dev_label)
print 'finish fitting...'
dev_score = clf.score(dev_data, dev_label)
test_score = clf.score(test_data, test_label)
print dev_score, test_score
