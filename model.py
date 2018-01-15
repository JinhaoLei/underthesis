import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ifcnn, c1_channel, c2_channel, c1_kernel, c2_kernel, 
        mlp_in, mlp_out, rnn_type, ntoken, ninp, hidden_size, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        #self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, hidden_size, nlayers, dropout=dropout, batch_first=True)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, hidden_size, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, ntoken)
        self.rnn_type = rnn_type
        self.ifcnn = ifcnn
  
        if ifcnn:
            self.conv1 = nn.Conv2d(1, c1_channel, kernel_size= c1_kernel)
            self.conv2 = nn.Conv2d(c1_channel, c2_channel, kernel_size=c2_kernel)
            self.fc1 = nn.Linear(mlp_in, mlp_out)

        if tie_weights:
            if hidden_size != ninp:
                raise ValueError('When using the tied flag, hidden_size must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        if self.ifcnn:
            self.fc1.bias.data.fill_(0)
            self.conv1.weight.data.uniform_(-initrange, initrange)
            self.conv2.weight.data.uniform_(-initrange, initrange)

    def forward(self, batch_size, input, hidden):
        if self.ifcnn:
            x = self.conv1(input)
            #print x.size()
            x = F.relu(x)
            #print x.size()
            x = F.max_pool2d(x, 2, stride=1)
            #print x.size()
            x = F.relu(self.conv2(x))
            #print 'after 2 conv:', x.size()
            x = x.view(-1, x.size()[-1] * x.size()[-2] * x.size()[-3])
            #print 'after 2 conv, transformed:', x.size()
            x = self.fc1(x)
            #print 'after 2 conv, fc:', x.size()
            x = x.view(batch_size, -1, x.size()[-1])
            #print 'input to lstm', x.size()
            emb = self.drop(x)
        else:

            emb = self.drop(input)
        output, hidden = self.rnn(emb, hidden)
        #output = self.drop(output)
        if self.rnn_type == 'LSTM':
            final_hidden = hidden[0][-1]   #[batch,hidden_size]
        else:
            final_hidden = hidden[-1]
        #print final_hidden.size()
        #print '=[batch, hidden_size]??'
        decoded = self.decoder(final_hidden)   #[batch_size, classes]
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.hidden_size).zero_().cuda()),
                    Variable(weight.new(self.nlayers, bsz, self.hidden_size).zero_().cuda()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.hidden_size).zero_().cuda())
