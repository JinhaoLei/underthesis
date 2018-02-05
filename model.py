import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, method, c1_channel, c2_channel, c1_kernel, c2_kernel, 
        mlp_in, mlp_out, rnn_type, ntoken, ninp, hidden_size, nlayers, att, dropout=0.5, tie_weights=False):
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
            self.rnn = nn.RNN(ninp, hidden_size, nlayers, nonlinearity=nonlinearity, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(hidden_size, ntoken)
        self.rnn_type = rnn_type
        self.method = method
        self.att = att
        '''if self.att == 1:
            self.attfc = nn.Linear(hidden_size, hidden_size)
            self.v = Variable(torch.zeros(hidden_size, 1).cuda())'''

        if method == 'cnn':
            self.conv1 = nn.Conv2d(1, c1_channel, kernel_size= c1_kernel, stride=(2, 1))
            self.conv2 = nn.Conv2d(c1_channel, c2_channel, kernel_size=c2_kernel)
            self.fc1 = nn.Linear(mlp_in, mlp_out)
        elif method == 'wdcnn':
            self.conv1 = nn.Conv2d(1, 16, kernel_size = (64, 1), stride = (16, 1), padding = (24, 0))
            self.conv2 = nn.Conv2d(16, 32, kernel_size = (3, 1), stride = (1, 1), padding = (1, 0))
            self.conv3 = nn.Conv2d(32, 64, kernel_size = (3, 1), stride = (1, 1), padding = (1, 0))
            self.conv4 = nn.Conv2d(64, 64, kernel_size = (3, 1), stride = (1, 1), padding = (1, 0))
            self.conv5 = nn.Conv2d(64, 64, kernel_size = (3, 1), stride = (1, 1))
            self.fc = nn.Linear(192, 100)
            self.fc2 = nn.Linear(100, 11)
            self.mlpdict = []
            self.mlpdict.extend([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.fc, self.fc2])

        elif method == 'mscnn':

            self.conv1 = nn.Conv2d(1, 10, kernel_size = (65, 1), stride = (1, 1))
            self.conv2 = nn.Conv2d(10, 15, kernel_size = (65, 1), stride = (1, 1))
            self.conv3 = nn.Conv2d(15, 15, kernel_size = (976, 1), stride = (1, 1))

            self.fc = nn.Linear(15, 11)
            
            self.mlpdict = []
            self.mlpdict.extend([self.conv1, self.conv2, self.conv3, self.fc])

        elif method == 'LR':
            #print ninp, ntoken
            #self.fclr = nn.Linear(ninp, 500)
            
            #self.fclr2 = nn.Linear(500, ntoken)
            self.fclr = nn.Linear(ninp, 2000)
            self.fclr2 = nn.Linear(2000, 200)
            self.fclr3 = nn.Linear(200, ntoken)
            self.mlpdict = []
            self.mlpdict.extend([self.fclr, self.fclr2, self.fclr3])

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
        if self.method == 'cnn':
            self.fc1.bias.data.fill_(0)
            self.fc1.weight.data.uniform_(-initrange, initrange)
            self.conv1.weight.data.uniform_(-initrange, initrange)
            self.conv2.weight.data.uniform_(-initrange, initrange)
        if (self.method == 'LR') or (self.method == 'wdcnn' or self.method == 'mscnn'):
            for mlp in self.mlpdict:
                mlp.bias.data.fill_(0)
                mlp.weight.data.uniform_(-initrange, initrange)
        if self.att == 1:
            self.attfc.bias.data.fill_(0)
            self.attfc.weight.data.uniform_(-initrange, initrange)
            self.v.data.uniform_(-initrange, initrange)
              

    def forward(self, batch_size, input, hidden):
        if self.method == 'cnn':

       
            x = input
            emb = x
            #print emb
            #print 'emb', emb.size()
        elif self.method == 'wdcnn':
            x = F.relu(self.conv1(input))
           
            x = F.max_pool2d(x, (2, 1))
           
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, (2, 1))
            #print x.size()
            x = F.relu(self.conv3(x))
            #print x.size()
            x = F.max_pool2d(x, (2, 1))
            #print x.size()
            x = F.relu(self.conv4(x))
            #print x.size()
            x = F.max_pool2d(x, (2, 1))
            #print x.size()
            x = F.relu(self.conv5(x))
            x = F.max_pool2d(x, (2, 1))
            #print x.size()
            x = x.view(x.size()[0], -1)
            x = self.fc(x)
            x = self.fc2(x)
            return x, ''

        elif self.method == 'mscnn':
            x = F.tanh(self.conv1(input))
            x = F.max_pool2d(x, (2, 1))
            x = F.tanh(self.conv2(x))
            x = F.max_pool2d(x, (2, 1))
            x = F.tanh(self.conv3(x))
        
        
           
            x = x.view(x.size()[0], -1)
            x = self.fc(x)
            x = F.relu(x)

            return x, ''

        elif self.method == 'LR':
            decoded = self.fclr(input)
            decoded = F.tanh(decoded)
            decoded = self.fclr2(decoded)
            decoded = F.tanh(decoded)
            decoded = self.fclr3(decoded)
            return decoded, ''
        else:
            emb = input            #emb = self.drop(input)
        output, hidden = self.rnn(emb, hidden)

        #output = self.drop(output)
        if self.att == 1:
            batch_size, seq_size = output.size()[0], output.size()[1]
            output = output.contiguous().view(output.size()[0] * output.size()[1], -1)
            att = F.tanh(self.attfc(output))
            att = torch.matmul(att, self.v)
            att = att.view(batch_size, seq_size, -1)
            att = F.softmax(att, dim=1)
            att = att.view(att.size()[0] * att.size()[1], -1)
            final_hidden = att * output
            final_hidden = final_hidden.view(batch_size, seq_size, -1)
            final_hidden = torch.sum(final_hidden, dim=1)
        else:
            if self.rnn_type == 'LSTM':
                final_hidden = hidden[0][-1]   #[batch,hidden_size]
            else:
                final_hidden = hidden[-1]
            #final_hidden = torch.mean(output, 1)
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
