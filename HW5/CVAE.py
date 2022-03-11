import torch
import torch.nn as nn
import numpy as np
import dataloader
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def KL_loss(m, logvar):
    return torch.sum(0.5 * (-logvar + (m**2) + torch.exp(logvar) - 1))

#Encoder
class EncoderRNN(nn.Module):
    def __init__(self, word_size, hidden_size, latent_size, num_condition, condition_size):
        super(EncoderRNN, self).__init__()
        
        self.word_size = word_size
        self.hidden_size = hidden_size
        self.condition_size = condition_size
        self.latent_size = latent_size

        self.condition_embedding = nn.Embedding(num_condition, condition_size)
        self.word_embedding = nn.Embedding(word_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1)

        self.mean = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)

        #init cell
        self.c_init = Variable(torch.zeros(1, 1, self.hidden_size, device=device))

    def forward(self, input, init_hidden, input_condition):
        c = self.condition(input_condition)

        # get (1,1,hidden_size)
        hidden = torch.cat((init_hidden, c), dim=2)

        # get (seq, 1, hidden_size)
        x = self.word_embedding(input).view(-1, 1, self.hidden_size)

        # get (seq, 1, hidden_size), (1, 1, hidden_size)
        output, (h, _) = self.lstm(x, (hidden, self.c_init))

        # get (1, 1, hidden_size)
        m = self.mean(h)
        logvar = self.logvar(h)

        # z = self.sample_z() * torch.exp(0.5*logvar) + m
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = m + eps*std

        return z, m, logvar

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size - self.condition_size, device=device)

    def condition(self, c):
        c = torch.LongTensor([c]).to(device)
        return self.condition_embedding(c).view(1,1,-1)
    
    def sample_z(self):
        return torch.normal(
            torch.FloatTensor([0]*self.latent_size), 
            torch.FloatTensor([1]*self.latent_size)
        ).to(device)

#Decoder
class DecoderRNN(nn.Module):
    def __init__(self, word_size, hidden_size, latent_size, condition_size):
        super(DecoderRNN, self).__init__()

        self.word_size = word_size
        self.hidden_size = hidden_size

        self.latent_to_hidden = nn.Linear(latent_size+condition_size, hidden_size)
        self.word_embedding = nn.Embedding(word_size, hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1)
        self.out = nn.Linear(hidden_size, word_size)

    def initHidden(self, z, c):
        latent = torch.cat((z, c), dim=2)
        c_init = Variable(torch.zeros(1, 1, self.hidden_size, device=device))
        return self.latent_to_hidden(latent), c_init

    def forward(self, x, hidden, c):
        # get (1, 1, hidden_size)
        x = self.word_embedding(x).view(1, 1, self.hidden_size)
        
        # get (1, 1, hidden_size) (1, 1, hidden_size)
        output, (hidden, c) = self.lstm(x, (hidden, c))

        # get (1, word_size)
        output = self.out(output).view(-1, self.word_size)
        
        return output, hidden, c

def test_encoder():
    CE_loss = 0  # Reset every print_every
    KLD_loss = 0  # Reset every plot_every
    train_dataset = dataloader.wordsDataset()

    sos_token = train_dataset.chardict.word2index['SOS']
    eos_token = train_dataset.chardict.word2index['EOS']
    hidden_size = 256
    latent_size = 32
    condition_size = 8
    #The number of vocabulary
    word_size = train_dataset.chardict.n_words
    num_condition = len(train_dataset.tenses)

    criterion = nn.CrossEntropyLoss()

    # encoder
    encoder = EncoderRNN(word_size, hidden_size, latent_size, num_condition, condition_size).to(device)
    data = train_dataset[0]
    input, c = data
    print(input)
    # print(c)
    z, m, logvar = encoder(input[1:-1].to(device), encoder.initHidden().to(device), c)

    # print(z)
    # print(m)
    # print(logvar)

    # decoder
    decoder = DecoderRNN(word_size, hidden_size, latent_size, condition_size).to(device)

    total_loss = 0
    x = torch.LongTensor([sos_token]).to(device)
    z = z.view(1,1,-1)
    hidden, c_0 = decoder.initHidden(z.to(device), encoder.condition(c))

    x = x.detach()
    output, hidden, c_0 = decoder(x, hidden, c_0)
    output_onehot = torch.max(torch.softmax(output, dim=1), 1)[1]
    x = output_onehot

    print(output)
    print(input[1])
    #loss
    total_loss += criterion(output, input[1].to(device).view(-1))

    CE_loss += total_loss
    KLD_loss += KL_loss(m, logvar)
    total_loss += KLD_loss

    print(CE_loss)
    print(KLD_loss)
    print(total_loss)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_encoder()