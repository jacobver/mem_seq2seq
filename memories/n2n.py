import torch
import torch.nn as nn
from torch.autograd import Variable
from onmt import Constants


class N2N(nn.Module):

    def __init__(self, opt):
        super(N2N, self).__init__()

        self.LS = opt.linear_start
        self.nhops = opt.hops
        self.word_vec_sz = opt.word_vec_size

        self.H = nn.Sequential(
            nn.Linear(opt.word_vec_size, opt.word_vec_size),
            nn.ReLU())
        self.W = nn.Sequential(
            nn.Linear(opt.word_vec_size, opt.word_vec_size),
            nn.Softmax())

        self.embed_Ta = nn.Embedding(100, opt.word_vec_size)
        self.embed_Tc = nn.Embedding(100, opt.word_vec_size)

        self.dropout = nn.Dropout(opt.dropout)
        self.softmax = nn.Softmax()
        self.net_data = {k: []
                         for k in ['p', 'M', 'C', 'Ta', 'Tc']} if opt.gather_net_data else None

    def forward(self, u, M, C, mask):
        U, O, P = [], [], []

        # for each mi : select row of temporal matrix
        t_select = Variable(u.data.new([range(M.size(1))] * u.size(0))).long()
        #print(' == t_select : ' + str(t_select.size()))
        t_select.masked_fill_(mask, 0)

        M = M.add(self.embed_Ta(t_select))
        C = C.add(self.embed_Tc(t_select))

        for h in range(self.nhops):
            u, (p, o) = self.hop(u, M, C)
            P += [p.data]
            U += [u]
            O += [o]

        out = self.W(u)

        if self.net_data:
            self.net_data['p'] += [torch.cat(P)]
            self.net_data['M'] += [M.data.squeeze().t()]
            self.net_data['C'] += [C.data.squeeze().t()]
            self.net_data['Ta'] += [self.embed_Ta.weight.data.squeeze().t()]
            self.net_data['Tc'] += [self.embed_Tc.weight.data.squeeze().t()]

        # return out, u
        return (out, o), torch.stack(U), torch.stack(O)

    def get_net_data(self):
        return self.net_data

    def hop(self, u, M, C):
        p = torch.bmm(M, u.unsqueeze(2)).squeeze(2)
        if not self.LS:
            p = self.softmax(p)
        o = torch.bmm(p.unsqueeze(1), C).squeeze(1)
        #u = self.H(u) + o
        u = torch.add(u, o)
        return u, (p, o)
