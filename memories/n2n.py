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

        self.H = nn.Linear(opt.word_vec_size, opt.word_vec_size)
        self.W = nn.Linear(opt.word_vec_size, opt.word_vec_size)

        self.embed_Ta = nn.Embedding(100, opt.word_vec_size)
        self.embed_Tc = nn.Embedding(100, opt.word_vec_size)

        self.dropout = nn.Dropout(opt.dropout)

        self.relu = nn.ReLU()
        self.softm = nn.Softmax()
        self.net_data = {k: []
                         for k in ['p', 'M', 'C', 'Ta', 'Tc']} if opt.gather_net_data else None

    def forward(self, u, M, C, mask):
        U, O, P = [], [], []

        # for each mi : select row of temporal matrix
        t_select = Variable(u.data.new([range(M.size(1))] * u.size(0))).long()
        t_select.masked_fill_(mask, 0)

        M = M.add(self.embed_Ta(t_select))
        C = C.add(self.embed_Tc(t_select))

        for h in range(self.nhops):
            u, (p, o) = self.hop(u, M, C)
            P += [p.data]
            U += [u]
            O += [o]

        out = self.softm(self.W(u))

        if self.net_data:
            self.net_data['p'] += [torch.cat(P)]
            self.net_data['M'] += [M.data.squeeze().t()]
            self.net_data['C'] += [C.data.squeeze().t()]
            self.net_data['Ta'] += [M.data.squeeze().t()]
            self.net_data['Tc'] += [C.data.squeeze().t()]

        return (out, o), torch.stack(U), torch.stack(O)

    def get_net_data(self):
        return self.net_data

    def hop(self, u, M, C):
        p = torch.bmm(M, u.unsqueeze(2)).squeeze(2)
        if not self.LS:
            p = self.softm(p)
        o = torch.bmm(p.unsqueeze(1), C).squeeze(1)
        u = self.H(u) + o
        return u, (p, o)
