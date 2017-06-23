import torch
import torch.nn as nn
from torch.autograd import Variable
from onmt.modules.GlobalAttention import GlobalAttention


class NSE(nn.Module):

    def __init__(self, opt):
        super(NSE, self).__init__()

        self.nlayers = opt.layers
        self.input_feed = opt.input_feed if opt.seq == 'decoder' else 0
        self.read_sz = opt.word_vec_size
        read_in = 2 * opt.word_vec_size if self.input_feed else opt.word_vec_size

        self.net_data = {'z': []} if opt.gather_net_data else None
        self.Z = None
        self.read_lstm = nn.LSTMCell(read_in, opt.rnn_size)

        self.dropout = nn.Dropout(opt.dropout)

        #self.sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax()

        compose_in_sz = opt.context_sz * opt.word_vec_size + \
            opt.word_vec_size

        # should be replaced with something more elaborate
        self.compose = nn.Sequential(
            nn.Linear(compose_in_sz, opt.rnn_size),
            nn.Softmax())

        self.write_lstm = nn.LSTMCell(opt.rnn_size, opt.word_vec_size)

        self.attn = None
        if opt.attn:
            self.attn = GlobalAttention(opt.word_vec_size)

    def forward(self, emb_utts, hidden, mem, M_que):

        M, mask = mem
        #(seq_sz, batch_sz, word_vec_sz) = emb_utts.size()
        outputs = []
        ((hr, cr), (hw, cw)) = hidden
        out = hr.clone()
        out.data.zero_()
        for w in emb_utts.split(1):
            w = w.squeeze(0)
            if self.input_feed:
                w = torch.cat((w, out), 1)

            hr, cr = self.read_lstm(w, (hr, cr))

            z = hr.unsqueeze(1).bmm(M.transpose(1, 2)).squeeze()
            z = self.softmax(z.masked_fill_(mask, float('-inf')))

            m = z.unsqueeze(1).bmm(M)
            cattet = torch.cat([hr, m.squeeze(1)], 1)
            comp = self.compose(cattet)
            out, cw = self.write_lstm(comp, (hw, cw))

            # Variable(M.data.new(torch.ones(*M.size())))
            M1 = Variable(M.clone().data.zero_() + 1)
            M = M1.sub(z.unsqueeze(2).expand(*M.size()))

            add1 = hw.unsqueeze(1).expand(*M.size())
            add2 = z.unsqueeze(2).expand(*M.size())
            M = M.addcmul(add1, add2)

            outputs += [out]

        if self.Z is not None:
            self.net_data['z'] += [self.Z]
            self.Z = None

        return torch.stack(outputs), ((hr, cr), (hw, cw)), M

    def get_net_data(self):
        # Z = torch.cat(self.net_data['z'],0)
        return {'z': self.net_data['z']}
