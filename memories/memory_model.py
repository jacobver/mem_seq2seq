from memories import nse, dnc, n2n, util
from onmt import Constants
from onmt import Models

import torch
import torch.nn as nn
from torch.autograd import Variable


class MemModel(nn.Module):

    def __init__(self, opt, dicts):
        super(MemModel, self).__init__()

        self.share_M = opt.share_M
        self.brnn = opt.brnn

        self.embed_in, self.embed_out = self.get_embeddings(opt, dicts)

        mem = opt.mem.split('_')

        # get encoder and decoder
        self.encoder = self.get_encoder(mem[0], opt, dicts)
        self.decoder = self.get_decoder(mem[1], opt, dicts)

        if self.brnn and not isinstance(self.encoder, Models.Encoder):
            self.bd_h = nn.Sequential(
                nn.Linear(2 * opt.word_vec_size, opt.word_vec_size),
                nn.ReLU())
            self.bd_context = nn.Sequential(
                nn.Linear(2 * opt.word_vec_size, opt.word_vec_size),
                nn.ReLU())
            if 'nse' in mem:
                self.bd_m = nn.Sequential(
                    nn.Linear(2 * opt.word_vec_size, opt.word_vec_size),
                    nn.ReLU())

        if 'n2n' == opt.mem[-3:]:
            self.n2n_cat_feed = nn.Sequential(
                nn.Linear(2 * opt.word_vec_size, opt.word_vec_size),
                nn.ReLU())

        self.forward = eval('self.' + opt.mem)

        self.generate = False

    def bienc(self, input, enc):

        context, enc_h, M = enc(input)
        context_rev, enc_h_rev, M_rev = enc(util.flip(input, 0))

        # dnc memory also bidirectional
        if isinstance(self.encoder, dnc.DNC):
            context_rev, enc_h_rev, M_rev = enc(util.flip(input, 0), M)

        '''
        if self.encoder.layers == 2:
            h_out = ((self.bd_h(torch.cat([context[0], enc_h_rev[0][0]], 1)),
                      self.bd_h(torch.cat([context[0], enc_h_rev[0][1]], 1))),
                     (self.bd_h(torch.cat([context[0], enc_h_rev[0][0]], 1)),
                      self.bd_h(torch.cat([context[0], enc_h_rev[0][1]], 1))))

        elif self.encoder.layers == 1:
            h_out = (self.bd_h(torch.cat([context[0], enc_h_rev[0][0]], 1)),
                     self.bd_h(torch.cat([context[0], enc_h_rev[0][1]], 1)))
        '''
        context_out = self.bd_context(torch.cat((
            util.flip(context, dim=0), context_rev), 2).view(-1, 2 * context.size(2)))

        if isinstance(self.encoder, nse.NSE):
            M_out = self.bd_m(
                torch.cat((util.flip(M, dim=1), M_rev), 2).view(-1, 2 * M.size(2))).view(*M.size())
        else:
            M_out = M_rev

        return context_out.view(*context.size()), enc_h_rev, M_out

    def embed_in_out(self, input):

        src = input[0][0]
        tgt = input[1][:-1]  # exclude last target from inputs

        emb_in = self.embed_in(src)
        emb_out = self.embed_out(tgt)

        return emb_in, emb_out

    def nse_enc(self, input):
        emb_in = self.embed_in(input)

        hidden = self.make_init_hidden(emb_in, 2)
        mask = input.t().eq(0).detach()

        M = emb_in.clone().transpose(0, 1).detach()

        return self.encoder(emb_in, hidden, (M, mask), None)  # self.M_que)

    def nse_n2n(self, input):

        if self.brnn:
            context, enc_h, enc_M = self.bienc(input[0][0], self.nse_enc)
        else:
            context, enc_h, enc_M = self.nse_enc(
                util.flip(input[0][0]))

        mask = input[0][0].t().eq(0).detach()

        M = self.embed_A(enc_M)
        C = self.embed_C(enc_M)
        emb_out = self.embed_out(input[1][:-1])

        # u = Variable(emb_out.data.new(*emb_out.size()
        #                              [1:]).zero_() + .1, requires_grad=True)
        u = enc_h[0][0]

        outputs = []
        for w in emb_out.split(1):
            dec_in = torch.cat((w.squeeze(), u), 1)
            u = self.n2n_cat_feed(dec_in)
            out, U, O = self.decoder(u, M, C, mask)
            outputs += [out[0]]

        return torch.stack(outputs)

    def lstm_n2n(self, input):

        enc_h, context = self.encoder(input[0][0])
        if self.brnn:
            enc_h = list(enc_h)
            for l in range(self.encoder.layers):
                enc_h[l] = self.fix_enc_hidden(enc_h[l])

        mask = input[0][0].t().eq(0).detach()

        emb_in = self.embed_out(input[0][0]).transpose(0, 1)
        M = self.embed_A(emb_in)
        C = self.embed_C(emb_in)
        emb_out = self.embed_out(input[1][:-1])

        u = Variable(emb_out.data.new(*emb_out.size()
                                      [1:]).zero_() + .1, requires_grad=True)

        outputs = []
        #u = enc_h[0][0].squeeze()
        for w in emb_out.split(1):
            dec_in = torch.cat((w.squeeze(), u), 1)
            u = self.n2n_cat_feed(dec_in)
            out, U, O = self.decoder(u, M, C, mask)
            outputs += [out]

        return torch.stack(outputs)

    def nse_lstm(self, input):

        if self.brnn:
            context, enc_h, M = self.bienc(input[0][0], self.nse_enc)
        else:
            context, enc_h, M = self.nse_enc(
                util.flip(input[0][0]))

        hidden = (torch.stack((enc_h[0][0], enc_h[1][0])),
                  torch.stack((enc_h[0][1], enc_h[1][1])))

        init_output = self.make_init_hidden(enc_h[0][0], 1)[0]

        out, dec_hidden, _attn = self.decoder(input[1][: -1], hidden,
                                              context, init_output)

        return out

    def nse_nse(self, input):
        if self.brnn:
            context, enc_h, enc_M = self.bienc(input[0][0], self.nse_enc)
        else:
            context, enc_h, enc_M = self.nse_enc(
                util.flip(input[0][0]))

        emb_out = self.embed_out(input[1][: -1])

        dec_M = enc_M.detach()
        mask = util.flip(input[0][0]).transpose(0, 1).eq(0).detach()
        # self.update_queue(M)

        outputs, _, _ = self.decoder(
            emb_out, enc_h, (dec_M, mask), None)

        # self.update_queue(M, mask)

        return outputs

    def dnc_enc(self, input, M=None):
        emb_in = self.embed_in(input)

        hidden = self.encoder.make_init_hidden(emb_in, *self.encoder.rnn_sz)
        init_M = self.encoder.make_init_M(emb_in.size(1)) if M is None else M

        return self.encoder(emb_in, hidden, init_M)

    def dnc_lstm(self, input):
        if self.brnn:
            context, enc_h, M = self.bienc(input[0][0], self.dnc_enc)
        else:
            context, enc_h, M = self.dnc_enc(input[0][0])

        hidden = (torch.stack((enc_h[0][0], enc_h[1][0])),
                  torch.stack((enc_h[0][1], enc_h[1][1])))

        init_output = self.make_init_hidden(enc_h[0][0], 1)[0]

        out, dec_hidden, _attn = self.decoder(input[1][: -1], hidden,
                                              context, init_output)
        return out

    def lstm_dnc(self, input):

        enc_h, context = self.encoder(input[0][0])
        if self.brnn:
            enc_h = list(enc_h)
            for l in range(self.encoder.layers):
                enc_h[l] = self.fix_enc_hidden(enc_h[l])

        if self.encoder.layers == 2:
            enc_h = ((enc_h[0][0].squeeze(0), enc_h[1][0].squeeze(0)),
                     (enc_h[0][1].squeeze(0), enc_h[1][1].squeeze(0)))

        emb_out = self.embed_out(input[1][: -1])
        M = self.decoder.make_init_M(emb_out.size(1))

        outputs, dec_hidden, M = self.decoder(emb_out, enc_h, M, context)

        return outputs

    def dnc_dnc(self, input):
        if self.brnn:
            context, enc_h, M = self.bienc(input[0][0], self.dnc_enc)
        else:
            context, enc_h, M = self.dnc_enc(input[0][0])

        emb_out = self.embed_out(input[1][: -1])
        if not self.share_M:
            M = self.decoder.make_init_M(emb_out.size(1))

        outputs, dec_hidden, M = self.decoder(emb_out, enc_h, M, context)

        return outputs

    def n2n_lstm(self, input):

        src = input[0][0]
        emb_out = self.embed_out(input[1][: -1])

        u = Variable(emb_out.data.new(*emb_out.size()
                                      [1:]).zero_() + .1, requires_grad=True)
        M = self.embed_A(src).transpose(0, 1)
        C = self.embed_C(src).transpose(0, 1)

        mask = input[0][0].t().eq(0)

        enc_h, U, O = self.encoder(u, M, C, mask)

        init_output = self.make_init_hidden(enc_h[0], 1)[0]

        enc_h = (enc_h[0].unsqueeze(0), enc_h[1].unsqueeze(0))
        out, dec_hidden, _attn = self.decoder(input[1][:-1], enc_h,
                                              U, init_output)
        return out

    def n2n_dnc(self, input):

        src = input[0][0]
        emb_out = self.embed_out(input[1][: -1])

        u = Variable(emb_out.data.new(*emb_out.size()
                                      [1:]).zero_() + .1, requires_grad=True)
        M = self.embed_A(src).transpose(0, 1)
        C = self.embed_C(src).transpose(0, 1)

        mask = input[0][0].t().eq(0)

        enc_h, U, O = self.encoder(u, M, C, mask)

        init_M = self.decoder.make_init_M(emb_out.size(1))

        out, dec_hidden, mem = self.decoder(emb_out, enc_h, init_M, U)
        return out

    def nse_dnc(self, input):
        if self.brnn:
            context, enc_h, enc_M = self.bienc(input[0][0], self.nse_enc)
        else:
            context, enc_h, enc_M = self.nse_enc(
                util.flip(input[0][0]))

        emb_out = self.embed_out(input[1][: -1])
        init_M = self.decoder.make_init_M(emb_out.size(1))

        out, dec_hidden, mem = self.decoder(emb_out, enc_h, init_M, context)

        return out

    def fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                .transpose(1, 2).contiguous() \
                .view(h.size(0) // 2, h.size(1), h.size(2) * 2)

    def update_queue(self, new_m):
        new_q = self.M_que[1:] + [new_m]
        self.nse_queue = torch.stack(new_q)

    def get_dump_data(self):
        return [mod.get_net_data() for mod in [self.encoder, self.decoder] if not isinstance(mod, Models.Decoder)]

    def set_generate(self, enabled):
        self.generate = enabled

    def make_init_hidden(self, inp, nlayers=1):
        h0 = Variable(inp.data.new(
            *inp.size()[-2:]).zero_(), requires_grad=False)
        if nlayers == 1:
            return (h0.clone(), h0.clone())
        elif nlayers == 2:
            return ((h0.clone(), h0.clone()), (h0.clone(), h0.clone()))

    def get_encoder(self, enc, opt, dicts):
        opt.seq = 'encoder'

        if enc == 'nse':
            opt.layers = 2
            opt.word_vec_size = self.embed_in.weight.size(1)
            opt.rnn_size = self.embed_in.weight.size(1)
            return nse.NSE(opt)

        elif enc == 'n2n':
            opt.layers = 1
            utt_emb_sz = (dicts['src'].size(), opt.word_vec_size)
            self.embed_A = nn.Embedding(*utt_emb_sz)
            self.embed_C = nn.Embedding(*utt_emb_sz)

            return n2n.N2N(opt)

        elif enc == 'dnc':
            if opt.mem == 'dnc_lstm':
                opt.rnn_size = opt.word_vec_size
            return dnc.DNC(opt)

        elif enc == 'lstm':
            if opt.mem != 'lstm_lstm':
                opt.layers = 2
            opt.rnn_size = opt.word_vec_size
            return Models.Encoder(opt, dicts['src'])

    def get_decoder(self, dec, opt, dicts):

        opt.seq = 'decoder'

        if dec == 'nse':
            opt.layers = 2
            return nse.NSE(opt)

        elif dec == 'n2n':  # implicit assumption encoder == nse
            self.embed_A = util.EmbMem(opt.word_vec_size, 'relu')
            self.embed_C = util.EmbMem(opt.word_vec_size, 'relu')

            return n2n.N2N(opt)

        elif dec == 'dnc':
            if opt.mem == 'lstm_dnc':
                opt.rnn_size = opt.word_vec_size
            return dnc.DNC(opt)

        elif dec == 'lstm':
            opt.rnn_size = opt.word_vec_size
            return Models.Decoder(opt, dicts['tgt'])

    def get_embeddings(self, opt, dicts):
        src, tgt = self.load_pretrained_vectors(opt)

        def emb_size(emb):
            return None if emb is None else emb.size(1)

        word_vec_size = emb_size(src) or emb_size(tgt)
        opt.word_vec_size = word_vec_size if word_vec_size is not None else opt.word_vec_size

        emb_in = nn.Embedding(
            dicts['src'].size(), opt.word_vec_size, padding_idx=Constants.PAD)
        emb_out = nn.Embedding(
            dicts['tgt'].size(), opt.word_vec_size, padding_idx=Constants.PAD)

        def set_emb_weight(l, w):
            if w is not None:
                l.weight.copy_ = Variable(w, requires_grad=False)
            return l

        return set_emb_weight(emb_in, src), set_emb_weight(emb_out, tgt)

    def load_pretrained_vectors(self, opt):
        src_emb, tgt_emb = None, None
        if opt.pre_word_vecs_enc is not None:
            print('* src pre embedding loaded')
            src_emb = torch.load(opt.pre_word_vecs_enc)
        if opt.pre_word_vecs_dec is not None:
            tgt_emb = torch.load(opt.pre_word_vecs_dec)
        return src_emb, tgt_emb

    def save_data(self, inp, outp, out_tensor):
        print(' activating th hook : ')
