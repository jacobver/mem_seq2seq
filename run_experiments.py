from __future__ import division
import train
import option_parse
import torch
from numpy.random import randint


def experiment(opt, n_exp):
    f = open('logs/experiments_res_' + str(opt.mem), 'a')
    print(' ==============================================================\n', file=f)
    print(' experiment %d - %s ' % (n_exp, str(opt.mem)))
    print(' start experiment: %d' % n_exp, file=f)
    print(' data: ' + str(opt.data), file=f)
    f.close()

    train.opt = opt
    #cur_ppl, epoch, trn_ppls, val_ppls, checkpoint, opt, nparams = 7 * [None]
    cur_ppl, epoch, trn_ppls, val_ppls, checkpoint, opt, nparams = train.main()
    opt = train.opt
    f = open('logs/experiments_res_' + str(opt.mem), 'a')

    print('low ppl: %f \n number of params: %d \n epoch: %d\n train ppls: %s \n vaild ppls: %s \n'
          % (cur_ppl, nparams, epoch, str(trn_ppls), str(val_ppls)), file=f)
    print(opt, file=f)
    print('\n===========================================================\n\n', file=f)
    f.close()

    dict_fn = 'logs/experiment_res_%s.pt' % str(opt.mem)

    try:
        res_dict = torch.load(dict_fn)
        last_exp_n = max(res_dict.keys()) + 1
    except FileNotFoundError:
        res_dict = {}
        last_exp_n = 0

    n_exp += last_exp_n
    res_dict[n_exp] = {}
    res_dict[n_exp]['nparams'] = nparams
    res_dict[n_exp]['trn_ppls'] = trn_ppls
    res_dict[n_exp]['val_ppls'] = trn_ppls
    res_dict[n_exp]['args'] = vars(opt)
    res_dict[n_exp]['checkpoint'] = checkpoint

    torch.save(res_dict, dict_fn)


def lstmdnc_vv():
    opt.share_M = 0
    opt.attn = 1
    opt.brnn = 1

    for mem in ['lstm_dnc', 'dnc_lstm']:
        opt.mem = mem
        for n_exp in range(3):
            experiment(opt, n_exp)


def dnc_dnc():
    for (m, b, a) in [(1, 0, 0), (0, 0, 0), (1, 1, 0), (0, 1, 1)]:
        opt.attn = a
        opt.brnn = b
        opt.share_M = m
        opt.mem = 'dnc_dnc'
        for n_exp in range(3):
            experiment(opt, n_exp)


def nse_nse():
    opt.mem = 'nse_nse'
    for br in [0, 1]:
        opt.brnn = br
        for n_exp in range(3):
            experiment(opt, n_exp)


def n2n_dnclstm():
    for mem in ['n2n_lstm', 'n2n_dnc']:
        for at in [0, 1]:
            opt.mem = mem
            opt.attn = at
            opt.brnn = 0
            opt.share_M = 0
            for n_exp in range(3):
                experiment(opt, n_exp)


def nse_other():
    for mem in ['nse_lstm', 'nse_dnc', 'nse_n2n']:
        for b in [0, 1]:
            opt.mem = mem
            opt.brnn = b
            opt.attn = 1 if mem != 'nse_n2n' else 0
            opt.share_M = 0 if mem != 'nse_n2n' else 1
            for n_exp in range(3):
                experiment(opt, n_exp)


def baseline():

    def go():
        opt.mem = None
        opt.brnn = 1
        opt.attn = 1
        for n_exp in range(3):
            experiment(opt, n_exp)

    go()
    opt.pre_word_vecs_enc = 'data/os_200k_atok_low.src.emb.pt'
    opt.pre_word_vecs_dec = 'data/os_200k_atok_low.tgt.emb.pt'
    opt.word_vec_size = 300
    opt.rnn_size = 300
    go()


if __name__ == "__main__":
    parser = option_parse.get_parser()
    opt = parser.parse_args()
    opt.seed = randint(1, 100)

    # baseline()
    # nse_other()
    # n2n_dnclstm()
    nse_nse()
    # lstmdnc_vv()
    # dnc_dnc()
