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
    for (m, a) in [(0, 1), (1, 0)]:
        opt.attn = a
        opt.brnn = 1
        opt.share_M = 1
        opt.mem = 'dnc_dnc'
        for n_exp in range(3):
            experiment(opt, n_exp)


def nse_nse():
    opt.mem = 'nse_nse'
    for br in [0, 1]:
        opt.brnn = br
        for n_exp in range(3):
            experiment(opt, n_exp)


if __name__ == "__main__":
    parser = option_parse.get_parser()
    opt = parser.parse_args()
    opt.seed = randint(1, 100)

    # nse_nse()
    # lstmdnc_vv()
    dnc_dnc()
