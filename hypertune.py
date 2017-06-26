from __future__ import division
import time
import train
import option_parse
from numpy.random import uniform, randint, choice
import torch


def check_params(opts, prev_opts):
    stds = {'dropout': .02, 'lr': 10**-6, 'lr_decay': .1, 'start_decay_at': 5, 'attn': 0,
            'cat_mo_spec': 0, 'mem_slots': 10, 'mem_size': 10, 'read_heads': 0, 'hops': 1,
            'input_feed': 0, 'curriculum': 0, 'share_M': 0, 'brnn': 0, 'layers': 0, 'word_vec_size': 0,
            'rnn_size': 0}

    if len(prev_opts) == 0:
        print(' new_opts ')
        print(opts)

        return True

    for prev_opt in prev_opts:
        same_vals = True
        for o in prev_opt:
            if isinstance(opts[o], str) or isinstance(opts[o], bool):
                if opts[o] != prev_opt[o]:
                    same_vals = False

            elif opts[o] < prev_opt[o] - stds[o] or opts[o] > prev_opt[o] + stds[o]:
                print(' checking %s : %.4f  ;;;  prev: %.4f  +- std : %.2f' % (str(o),
                                                                               opts[o], prev_opt[o],
                                                                               stds[o]))
                same_vals = False

        if same_vals:

            print(' --- no new options ')
            print(opts)
            return False
    print(' ---- new options')
    print(opts)
    return True


if __name__ == "__main__":
    parser = option_parse.get_parser()
    opt = parser.parse_args()

    tries = 0
    low_ppl = 100000
    f = open('logs/hypertune_res_' + str(opt.mem), 'a')
    print(' data: ' + str(opt.data), file=f)
    f.close()

    # prev_opts: {mem_type: list of {opt : value}}
    if opt.prev_opts:
        try:
            prev_opts = torch.load(opt.prev_opts)
        except FileNotFoundError:
            prev_opts = []
    else:
        prev_opts = []

    while True:  # low_ppl > 4 or tries < 64:
        ok_params = False
        while not ok_params:

            opt.brnn = randint(2)
            opt.share_M = randint(2)
            opt.dropout = round(uniform(.1, .5), 2)
            opt.learning_rate = 5 ** uniform(-5, -1)
            opt.learning_rate_decay = round(uniform(.3, .8), 2)
            opt.start_decay_at = randint(8, 32)
            opt.curriculum = randint(2, 10)
            opt.attn = uniform() // .7
            opt.input_feed = uniform() // .3
            #opt.not_optim_tgt_emb = uniform() // .5

            opts = {'share_M': opt.share_M, 'dropout': opt.dropout, 'lr': opt.learning_rate,
                    'lr_decay': opt.learning_rate_decay, 'start_decay_at': opt.start_decay_at,
                    'attn': opt.attn, 'input_feed': opt.input_feed, 'curriculum': opt.curriculum}

            if 'nse' in opt.mem.split('_'):
                opt.cat_mo_special = uniform() // .5 == 1
                opts['cat_mo_spec'] = opt.cat_mo_special

            if 'dnc' in opt.mem.split('_'):

                ok_size = False

                opt.word_vec_size = int(choice([200, 300, 400, 500]))
                opt.rnn_size = int(choice([200, 300, 400, 500]))
                opt.layers = randint(2) + 1
                while not ok_size:
                    opt.read_heads = randint(1, 4)
                    opt.mem_slots = randint(10, 50)
                    opt.mem_size = randint(50, 500)
                    ok_size = opt.read_heads * opt.mem_slots * opt.mem_size < 6000

                opts['mem_slots'] = opt.mem_slots
                opts['mem_size'] = opt.mem_size
                opts['read_heads'] = opt.read_heads

            if 'n2n' == opt.mem.split('_')[0]:
                opt.nr_of_hops = randint(4, 20)
                opts['hops'] = opt.nr_of_hops

            ok_params = check_params(opts, prev_opts)

        print(' start try: ' + str(tries))

        train.opt = opt
        cur_ppl, epoch, trn_ppls, val_ppls, opt, nparams = train.main()
        f = open('logs/hypertune_res_' + str(opt.mem), 'a')
        if cur_ppl < low_ppl:
            low_ppl = cur_ppl
            print(' =====  better result ====\n', file=f)
        print('low ppl: %f \n number of params: %d \n epoch: %d\n train ppls: %s \n vaild ppls: %s \n'
              % (cur_ppl, nparams, epoch, str(trn_ppls), str(val_ppls)), file=f)
        print(opt, file=f)
        print('\n===========================================================\n\n', file=f)
        f.close()

        if opt.prev_opts:
            prev_opts += [opts]
            torch.save(prev_opts, opt.prev_opts)

        tries += 1
