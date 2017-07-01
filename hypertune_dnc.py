from __future__ import division
import time
import train
import option_parse
from numpy.random import uniform, randint, choice
import torch


def check_params(opts, prev_opts):
    stds = {'dropout': .02, 'lr': 10**-6, 'lr_decay': .1, 'start_decay_at': 5,
            'input_feed': 0, 'curriculum': 0, 'brnn': 0, 'layers': 0,
            'read_heads': 1, 'mem_slots': 5, 'mem_size': 50}

    if len(prev_opts) == 0:
        print(' new_opts ')
        print(opts)

        return True

    for prev_opt in prev_opts:
        same_vals = True
        for o in opts:
            if isinstance(opts[o], str) or isinstance(opts[o], bool):
                if opts[o] != prev_opt[o]:
                    same_vals = False
            elif o in stds:
                if opts[o] < prev_opt[o] - stds[o] or opts[o] > prev_opt[o] + stds[o]:
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
            parser = option_parse.get_parser()
            opt = parser.parse_args()

            opt.brnn = randint(2)
            opt.dropout = round(uniform(.1, .5), 2)
            opt.learning_rate = round(uniform(1e-5, 5e-3), 6)
            opt.learning_rate_decay = round(uniform(.3, .8), 2)
            opt.start_decay_at = randint(8, 32)
            opt.curriculum = randint(2, 10)
            opt.input_feed = uniform() // .3

            opt.share_M = randint(2)
            opt.attn = uniform() // .7

            opt.word_vec_size = int(choice([200, 300, 400, 500]))
            opt.rnn_size = int(choice([200, 300, 400, 500]))
            opt.layers = randint(2) + 1

            ok_size = False
            while not ok_size:
                opt.read_heads = randint(1, 4)
                opt.mem_slots = randint(10, 50)
                opt.mem_size = randint(50, 500)
                ok_size = opt.read_heads * opt.mem_slots * opt.mem_size < 6000

            opt_dict = vars(opt)

            ok_params = check_params(opt_dict, prev_opts)

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
            prev_opts += [opt_dict]
            torch.save(prev_opts, opt.prev_opts)

        tries += 1
