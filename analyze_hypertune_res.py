import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='analyze_hypertune_res.py')

parser.add_argument('-fn', default='')


def main():
    parse(args.fn)


def parse(res_f):
    with open(res_f) as f:
        res_f_arrs = [line.split() for line in f]

    results = {}

    for line_arr in res_f_arrs:
        l_key = line_arr[0]
        if l_key == 'low':
            cur_res_key = float(line_arr[2])
            results[cur_res_key] = {}
        if l_key == 'valid':
            results[cur_res_key]['ppls'] = [
                float(line_arr[i]) for i in range(1, len(line_arr))]
        if 'Namespace' in l_key:
            parse_opts()

    return val_ppl, trn_ppl


if __name__ == '__main__':
    args = parser.parse_args()
    main()
