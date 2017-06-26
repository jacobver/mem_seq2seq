import torch
import matplotlib.pyplot as plt
import numpy as np
import train
import argparse


def process_data(net_data):
    # vocab = net_data['dicts']['vdict']
    dicts = (net_data['dicts']['src'], net_data['dicts']['tgt'])

    (src, tgt) = net_data['data']  # ['src']

    plt.ion()
    print('\n == visualize \'%s\' net activity ====\n  1) %s\n  2) %s\n  q) quit' % (
        new_opt.mem, *new_opt.mem.split('_')))
    i = input(' \n\n --> ')
    if i == 'q':
        print()
        return(0)
    else:
        i = int(i) - 1
        net = new_opt.mem.split('_')[i]

    viz_txt = sort_data(
        net, net_data['modules'][i], net_data['data'][i], dicts[i])
    if net == 'nse':
        show_z_data(*viz_txt)
    elif net == 'n2n':
        show_n2n_data(*viz_txt)


def sort_data(net, data, utts, vocab):
    txt = []
    utt_pix = []
    print(' data: ' + str(data))
    for i, utt in enumerate(utts):
        utt_pix += [{param: data[param][i].tolist()
                     for param in data}]
        txt += [vocab.convertToLabels(utt.squeeze().data.tolist(), 0)]

    return utt_pix, txt


def show_z_data(visuals, src_sens):

    for z_map, src_sen in zip(visuals, src_sens):
        z = z_map['z']
        print(' z sz: ' + str(np.shape(z)))
        print(' '.join(src_sen))
        # plt.figure()
        plt.matshow(z)
        plt.xticks(range(len(src_sen)), src_sen)
        plt.yticks(range(len(src_sen)), src_sen)
        plt.xlabel(' memory cell ')
        plt.ylabel(' t ')
        i = input('press enter, or \'q\' to quit : ')
        if i == 'q':
            break
        plt.close()


def show_n2n_data(visuals, src_sens):

    for pix, src_sen in zip(visuals, src_sens):
        print(' p sz: ' + str(np.shape(pix['p'])))
        print(' M sz: ' + str(np.shape(pix['C'])))
        print(' C sz: ' + str(np.shape(pix['M'])))
        print(' '.join(src_sen))
        # plt.figure()
        plt.matshow(pix['p'])
        plt.xticks(range(len(src_sen)), src_sen)
        #plt.yticks(range(len(src_sen)), src_sen)
        plt.xlabel(' memory cell ')
        plt.ylabel(' hops ')
        i = input('press enter, or \'q\' to quit : ')
        if i == 'q':
            break
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str)
    opt = parser.parse_args()
    new_opt = torch.load(opt.model)['opt']
    new_opt.batch_size = 1
    new_opt.train_from_state_dict = opt.model
    new_opt.gather_net_data = 1
    new_opt.n_samples = 10

    train.opt = new_opt
    net_data = train.main()

    process_data(net_data)
