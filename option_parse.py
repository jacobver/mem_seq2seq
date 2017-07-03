import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        add_help=False, description=' memory seq 2 seq ')

    # Data options

    parser.add_argument('-data', required=True,
                        help='Path to the *-train.pt file from preprocess.py')
    parser.add_argument('-save_model', default='models/model',
                        help="""Model filename (the model will be saved as
                        <save_model>_epochN_PPL.pt where PPL is the
                        validation perplexity""")
    parser.add_argument('-train_from_state_dict', default='', type=str,
                        help="""If training from a checkpoint then this is the
                        path to the pretrained model's state_dict.""")
    parser.add_argument('-train_from', default='', type=str,
                        help="""If training from a checkpoint then this is the
                        path to the pretrained model.""")

    # Model options

    parser.add_argument('-layers', type=int, default=2,
                        help='Number of layers in the LSTM encoder/decoder')
    parser.add_argument('-rnn_size', type=int, default=500,
                        help='Size of LSTM hidden states')
    parser.add_argument('-word_vec_size', type=int, default=500,
                        help='Word embedding sizes')
    parser.add_argument('-input_feed', type=int, default=1,
                        help="""Feed the context vector at each time step as
                        additional input (via concatenation with the word
                        embeddings) to the decoder.""")
    parser.add_argument('-attn', type=int, default=0)
    parser.add_argument('-brnn', type=int, default=0,
                        help='Use a bidirectional encoder')
    parser.add_argument('-brnn_merge', default='concat',
                        help="""Merge action for the bidirectional hidden states:
                        [concat|sum]""")

    # Optimization options
    parser.add_argument('-encoder_type', default='text',
                        help="Type of encoder to use. Options are [text|img].")
    parser.add_argument('-batch_size', type=int, default=64,
                        help='Maximum batch size')
    parser.add_argument('-max_generator_batches', type=int, default=32,
                        help="""Maximum batches of words in a sequence to run
                        the generator on in parallel. Higher is faster, but uses
                        more memory.""")
    parser.add_argument('-epochs', type=int, default=79,
                        help='Number of training epochs')
    parser.add_argument('-start_epoch', type=int, default=1,
                        help='The epoch from which to start')
    parser.add_argument('-param_init', type=float, default=0.1,
                        help="""Parameters are initialized over uniform distribution
                        with support (-param_init, param_init)""")
    parser.add_argument('-optim', default='adam',
                        help="Optimization method. [sgd|adagrad|adadelta|adam]")
    parser.add_argument('-max_grad_norm', type=float, default=5,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to max_grad_norm""")
    parser.add_argument('-dropout', type=float, default=0.3,
                        help='Dropout probability; applied between LSTM stacks.')
    parser.add_argument('-curriculum', action="store_true",
                        help="""For this many epochs, order the minibatches based
                        on source sequence length. Sometimes setting this to 1 will
                        increase convergence speed.""")
    parser.add_argument('-extra_shuffle', action="store_true",
                        help="""By default only shuffle mini-batch order; when true,
                        shuffle and re-assign mini-batches""")

    # learning rate
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Starting learning rate. If adagrad/adadelta/adam is
                        used, then this is the global learning rate. Recommended
                        settings: sgd = 1, adagrad = 0.1,
                        adadelta = 1, adam = 0.001""")
    parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set or (ii) epoch has gone past
                        start_decay_at""")
    parser.add_argument('-start_decay_at', type=int, default=23,
                        help="""Start decaying every epoch after and including this
                        epoch""")

    # pretrained word vectors
    parser.add_argument('-pre_word_vecs_enc',
                        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the encoder side.
                        See README for specific formatting instructions.""")
    parser.add_argument('-pre_word_vecs_dec',
                        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the decoder side.
                        See README for specific formatting instructions.""")

    # GPU
    parser.add_argument('-gpus', default=[0], nargs='+', type=int,
                        help="Use CUDA on the listed devices.")

    parser.add_argument('-log_interval', type=int, default=50,
                        help="Print stats at this interval.")

    # Memory options
    parser.add_argument('-mem', default=None,
                        help='which type of memory to use, default: None')
    parser.add_argument('-hops', type=int, default=8,
                        help='in case of [n2n]: number of computational hops')
    parser.add_argument('-linear_start', type=int, default=1,
                        help='in case of [n2n]: completely linear model to quickstart training')
    parser.add_argument('-context_sz', type=int, default=1,
                        help='in case of [nse]: number of encoded memories to read')

    # DNC options
    parser.add_argument('-mem_slots', type=int, default=20,
                        help='in case of [dnc]: number of memory slots')
    parser.add_argument('-mem_size', type=int, default=100,
                        help='in case of [dnc]: size of memory slots')
    parser.add_argument('-read_heads', type=int, default=1,
                        help='in case of [dnc]: number of read heads')
    parser.add_argument('-share_M', type=int, default=1,
                        help='whther to share the memory between en- and decoder')

    # hypertune specific
    parser.add_argument('-prev_opts', default=None,
                        help='pkl file with previously tried options')

    # vizualization
    parser.add_argument('-gather_net_data', type=int, default=0,
                        help='save hidden states and memory specific data')
    parser.add_argument('-n_samples', type=int, default=10)

    return parser
