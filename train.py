from __future__ import division

import onmt
import memories
import onmt.Markdown
import onmt.modules
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import option_parse
from math import isnan


def NMTCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[onmt.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.gpus:
        crit.cuda()
    return crit


def memoryEfficientLoss(outputs, targets, generator, crit, eval=False):
    # compute generations one piece at a time
    num_correct, loss = 0, 0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)

    batch_size = outputs.size(1)
    outputs_split = torch.split(outputs, opt.max_generator_batches)
    targets_split = torch.split(targets, opt.max_generator_batches)
    for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
        out_t = out_t.view(-1, out_t.size(2))
        scores_t = generator(out_t)
        loss_t = crit(scores_t, targ_t.view(-1))
        pred_t = scores_t.max(1)[1]
        num_correct_t = pred_t.data.eq(targ_t.data) \
                                   .masked_select(
                                       targ_t.ne(onmt.Constants.PAD).data) \
                                   .sum()
        num_correct += num_correct_t
        loss += loss_t.data[0]
        if not eval:
            loss_t.div(batch_size).backward()

    grad_output = None if outputs.grad is None else outputs.grad.data
    return loss, grad_output, num_correct


def eval(model, criterion, data):
    total_loss = 0
    total_words = 0
    total_num_correct = 0

    model.eval()
    for i in range(len(data)):
        # exclude original indices
        batch = data[i][:-1]
        outputs = model(batch)
        # exclude <s> from targets
        targets = batch[1][1:]
        loss, _, num_correct = memoryEfficientLoss(
            outputs, targets, model.generator, criterion, eval=True)
        total_loss += loss
        total_num_correct += num_correct
        total_words += targets.data.ne(onmt.Constants.PAD).sum()

    model.train()
    return total_loss / total_words, total_num_correct / total_words


def trainModel(model, trainData, validData, dataset, optim):
    print(model)
    model.train()

    low_ppl = 10**8
    tollerance = 0
    best_e = 0
    trn_ppls = []
    val_ppls = []

    # Define criterion of each GPU.
    criterion = NMTCriterion(dataset['dicts']['tgt'].size())

    start_time = time.time()

    def trainEpoch(epoch):

        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        # Shuffle mini batch order.
        batchOrder = torch.randperm(len(trainData))

        total_loss, total_words, total_num_correct = 0, 0, 0
        report_loss, report_tgt_words = 0, 0
        report_src_words, report_num_correct = 0, 0
        start = time.time()
        for i in range(len(trainData)):

            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            # Exclude original indices.
            batch = trainData[batchIdx][:-1]

            model.zero_grad()
            outputs = model(batch)
            # Exclude <s> from targets.
            targets = batch[1][1:]
            loss, gradOutput, num_correct = memoryEfficientLoss(
                outputs, targets, model.generator, criterion)

            outputs.backward(gradOutput)

            # Update the parameters.
            optim.step()

            num_words = targets.data.ne(onmt.Constants.PAD).sum()
            report_loss += loss
            report_num_correct += num_correct
            report_tgt_words += num_words
            report_src_words += batch[0][1].data.sum()
            total_loss += loss
            total_num_correct += num_correct
            total_words += num_words
            if i % opt.log_interval == -1 % opt.log_interval:
                print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f;\t %6.0f s elapsed") %
                      (epoch, i + 1, len(trainData),
                       report_num_correct / report_tgt_words * 100,
                       math.exp(report_loss / report_tgt_words),
                       #report_src_words / (time.time() - start),
                       #report_tgt_words / (time.time() - start),
                       time.time() - start_time))

                report_loss, report_tgt_words = 0, 0
                report_src_words, report_num_correct = 0, 0
                start = time.time()
            if isnan(loss):
                break
        return total_loss / total_words, total_num_correct / total_words

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')
        checkpoint = None

        #  (1) train for one epoch on the training set
        train_loss, train_acc = trainEpoch(epoch)
        train_ppl = math.exp(min(train_loss, 100))
        print('Train perplexity: %g' % train_ppl)
        print('Train accuracy: %g' % (train_acc * 100))

        #  (2) evaluate on the validation set
        valid_loss, valid_acc = eval(model, criterion, validData)
        valid_ppl = math.exp(min(valid_loss, 100))
        print('Validation perplexity: %g' % valid_ppl)
        print('Validation accuracy: %g' % (valid_acc * 100))
        #valid_ppl = 0

        trn_ppls += [train_ppl]
        val_ppls += [valid_ppl]

        if valid_ppl < low_ppl:
            low_ppl = valid_ppl
            best_e = epoch

        #  (3) update the learning rate
            optim.updateLearningRate(valid_ppl, epoch)

            model_state_dict = (model.module.state_dict() if len(opt.gpus) > 1
                                else model.state_dict())
            model_state_dict = {k: v for k, v in model_state_dict.items()
                                if 'generator' not in k}
            generator_state_dict = (model.generator.module.state_dict()
                                    if len(opt.gpus) > 1
                                    else model.generator.state_dict())
            #  (4) drop a checkpoint
            checkpoint = {
                'model': model_state_dict,
                'generator': generator_state_dict,
                'dicts': dataset['dicts'],
                'opt': opt,
                'epoch': epoch,
                'optim': optim
            }
            torch.save(checkpoint,
                       '%s_%s.pt'  # _ppl_%.2f_e%d.pt'
                       % (opt.save_model, opt.mem))  # , valid_ppl, epoch))
            tollerance = 0

        elif tollerance > 1 or isnan(valid_ppl):
            return low_ppl, best_e, trn_ppls, val_ppls, checkpoint
        else:
            low_ppl = valid_ppl
            tollerance += 1

    return low_ppl, best_e, trn_ppls, val_ppls, checkpoint


def main():
    if torch.cuda.is_available() and not opt.gpus:
        print("WARNING: You have a CUDA device, so you should probably run with -gpus 0")

    if opt.gpus:
        cuda.set_device(opt.gpus[0])

    print(opt)

    print("Loading data from '%s'" % opt.data)

    dataset = torch.load(opt.data)
    dict_checkpoint = (opt.train_from if opt.train_from
                       else opt.train_from_state_dict)
    if dict_checkpoint:
        print('Loading dicts from checkpoint at %s' % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint)
        dataset['dicts'] = checkpoint['dicts']

    trainData = onmt.Dataset(dataset['train']['src'],
                             dataset['train']['tgt'], opt.batch_size, opt.gpus,
                             data_type=dataset.get("type", "text"))
    validData = onmt.Dataset(dataset['valid']['src'],
                             dataset['valid']['tgt'], opt.batch_size, opt.gpus,
                             volatile=True,
                             data_type=dataset.get("type", "text"))

    dicts = dataset['dicts']
    print(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
    print(' * number of training sentences. %d' %
          len(dataset['train']['src']))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')

    if opt.mem is None:
        if opt.encoder_type == "text":
            encoder = onmt.Models.Encoder(opt, dicts['src'])
        elif opt.encoder_type == "img":
            encoder = onmt.modules.ImageEncoder(opt)
            assert("type" not in dataset or dataset["type"] == "img")
        else:
            print("Unsupported encoder type %s" % (opt.encoder_type))

        decoder = onmt.Models.Decoder(opt, dicts['tgt'])

        model = onmt.Models.NMTModel(encoder, decoder)

    else:
        model = memories.memory_model.MemModel(opt, dicts)

        encoder = model
        decoder = model

    generator = nn.Sequential(
        nn.Linear(opt.word_vec_size, dicts['tgt'].size()),
        nn.LogSoftmax())

    if opt.train_from:
        print('Loading model from checkpoint at %s' % opt.train_from)
        chk_model = checkpoint['model']
        generator_state_dict = chk_model.generator.state_dict()
        model_state_dict = {k: v for k, v in chk_model.state_dict().items()
                            if 'generator' not in k}
        model.load_state_dict(model_state_dict)
        generator.load_state_dict(generator_state_dict)
        opt.start_epoch = checkpoint['epoch'] + 1

    if opt.train_from_state_dict:
        print('Loading model from checkpoint at %s'
              % opt.train_from_state_dict)
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
        opt.start_epoch = checkpoint['epoch'] + 1

    if len(opt.gpus) >= 1:
        model.cuda()
        generator.cuda()
    else:
        model.cpu()
        generator.cpu()

    if len(opt.gpus) > 1:
        model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)
        generator = nn.DataParallel(generator, device_ids=opt.gpus, dim=0)

    model.generator = generator

    if not opt.train_from_state_dict and not opt.train_from:
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

        # encoder.load_pretrained_vectors(opt)
        # decoder.load_pretrained_vectors(opt)

        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at
        )
    else:
        print('Loading optimizer from checkpoint:')
        optim = checkpoint['optim']
        print(optim)

    optim.set_parameters(model.parameters())

    if opt.train_from or opt.train_from_state_dict:
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    if opt.gather_net_data:
        # , opt.n_samples)
        return gather_data(model, validData, dataset['dicts'])

    low_ppl, best_e, trn_ppls, val_ppls, checkpoint = trainModel(
        model, trainData, validData, dataset, optim)
    return low_ppl, best_e, trn_ppls, val_ppls, checkpoint, opt, nParams


def gather_data(model, data, dicts):  # , n_samples):
    print(' gathering  data ... ')

    srcData = []
    tgtData = []
    criterion = NMTCriterion(dicts['tgt'].size())
    # criterion = NMTCriterion(dicts['vdict'].size())

    total_loss = 0
    total_words = 0
    total_num_correct = 0

    model.eval()

    sample_idxs = torch.Tensor(opt.n_samples).uniform_(0, len(data)).long()
    for i in sample_idxs:
        batch = data[i][:-1]  # exclude original indices
        srcData += [batch[0][0]]
        tgtData += [batch[1]]
        if batch[0][0].size(1) != opt.batch_size:
            continue
        # torch.save(batch, 'data/net_data/batch')

        outputs = model(batch)
        targets = batch[1][1:]  # exclude <s> from targets
        loss, _, num_correct = memoryEfficientLoss(
            outputs, targets, model.generator, criterion, eval=True)
        total_loss += loss
        total_num_correct += num_correct
        total_words += targets.data.ne(onmt.Constants.PAD).sum()

    model.train()
    ppl = math.exp(min(total_loss / total_words, 100))
    print(' perplexity: %g' % ppl)
    net_data = {'modules': model.get_dump_data(),
                'dicts': dicts,
                'data': (srcData, tgtData)}

    torch.save(net_data, 'data/net_data/tst_ipy.pt')

    return net_data
    # torch.save(net_data, 'data/net_data/%s.%s.pt' %
    #           (opt.mem, '.'.join(opt.data[5:].split('.')[:2])))


if __name__ == "__main__":
    parser = option_parse.get_parser()
    opt = parser.parse_args()
    opt.gpus = [0]

    main()
