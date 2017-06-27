import torch
import torch.nn as nn
from torch.autograd import Variable
from memories.dnc_controller import Controller
from memories.dnc_memory import Memory
from onmt.modules import GlobalAttention


class DNC(nn.Module):
    def __init__(self, opt):
        super(DNC, self).__init__()

        self.input_feed = opt.input_feed if opt.seq == 'decoder' else 0
        self.rnn_sz = (opt.word_vec_size, None) if opt.layers == 1 else (
            opt.rnn_size, opt.word_vec_size)
        self.layers = opt.layers
        self.net_data = [] if opt.gather_net_data else None

        use_cuda = len(opt.gpus) > 0
        self.memory = Memory(opt.mem_slots, opt.mem_size,
                             opt.read_heads, opt.batch_size, use_cuda)

        input_sz = 2 * opt.word_vec_size if self.input_feed else opt.word_vec_size
        self.controller = Controller(input_sz, opt.word_vec_size, opt.read_heads, opt.rnn_size,
                                     opt.mem_size, opt.batch_size, opt.dropout, opt.layers)

        self.attn = GlobalAttention(
            opt.word_vec_size) if opt.attn and opt.seq == 'decoder' else None

    '''
    input: embedded sequence (seq_sz x batch_sz x word_vec_sz)
    output:
    '''

    def forward(self, emb_utt, hidden, mem, context=None):

        #hidden = self.make_init_hidden(emb_utt, *self.rnn_sz)
        seq_len = emb_utt.size(0)
        # batch_sz = emb_utt.size(1)

        controller_state = hidden

        outputs_time = [[]] * seq_len
        free_gates_time = [[]] * seq_len
        allocation_gates_time = [[]] * seq_len
        write_gates_time = [[]] * seq_len
        read_weights_time = [[]] * seq_len
        write_weights_time = [[]] * seq_len
        usage_vectors_time = [[]] * seq_len

        last_read_vectors = mem['read_vec']
        mem_mat = mem['mem_mat']
        mem_usage = mem['mem_usage']
        read_weight = mem['read_weight']
        write_weight = mem['write_weight']
        pre_vec = mem['pre_vec']
        link_mat = mem['link_mat']

        pre_output, interface = None, None

        # TODO: perform matmul(input, W) before loops
        out = emb_utt[0].clone()
        out.data.zero_()

        for time, emb_w in enumerate(emb_utt.split(1)):
            # print(' emb_w : ' + str(emb_w.size()))
            step_input = emb_w.squeeze(0)
            if self.input_feed:
                step_input = torch.cat((step_input, out), 1)

            pre_output, interface, controller_state = self.controller.process_input(step_input,
                                                                                    last_read_vectors, controller_state)

            usage_vector, write_weight, mem_mat, \
                link_mat, pre_vec = self.memory.write(
                    mem_mat,
                    mem_usage,
                    read_weight,
                    write_weight,
                    pre_vec,
                    link_mat,

                    interface['write_key'],
                    interface['write_strength'],
                    interface['free_gates'],
                    interface['allocation_gate'],
                    interface['write_gate'],
                    interface['write_vector'],
                    interface['erase_vector']
                )

            read_weight, last_read_vectors = self.memory.read(
                mem_mat,
                read_weight,
                interface['read_keys'],
                interface['read_strengths'],
                link_mat,
                interface['read_modes'],
            )

            out = self.controller.final_output(
                pre_output, last_read_vectors).clone()

            if context is not None and self.attn is not None:
                out, attn = self.attn(out, context.t())

            outputs_time[time] = out
            free_gates_time[time] = interface['free_gates'].clone()
            allocation_gates_time[time] = interface['allocation_gate'].clone()
            write_gates_time[time] = interface['write_gate'].clone()
            read_weights_time[time] = read_weight.clone()
            write_weights_time[time] = write_weight.clone()
            usage_vectors_time[time] = usage_vector.clone()

        packed_output = torch.stack(outputs_time)

        # mem = {}
        mem['read_vec'] = last_read_vectors.clone()
        mem['mem_mat'] = mem_mat
        mem['mem_usage'] = usage_vector.clone()
        mem['read_weight'] = read_weight.clone()
        mem['write_weight'] = write_weight.clone()
        mem['pre_vec'] = pre_vec
        mem['link_mat'] = link_mat

        # apply_dict(locals())

        if self.net_data is not None:
            packed_memory_view = {
                'free_gates':       torch.stack(free_gates_time),
                'allocation_gates': torch.stack(allocation_gates_time),
                'write_gates':      torch.stack(write_gates_time),
                'read_weights':     torch.stack(read_weights_time),
                'write_weights':    torch.stack(write_weights_time),
                'usage_vectors':    torch.stack(usage_vectors_time)
            }
            self.net_data += [(packed_memory_view, mem)]

        return packed_output, controller_state, mem  # _memory_view

    def make_init_M(self, batch_sz):
        memory_state = self.memory.init_memory(batch_sz)
        mem = {}
        mem['read_vec'] = memory_state.read_vec
        mem['mem_mat'] = memory_state.mem_mat
        mem['mem_usage'] = memory_state.mem_usage
        mem['read_weight'] = memory_state.read_weight
        mem['write_weight'] = memory_state.write_weight
        mem['pre_vec'] = memory_state.pre_vec
        mem['link_mat'] = memory_state.link_mat
        return mem

    def make_init_hidden(self, inp, l1_sz, l2_sz=None):
        h0 = Variable(inp.data.new(inp.size(1), l1_sz).zero_(),
                      requires_grad=False)
        if l2_sz is not None:
            h1 = Variable(inp.data.new(inp.size(1), l2_sz).zero_(),
                          requires_grad=False)
            return ((h0, h0.clone()), (h1, h1.clone()))
        return (h0, h0.clone())

    def get_net_data(self):
        nd = {'packed_view': [], 'memory': []}
        for (pv, m) in self.net_data:
            nd['packed_view'] += [pv]
            nd['memory'] += [m]

        return nd
