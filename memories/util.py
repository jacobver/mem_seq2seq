import torch
from torch.autograd import Variable
from onmt import Dict
from onmt import Constants


'''
input: vector (batch size x vector length)
       matrix (batch size x sequence lenth x word vector length)
'''


class EmbMem(torch.nn.Module):

    def __init__(self, input_size, act):
        super(EmbMem, self).__init__()

        if act == 'relu':
            activation_layer = torch.nn.ReLU()
        elif act == 'softmax':
            activation_layer = torch.nn.Softmax()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size),
            torch.nn.Linear(input_size, input_size),
            activation_layer)

    def forward(self, input):
        ret = []

        # input size is size of M (b x s x w)
        for w in input.split(1, 1):
            ret += [self.mlp(w.squeeze(1))]

        return torch.stack(ret).transpose(0, 1)


def flip(t, dim=0):

    idxs = Variable(torch.Tensor(
        range(t.size(dim) - 1, -1, -1)).long(), requires_grad=False)
    if t.is_cuda:
        idxs = idxs.cuda()
    return t.index_select(dim, idxs)


def similarity(vec, mat, eps=1e-6):
    vec_norm = torch.norm(vec, 2, 1)
    mat_norm = torch.norm(mat, 2, 2)
    normalized_vec = torch.div(vec, vec_norm.expand_as(vec).clamp(min=eps))
    normalized_mat = torch.div(mat, mat_norm.expand_as(mat).clamp(min=eps))
    return torch.bmm(normalized_mat, normalized_vec.unsqueeze(2)).squeeze(2)

# same as


def cosine_similarity(vec, mat, eps=1e-6):
    w12 = torch.bmm(mat, vec.unsqueeze(2))
    w1 = torch.norm(vec, 2, 1)
    w2 = torch.norm(mat, 2, 2)
    return (w12 / (torch.bmm(w2, w1.unsqueeze(2)).clamp(min=eps))).squeeze()


def parse_word_vecs(vecs_text_file, dimension):
    word_vecs = {}
    with open(vecs_text_file) as f:
        for line in f.readlines():
            word_vec_line = line.split(' ')
            vec = [float(element) for element in word_vec_line[1:]]
            assert len(vec) == dimension
            word_vecs[word_vec_line[0]] = torch.Tensor(vec)

    vecs_pt_file = '.'.join(vecs_text_file.split('.')[:-1] + ['pt'])
    torch.save(word_vecs, vecs_pt_file)


def make_pre_embedding(dict_file, vectors, vec_size):

    dictionary = Dict()
    dictionary.loadFile(dict_file)
    vecs = vectors if isinstance(vectors, dict) else torch.load(vectors)
    assert vecs['any'].size(0) == vec_size
    embedding = torch.Tensor(dictionary.size(), vec_size)
    for i in range(dictionary.size()):
        word = dictionary.getLabel(i)
        if word == Constants.PAD_WORD:
            embedding[i] = torch.zeros(vec_size)
        else:
            embedding[i] = vecs[word] if word in vecs else vecs[Constants.UNK_WORD]

    emb_file = '.'.join(dict_file.split('.')[:-1] + ['emb.pt'])
    torch.save(embedding, emb_file)
    return embedding
