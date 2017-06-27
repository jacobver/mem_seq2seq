import torch
from onmt import Dict
from onmt import Constants

'''
input: vector (batch size x vector length)
       matrix (batch size x sequence lenth x word vector length)
'''


def similarity(vec, mat):
    vec_norm = torch.norm(vec, 2, 1)
    mat_norm = torch.norm(mat, 2, 2)
    normalized_vec = torch.div(vec, vec_norm.expand_as(vec))
    normalized_mat = torch.div(mat, mat_norm.expand_as(mat))
    return torch.bmm(normalized_mat, normalized_vec.unsqueeze(2)).squeeze(2)


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
        word = dictionary.idxToLabel[i]
        if word == Constants.PAD_WORD:
            embedding[i] = torch.zeros(vec_size)
        embedding[i] = vecs[word] if word in vecs else vecs[Constants.UNK_WORD]

    emb_file = '.'.join(dict_file.split('.')[:-1] + ['emb.pt'])
    torch.save(embedding, emb_file)
    return embedding
