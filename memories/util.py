import torch

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
