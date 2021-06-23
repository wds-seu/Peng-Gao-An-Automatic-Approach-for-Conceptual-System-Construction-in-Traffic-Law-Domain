import torch
from torch import Tensor
from torch import nn
from torch import functional as F
from typing import Union, Tuple, List, Iterable, Dict
import os
import json
from ..util import fullname, import_from_string


class TensorLayer(nn.Module):
    """
    This layer apply tensor layer to aggregrate left and right context embedding
    """
    def __init__(self,
                 batch_size: int,
                 sentence_embedding_dimension: int,
                 num_labels: int = 2,
                 seq_len: int = 128,
                 add_bias1: bool = True,
                 add_bias2: bool = True):
        super(TensorLayer, self).__init__()

        self.config_keys = ['batch_size',
                            'sentence_embedding_dimension',
                            'num_labels',
                            'seq_len']
        self.batch_size = batch_size
        self.sentence_embedding_dimension = sentence_embedding_dimension
        self.seq_len = seq_len
        self.num_labels = num_labels
        self.add_bias1 = add_bias1
        self.add_bias2 = add_bias2
        self.bilinear_W = nn.Parameter(torch.randn(sentence_embedding_dimension+add_bias1,
                                                   num_labels,
                                                   sentence_embedding_dimension+add_bias2))
        self.bias1_ones = nn.Parameter(torch.ones((batch_size, seq_len, 1)))
        self.bias2_ones = nn.Parameter(torch.ones((batch_size, seq_len, 1)))
        nn.init.orthogonal(self.bilinear_W)

    def forward(self, features: Dict[str, Tensor]):
        reps = [v for k, v in features.items() if k == 'sentence_embedding']
        assert len(reps) == 2
        rep_a, rep_b = reps
        if self.add_bias1:
            rep_a = torch.cat([rep_a, self.bias1_ones], dim=2)
        if self.add_bias2:
            rep_b = torch.cat([rep_b, self.bias2_ones], dim=2)
        output_shape = [self.batch_size, self.seq_len, self.num_labels, self.seq_len]
        lin = torch.matmul(rep_a.resize([-1,
                                         self.sentence_embedding_dimension+self.add_bias1]),
                           self.bilinear_W)
        bilin = torch.matmul(lin.resize([self.batch_size,
                                         self.seq_len,
                                         self.sentence_embedding_dimension+self.add_bias2]),
                             torch.transpose(rep_b, 1, 2))
        bilin = bilin.resize([-1, self.num_labels, self.seq_len])
        pairwise_scores = bilin.resize(output_shape)
        softmax = torch.nn.Softmax(dim=2)
        result = softmax(pairwise_scores)
        result = result.resize([0, 1, 3, 2])
        output = self.logsumexp(result, [1, 2])
        features.update({'tensor_scores': output})

        return features

    def logsumexp(self, x, dim=None, keepdim=False):
        if dim is None:
            x, dim = x.view(-1), 0
        xm, _ = torch.max(x, dim, keepdim=True)
        x = torch.where(
            (xm == float('inf')) | (xm == float('inf')),
            xm,
            xm + torch.log(torch.sum(torch.exp(x-xm), dim, keepdim=True))
        )

        return x if keepdim else x.squeeze(dim)

    def get_sentence_embedding_dimension(self) -> int:
        return self.sentence_embedding_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        model = TensorLayer(**config)
        return model
