import torch
import os
import json
from torch import Tensor
from torch import nn
from typing import Dict
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    This layer apply multi-head attention to sentence embedding
    """
    def __init__(self,
                 query_dim: int,
                 key_dim: int,
                 num_units: int,
                 dropout_p: float = 0.5,
                 h: int = 8):
        super(MultiHeadAttention, self).__init__()

        self.config_keys = ['query_dim',
                            'key_dim',
                            'num_units',
                            'dropout_p',
                            'h']

        if query_dim != key_dim:
            raise ValueError("query_dim and key_dim must be the same")
        if num_units % h != 0:
            raise ValueError("num_units must be dividable by h")
        if query_dim != num_units:
            raise ValueError("to employ residual connection, the number of "
                             "query_dim and num_units must be the same")

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.num_units = num_units
        self.h = h
        self.dropout_p = dropout_p

        self.key_tensor = torch.tensor(self.key_dim, requires_grad=False).float()
        self.query_layer = nn.Linear(self.query_dim, self.num_units, bias=False)
        self.key_layer = nn.Linear(self.key_dim, self.num_units, bias=False)
        self.value_layer = nn.Linear(self.key_dim, self.num_units, bias=False)
        self.bn = nn.BatchNorm1d(self.num_units)
        self.ln = nn.LayerNorm(self.num_units)
        self.dropout_layer = nn.Dropout(self.dropout_p)

    def forward(self, features: Dict[str, Tensor]):
        # query: [bs, max_seq_len, hidden_dim]
        query = features['token_embeddings']
        # print('multi head attention query size:', query.shape)
        keys = features['token_embeddings']
        mask = features['attention_mask']
        bs = mask.shape[0]
        max_seq_len = mask.shape[1]
        mask = mask.unsqueeze(1).expand(bs, max_seq_len, max_seq_len)
        Q = self.query_layer(query)
        # print('Before Q size:', Q.shape)
        K = self.key_layer(keys)
        V = self.value_layer(keys)

        l = Q.size(0)

        # split each Q, K and V into h different values from dim 2
        # and then merge them back together in dim 0
        chunk_size = int(self.num_units / self.h)
        Q = torch.cat(Q.split(split_size=chunk_size, dim=2), dim=0)
        K = torch.cat(K.split(split_size=chunk_size, dim=2), dim=0)
        V = torch.cat(V.split(split_size=chunk_size, dim=2), dim=0)

        # calculate QK^T
        # print('multi head attention Q size:', Q.shape)
        attention = torch.matmul(Q, K.transpose(1, 2))
        # normalize with sqrt(dk)
        attention = attention / torch.sqrt(self.key_tensor).cuda()

        if mask is not None:
          mask = mask.repeat(self.h, 1, 1)
          # print('multi head attention mask size:', mask.shape)
          # print('multi head attention attention size:', attention.shape)
          attention.masked_fill_(mask == 0, -float('inf'))

        attention = F.softmax(attention, dim=-1)
        # apply dropout
        attention = self.dropout_layer(attention)
        # multiplyt it with V
        attention = torch.matmul(attention, V)
        # convert attention back to its input original size
        restore_chunk_size = int(attention.size(0) / self.h)
        attention = torch.cat(
            attention.split(split_size=restore_chunk_size, dim=0), dim=2)
        # residual connection
        attention += query
        # apply batch normalization
        #attention = self.bn(attention.transpose(1, 2)).transpose(1, 2)
        # apply layer normalization
        attention = self.ln(attention)
        # print('attention size:', attention.shape)

        features.update({'token_embeddings': attention})
        return features

    def get_sentence_embedding_dimension(self):
        return self.key_dim

    def get_max_seq_len(self):
        return self.max_seq_len

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return MultiHeadAttention(**config)
