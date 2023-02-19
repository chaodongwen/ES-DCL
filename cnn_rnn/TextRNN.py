# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np


class Config(object):
    def __init__(self, args):
        # 必须要的
        self.vocab_size = args.vocab_size
        self.num_classes = args.num_classes
        self.device = args.device
        self.embedding_path = args.embedding_path
        self.embedding_pretrained = torch.tensor(
            np.load(self.embedding_path)["embeddings"].astype('float32')) if self.embedding_path != "" else None   # 预训练词向量
        self.embedding_size = self.embedding_pretrained.size(1) if self.embedding_pretrained is not None else 300

        self.max_seq_length = 40
        self.num_epochs = 30
        self.batch_size = 64
        self.log_interval = 20
        self.eval_interval = 100
        self.early_stop = 2000                                          # 若超过2000step效果还没提升，则提前结束训练
        self.learning_rate = 5e-4

        self.dropout = 0.5                                              # 随机失活
        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            try:
                self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
            except:
                self.embedding = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.vocab_size - 1)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.vocab_size - 1)
        # 定义模型
        self.lstm = nn.LSTM(config.embedding_size, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        out = self.embedding(x)  # [batch_size, seq_len, embeding]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out

    '''变长RNN，效果差不多，甚至还低了点...'''
    # def forward(self, x):
    #     x, seq_len = x
    #     out = self.embedding(x)
    #     _, idx_sort = torch.sort(seq_len, dim=0, descending=True)  # 长度从长到短排序（index）
    #     _, idx_unsort = torch.sort(idx_sort)  # 排序后，原序列的 index
    #     out = torch.index_select(out, 0, idx_sort)
    #     seq_len = list(seq_len[idx_sort])
    #     out = nn.utils.rnn.pack_padded_sequence(out, seq_len, batch_first=True)
    #     # [batche_size, seq_len, num_directions * hidden_size]
    #     out, (hn, _) = self.lstm(out)
    #     out = torch.cat((hn[2], hn[3]), -1)
    #     # out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    #     out = out.index_select(0, idx_unsort)
    #     out = self.fc(out)
    #     return out
