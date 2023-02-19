# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.hidden_size = 256                                          # lstm隐藏层
        self.num_layers = 1                                             # lstm层数


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
        self.maxpool = nn.MaxPool1d(config.max_seq_length)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embedding_size, config.num_classes)

    def forward(self, x):
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]
        out, _ = self.lstm(embed)
        # 将64*30*300和64*30*512在第2个维度进行拼接
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        # permute函数用于维度换位，便于对中间一个维度进行最大池化
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out
