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
        self.hidden_size = 128                                          # lstm隐藏层
        self.hidden_size2 = 64                                          # 中间层隐藏层
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
        self.tanh = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2)
        self.fc = nn.Linear(config.hidden_size2, config.num_classes)

    def forward(self, x):
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 150, 128]
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[64, 150, 256]

        M = self.tanh(H)  # [64, 150, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [64, 150, 1]
        out = H * alpha  # [64, 150, 256]
        out = torch.sum(out, 1)  # [64, 256]
        out = F.relu(out)
        out = self.fc1(out)  # [64, 64]
        out = self.fc(out)  # [64, 3]
        return out
