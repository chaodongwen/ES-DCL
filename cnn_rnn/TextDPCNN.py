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
            np.load(self.embedding_path)["embeddings"].astype('float32')) if self.embedding_path != "" else None  # 预训练词向量
        self.embedding_size = self.embedding_pretrained.size(1) if self.embedding_pretrained is not None else 300

        self.max_seq_length = 40
        self.num_epochs = 30
        self.batch_size = 64
        self.log_interval = 20
        self.eval_interval = 100
        self.early_stop = 2000                                      # 若超过2000step效果还没提升，则提前结束训练
        self.learning_rate = 5e-4

        self.dropout = 0.5                                          # 随机失活
        self.num_filters = 250


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
        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.embedding_size), stride=1)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        # nn.ZeroPad2d括号内的四个参数依次是左右上下填充的个数
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(config.num_filters, config.num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, seq_len, embedding_size]=[64, 150, 128]
        x = x.unsqueeze(1)   # [batch_size, 1, seq_len, embedding_size]=[64, 1, 150, 128]
        x = self.conv_region(x)  # [batch_size, num_filters, seq_len-3+1, 1]=[64, 100, 148, 1]
        x = self.padding1(x)  # [batch_size, num_filters, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, num_filters, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, num_filters, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, num_filters, seq_len-3+1, 1]
        # 由于1/2池化层的存在，文本序列长度会随着block数量的增加呈指数级减少，这会导致序列长度随着网络加深呈金字塔形状
        while x.size()[2] >= 2:
            x = self._block(x)
        # 当x.size()[2]为1时，x.squeeze()将会直接将x.size()[2]去掉
        x = x.squeeze()  # [batch_size, num_filters(250)]
        x = self.fc(x)
        return x

    def _block(self, x):
        # 设进来时x的size为[batch_size, num_filters, len, 1]
        x = self.padding2(x)    # [batch_size, num_filters, len+1, 1]
        px = self.max_pool(x)   # [batch_size, num_filters, (len+1-3)/2 + 1, 1]
        x = self.padding1(px)   # [batch_size, num_filters, (len+1-3)/2 + 3, 1]
        x = F.relu(x)
        x = self.conv(x)        # [batch_size, num_filters, (len+1-3)/2 + 1, 1]
        x = self.padding1(x)    # [batch_size, num_filters, (len+1-3)/2 + 3, 1]
        x = F.relu(x)
        x = self.conv(x)        # [batch_size, num_filters, (len+1-3)/2 + 1, 1]
        # Short Cut
        x = x + px              # [batch_size, num_filters, (len+1-3)/2 + 1, 1]
        return x
