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

        self.dropout = 0.5                                               # 随机失活
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)



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
        # 构建三种不同的卷积操作
        self.convs = nn.ModuleList([nn.Conv2d(1, config.num_filters, (k, config.embedding_size)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # 删掉第3个维度，如(64,100,298,1)变成了(64,100,298)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)   # 删掉第3个维度，如(64,100,1)变成了(64,100)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)  # 增加第1个维度，如(64,300,128)变成了(64,1,300,128)
        # 在第1个维度进行叠加，如三个(64,100)进行拼接变成了(64,300)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
