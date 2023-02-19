import numpy as np
import torch


UNK, PAD = '<UNK>', '<PAD>'


class Full_All_MOOC(object):
    def __init__(self):
        self.processed_train_data = "../data_preprocess/processed_data/processed_full_mooc_data/all_labeled_data/uncut_train_data.txt"
        self.processed_test_data = "../data_preprocess/processed_data/processed_full_mooc_data/all_labeled_data/uncut_test_data.txt"
        self.label_list = [str(i) for i in range(8)]


class Full_Medical_MOOC(object):
    def __init__(self):
        self.processed_train_data = "../data_preprocess/processed_data/processed_full_mooc_data/医药卫生类_labeled_data/uncut_train_data.txt"
        self.processed_test_data = "../data_preprocess/processed_data/processed_full_mooc_data/医药卫生类_labeled_data/uncut_test_data.txt"
        self.label_list = [str(i) for i in range(8)]


class Full_Art_MOOC(object):
    def __init__(self):
        self.processed_train_data = "../data_preprocess/processed_data/processed_full_mooc_data/文学艺术类_labeled_data/uncut_train_data.txt"
        self.processed_test_data = "../data_preprocess/processed_data/processed_full_mooc_data/文学艺术类_labeled_data/uncut_test_data.txt"
        self.label_list = [str(i) for i in range(8)]


class Full_Science_MOOC(object):
    def __init__(self):
        self.processed_train_data = "../data_preprocess/processed_data/processed_full_mooc_data/理工类_labeled_data/uncut_train_data.txt"
        self.processed_test_data = "../data_preprocess/processed_data/processed_full_mooc_data/理工类_labeled_data/uncut_test_data.txt"
        self.label_list = [str(i) for i in range(8)]


class Full_Computer_MOOC(object):
    def __init__(self):
        self.processed_train_data = "../data_preprocess/processed_data/processed_full_mooc_data/计算机类_labeled_data/uncut_train_data.txt"
        self.processed_test_data = "../data_preprocess/processed_data/processed_full_mooc_data/计算机类_labeled_data/uncut_test_data.txt"
        self.label_list = [str(i) for i in range(8)]


# 把一个句子分词，然后根据词汇表各个词对应的位置（ID），转换词序列为ID序列
def sentence_to_token_ids(line, vocab_dict):
    words = line.strip().split(' ')[1:]
    word_ids = [vocab_dict.get(w, vocab_dict.get(UNK)) for w in words]
    return word_ids


#  数据准备函数
def create_id_dataset(vocab_dict, processed_train_data, processed_test_data, max_seq_length):
    # 从原始文本数据得到学习语料
    train_data, test_data = [], []
    processed_data = [processed_train_data, processed_test_data]
    real_max_length = 0
    real_total_length = []
    for index, data in enumerate(processed_data):
        # 获取最大序列长度
        with open(data, "r", encoding='utf-8') as f_reader:
            for line in f_reader.readlines():
                word_ids = sentence_to_token_ids(line, vocab_dict)
                label = line.split("__")[-1].split(" ")[0]
                real_total_length.append(len(word_ids))
                # 记录真实数据的最大长度，方便设置max_seq_length
                if len(word_ids) > real_max_length:
                    real_max_length = len(word_ids)
                # 进行短填长切
                if len(word_ids) > max_seq_length:
                    word_ids = word_ids[:max_seq_length]
                else:
                    word_ids = word_ids + [vocab_dict.get(PAD)] * (max_seq_length - len(word_ids))
                if index == 0:
                    train_data.append((word_ids, int(label)))
                else:
                    test_data.append((word_ids, int(label)))

    real_avg_length = np.mean(np.asarray(real_total_length))
    print("最大序列长度为:{},平均序列长度为:{}".format(real_max_length, real_avg_length))
    return train_data, test_data


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        return x, y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


if __name__ == '__main__':
    data_class = Full_All_MOOC()
