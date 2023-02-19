import os
import argparse
import datetime
import sys
# 引入上层目录
sys.path.append("..")
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import pandas as pd
import csv
# import torchtext.data as data
# import torchtext.datasets as datasets
import torch.nn as nn
from importlib import import_module
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
import data_utils
import time
import pickle as pkl
import jieba
import re
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser(description="Text classifier")

# option
parser.add_argument('--model_name', type=str, default="TextCNN", help='which model to run')
parser.add_argument('--do_train', type=bool, default=True, help='if train')
parser.add_argument('--do_test', type=bool, default=True, help='if test')
parser.add_argument('--task_name', type=str, default="full_all_mooc", help='which task to execute')
parser.add_argument('--k_num', type=int, default=5, help='KFold cross validation')
args = parser.parse_args()


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter):
    # 进入训练模式
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    steps, last_improve_step = 0, 0
    dev_best_loss = float('inf')
    flag = False
    # 提前建立文件夹，以免出现找不到文件夹的错误
    if not os.path.isdir(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    writer = SummaryWriter(log_dir=args.model_save_dir + '/log/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for i in range(1, config.num_epochs + 1):
        for x_train_tensor, y_train_tensor in train_iter:
            if len(y_train_tensor.data.cpu().numpy()) == 1:
                continue
            logit = model(x_train_tensor)
            loss = F.cross_entropy(logit, y_train_tensor)
            # 重置梯度
            model.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1
            # 输出日志信息
            if steps % config.log_interval == 0:
                accuracy = np.mean(torch.max(logit, 1)[1].cpu().numpy() == y_train_tensor.data.cpu().numpy()) * 100
                print("Epoch[{}/{}]\tStep[{}]\tloss:{:.6f}\tacc:{:.2f}%".format(i, config.num_epochs, steps, loss.item(), accuracy))
            # 进行验证
            if steps % config.eval_interval == 0:
                # 进行验证时要切换到验证模式
                model.eval()
                sample_num, dev_corrects, total_dev_loss = 0, 0, 0
                for x_dev_tensor, y_dev_tensor in dev_iter:
                    dev_logit = model(x_dev_tensor)
                    # size_average=False时得到的是一个batch中所有样本总的loss，size_average=True时得到的是一个batch中平均每个样本的loss
                    dev_loss = F.cross_entropy(dev_logit, y_dev_tensor, size_average=False)
                    total_dev_loss += dev_loss.item()
                    sample_num += len(x_dev_tensor)
                    dev_corrects += (torch.max(dev_logit, 1)[1].cpu().numpy() == y_dev_tensor.data.cpu().numpy()).sum()
                dev_accuracy = (dev_corrects / sample_num) * 100
                avg_dev_loss = total_dev_loss / sample_num
                # 以下accuracy是训练集的准确率
                accuracy = np.mean(torch.max(logit, 1)[1].cpu().numpy() == y_train_tensor.data.cpu().numpy()) * 100
                # 写入验证日志
                writer.add_scalar("loss/train", loss.item(), steps)
                writer.add_scalar("loss/dev", avg_dev_loss, steps)
                writer.add_scalar("acc/train", accuracy, steps)
                writer.add_scalar("acc/dev", dev_accuracy, steps)
                # 是否保存
                # 判断效果是否有提升(采用loss来判断模型是否最优要科学一点；因为如果采用acc判断，虽然最终的准确率高一点，但是泛化能力差)
                # if dev_accuracy > best_dev_accuracy:
                #     best_dev_accuracy = dev_accuracy
                if avg_dev_loss < dev_best_loss:
                    dev_best_loss = avg_dev_loss
                    last_improve_step = steps
                    save_model(model, args.model_save_dir, "best", steps)
                    improve = '*'
                else:
                    # save_model(model, args.model_save_dir, "snapshot", steps)
                    improve = ''
                print("==" * 50)
                print("Epoch[{}/{}]\tEvaluation\tStep[{}]\tloss:{:.6f}\tacc:{:.2f}%\t{}".format(i, config.num_epochs, steps, avg_dev_loss, dev_accuracy, improve))
                print("==" * 50)
                # 验证完后模型要切换到训练模式
                model.train()
            if steps - last_improve_step > config.early_stop:
                print("early stop by {} steps.".format(config.early_stop))
                flag = True
                break
        if flag:
            break
    writer.close()


def save_model(model, model_save_dir, save_prefix, steps):
    save_prefix = model_save_dir + "/" + save_prefix
    save_path = "{}_steps_{}.ckpt".format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)


# 用于对每一折数据的评价指标求平均
def metric_k_fold(k_num, result_save_path, statistic_save_path):
    # 定义列表用于存储k个子模块的评价指标值
    k_accuracy = []
    k_precision_list, k_recall_list = [], []
    k_precision_macro, k_recall_macro = [], []
    for fold_index in range(1, k_num + 1):
        # pip install xlrd==1.2.0（高版本没法读取xlsx）
        saved_data = pd.read_excel(result_save_path, sheet_name="{}_fold".format(fold_index))
        labels_all = saved_data['labels_all']
        predict_all = saved_data['predict_all']

        y_test, y_predict = list(labels_all), list(predict_all)
        label_set = sorted(list(set(y_test + y_predict)))
        # 对label进行映射，防止有些label不是从0开始
        label_map = {}
        for (i, label) in enumerate(label_set):
            label_map[label] = i
        label_set_maped = [label_map[item] for item in label_set]
        # 求混淆矩阵，一行一行地计算个数(行是真实值，列是预测值)
        confusion_matrix = np.zeros((len(label_set_maped), len(label_set_maped)), dtype=np.int)
        for flag in label_set_maped:
            for i in range(len(y_test)):
                if label_map[y_test[i]] == flag:
                    confusion_matrix[flag][label_map[y_predict[i]]] += 1
        # 根据混淆矩阵求解微平均和宏平均指标
        TP_list, FP_list, FN_list = [], [], []
        precision_list, recall_list = [], []
        for i in range(confusion_matrix.shape[0]):
            TP = confusion_matrix[i][i]
            FP = np.sum([confusion_matrix[j][i] for j in range(confusion_matrix.shape[0])]) - TP
            FN = np.sum([confusion_matrix[i][j] for j in range(confusion_matrix.shape[1])]) - TP
            TP_list.append(TP)
            FP_list.append(FP)
            FN_list.append(FN)
            if TP == 0:
                precision_list.append(0), recall_list.append(0)
                continue
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            precision_list.append(precision), recall_list.append(recall)
        # 宏平均指标
        precision_macro = np.mean(precision_list)
        recall_macro = np.mean(recall_list)
        # 准确率
        accuracy = np.mean(np.array(y_test) == np.array(y_predict))

        # 将k个子模块的评价指标值加入到列表中
        k_accuracy.append(accuracy)
        k_precision_list.append(np.array(precision_list)), k_recall_list.append(np.array(recall_list))
        k_precision_macro.append(precision_macro), k_recall_macro.append(recall_macro)
    # 求k个子模块的评价指标平均值
    mean_accuracy = np.mean(np.array(k_accuracy))
    mean_precision_list, mean_recall_list = list(np.mean(np.array(k_precision_list), axis=0)), list(np.mean(np.array(k_recall_list), axis=0))
    mean_f1_list = []
    for i in range(len(mean_precision_list)):
        mean_f1 = (2 * mean_precision_list[i] * mean_recall_list[i]) / (mean_precision_list[i] + mean_recall_list[i])
        mean_f1_list.append(mean_f1)
    mean_precision_macro, mean_recall_macro = np.mean(np.array(k_precision_macro)), np.mean(np.array(k_recall_macro))
    mean_f1_macro = (2 * mean_precision_macro * mean_recall_macro) / (mean_precision_macro + mean_recall_macro)

    print("K折平均后的准确率为：%.2f%%" % (mean_accuracy * 100))
    print("K折平均后的宏平均的精确率为：%.4f，召回率为：%.4f，F1值为：%.4f" % (mean_precision_macro, mean_recall_macro, mean_f1_macro))
    # 将统计信息写入csv文件中
    with open(statistic_save_path, 'w', encoding='utf-8-sig', newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["情绪类别", "惊讶", "好奇", "享受", "困惑", "焦虑", "沮丧", "无聊", "中性"])
        mean_precision_list = [round(number, 4) for number in mean_precision_list]
        mean_recall_list = [round(number, 4) for number in mean_recall_list]
        mean_f1_list = [round(number, 4) for number in mean_f1_list]
        csv_writer.writerow(["精确率（P）"] + mean_precision_list)
        csv_writer.writerow(["召回率（R）"] + mean_recall_list)
        csv_writer.writerow(["F1值"] + mean_f1_list)
        mean_macro_str = ["精确率：", str(round(mean_precision_macro, 4)), "召回率：", str(round(mean_recall_macro, 4)), "F1值：",
                     str(round(mean_f1_macro, 4))]
        csv_writer.writerow(["宏平均指标"] + mean_macro_str)
        csv_writer.writerow(["准确率（Acc）"] + [round(mean_accuracy, 4)])


if __name__ == '__main__':
    processors = {
        'full_all_mooc': data_utils.Full_All_MOOC,
        'full_medical_mooc': data_utils.Full_Medical_MOOC,
        'full_art_mooc': data_utils.Full_Art_MOOC,
        'full_science_mooc': data_utils.Full_Science_MOOC,
        'full_computer_mooc': data_utils.Full_Computer_MOOC,
    }
    task_list = ["full_all_mooc", "full_medical_mooc", "full_art_mooc", "full_science_mooc", "full_computer_mooc"]
    for args.task_name in task_list:
        # 获取数据集
        print("Loading data...")
        vocab_path = "../data_preprocess/processed_data/vocabulary_embedding/vocab.pkl"
        embedding_path = ""
        data_class = processors[args.task_name]()
        # 加载字典
        vocab_dict = pkl.load(open(vocab_path, 'rb'))
        # 添加参数
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')      # 是否使用GPU
        args.label_list = data_class.label_list
        args.num_classes = len(data_class.label_list)
        args.vocab_size = len(vocab_dict)
        args.embedding_path = embedding_path
        print("vocab_size is：", args.vocab_size)

        # 选择所使用的模型
        model_list = ["TextCNN", "TextRNN", "TextRCNN", "TextRNN_Att", "TextDPCNN", "Transformer"]
        for model_name in model_list:
            args.model_save_dir = "../model/" + model_name + "/" + args.task_name
            # origin_model_save_dir用于记住模型保存的最外层目录
            origin_model_save_dir = args.model_save_dir
            # 定义保存数据的DataFrame
            k_num = args.k_num
            result_save_path = origin_model_save_dir + '/kfold_predict.xlsx'
            writer = pd.ExcelWriter(result_save_path)
            statistic_save_path = origin_model_save_dir + '/kfold_statistic.csv'

            # 设置静态随机数，保证每次结果一样
            torch.manual_seed(1)
            torch.cuda.manual_seed_all(1)
            torch.backends.cudnn.deterministic = True
            # 加载模型
            model_path = import_module('cnn_rnn_vocab.' + model_name)
            config = model_path.Config(args)
            model = model_path.Model(config).to(config.device)
            # 输出参数详情
            print("参数详情：")
            for att, value in sorted(args.__dict__.items()):
                print("\t{}={}".format(att.upper(), value))
            # 加载数据
            train_data, test_data = data_utils.create_id_dataset(vocab_dict, data_class.processed_train_data,
                                                                 data_class.processed_test_data, config.max_seq_length)
            all_data = train_data + test_data
            all_data = np.asarray(all_data)
            k_fold = KFold(n_splits=k_num, shuffle=False)
            fold_index = 0
            for train_index, dev_index in k_fold.split(all_data):
                fold_index += 1
                args.model_save_dir = origin_model_save_dir + "/" + "{}_fold".format(fold_index)
                # 由于是交叉验证，则可令test_data_k=dev_data_k
                train_data_k = all_data[train_index]
                dev_data_k = all_data[dev_index]
                test_data_k = dev_data_k
                train_iter = data_utils.build_iterator(train_data_k, config)
                dev_iter = data_utils.build_iterator(dev_data_k, config)
                test_iter = data_utils.build_iterator(test_data_k, config)
                # 每重新开始训练时，初始化模型权重（Transformer在进行初始化时会出问题）
                if model_name != "Transformer":
                    init_network(model)
                if args.do_train:
                    train(config, model, train_iter, dev_iter)
                if args.do_test:
                    # 加载最后保存的那个模型
                    model_weight_list = os.listdir(args.model_save_dir)
                    model_weight_list = [item for item in model_weight_list if ".ckpt" in item]
                    model_weight_list = sorted(model_weight_list, key=lambda x: int(x.strip(".ckpt").split("_")[-1]))
                    last_saved_model = model_weight_list[-1]
                    model_load_path = args.model_save_dir + "/" + last_saved_model
                    # 删除多余的模型文件
                    for delete_model_name in model_weight_list[:-1]:
                        delete_model = args.model_save_dir + "/" + delete_model_name
                        if os.path.exists(delete_model):
                            os.remove(delete_model)

                    if not os.path.exists(model_load_path):
                        print("已训练的模型加载失败")
                        sys.exit()
                    print("所加载的模型为：", model_load_path)
                    model.load_state_dict(torch.load(model_load_path))
                    model.eval()
                    # 加载测试数据集
                    y_predict, y_test = np.array([], dtype=int), np.array([], dtype=int)
                    with torch.no_grad():
                        for x_test_tensor, y_test_tensor in test_iter:
                            logit = model.forward(x_test_tensor)
                            y_ = torch.max(logit, 1)[1].cpu().numpy()
                            y_predict = np.append(y_predict, y_)
                            y_test = np.append(y_test, y_test_tensor.data.cpu().numpy())
                    save_data = pd.DataFrame(columns=['labels_all', 'predict_all'])
                    save_data['labels_all'] = y_test
                    save_data['predict_all'] = y_predict
                    save_data.to_excel(excel_writer=writer, index=None, encoding='utf-8-sig', sheet_name="{}_fold".format(fold_index))
                    writer.save()
                    writer.close()
                    print("第{}折训练完成！".format(fold_index))
                    print("==" * 50)
            # K折训练完了以后，需要计算平均准确率和F1值
            metric_k_fold(k_num, result_save_path, statistic_save_path)