from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, classification_report, precision_score
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
warnings.filterwarnings('ignore')
dic = {'1': 'Cotton',
       '2': 'Corn',
       '3': 'Pepper',
       '4': 'Jujube',
       '5': 'Pear',
       '6': 'Apricot',
       '7': 'Tomato',
       '8': 'Others',}


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        print('==================',flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        folder_path1= './test_results/' + setting + '/'
        if not os.path.exists(folder_path1):
            os.mkdir(folder_path1)
        # 保存混淆矩阵
        cm = confusion_matrix(trues, predictions)

        # 计算百分比混淆矩阵
        cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # 按指定顺序排列标签
        order = ['Cotton', 'Corn', 'Pepper', 'Jujube', 'Pear', 'Apricot', 'Tomato', 'Others']
        ordered_indices = [list(dic.values()).index(label) for label in order]
        cm_perc = cm_perc[ordered_indices, :][:, ordered_indices]
        cm = cm[ordered_indices, :][:, ordered_indices]

        # 保存百分比混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_perc, annot=False, cmap='Blues', xticklabels=order, yticklabels=order, cbar_kws={"aspect": 40})
        plt.xlabel('Predicted', fontsize=16)
        plt.ylabel('True', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.title('Confusion Matrix (Percentage)', fontsize=16)
        plt.savefig(os.path.join(folder_path1, 'confusion_matrix_percentage.png'))
        plt.close()

        # 保存个数混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=order, yticklabels=order, cbar_kws={"aspect": 40})
        plt.xlabel('Predicted', fontsize=16)
        plt.ylabel('True', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.title('Confusion Matrix (Count)', fontsize=16)
        plt.savefig(os.path.join(folder_path1, 'confusion_matrix_count.png'))
        plt.close()

        # 计算F1得分并保存到文件
        f1_scores = f1_score(trues, predictions, average=None)
        f1_scores_dict = {order[i]: f1_scores[ordered_indices[i]] for i in range(len(order))}

        # 计算每个类别的OA
        oa_scores = {}
        for i, label in enumerate(order):
            tp = cm[i, i]
            total = cm.sum(axis=1)[i]
            oa_scores[label] = tp / total if total > 0 else 0

        # 保存F1和OA得分到文件
        with open(os.path.join(folder_path1, 'f1_and_oa_scores.txt'), 'w') as f:
            for crop in order:
                f.write(f'{crop} - F1 Score: {f1_scores_dict[crop]:.4f}, OA: {oa_scores[crop]:.4f}\n')

        # 计算平均精度和平均F1得分
        avg_precision = precision_score(trues, predictions, average='macro')
        avg_f1 = f1_score(trues, predictions, average='macro')

        # 保存平均精度和平均F1得分到文件
        with open(os.path.join(folder_path1, 'average_precision_and_f1.txt'), 'w') as f:
            f.write(f'Average Precision: {avg_precision:.4f}\n')
            f.write(f'Average F1 Score: {avg_f1:.4f}\n')

        # 保存测试结果
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('accuracy:{}'.format(accuracy))
        file_name = 'result_classification.txt'
        f = open(os.path.join(folder_path, file_name), 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('\n')
        f.close()
        return


    def predict(self, setting, test=0):

        paths = ['WeiganFarmland']
        for path in paths:
            flag = 'TEST'
            ii=path
            path = './data/'+str(path)+'/'
            print(path)
            # test_data, test_loader = data_provider1(path, flag)
            # print('test_loader', test_loader)
            # print('test_data:', len(test_data), type(test_data))
            # print('*****', test_data[10])
            #print('*****', test_data[50])
            #print('*****', test_data[100])
            def extract_doy_ndvi(rowcol_DOY_NDVI):
                elements = rowcol_DOY_NDVI.split(',')
                # 检查元素数量，如果少于3个，则跳过处理
                if len(elements) < 3:
                    return None
                if not elements[2].strip():
                    return None
                row = rowcol_DOY_NDVI.split(',')[0]
                row = 0
                col = rowcol_DOY_NDVI.split(',')[1]
                col = 0

                rowcol = str(int(row)) + '_' + str(int(col))
                return rowcol

            def readDoyNDVI(filepath):
                with open(filepath, "r") as f:  # 打开文件
                    lrowCol_doy_ndvi = f.readlines()[14:]  # 读取文件
                l_rc_doy_ndvi = list(filter(None, map(extract_doy_ndvi, lrowCol_doy_ndvi)))
                return l_rc_doy_ndvi

            path2=path+'WeiganFarmland_TEST.ts'
            rowcol = readDoyNDVI(path2)

            if test:
                print('loading model')
                self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

            preds = []
            trues = []

            test_data, test_loader = self._get_data(flag='TEST')

            self.model.eval()
            with torch.no_grad():
                for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    padding_mask = padding_mask.float().to(self.device)
                    label = label.to(self.device)

                    outputs = self.model(batch_x, padding_mask, None, None)

                    preds.append(outputs.detach())
                    trues.append(label)

            preds = torch.cat(preds, 0)
            trues = torch.cat(trues, 0)
            print('test shape:', preds.shape, trues.shape)

            # 保存每个样本的预测标签和概率到文
            # 计算每个类别的概率
            probs = torch.nn.functional.softmax(preds, dim=1)  # (total_samples, num_cl

            # 获取每个样本最可能的类别的索引
            max_prob_indices = torch.argmax(probs, dim=1)  # (total_samples,)

            # 使用 torch.gather 来获取每个样本最可能的类别的最大概率
            # probs 的第一个维度是样本，第二个维度是类别
            max_probs = torch.gather(probs, 1, max_prob_indices.unsqueeze(1)).squeeze(1)  # (total_samples,)

            # 接下来，您可以继续您的代码，例如计算准确率等
            # ...
            # 获取最可能的类别索引（即预测标签）
            predictions = torch.argmax(preds, dim=1).cpu().numpy()  # (total_samples,)

            # 将 trues 展平成一维张量，并转换为 numpy 数组
            # trues = trues.flatten().cpu().numpy()

            # 计算准确率
            # accuracy = cal_accuracy(predictions, trues)

            # 将预测概率转换为 numpy 数组，用于输出
            # 注意：这里需要指定具体的概率索引，或者使用其他方式获取所有样本的概率
            probs = max_probs.cpu().numpy()
            print(f'Length of rowcol: {len(rowcol)}')
            print(f'Length of max_probs: {len(max_probs)}')

            # 打印预测标签数组的长度
            print(f'Length of predictions: {len(predictions)}')

            # 保存每个样本的预测标签和概率到文件
            # 假设 rowcol 是一个与 predictions 和 probs 长度相同的列表
            predictions_and_probs = list(zip(predictions+1, probs)) # +1是为了保证序号从1开始

            # result save
            folder_path = './predict_results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            with open(os.path.join(folder_path, 'predictions_and_probs'+str(ii)+'.txt'), 'w') as f:
                for label, prob in predictions_and_probs:
                    # 将概率列表中的每个元素转换为字符串，然后用逗号和空格分隔
                    # f.write('{},{} \n'.format(label, str(prob)))
                    f.write('{}\n'.format(label)) # 仅输出最终预测标签值
            print('============over===============')
            # with open(os.path.join(folder_path, 'accuracy.txt'), 'w') as f:
            #     f.write('accuracy:{}'.format(accuracy))
            # print('accuracy:{}'.format(accuracy))


        return
