import time
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, fbeta_score, \
     roc_auc_score, roc_curve, average_precision_score, precision_recall_curve, \
     log_loss, mean_absolute_error, recall_score,accuracy_score,auc


def mask_data(x:np.array, H:int, W:int, mask):       # for evaluation only on selected grids
    assert (len(x.shape) == 4)|(len(x.shape) == 5)        # graph: (num_samples, horizon, N, C) / grid: (num_samples, horizon, C, H, W)
    if len(x.shape) == 4:   # graph
        assert x.shape[-2]==H*W
        x = x.reshape(x.shape[0], x.shape[1], H, W, x.shape[-1])
    else:   # len=5
        if (x.shape[-2]==H)&(x.shape[-1]==W):   # grid
            x = x.transpose((0, 1, 3, 4, 2))    # switch to channel last
        else:
            assert (x.shape[2]==H)&(x.shape[3]==W)
            pass
    mask=None
    if mask is not None:
        mask_count = 0
        x_masked = list()
        for h in range(H):
            for w in range(W):
                if (h, w) in mask:
                    mask_count += 1
                    continue
                x_masked.append(x[:, :, h, w, :])
        # print('    Number of masked grids:', mask_count)
        x_masked = np.array(x_masked).transpose((1, 2, 0, 3))  # (num_samples, horizon, masked_N, C)
    else:
        x_masked = x.reshape(x.shape[0], x.shape[1], -1, x.shape[-1])       # unmasked
    return x_masked

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point
    
class ModelEvaluator(object):
    def __init__(self, params:dict, precision=4, beta=2, epsilon=1e-0):
        self.params = params
        self.precision = precision      # digits behind decimal
        self.beta = beta        # for F-beta score
        self.epsilon = epsilon      # avoid zero division
        self.softmax = nn.Softmax(dim=-1)
        self.device='cuda:0'
    def evaluate_binary(self, y_pred_prob: np.array, y_true_bi: np.array, threshold:list, mode:str):
        '''
        Evaluate on binary metrics: macro/micro-F1/F2, recall, ROC-AUC/PR-AUC, BCE, MAE
        :param y_pred_prob: (num_samples, horizon, masked_N, C) \in [0, 1]
        :param y_true_bi: (num_samples, horizon, masked_N, C) \in {0, 1}
        :param threshold: historical avg.
        :param mode: [train, test]
        :return:
        '''
        assert (y_pred_prob.shape == y_true_bi.shape) and (y_true_bi.shape[-1] == len(threshold))

        with open(self.params['output_dir'] + f'/{self.params["model"]}_eval-bi-metrics-xiaorong.csv', 'a') as cf:
            print(' '.join(['*' * 10, f'Evaluation on {mode} set started at', time.ctime(), '*' * 10]))
            cf.write(f'*****, Evaluation starts, {mode}, {time.ctime()}, ***** \n')
            for param in self.params.keys():        # write model parameters
                cf.write(f'{param}: {self.params[param]},')
            cf.write('\n')

        # stepwise through horizon
        multistep_metrics = list()
        #y_pred_prob=np.load('Result-Wave-12.npz')['result']
        #y_pred_prob=torch.Tensor(y_pred_prob).to(self.device)
        #y_pred_prob=self.softmax(torch.relu(y_pred_prob))
        #y_true_bi=np.load('Test-Wave-12.npz')['test']
        #print(y_pred_prob.shape)
        #y_pred_prob=y_pred_prob-1
        #y_pred_prob=y_pred_prob.cpu().numpy()
        """
        y_pred_prob=np.ones(shape=(181,3,77,5))
        y_true_bi=np.ones(shape=(181,3,77,5))
        Y_pred_prob=np.load('Result-SVR-3-qian.npz')['result']
        Y_pred_prob=Y_pred_prob.reshape(181,77,3,4)
        Y_pred_prob=Y_pred_prob.transpose(0,2,1,3)
        Y_true_bi=np.load('Test-SVR-3-qian.npz')['test']
        Y_true_bi=Y_true_bi.reshape(181,77,3,4)
        Y_true_bi=Y_true_bi.transpose(0,2,1,3)
        y_pred_prob[:,:,:,0:4]=Y_pred_prob
        y_true_bi[:,:,:,0:4]=Y_true_bi
        Y_pred_prob=np.load('Result-SVR-3-hou.npz')['result']
        Y_pred_prob=Y_pred_prob.reshape(181,77,3,4)
        Y_pred_prob=Y_pred_prob.transpose(0,2,1,3)
        Y_true_bi=np.load('Test-SVR-3-hou.npz')['test']
        Y_true_bi=Y_true_bi.reshape(181,77,3,4)
        Y_true_bi=Y_true_bi.transpose(0,2,1,3)
        y_pred_prob[:,:,:,4]=Y_pred_prob[:,:,:,3]
        y_true_bi[:,:,:,4]=Y_true_bi[:,:,:,3]
        print(y_pred_prob.shape)
        print(y_true_bi[0,0,:,0])
        """
        print(y_pred_prob.shape)

        y_pred_0=pd.DataFrame(y_pred_prob[-1,0,:,:])
        y_pred_1=pd.DataFrame(y_pred_prob[-1,1,:,:])
        y_pred_2=pd.DataFrame(y_pred_prob[-1,2,:,:])
        y_pred_0.to_csv('y_pred_0.csv')
        y_pred_1.to_csv('y_pred_1.csv')
        y_pred_2.to_csv('y_pred_2.csv')

        y_true_0=pd.DataFrame(y_true_bi[-1,0,:,:])
        y_true_1=pd.DataFrame(y_true_bi[-1,1,:,:])
        y_true_2=pd.DataFrame(y_true_bi[-1,2,:,:])
        y_true_0.to_csv('y_true_0.csv')
        y_true_1.to_csv('y_true_1.csv')
        y_true_2.to_csv('y_true_2.csv')

        print(y_true_bi.shape)
        optimal_th=[0]*5
        for i in range(0,5):
            aa=y_pred_prob[:,:,:,i].flatten()
            bb=y_true_bi[:,:,:,i].flatten()
            fpr, tpr, thresholds = roc_curve(bb,aa)
            roc_auc = auc(fpr, tpr)
            optimal_th[i], optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
            print(optimal_th[i])
            plt.subplot(2,3,i+1)
            plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
            plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
            plt.xlim([-0.05, 1.05])  
            plt.ylim([-0.05, 1.05])
            plt.savefig('a.png')
        for step in range(self.params['pred_len']):
            #step_metrics = self.one_step_eval_bi(y_pred_prob[:,step,:,:], y_true_bi[:,step,:,:], threshold)
            step_metrics = self.one_step_eval_bi(y_pred_prob[:,step,:,:], y_true_bi[:,step,:,:], optimal_th)
            print(f'Step {step}: \n'
                  f'   Macro-F1: {step_metrics["Macro-F1"]}, Micro-F1: {step_metrics["Micro-F1"]} \n'
                  f'   Macro-F2: {step_metrics["Macro-F2"]}, Micro-F2: {step_metrics["Micro-F2"]} \n'
                  f'   Overall recall: {step_metrics["Recall"]}, ROC-AUC: {step_metrics["ROC-AUC"]}, PR-AUC: {step_metrics["PR-AUC"]} \n'
                  f'   Binary Cross-Entropy: {step_metrics["BCE"]}, MAE: {step_metrics["MAE"]} \n')
            multistep_metrics.append(step_metrics)

        with open(self.params['output_dir'] + f'/{self.params["model"]}_eval-bi-metrics-new.csv', 'a') as cf:
            '''
            table_writer = csv.DictWriter(cf, fieldnames=col_names)
            table_writer.writeheader()
            for metrics in multistep_metrics:
                table_writer.writerow(metrics)
            '''
            col_names = [' '] + list(step_metrics.keys())
            cf.write(','.join(col_names) + '\n')
            for step in range(self.params['pred_len']):
                row_items = [f'Step {step}'] + list(multistep_metrics[step].values())
                cf.write(','.join([str(item) for item in row_items]) + '\n')
            cf.write(f'*****, Evaluation ends, {mode}, {time.ctime()}, ***** \n \n')
            print(' '.join(['*' * 10, f'Evaluation on {mode} set ended at', time.ctime(), '*' * 10]))

        return

    def one_step_eval_bi(self, step_y_pred_prob: np.array, step_y_true_bi: np.array, threshold:list):
        '''
        Single step binary evaluation through categories
        :param step_y_pred_prob: (num_samples, masked_N, C) \in [0, 1]
        :param step_y_true_bi: (num_samples, masked_N, C) \in {0, 1}
        :param threshold: historical avg.
        :return:
        '''
        assert step_y_pred_prob.shape == step_y_true_bi.shape

        step_y_pred_bi = []
        tp_lst, fn_lst, fp_lst = [], [], []  # to calculate Macro-F1/F2
        f1_lst = []  # to calculate Micro-F1
        f2_lst = []  # to calculate Micro-F2

        # loop category
        for c in range(self.params['C']):
            c_y_true_bi = step_y_true_bi[:,:,c].flatten()
            c_y_pred_prob = step_y_pred_prob[:,:,c].flatten()
            c_y_pred_bi = np.where(c_y_pred_prob >= threshold[c], 1, 0)
            step_y_pred_bi.append(c_y_pred_bi.reshape(step_y_pred_prob[:,:,c].shape))
            

            # single category evaluation
            '''
            print(f"{self.params['C_inc'][c].upper()} \n \
                                 F1-score: {round(f1_score(c_y_true_bi, c_y_pred_bi), self.precision)} \
                                 F2-score: {round(fbeta_score(c_y_true_bi, c_y_pred_bi, self.beta), self.precision)} \n \
                                 ROC-score: {round(roc_auc_score(c_y_true_bi, c_y_pred_prob), self.precision)} \
                                 PR-score: {round(average_precision_score(c_y_true_bi, c_y_pred_prob), self.precision)} \n")
            '''
            # check recall vs. precision

            f1_report = classification_report(c_y_true_bi, c_y_pred_bi, labels=np.unique(c_y_pred_bi))
            #print('   Classification report: \n', f1_report)
            # confusion matrix
            #print('   Confusion matrix: \n', confusion_matrix(c_y_true_bi, c_y_pred_bi))
            TN, FP, FN, TP = confusion_matrix(c_y_true_bi, c_y_pred_bi).ravel()

            tp_lst.append(TP)
            fn_lst.append(FN)
            fp_lst.append(FP)
            f1_lst.append(2 * TP / (2 * TP + FN + FP))
            f2_lst.append((1 + self.beta ** 2) * TP / ((1 + self.beta ** 2) * TP + (self.beta ** 2) * FN + FP))
        # F1
        macro_f1 = 2 * sum(tp_lst) / (2 * sum(tp_lst) + sum(fn_lst) + sum(fp_lst))
        micro_f1 = sum(f1_lst) / len(f1_lst)
        # F2
        macro_f2 = (1 + self.beta ** 2) * sum(tp_lst) / ((1 + self.beta ** 2) * sum(tp_lst) + (self.beta ** 2) * sum(fn_lst) + sum(fp_lst))
        micro_f2 = sum(f2_lst) / len(f2_lst)

        # dict of metrics
        step_metrics = dict()
        step_metrics['Macro-F1'] = round(macro_f1, self.precision)
        step_metrics['Micro-F1'] = round(micro_f1, self.precision)
        step_metrics[f'Macro-F{self.beta}'] = round(macro_f2, self.precision)
        step_metrics[f'Micro-F{self.beta}'] = round(micro_f2, self.precision)

        # evaluate on all categories
        step_y_pred_bi = np.stack(step_y_pred_bi, axis=-1)
        step_y_pred_prob, step_y_pred_bi, step_y_true_bi = step_y_pred_prob.flatten(), step_y_pred_bi.flatten(), step_y_true_bi.flatten()
        step_metrics['Recall'] = round(recall_score(step_y_true_bi, step_y_pred_bi), self.precision)
        step_metrics['Accuracy'] = round(accuracy_score(step_y_true_bi, step_y_pred_bi), self.precision)
        step_metrics['ROC-AUC'] = round(roc_auc_score(step_y_true_bi, step_y_pred_prob), self.precision)
        step_metrics['PR-AUC'] = round(average_precision_score(step_y_true_bi, step_y_pred_prob), self.precision)
        step_metrics['BCE'] = round(log_loss(step_y_true_bi, step_y_pred_prob), self.precision)
        step_metrics['MAE'] = round(mean_absolute_error(step_y_true_bi, step_y_pred_prob), self.precision)

        return step_metrics

