import time
import numpy as np
import torch
import pandas as pd
from torch import nn, optim
from STC_GNN import STCGNN
import torch.nn.functional as F
from Metrics import mask_data, ModelEvaluator
from torch.nn.functional import normalize
class GraphStructuralEncoder(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,):
        super(GraphStructuralEncoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward,)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src):

        src2=self.self_attn(src,src, src)[0]
        #src2 = self.self_attn(src,src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
class ComboLoss(nn.Module):
    def __init__(self):
        super(ComboLoss, self).__init__()
        self.binary_crossentropy = nn.BCELoss(reduction='mean')

    def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor):
        bce_loss = self.binary_crossentropy(y_pred, y_true)
        dice_loss = self.dice_loss(y_pred, y_true)
        return bce_loss + dice_loss

    @staticmethod
    def dice_loss(y_pred:torch.Tensor, y_true:torch.Tensor):
        numerator = 2 * torch.reshape(y_pred * y_true, (y_pred.shape[0], -1)).sum(-1)
        denominator = torch.reshape(y_pred + y_true, (y_pred.shape[0], -1)).sum(-1)
        return torch.mean(1 - numerator / denominator)


class ModelTrainer(object):
    def __init__(self, params:dict, data:dict):
        self.params = params
        self.mask = data['mask']    # for evaluation
        self.threshold = data['HA']
        self.prior_graph = [torch.from_numpy(data['s_adj']).float().to(self.params['device']),
                            torch.from_numpy(data['c_cor']).float().to(self.params['device'])]
        self.model = self.get_model().to(params['device']) if params['device'].startswith('cuda') else self.get_model()
        self.criterion = ComboLoss()
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.params['learn_rate'], weight_decay=self.params['decay_rate'])

        
        self.length=96
        self.params_0= self.init_params(self.params['H'] * self.params['W'], self.params['hidden_dim'])
        self.relu=nn.ReLU()
        self.num_multi_pattern_encoder = 3
        self.num_cross_graph_encoder = 1
        self.softmax = nn.Softmax(dim=-1)
        self.para1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)#the size is [1]
        self.para1.data.fill_(0.5)
        self.para2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)#the size is [1]
        self.para2.data.fill_(0.2)
        self.para3 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)#the size is [1]
        self.para3.data.fill_(0.4)
        self.device='cuda:0'
        self.out_proj = nn.Sequential(nn.Linear(in_features=self.params['hidden_dim'], out_features=self.params['hidden_dim']//2, bias=True),
                                      nn.Linear(in_features=self.params['hidden_dim']//2, out_features=1, bias=True))
        self.multi_pattern_blocks = nn.ModuleList(
            [GraphStructuralEncoder(d_model=self.length, nhead=1) for _ in range(self.num_multi_pattern_encoder)]).to(params['device'])
        self.cross_graph_blocks = nn.ModuleList(
            [GraphStructuralEncoder(d_model=self.length, nhead=1) for _ in range(self.num_cross_graph_encoder)]).to(params['device'])
    def init_params(self, in_dim:int, hidden_dim:int):
        params_0= nn.ParameterDict()
        params_0['covid'] = nn.Parameter(torch.randn(12, self.length), requires_grad=True)
        params_0['census'] = nn.Parameter(torch.randn(7, self.length), requires_grad=True)
        params_0['health'] = nn.Parameter(torch.randn(23, self.length), requires_grad=True)
        for param in params_0.values():
            nn.init.xavier_normal_(param)
        return params_0
    def get_model(self):
        if self.params['model'] == 'STC-GNN':
            model = STCGNN(num_nodes=self.params['H'] * self.params['W'],
                           num_categories=self.params['C'],
                           Ks=self.params['cheby_order'],
                           Kc=self.params['cheby_order'],
                           input_dim=1,
                           hidden_dim=self.params['hidden_dim'],
                           num_layers=self.params['nn_layers'],
                           out_horizon=self.params['pred_len'])
        else:
            raise NotImplementedError('Invalid model name.')

        return model

    def train(self, data_loader:dict,modes:list, early_stop_patience=10):
        checkpoint = {'epoch': 0, 'train_loss': np.inf, 'val_loss': np.inf, 'state_dict': self.model.state_dict()}
        patience_count = early_stop_patience
        val_loss = np.inf       # initialize validation loss
        loss_curve = {mode: [] for mode in modes}
        run_time = {mode: [] for mode in modes}



        print('\n', time.ctime())
        print(f'     {self.params["model"]} model training begins:')
        for epoch in range(1, 1 + self.params['num_epochs']):
            running_loss = {mode: 0.0 for mode in modes}
            for mode in modes:
                if mode == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                step = 0
                start_time = time.time()


                for x_seq, y_true,trip_x,trip_y in data_loader[mode]:
                    with torch.set_grad_enabled(mode=(mode=='train')):
                        

                        if self.params['model'] == 'STC-GNN':
                            y_pred = self.model(X_seq=x_seq, As=self.prior_graph[0], Ac=self.prior_graph[1],trip_x=trip_x,trip_y=trip_y)
                            #y_pred = self.model(X_seq=x_seq, As=As, Ac=Ac,trip_x=trip_x,trip_y=trip_y)
                        else:
                            raise NotImplementedError('Invalid model name.')

                        loss = self.criterion(y_pred, y_true)
                        if mode == 'train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                    running_loss[mode] += loss * y_true.shape[0]    # loss reduction='mean': batchwise average
                    step += y_true.shape[0]
                    torch.cuda.empty_cache()

                # epoch mode end
                end_time = time.time()
                run_time[mode].append(end_time - start_time)
                loss_curve[mode].append(running_loss[mode]/step)

            # epoch end
            log = f'Epoch {epoch}: training time: {run_time["train"][-1]:.4} s/epoch, training loss: {loss_curve["train"][-1]:.4}; ' \
                  f'inference time: {run_time["validate"][-1]:.4} s, '
            if loss_curve["validate"][-1] < val_loss:
                add_log = f'validation loss drops from {val_loss:.4} to {loss_curve["validate"][-1]:.4}. Update model checkpoint..'
                val_loss = loss_curve["validate"][-1]
                checkpoint.update(epoch=epoch,
                                  train_loss = loss_curve["train"][-1],
                                  val_loss = loss_curve["validate"][-1],
                                  state_dict=self.model.state_dict())
                torch.save(checkpoint, self.params['output_dir'] + f'/{self.params["model"]}-{self.params["time_slice"]}.pkl')
                patience_count = early_stop_patience
            else:
                add_log = f'validation loss does not improve from {val_loss:.4}.'
                patience_count -= 1
                if patience_count == 0:     # early stopping
                    print('\n', time.ctime())
                    print(f'    Early stopping triggered at epoch {epoch}.')
                    break
            print(log + add_log)

        # training end
        print('\n', time.ctime())
        print(f'     {self.params["model"]} model training ends.')
        # torch.save(checkpoint, self.params['output_dir']+f'/{self.params["model"]}.pkl')
        print(f'    Average training time: {np.mean(run_time["train"]):.4} s/epoch.')
        print(f'    Average inference time: {np.mean(run_time["validate"]):.4} s.')
        return


    def test(self, data_loader:dict, modes:list):
        # load trained model
        trained_checkpoint = torch.load(self.params['output_dir']+f'/{self.params["model"]}-{self.params["time_slice"]}.pkl')
        print(f'Successfully loaded trained {self.params["model"]} model - epoch: {trained_checkpoint["epoch"]}, training loss: {trained_checkpoint["train_loss"]}, validation loss: {trained_checkpoint["val_loss"]}')
        self.model.load_state_dict(trained_checkpoint['state_dict'])
        self.model.eval()

        model_evaluator = ModelEvaluator(self.params)
        for mode in modes:
            print('\n', time.ctime())
            print(f'     {self.params["model"]} model testing on {mode} data begins:')
            forecast, ground_truth = [], []
            for x_seq, y_true,trip_x,trip_y in data_loader[mode]:
                if self.params['model'] == 'STC-GNN':
                    y_pred = self.model(X_seq=x_seq, As=self.prior_graph[0], Ac=self.prior_graph[1],trip_x=trip_x,trip_y=trip_y)
                else:
                    raise NotImplementedError('Invalid model name.')

                forecast.append(y_pred.cpu().detach().numpy())
                ground_truth.append(y_true.cpu().detach().numpy())

            forecast = np.concatenate(forecast, axis=0)
            ground_truth = np.concatenate(ground_truth, axis=0)

            # evaluate
            forecast_masked = mask_data(x=forecast, H=self.params['H'], W=self.params['W'], mask=self.mask)
            ground_truth_masked = mask_data(x=ground_truth, H=self.params['H'], W=self.params['W'], mask=self.mask)
            model_evaluator.evaluate_binary(y_pred_prob=forecast_masked,
                                            y_true_bi=ground_truth_masked,
                                            threshold=self.threshold,
                                            mode=mode)

        print('\n', time.ctime())
        print(f'     {self.params["model"]} model testing ends.')
        return


