import torch
from torch import nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
import sklearn
from torch.nn.functional import normalize
import os
import torch as th
from torch.autograd import Variable
from depthwise import DepthwiseNet
from torch.nn.utils import weight_norm

os.environ['CUDA_VISIBLE_DEVICES']='0,1'
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
        #print(src.shape)
        src2=self.self_attn(src,src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class BDG_Dif(nn.Module):        # 2D graph convolution operation: 1 input
    def __init__(self, Ks:int, Kc:int, input_dim:int, hidden_dim:int, use_bias=True, activation=None):
        super(BDG_Dif, self).__init__()
        self.Ks = 3
        self.Kc = 3
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = True
        self.num_multi_pattern_encoder = 3
        self.length=16
        self.activation = activation() if activation is not None else None
        self.params= self.init_params()
        self.self_attn_n = nn.MultiheadAttention(2*self.length,1 ,dropout=0.1,batch_first=True)

        self.self_attn_c = nn.MultiheadAttention(self.length,1, dropout=0.1,batch_first=True)
        self.softmax = nn.Softmax(dim=-1)
        self.dwn = DepthwiseNet(5, 1, kernel_size=4, dilation_c=4)



    def init_params(self, b_init=0.0):
        self.W = nn.Parameter(torch.empty((self.input_dim)*self.Ks*self.Kc, self.hidden_dim), requires_grad=True)
        nn.init.xavier_normal_(self.W)
        params = nn.ParameterDict()
        params['N'] = nn.Parameter(torch.randn(5,self.input_dim,self.length), requires_grad=True)
        params['C'] = nn.Parameter(torch.randn(77,self.input_dim,self.length), requires_grad=True)
        params['in'] = nn.Parameter(torch.randn(77, self.length), requires_grad=True)
        params['out'] = nn.Parameter(torch.randn(77, self.length), requires_grad=True)
        for param in params.values():
            nn.init.xavier_normal_(param)
        if self.use_bias:
            self.b = nn.Parameter(torch.empty(self.hidden_dim), requires_grad=True)
            nn.init.constant_(self.b, val=b_init)
        return params

    @staticmethod
    def cheby_poly(G:torch.Tensor, cheby_K:int):
        G_set = [torch.eye(G.shape[0]).to(G.device), G]     # order 0, 1
        for k in range(2, cheby_K):
            G_set.append(torch.mm(2 * G, G_set[-1]) - G_set[-2])
        return G_set

    def forward(self, X:torch.Tensor,As:torch.Tensor, Gs:torch.Tensor, Gc:torch.Tensor,X_trip:torch.Tensor):
        #print(X.shape)
        X1= torch.einsum('bncl,clm->bnm', X, self.params['N'])
        X=X.transpose(1,2)   
        X2= torch.einsum('bncl,clm->bnm', X, self.params['C'])
        X=X.transpose(1,2)
        
        X2 = self.dwn(X2)

        X_trip_out=normalize(X_trip,p=2.0,dim=2)
        X_trip_in=normalize(X_trip,p=2.0,dim=1)
        X_in= torch.einsum('bnc,cm->bnm', X_trip_in.transpose(1,2), self.params['in'])
        X_out= torch.einsum('bnc,cm->bnm', X_trip_out, self.params['out'])
        X1_in=torch.cat([X1,X_in],2)
        X1_out=torch.cat([X1,X_out],2)
        Src_att_n=torch.einsum('bnl,blm->bnm', X1_out, X1_in.transpose(1,2))

        Src_att_n=self.softmax(torch.relu(Src_att_n))
        Src_C=self.self_attn_c(X2,X2,X2)   
        Src_att_c=Src_C[1]
        Src_att_c=self.softmax(torch.relu(Src_att_c))

        Gs_set = [torch.eye(Src_att_n.shape[1]).to(Gs.device),Gs,Src_att_n]
        Gc_set = [torch.eye(Src_att_c.shape[1]).to(Gc.device),Gc,Src_att_c] 

        feat_coll = list()
        for n in range(self.Ks):
            for c in range(self.Kc):
                if n==0 or n==1:
                    _1_mode_product = torch.einsum('bncl,nm->bmcl', X, Gs_set[n])
                else:
                    _1_mode_product = torch.einsum('bncl,bnm->bmcl', X, Gs_set[n])
                if c==0 or c==1:
                    _2_mode_product = torch.einsum('bmcl,cd->bmdl', _1_mode_product, Gc_set[c])
                else:
                    _2_mode_product = torch.einsum('bmcl,bcd->bmdl', _1_mode_product, Gc_set[c])
                feat_coll.append(_2_mode_product)
        _2D_feat = torch.cat(feat_coll, dim=-1)
        _3_mode_product = torch.einsum('bmdk,kh->bmdh', _2D_feat, self.W)
        if self.use_bias:
            _3_mode_product += self.b
        H = self.activation(_3_mode_product) if self.activation is not None else _3_mode_product
        return H



class STC_Cell(nn.Module):
    def __init__(self, num_nodes:int, num_categories:int, Ks:int, Kc:int, input_dim:int, hidden_dim:int, use_bias=True, activation=None):
        super(STC_Cell, self).__init__()
        self.num_nodes = num_nodes
        self.num_categories = num_categories
        self.hidden_dim = hidden_dim
        self.gates = BDG_Dif(Ks, Kc, input_dim+hidden_dim, hidden_dim*2, use_bias, activation)
        self.candi = BDG_Dif(Ks, Kc, input_dim+hidden_dim, hidden_dim, use_bias, activation)

    def init_hidden(self, batch_size:int):
        weight = next(self.parameters()).data
        hidden = (weight.new_zeros(batch_size, self.num_nodes, self.num_categories, self.hidden_dim))
        return hidden

    def forward(self,As:torch.Tensor,Gs:torch.Tensor, Gc:torch.Tensor,X_trip:torch.Tensor,Xt:torch.Tensor, Ht_1:torch.Tensor):
        assert len(Xt.shape) == len(Ht_1.shape) == 4, 'STC-cell must take in 4D tensor as input [Xt, Ht-1]'
        #print('Xt',Xt.shape)
        #print('Ht',Ht_1.shape)
        XH = torch.cat([Xt, Ht_1], dim=-1)
        #print('XH',XH.shape)
        XH_conv = self.gates(X=XH, As=As,Gs=Gs, Gc=Gc,X_trip=X_trip)

        u, r = torch.split(XH_conv, self.hidden_dim, dim=-1)
        update = torch.sigmoid(u)
        reset = torch.sigmoid(r)

        candi = torch.cat([Xt, reset*Ht_1], dim=-1)
        candi_conv = torch.tanh(self.candi(X=candi,As=As,Gs=Gs, Gc=Gc,X_trip=X_trip))

        Ht = (1.0 - update) * Ht_1 + update * candi_conv
        return Ht



class STC_Encoder(nn.Module):
    def __init__(self, num_nodes:int, num_categories:int, Ks:int, Kc:int, input_dim:int, hidden_dim:int, num_layers:int,
                 use_bias=True, activation=None, return_all_layers=True):
        super(STC_Encoder, self).__init__()
        self.hidden_dim = self._extend_for_multilayers(hidden_dim, num_layers)
        self.num_layers = num_layers
        self.return_all_layers = return_all_layers
        assert len(self.hidden_dim) == self.num_layers, 'Input [hidden, layer] length must be consistent'

        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = input_dim if i==0 else self.hidden_dim[i-1]
            self.cell_list.append(STC_Cell(num_nodes, num_categories, Ks, Kc, cur_input_dim, self.hidden_dim[i], use_bias=use_bias, activation=activation))

    def forward(self,As:torch.Tensor,Gs:torch.Tensor, Gc:torch.Tensor, X_seq:torch.Tensor,trip_x:torch.Tensor,H0_l=None):
        assert len(X_seq.shape) == 5, 'STC-encoder must take in 5D tensor as input X_seq'
        batch_size, seq_len, _, _, _ = X_seq.shape
        if H0_l is None:
            H0_l = self._init_hidden(batch_size)
        out_seq_lst = list()    # layerwise output seq
        Ht_lst = list()        # layerwise last state
        in_seq_l = X_seq        # current input seq
        #print(in_seq_l[:,0,...].shape)
        #print(H0_l[0].shape)
        for l in range(self.num_layers):
            Ht = H0_l[l]
            out_seq_l = list()
            for t in range(seq_len):
                #print('t',t)
                #print('Xt',in_seq_l[:,t,...].shape)
                Ht = self.cell_list[l](As=As,Gs=Gs, Gc=Gc,X_trip=trip_x[:,t,...],Xt=in_seq_l[:,t,...], Ht_1=Ht)
                out_seq_l.append(Ht)

            out_seq_l = torch.stack(out_seq_l, dim=1)  # (B, T, N, C, h)
            in_seq_l = out_seq_l    # update input seq
            out_seq_lst.append(out_seq_l)
            Ht_lst.append(Ht)

        if not self.return_all_layers:
            out_seq_lst = out_seq_lst[-1:]
            Ht_lst = Ht_lst[-1:]
        return out_seq_lst, Ht_lst

    def _init_hidden(self, batch_size):
        H0_l = []
        for i in range(self.num_layers):
            H0_l.append(self.cell_list[i].init_hidden(batch_size))
        return H0_l

    @staticmethod
    def _extend_for_multilayers(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param



class STC_Decoder(nn.Module):
    def __init__(self, num_nodes:int, num_categories:int, Ks:int, Kc:int, output_dim:int, hidden_dim:int, num_layers:int,
                 out_horizon:int, use_bias=True, activation=None):
        super(STC_Decoder, self).__init__()
        self.out_horizon = out_horizon      # output steps
        self.hidden_dim = self._extend_for_multilayers(hidden_dim, num_layers)
        self.num_layers = num_layers
        assert len(self.hidden_dim) == self.num_layers, 'Input [hidden, layer] length must be consistent'

        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = output_dim if i==0 else self.hidden_dim[i-1]
            self.cell_list.append(STC_Cell(num_nodes, num_categories, Ks, Kc, cur_input_dim, self.hidden_dim[i], use_bias=use_bias, activation=activation))
        # self.out_projector = nn.Linear(in_features=self.hidden_dim[-1], out_features=output_dim, bias=use_bias)

    def forward(self, As:torch.Tensor,Gs:torch.Tensor, Gc:torch.Tensor, Xt:torch.Tensor, H0_l:list,trip_y:torch.Tensor):
        assert len(Xt.shape) == 4, 'STC-decoder must take in 4D tensor as input Xt'

        Ht_lst = list()        # layerwise hidden state
        Xin_l = Xt
        for l in range(self.num_layers):
            Ht_l = self.cell_list[l](As=As,Gs=Gs, Gc=Gc,X_trip=trip_y,Xt=Xin_l, Ht_1=H0_l[l])
            Ht_lst.append(Ht_l)
            Xin_l = Ht_l      # update input for next layer

        # output = self.out_projector(Ht_l)      # output
        return Ht_l, Ht_lst

    @staticmethod
    def _extend_for_multilayers(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class DeepFc(nn.Module):
    def __init__(self, input_dim, output_dim):
        # 输入层，隐藏层*2,输出层.隐藏层节点数目为输入层两倍
        super(DeepFc, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            nn.Linear(input_dim * 2, output_dim),
            nn.LeakyReLU(negative_slope=0.3, inplace=True), )

        self.output = None

    def forward(self, x):
        output = self.model(x)
        self.output = output
        return output

    def out_feature(self):
        return self.output
class STCGNN(nn.Module):
    def __init__(self, num_nodes:int, num_categories:int, Ks:int, Kc:int, input_dim:int, hidden_dim:int, num_layers:int, out_horizon:int,use_bias=True, activation=None):
        super(STCGNN, self).__init__()
        self.length=16
        self.encoder = STC_Encoder(num_nodes, num_categories, Ks, Kc, input_dim, hidden_dim, num_layers, use_bias, activation, return_all_layers=True)
        self.decoder = STC_Decoder(num_nodes, num_categories, Ks, Kc, hidden_dim, hidden_dim, num_layers, out_horizon, use_bias, activation)
        self.params= self.init_params(num_nodes, hidden_dim)
        self.relu=nn.ReLU()
        self.num_multi_pattern_encoder = 1
        self.num_cross_graph_encoder = 1
        self.softmax = nn.Softmax(dim=-1)
        self.para1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)#the size is [1]
        self.para1.data.fill_(0.5)

        self.device='cuda:0'
        self.out_proj = nn.Sequential(nn.Linear(in_features=hidden_dim, out_features=hidden_dim//2, bias=use_bias),
                                      nn.Linear(in_features=hidden_dim//2, out_features=input_dim, bias=use_bias))
        #self.multi_pattern_blocks = nn.ModuleList(
            #[GraphStructuralEncoder(d_model=self.length, nhead=1) for _ in range(self.num_multi_pattern_encoder)])
        self.cross_graph_blocks = nn.ModuleList(
            [GraphStructuralEncoder(d_model=self.length, nhead=1) for _ in range(self.num_cross_graph_encoder)])


    def init_params(self, in_dim:int, hidden_dim:int):
        params = nn.ParameterDict()
        params['covid'] = nn.Parameter(torch.randn(12, self.length), requires_grad=True)
        params['census'] = nn.Parameter(torch.randn(7, self.length), requires_grad=True)
        params['health'] = nn.Parameter(torch.randn(23, self.length), requires_grad=True)
        for param in params.values():
            nn.init.xavier_normal_(param)
        return params
        
    def forward(self, X_seq:torch.Tensor, As:torch.Tensor, Ac:torch.Tensor,trip_x:torch.Tensor,trip_y:torch.Tensor):
        assert len(X_seq.shape) == 4
        df_covid = pd.read_csv("../data/covid_count.csv",header = None)
        covid=df_covid.to_numpy()
        embed_covid=torch.Tensor(covid).to(self.device)
        covid=normalize(embed_covid,p=2.0,dim = 0)
        df_census = pd.read_csv("../data/census_count.csv",header = None)
        census=df_census.to_numpy()
        embed_census=torch.Tensor(census).to(self.device)
        census=normalize(embed_census,p=2.0,dim = 0)
        df_health = pd.read_csv("../data/health_count.csv",header = None)
        health=df_health.to_numpy()
        embed_health=torch.Tensor(health).to(self.device)
        health=normalize(embed_health,p=2.0,dim=0)
        W_covid=self.params['covid']
        W_census=self.params['census']
        W_health=self.params['health']
        As=np.zeros(shape=(3,77,self.length))
        As=torch.Tensor(As).to(self.device)
        As[0,:,:]=torch.mm(covid,W_covid)
        As[1,:,:]=torch.mm(census,W_census)
        As[2,:,:]=torch.mm(health,W_health)
        As_t=As
        As=As.transpose(0,1)
        for cross_graph in self.cross_graph_blocks:
            As = cross_graph(As)
        As=As.transpose(0,1)
        As=As*(1-self.para1) + As_t*self.para1
        Gs=(As[0,:,:]+As[1,:,:]+As[2,:,:])/3
        As_yuan=Gs
        As_yuan=As_yuan.repeat(X_seq.shape[0],X_seq.shape[3],1,1).permute(0,2,1,3)
        Gs=torch.mm(Gs,Gs.T)
        Gs_csv=pd.DataFrame(Gs.cpu().detach().numpy())
        Gs_csv.to_csv('Gs.csv')
        Gs=self.softmax(torch.relu(Gs))
        Gc=self.softmax(torch.relu(Ac))

        X_seq = X_seq.unsqueeze(dim=-1)  # for encoder input

        H0_l=[As_yuan,As_yuan]

        _, Ht_lst = self.encoder(As=As_yuan,Gs=Gs, Gc=Gc, X_seq=X_seq,trip_x=trip_x)
        # initiate decoder input
        deco_input = Ht_lst[-1]
        outputs = list()
        for t in range(self.decoder.out_horizon):
            Ht_l, Ht_lst = self.decoder(As=As_yuan,Gs=Gs, Gc=Gc, Xt=deco_input, H0_l=Ht_lst,trip_y=trip_y[:,t,...])
            output = Ht_l
            deco_input = output     # update decoder input
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)  # (B, horizon, N, C, h)
        outputs = torch.sigmoid(self.out_proj(outputs))
        return outputs.squeeze(dim=-1)

class MixedFusion(nn.Module):
    def __init__(self, in_dim:int):
        super(MixedFusion, self).__init__()
        self.in_dim = in_dim
        self.lin_A = nn.Linear(in_dim**2, in_dim**2)

    def forward(self, A:torch.Tensor, P:torch.Tensor):
        #assert len(A.shape) == len(P.shape) == 2
        a_A = self.lin_A(A.reshape(A.shape[0],self.in_dim*self.in_dim))
        a_P = self.lin_A(P.repeat(A.shape[0],1).reshape(A.shape[0],self.in_dim*self.in_dim))
        a = torch.sigmoid(torch.add(a_A, a_P)).reshape(A.shape[0],self.in_dim, self.in_dim)
        P=a_P.reshape(A.shape[0],self.in_dim, self.in_dim)
        G = torch.add(torch.einsum('nab,nbc->nac',a, A), torch.einsum('nab,nbc->nac',1-a, P))
        return G

