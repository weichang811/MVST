import torch as th
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from depthwise import DepthwiseNet
from torch.nn.utils import weight_norm
import numpy as np

class ADDSTCN(nn.Module):
    def __init__(self, target, input_size, num_levels, kernel_size, cuda, dilation_c):
        super(ADDSTCN, self).__init__()

        self.target=target

        self.dwn = DepthwiseNet(self.target, input_size, num_levels, kernel_size=kernel_size, dilation_c=dilation_c)
        self.pointwise = nn.Conv1d(input_size, 1, 1)

        self._attention = th.ones(input_size,1)
        self._attention = Variable(self._attention, requires_grad=False)

        self.fs_attention = th.nn.Parameter(self._attention.data)
        
        if cuda:
            self.dwn = self.dwn.cuda()
            self.pointwise = self.pointwise.cuda()
            self._attention = self._attention.cuda()
                  
    #def init_weights(self):
        #self.pointwise.weight.data.normal_(0, 0.1)       
        
    def forward(self, x):
        #print(x.shape)
        #print(F.softmax(self.fs_attention, dim=0).shape)
        #y=x*F.softmax(self.fs_attention, dim=0)
        #print(y.shape)
        print('x',x.shape)
        y1=self.dwn(x*F.softmax(self.fs_attention, dim=0))
        #y1=self.dwn(x)
        y1 = self.pointwise(y1)
        print('y',y1.shape)
        return y1.transpose(1,2)