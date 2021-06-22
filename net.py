import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class Periodic(nn.Module):
    def __init__(self, device=None):
        super(Periodic, self).__init__()
        self.device = device

    def forward(self, x):
       n, c, l = x.size()
       y = torch.empty(n,c,l+2, device=self.device)
       y[:,:,1:-1] = x
       y[:,:,0] = x[:,:,-1]
       y[:,:,-1] = x[:,:,0]
       return y



class Net(nn.Module):

    def __init__(self, ne: int, nitr: int, device=None, p=0.5):
        super(Net, self).__init__()

        self.nitr = nitr
        self.drop = None
        self.p = p

        block11 = [None] * nitr
        block12 = [None] * nitr
        block13 = [None] * nitr
        block14 = [None] * nitr
        block5  = [None] * nitr
        block25 = [None] * nitr
        block24 = [None] * nitr
        block23 = [None] * nitr
        block22 = [None] * nitr
        block21 = [None] * nitr

        for n in range(nitr):

            # L=40
            block12[n] = nn.Sequential(
                Periodic(device),
                nn.Conv1d(4, 8, kernel_size=3, bias=False),
                nn.BatchNorm1d(8))
            # L=20
            block13[n] = nn.Sequential(
                nn.ReLU(inplace=True),
                Periodic(device),
                nn.Conv1d(8, 8, kernel_size=3, bias=False),
                nn.BatchNorm1d(8),
                nn.ReLU(inplace=True),
                Periodic(device),
                nn.Conv1d(8, 16, kernel_size=3, bias=False),
                nn.BatchNorm1d(16))
            # L=10
            block14[n] = nn.Sequential(
                nn.ReLU(inplace=True),
                Periodic(device),
                nn.Conv1d(16, 16, kernel_size=3, bias=False),
                nn.BatchNorm1d(16),
                nn.ReLU(inplace=True),
                Periodic(device),
                nn.Conv1d(16, 16, kernel_size=3, bias=False),
                nn.BatchNorm1d(16),
                nn.ReLU(inplace=True),
                Periodic(device),
                nn.Conv1d(16, 32, kernel_size=3, bias=False),
                nn.BatchNorm1d(32))
            # L=5
            block5[n] = nn.Sequential(
                nn.ReLU(inplace=True),
                Periodic(device),
                nn.Conv1d(32, 32, kernel_size=3, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                Periodic(device),
                nn.Conv1d(32, 32, kernel_size=3, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                Periodic(device),
                nn.Conv1d(32, 32, kernel_size=3, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                Periodic(device),
                nn.Conv1d(32, 32, kernel_size=3, bias=False))
            # L=10
            block24[n] = nn.Sequential(
                nn.ReLU(inplace=True),
                Periodic(device),
                nn.Conv1d(32, 16, kernel_size=3, bias=False),
                nn.BatchNorm1d(16),
                nn.ReLU(inplace=True),
                Periodic(device),
                nn.Conv1d(16, 16, kernel_size=3, bias=False),
                nn.BatchNorm1d(16),
                nn.ReLU(inplace=True),
                Periodic(device),
                nn.Conv1d(16, 16, kernel_size=3, bias=False),
                nn.BatchNorm1d(16))
            # L=20
            block23[n] = nn.Sequential(
                nn.ReLU(inplace=True),
                Periodic(device),
                nn.Conv1d(16, 8, kernel_size=3, bias=False),
                nn.BatchNorm1d(8),
                nn.ReLU(inplace=True),
                Periodic(device),
                nn.Conv1d(8, 8, kernel_size=3, bias=False),
                nn.BatchNorm1d(8))
            # L=40
            block22[n] = nn.Sequential(
                nn.ReLU(inplace=True),
                Periodic(device),
                nn.Conv1d(8, 4, kernel_size=3, bias=False),
                nn.BatchNorm1d(4))

        # L=40
        block11 = nn.Sequential(
            nn.Conv1d(1,4, kernel_size=1, bias=False),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True))
        # L=40
        block21 = nn.Sequential(
            nn.Conv1d(4,1, kernel_size=1))

        self.block11 = block11
        self.block12 = nn.ModuleList(block12)
        self.block13 = nn.ModuleList(block13)
        self.block14 = nn.ModuleList(block14)
        self.block5  = nn.ModuleList(block5)
        self.block25 = nn.ModuleList(block25)
        self.block24 = nn.ModuleList(block24)
        self.block23 = nn.ModuleList(block23)
        self.block22 = nn.ModuleList(block22)
        self.block21 = block21


    def __drop(self):
        if self.drop is not None:
            return self.drop
        else:
            return self.training

    def forward(self, x):
        x = x[:,None,:]
        x1 = self.block11(x)
        for n in range(self.nitr):
            x2 = self.block12[n](x1) # L=40, C=4->8
            x3, i3 = F.max_pool1d(x2, kernel_size=2, return_indices=True)
            x3 = self.block13[n](x3) # L=20, C=8->16
            x4, i4 = F.max_pool1d(x3, kernel_size=2, return_indices=True)
            x4 = self.block14[n](x4) # L=10, C=16->32
            x5, i5 = F.max_pool1d(x4, kernel_size=2, return_indices=True)
            x5 = self.block5[n](x5) # L=5, C=32
            x5 = F.dropout(x5, training=self.__drop(), p=self.p)
            x5 = F.max_unpool1d(x5, i5, kernel_size=2)
            x4 = x5 + x4
            x4 = self.block24[n](x4) # L=10, C=32->16
            x4 = F.max_unpool1d(x4, i4, kernel_size=2)
            x3 = x4 + x3
            x3 = self.block23[n](x3) # L=20, C=16->8
            x3 = F.max_unpool1d(x3, i3, kernel_size=2)
            x2 = x3 + x2
            x2 = self.block22[n](x2) # L=40, C=8->4
            x1 = F.relu(x2 + x1)
            if n != self.nitr-1:
                x1 = F.dropout(x1, training=self.__drop(), p=self.p)

        x = self.block21(x1)
        x = x[:,0,:]

        return x


