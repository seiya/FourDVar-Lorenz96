from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import sys
import time
import math

import numpy as np
import lorenz96
import net




np.random.seed(0)
torch.manual_seed(0)



k = 40
f = 8.0
dt = 0.01

nt = 50
nt2 = 100
#nt2 = 200

int_obs = 5

nobs = int(nt/int_obs) + 1
nobs2 = int(nt2/int_obs) + 1


nf = nobs2 - nobs + 1



sigma = 1e-1

nitr = 2

double = False
#double = True


args = sys.argv
argn = len(args)

if argn==0:
    print("Usage: train_10step.py [ndata] [batch_size] [init]")
    exit()


ndata_t = int(args[1]) if argn>1 else 100

batch_size_t = int(args[2]) if argn>2 else ndata_t * nf

init = args[3]=="True" if argn>3 else False


#path_net = "./train_1step_ens200_bsize4000.pth"
#path_net = "./train_1step_ens400_bsize4000.pth"
path_net = f"./train_1step_ens{ndata_t}_bsize{ndata_t*10}.pth"




print(f"pth file is {path_net}")
print(f"# of ensembles is {ndata_t}")
print(f"# of batch_size is {batch_size_t}")
print(f"init is {init}")







max_norm = 0.01

max_epoch = 50000
#max_epoch = 1000
#max_epoch = 500
#max_epoch = 1

#ndata_t = 200
#batch_size_t = 200



#ndata_e = 1
ndata_e = 100
batch_size_e = ndata_e * nf

batch_num_t = ndata_t * nf / batch_size_t
batch_num_e = ndata_e * nf / batch_size_e

model = lorenz96.Lorenz96(k, f, dt)
x0 = model.init(f, 0.01)


class DataSet:
    def __init__(self, len: int, nobs: int, nobs2: int, k: int):
        self.len = len
        self.nobs = nobs
        self.nobs2 = nobs2
        self.k = k
        if double:
            self.data = np.zeros([len,nobs2,k])
        else:
            self.data = np.zeros([len,nobs2,k], dtype="float32")
    def __len__(self):
        return self.len * (nobs2 - nobs + 1)
    def __getitem__(self, index):
        nx = self.nobs2 - self.nobs + 1
        i = int(index/nx)
        n = index % nx
        return self.data[i,n:n+self.nobs,:]
    def push(self, i, n, data):
        self.data[i,n,:] = data

#trainning data
print("prepare trainning data")
data_t = DataSet(ndata_t, nobs, nobs2, k)
for m in range(ndata_t):
    x = x0 + np.random.randn(k) * sigma
    # spinup
    for n in range(100):
        x = model.forward(x)

    data_t.push(m,0,x)
    for n in range(nt2):
        x = model.forward(x)
        if (n+1)%int_obs == 0:
            data_t.push(m,int((n+1)/int_obs),x)


# evaluation data
print("prepare evaluation data")
data_e = DataSet(ndata_e, nobs, nobs2, k)
for m in range(ndata_e):
    x = x0 + np.random.randn(k) * sigma
    # spinup
    for n in range(100):
        x = model.forward(x)

    data_e.push(m,0,x)
    for n in range(nt2):
        x = model.forward(x)
        if (n+1)%int_obs == 0:
            data_e.push(m,int((n+1)/int_obs),x)



loader_t = torch.utils.data.DataLoader(data_t, batch_size=batch_size_t, shuffle=True)
loader_e = torch.utils.data.DataLoader(data_e, batch_size=batch_size_e)



stat = torch.load(path_net)


net = net.Net(k, nitr)
#net = net.Net(k, nitr, 0.0001)
if not init:
    net.load_state_dict(stat['net'])
if double:
    net = net.double()

criterion = nn.MSELoss()

if init:
    lr = 0.001 * batch_size_t / 1000
else:
    lr = 0.01 * batch_size_t / 1000

optimizer = optim.Adam(net.parameters(), lr=lr)
#optimizer = optim.Adam(net.parameters(), lr=0.01)
#optimizer = optim.Adam(net.parameters(), lr=0.0001)
#optimizer.load_state_dict(stat['opt'])

#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)


nint = 10
nint2 = 50
#nint2 = 500

print("start trainning")

start = time.time()


loss_t = np.zeros(int(max_epoch/nint+1))
loss_e = np.zeros(int(max_epoch/nint+1))
large = 999.9e5
min = [large, 0, 0]
min_tmp = large
unchange = 0

for epoch in range(max_epoch):

    if init:
        net.train()
        net.drop = True
    else:
        net.eval()
        net.drop = False

    optimizer.zero_grad()
    running_loss_t = 0.0
    for data in loader_t:

        out = data[:,0,:]
        #out.requires_grad = True
        #tmp = out
        loss = 0.0
        for n in range(nobs-1):
            out = net(out)
            true = data[:,n+1,:]
            norm = criterion(out, true)
            loss += norm
            #print(n, norm.item())

        loss.backward()

        nn.utils.clip_grad_norm_(net.parameters(), max_norm)
        optimizer.step()
        running_loss_t += loss.item()

    scheduler.step()

    if (epoch+1)%nint == 0 or epoch==0:

        #print(torch.norm(tmp).item(), torch.norm(tmp.grad).item())

        net.eval()
        net.drop = False

        with torch.no_grad():
            running_loss_e = 0.0
            for data in loader_e:
                loss = 0.0
                out = data[:,0,:]
                for n in range(nobs-1):
                    out = net(out)
                    loss += criterion(out, data[:,n+1,:])
                running_loss_e += loss.item()

        l_t = running_loss_t / ( batch_num_t * (nobs-1) )
        l_e = running_loss_e / ( batch_num_e * (nobs-1) )
        loss_t[int((epoch+1)/nint)] = l_t
        loss_e[int((epoch+1)/nint)] = l_e
        if epoch > 0 and l_e < min[0]:
            min = [l_e, l_t, epoch+1]
            state = {
                'net': net.state_dict(),
                'opt': optimizer.state_dict(),
                'sch': scheduler.state_dict(),
            }
            unchange = 0

        if l_e < min_tmp:
            min_tmp = l_e

        if (epoch+1)%(math.ceil(max_epoch/nint2)) == 0 or epoch==0:
            print('[%d] lr: %.2e, training: %.6f, eval: %.6f (%.6f, %.6f)' % (epoch + 1, scheduler.get_last_lr()[0], l_t, l_e, min_tmp, min[0]))
            if min_tmp > min[0]:
                unchange += 1
            if ( epoch > 5000 and min_tmp > min[0] * 1.5 ) or unchange >= 10:
                break
            min_tmp = large


print("minimam loss: %.6f, %.6f, %d"%(min[0], min[1], min[2]))
print(f"elapsed time: %d sec"%(time.time() - start))


fname = f"train_ens{ndata_t}_bsize{batch_size_t}_init{init}"

path = fname+".pth"
torch.save(state, path)

np.savez(fname, loss_t, loss_e)
