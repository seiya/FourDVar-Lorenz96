from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import math
import sys
import time

import numpy as np
import lorenz96
import net


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = None


#torch.autograd.set_detect_anomaly(True)



np.random.seed(0)
torch.manual_seed(0)


k = 40
f = 8.0
dt = 0.01

nt = 100
int_obs = 5
nobs = int(nt/int_obs) + 1

sigma = 1e-1

nitr = 2



max_epoch = 50000

args = sys.argv
argn = len(args)

ndata_t = int(args[1]) if argn>1 else 100
#ndata_t = 800
#ndata_t = 400
#ndata_t = 200
#ndata_t = 100
#ndata_t = 20
#ndata_t = 10
#ndata_t = 5
#ndata_t = 1

batch_size_t = int(args[2]) if argn>2 else 100
#batch_size_t = ndata_t * (nobs-1)
#batch_size_t = 16000
#batch_size_t = 8000
#batch_size_t = 4000
#batch_size_t = 2000
#batch_size_t = 400
#batch_size_t = 200
#batch_size_t = 100

if batch_size_t > ndata_t * (nobs-1):
    batch_size_t = ndata_t * (nobs-1)


ndata_e = 100
batch_size_e = ndata_e * (nobs-1)

batch_num_t = (nobs-1) * ndata_t // batch_size_t
batch_num_e = (nobs-1) * ndata_e // batch_size_e


print(f"# of ensembles is {ndata_t}")
print(f"# of batch_size is {batch_size_t}")



model = lorenz96.Lorenz96(k, f, dt)
x0 = model.init(f, 0.01)



class DataSet:
    def __init__(self, len: int, nobs: int, k: int):
        self.len = len
        self.nobs = nobs
        self.k = k
        self.data = np.zeros([len,nobs,k], dtype="float32")
    def __len__(self):
        return self.len * (self.nobs-1)
    def __getitem__(self, index):
        i = int(index/(self.nobs-1))
        n = index % (self.nobs-1)
        return self.data[i,n,:], self.data[i,n+1,:]
    def push(self, i, n, data):
        self.data[i,n,:] = data

#training data
print("prepare training data")
data_t = DataSet(ndata_t, nobs, k)
for m in range(ndata_t):
    x = x0 + np.random.randn(k) * sigma
    # spinup
    for n in range(100):
        x = model.forward(x)

    data_t.push(m,0,x)
    for n in range(nt):
        x = model.forward(x)
        if (n+1)%int_obs == 0:
            data_t.push(m,int((n+1)/int_obs),x)


# evaluation data
print("prepare evaluation data")
data_e = DataSet(ndata_e, nobs, k)
for m in range(ndata_e):
    x = x0 + np.random.randn(k) * sigma
    # spinup
    for n in range(100):
        x = model.forward(x)
    data_e.push(m,0,x)
    for n in range(nt):
        x = model.forward(x)
        if (n+1)%int_obs == 0:
            data_e.push(m,int((n+1)/int_obs),x)


loader_t = torch.utils.data.DataLoader(data_t, batch_size=batch_size_t, shuffle=True)
loader_e = torch.utils.data.DataLoader(data_e, batch_size=batch_size_e)


net = net.Net(k, nitr, device=device)
if device:
    net = net.to(device)

criterion = nn.MSELoss()

lr = 0.01 * batch_size_t / 1000
optimizer = optim.Adam(net.parameters(), lr=lr)
#optimizer = optim.Adam(net.parameters(), lr=0.01)
#optimizer = optim.Adam(net.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)


nint = 10

print("start training")

start = time.time()

loss_t = np.zeros(int(max_epoch/nint+1))
loss_e = np.zeros(int(max_epoch/nint+1))
min = [999.9, 0, 0]
min_tmp = 999.9
unchange = 0


for epoch in range(max_epoch):

    net.train()

    optimizer.zero_grad()
    running_loss_t = 0.0
    for data in loader_t:
        if device:
            data = [data[0].to(device), data[1].to(device)]
        out = net(data[0])
        loss = criterion(out, data[1])
        loss.backward()
        # nn.utils.clip_grad_norm_(net.parameters(), max_norm)
        optimizer.step()
        running_loss_t += loss.item()

    scheduler.step()

    if (epoch+1)%nint == 0 or epoch==0:

        net.eval()
        net.drop = False

        with torch.no_grad():
            running_loss_e = 0.0
            for data in loader_e:
                if device:
                    data = [data[0].to(device), data[1].to(device)]
                out = net(data[0])
                loss = criterion(out, data[1])
                running_loss_e += loss.item()

        l_t = running_loss_t / batch_num_t
        l_e = running_loss_e / batch_num_e
        loss_t[int((epoch+1)/nint)] = l_t
        loss_e[int((epoch+1)/nint)] = l_e
        if l_e < min[0]:
            min = [l_e, l_t, epoch+1]
            state = {
                'net': net.state_dict(),
                'opt': optimizer.state_dict(),
                'sch': scheduler.state_dict(),
            }
            unchange = 0

        if l_e < min_tmp:
            min_tmp = l_e

        if (epoch+1)%(math.ceil(max_epoch/50)) == 0 or epoch==0:
            print('[%d] lr: %.2e, training: %.6f, eval: %.6f (%.6f, %.6f)' % (epoch + 1, scheduler.get_last_lr()[0], l_t, l_e, min_tmp, min[0]))
            if min_tmp > min[0]:
                unchange += 1
#            if min_tmp > min[0] * 1.5 or unchange >= 5:
            if unchange >= 10:
                break
            min_tmp = 999.9




#print(min)
print("minimam loss: %.6f, %.6f, %d"%(min[0], min[1], min[2]))
print(f"elapsed time: %d sec"%(time.time() - start))


fname = f"train_1step_ens{ndata_t}_bsize{batch_size_t}"

#path = "./perfect_1step.pth"
path = fname+".pth"
torch.save(state, path)

np.savez(fname, loss_t, loss_e)

print("finish saving")

exit()

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


fig = plt.figure()

xx = np.linspace(0,max_epoch,int(max_epoch/nint+1))
xx[0] = 1

mask = loss_t>0
xx = xx[mask]
loss_t = loss_t[mask]
loss_e = loss_e[mask]


plt.plot(xx, loss_t, color="blue", label="train")
plt.plot(xx, loss_e, color="red", label="test")
plt.yscale("log")
plt.title("Learning curve")
plt.xlim([0,max_epoch+1])
plt.ylim([0.0001,1.0])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(loc="upper right")
#plt.show()

path = fname+".png"
fig.savefig(path)
