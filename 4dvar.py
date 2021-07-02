from __future__ import print_function

import numpy as np
import lorenz96


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import net

import sys
import math


args = sys.argv
argn = len(args)

ens = int(args[1]) if argn>1 else 800
bsize = int(args[2]) if argn>2 else 4400
lint = int(args[3]) if argn>3 else 10
lr = float(args[4]) if argn>4 else 0.0001
cuda = int(args[5]) if argn>5 else None

path = f"train_10step_ens{ens}_bsize{bsize}_initFalse_1.pth"

np.random.seed(0)
torch.manual_seed(0)


k = 40
f = 8.0
dt = 0.01

fact0 = 1.0

gfact = ( 1.0 + math.sqrt(5) ) * 0.5
gfact1 = gfact + 1.0

nitr = 2

nt = 50
int_obs = 5
nobs = int(nt/int_obs) + 1

nstep = 1000



double = False
#double = True

def phy(x):
    l = 0.0
    for n in range(nt):
        x = model.forward(x)
        if (n+1)%int_obs == 0:
            diff = x - xt_obs[(n+1)//int_obs,:]
            l += (diff**2).sum() / k
    return l

def sur(x):
    l = 0.0
    with torch.no_grad():
        for n in range(nobs-1):
            x = net(x)
            diff = x - xt_obs2[n+1,:]
            l += (diff**2).sum() / k
    return l.item()


def min_exp(x0, dldx, fact, l, met):
    f = [0.0, fact/gfact1, fact]
    ll = [l, 0.0, 0.0]
    for i in range(2):
        x = x0 - dldx * f[i+1]
        ll[i+1] = met(x)
    #print(f,ll)
    if ll[0] < ll[1] and ll[0] < ll[2]:
        for i in range(20):
            ll[2] = ll[1]
            f[2] = f[1]
            f[1] = ( f[1] - f[0] ) / gfact1
            if f[1] < 1e-5:
                #print("why???")
                return [f[2], ll[2]]
            x = x0 - dldx * f[1]
            ll[1] = met(x)
            if ll[1] < ll[0]:
                break
            #print(f,ll)

    if ll[0] < ll[1] and ll[0] < ll[2]:
        print("grad is not valid", ll, f)
        exit()

    if ll[2] < ll[1] and ll[2] < ll[0]:
        for i in range(20):
            ll[1] = ll[2]
            f[1] = f[2]
            f[2] = f[1] * gfact1
            x = x0 - dldx * f[2]
            ll[2] = met(x)
            if ll[2] > ll[1]:
                break
    if ll[2] < ll[1] and ll[2] < ll[0]:
        print("fact is quite large", ll, f)
        exit()

    for i in range(100):
        if f[1] - f[0] > f[2] - f[1]:
            ff = f[1] - ( f[2] - f[1] ) / gfact
            x = x0 - dldx * ff
            lll = met(x)
            if lll > ll[1]:
                f[0] = ff
                ll[0] = lll
            elif lll < ll[1]:
                f[2] = f[1]
                ll[2] = ll[2]
                f[1] = ff
                ll[1] = lll
            else:
                break
        else:
            ff = f[1] + ( f[1] - f[0] ) / gfact
            x = x0 - dldx * ff
            lll = met(x)
            if lll > ll[1]:
                f[2] = ff
                ll[2] = lll
            elif lll < ll[1]:
                f[0] = f[1]
                ll[0] = ll[0]
                f[1] = ff
                ll[1] = lll
            else:
                break

    return [f[1],ll[1]]




model = lorenz96.Lorenz96(k, f, dt)
x0 = model.init(f, 0.01)

np.random.seed(100)

xt = x0
x = x0 + np.random.randn(k) * 1e-1
# spinup
for n in range(100):
    xt = model.forward(xt)
    x = model.forward(x)


xt_obs = np.zeros([nobs,k])
xt_obs[0,:] = xt
for n in range(nt):
    xt = model.forward(xt)
    if (n+1)%int_obs == 0:
        xt_obs[(n+1)//int_obs,:] = xt



if torch.cuda.is_available():
    if cuda:
        dev = f"cuda:{cuda}"
    else:
        dev = f"cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)

net = net.Net(k, nitr, device=device)
stat = torch.load(path, map_location=device)
net.load_state_dict(stat['net'])
if double:
    net.double()

net = net.to(device)

net.eval()
for param in net.parameters():
    param.requires_grad = False

optimizer = optim.SGD(net.parameters(), lr)
criterion = nn.MSELoss()

if double:
    rp = np.float64
    torch.set_default_tensor_type(torch.DoubleTensor)
else:
    rp = np.float32
    torch.set_default_tensor_type(torch.FloatTensor)

xt_obs2 = torch.from_numpy(xt_obs.astype(rp)).to(device)


fact = fact0


cost_sur = np.empty(nstep)
cost_phy = np.empty(nstep)



x = torch.from_numpy(x.astype(rp)).reshape(1,k).to(device)

x = x.detach()
for m in range(nstep):

    #x = x.requires_grad_()
    x.requires_grad = True
    x0 = x

    cost = 0.0
    for n in range(nobs-1):
        x = net(x)
        cost += criterion(x, xt_obs2[n+1:n+2,:])

    cost.backward()
    dldx = x0.grad

    cost = cost.item()
    if math.isnan(cost):
        break

    cost_sur[m] = cost / (nobs-1)
    cost_phy[m] = phy(x0[0,:].to("cpu").detach().numpy()) / (nobs-1)

    if m==0 or (m+1)%20 == 0:
        print( f"[%04d]: %e, %e"%(m+1, cost_sur[m], cost_phy[m]) )

    with torch.no_grad():
        fact, _ = min_exp(x0, dldx, fact, cost, sur)
        #print(fact)
        x = x0 - dldx * fact

    x = x.detach()
    fact = max(fact * gfact, 0.1)

    # update surrogate model
    if (m+1)%lint == 0:
        obs = np.zeros([nobs,k])
        x1 = x[0,:].to("cpu").numpy()
        obs[0,:] = x1
        for n in range(nt):
            x1 = model.forward(x1)
            if (n+1)%int_obs == 0:
                obs[(n+1)//int_obs,:] = x1
        obs = torch.from_numpy(obs.astype(rp)).to(device)

        for param in net.parameters():
            param.requires_grad = True
        optimizer.zero_grad()
        loss = 0.0
        out = x
        for n in range(nobs-1):
            out = net(out)
            loss += criterion(out, obs[n+1:n+2,:])
        loss.backward()
        optimizer.step()
        #scheduler.step()
        for param in net.parameters():
            param.requires_grad = False



fname = f"4dvar_ens{ens}_bsize{bsize}_lint{lint}_lr{lr}"
path = fname+".pth"
state = {
    'net': net.state_dict(),
}
torch.save(state, path)
np.savez(fname, cost_sur, cost_phy)


exit()


import matplotlib as mpl
import matplotlib.pyplot as plt


fig = plt.figure()

xx = np.arange(nstep)



plt.plot(xx, cost_sur, color="blue", label="surrogate")
plt.plot(xx, cost_phy, color="red", label="physical")
plt.yscale("log")
plt.title("error")
#plt.xlim([0,max_epoch+1])
#plt.ylim([0.0008,0.1])
plt.xlabel("iteration")
plt.ylabel("loss")
plt.legend(loc="upper right")
plt.show()

