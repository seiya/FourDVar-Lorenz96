from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import math

import numpy as np
import lorenz96
import net




import sys
import re


args = sys.argv


path = "./train_1step_ens800_bsize4000" if len(args)==1 else args[1]
path = path + ".pth"


if not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = None


np.random.seed(0)
#np.random.seed(1)

#torch.manual_seed(0)
#torch.manual_seed(1)



k = 40
f = 8.0
dt = 0.01

#nt = 50
nt = 100
int_obs = 5
nobs = int(nt/int_obs) + 1


sigma = 1e-1


nitr = 2


#ndata = 1
ndata = 1000


model = lorenz96.Lorenz96(k, f, dt)
x = model.init(f, 0.01)

xa = np.zeros([nobs,k])
x0 = x




net = net.Net(k, nitr)
stat = torch.load(path, map_location=device)
net.load_state_dict(stat['net'])
net.eval()



# spinup
for n in range(100):
    x = model.forward(x)
xa[0,:] = x
for n in range(nt):
    x = model.forward(x)
    if (n+1)%int_obs == 0:
        no = (n+1)//int_obs
        xa[no,:] = x



#np.random.seed(1)
np.random.seed(2)
for n in range(200):
    np.random.randn(k)



error = np.empty([nobs,ndata])
error2 = np.empty([nobs,ndata])
xall = np.empty([nobs,ndata,k])

for m in range(ndata):
    x = x0 + np.random.randn(k) * sigma
    # spinup
    for n in range(100):
        x = model.forward(x)

    xall[0,m,:] = x
    error2[0,m] = ( ( x - xa[0,:] )**2 ).mean()
    for n in range(nt):
        x = model.forward(x)
        if (n+1)%int_obs == 0:
            no = (n+1)//int_obs
            xall[no,m,:] = x
            error2[no,m] = ( ( x - xa[no,:] )**2 ).mean()

error[0,:] = 0.0
x_m = torch.from_numpy(xall[0,:,:].astype(np.float32))
for n in range(nobs-1):
    with torch.no_grad():
        x_m = net(x_m)
    tmp = x_m.detach().numpy()
    error[n+1,:] = ( ( tmp - xall[n+1,:,:] )**2 ).mean(axis=1)



error = error.mean(axis=1)
error2 = error2.mean(axis=1)

error2 = error2 * error[1] / error2[0]


print(error)
print(error2)

pat = r"train_(.+).pth"
res = re.search(pat, path)
fname = "run_err_" + res.group(1)
print(fname)
np.savez(fname, error)

import matplotlib.pyplot as plt
xx = np.linspace(0, dt*int_obs*(nobs-1), nobs)
plt.plot(xx[1:], error[1:], label="Surogate")
plt.plot(xx, error2, label="Physics")
plt.xlim([0,dt*nt*1.1])
#plt.ylim([1e-3,0.1])
plt.ylim([0.0002, 0.2])
plt.yscale("log")
plt.grid(which="both")
plt.legend()


plt.show()
