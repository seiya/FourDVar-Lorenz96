import sys

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt



args = sys.argv
argn = len(args)


ens = 400
bsize = 4400
lint = 10

fname = args[1] if argn>1 else f"4dvar_ens{ens}_bsize{bsize}_l{lint}"
nstep = int(args[2]) if argn>2 else -1


path = fname+".npz"
npz = np.load(path)

cost_sur = npz['arr_0']
cost_phy = npz['arr_1']


if nstep == -1:
    nstep = len(cost_sur)


cost_sur = cost_sur[:nstep]
cost_phy = cost_phy[:nstep]



fig = plt.figure()

xx = np.arange(nstep)

mask = cost_sur < 10


cost_sur = cost_sur[mask]
cost_phy = cost_phy[mask]
xx = xx[mask]

cost_sur = cost_sur[1:]
cost_phy = cost_phy[1:]
xx = xx[1:]


plt.plot(xx, cost_sur, color="blue", label="surrogate")
plt.plot(xx, cost_phy, color="red", label="physical")
#plt.xscale("log")
plt.yscale("log")
plt.title("error")
#plt.xlim([0,max_epoch+1])
#plt.ylim([0.0008,0.1])
plt.xlabel("iteration")
plt.ylabel("loss")
plt.legend(loc="upper right")
plt.show()

