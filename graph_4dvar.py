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


path = fname+".npz"
npz = np.load(path)

cost_sur = npz['arr_0']
cost_phy = npz['arr_1']


nstep = len(cost_sur)


fig = plt.figure()

xx = np.arange(nstep)

mask = cost_sur > 1e-10


xx = xx[mask]


plt.plot(xx, cost_sur[mask], color="blue", label="surrogate")
plt.plot(xx, cost_phy[mask], color="red", label="physical")
plt.yscale("log")
plt.title("error")
#plt.xlim([0,max_epoch+1])
#plt.ylim([0.0008,0.1])
plt.xlabel("iteration")
plt.ylabel("loss")
plt.legend(loc="upper right")
plt.show()

