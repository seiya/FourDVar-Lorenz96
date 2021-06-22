import sys

import numpy as np

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt


args = sys.argv
argn = len(args)

fname = args[1] if argn>1 else "perfect_1step_ens400_bsize4000"

#ndata_t = int(args[1]) if argn>1 else 100
#batch_size_t = int(args[2]) if argn>2 else 2000
#fname = f"perfect_1step_ens{ndata_t}_bsize{batch_size_t}"

path = fname+".npz"
npz = np.load(path)

files = npz.files

loss_t = npz['arr_0']
loss_e = npz['arr_1']

nint = 10
nlen = len(loss_t)


fig = plt.figure()

xx = np.linspace(0, (nlen-1)*nint, nlen)

xx[0] = 1

mask = loss_t>0
xx = xx[mask]
loss_t = loss_t[mask]
loss_e = loss_e[mask]


idx = np.argmin(loss_e)
print(xx[idx], loss_e[idx], loss_t[idx])

plt.plot(xx, loss_t, color="blue", label="train")
plt.plot(xx, loss_e, color="red", label="test")
plt.yscale("log")
plt.title("Learning curve")
plt.xlim([0,nlen*nint+1])
#plt.ylim([0.008, 0.05])
#plt.ylim([0.0008,0.1])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(loc="upper right")
plt.show()

#path = fname+".png"
#fig.savefig(path)
