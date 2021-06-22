import glob
import re

import numpy as np

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt


nens = {}
loss_e = {}
num_ep = {}
num_eq = {}
elapse = {}

nint = 10
nlen = 5001
xx = np.linspace(0, (nlen-1)*nint, nlen)
xx[0] = 1


fnames = glob.glob("./train_1step_ens*.npz")
pat = re.compile(r'ens(\d+)_bsize(\d+)\.npz')
pat2 = re.compile(r'elapsed time: (\d+) sec')
for f in fnames:
    mat = pat.findall(f)[0]
    ne = int(mat[0])
    nb = int(mat[1])
    if not nb in nens:
        nens[nb] = []
        loss_e[nb] = {}
        num_ep[nb] = {}
        num_eq[nb] = {}
        elapse[nb] = {}

    npz = np.load(f)
    #l_t = npz['arr_0']
    l_e = npz['arr_1']
    mask = l_e > 0

    l_e = l_e[mask]
    
    idx = np.argmin(l_e)
    lmin = l_e[idx]
    i = np.where(l_e < lmin*1.1)[0][0]

    nens[nb].append(ne)
    loss_e[nb][ne] = lmin
    num_ep[nb][ne] = xx[mask][-1]
    num_eq[nb][ne] = xx[i]

    with open(f"log_ens{ne}_bsize{nb}","r") as f:
        str = f.read()
        mat = pat2.findall(str)[0]
        elapse[nb][ne] = int(mat) / 3600

nbs = nens.keys()



fig = plt.figure()
pl1 = fig.add_subplot(2, 2, 1)
pl2 = fig.add_subplot(2, 2, 2)
pl3 = fig.add_subplot(2, 2, 3)
pl4 = fig.add_subplot(2, 2, 4)


nt = 100 / 5


# batch_size
color = {125: "gray", 250: "red", 500: "green", 1000: "orange", 2000: "purple", 4000: "cyan", 8000: "navy", 16000: "blue"}
# ensemble size
mark = {50: "o", 100: "s", 200: "+", 400: "x", 800: "d"}


for nb in nbs:
    for ne in nens[nb]:
        pl1.scatter([ne*nt], [loss_e[nb][ne]], c=color[nb], marker=mark[ne])
pl1.set(xlabel="# of data", ylabel="loss", xscale="log", yscale="log")

for nb in nbs:
    for ne in nens[nb]:
        pl2.scatter([nb], [loss_e[nb][ne]], c=color[nb], marker=mark[ne])
pl2.set(xlabel="batch size", ylabel="loss", xscale="log", yscale="log")


for nb in nbs:
    for ne in nens[nb]:
        #print(ne,nb)
        el = elapse[nb][ne] * num_eq[nb][ne] / num_ep[nb][ne]
        pl3.scatter([el], [loss_e[nb][ne]], c=color[nb], marker=mark[ne])
pl3.set(xlabel="elapsed time", ylabel="loss", xscale="log", yscale="log")


for nb in nbs:
    for ne in nens[nb]:
        pl4.scatter([nb], [num_eq[nb][ne]], c=color[nb], marker=mark[ne])
pl4.set(xlabel="batch size", ylabel="epoch num", xscale="log", yscale="log")





#plt.title = "Learning curve"

fig.tight_layout()

plt.show()


fig.savefig("learning_curve_1step.png")
