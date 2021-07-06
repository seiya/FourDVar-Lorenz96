import glob
import re

import numpy as np

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt


key = []
acc = []
nit = []


fnames = glob.glob("./4dvar_ens*.npz")
pat = re.compile(r'ens(\d+)_bsize(\d+)_lint(\d+)_lr(.+)\.npz')
for f in fnames:
    mat = pat.findall(f)[0]
    ne = int(mat[0])
    nb = int(mat[1])
    li = int(mat[2])
    lr = float(mat[3])

    npz = np.load(f)
    l_p = npz['arr_1']

    if l_p.max() < 1:
        key.append([ne,li,lr])
        acc.append(l_p[-1])
        tmp = np.where(l_p<1e-3)[0]
        if tmp.size > 0:
            nit.append(tmp[0])
        else:
            nit.append(-999)

nacc = len(acc)


fig = plt.figure()
pl1 = fig.add_subplot(2, 2, 1)
pl2 = fig.add_subplot(2, 2, 2)
pl3 = fig.add_subplot(2, 2, 3)
pl4 = fig.add_subplot(2, 2, 4)



# [ens, lr]
mark = {
    50:  {1e-4:"h", 3e-5: "o", 1e-5:"."},
    100: {1e-4:">", 3e-5: "<", 1e-5:"^"},
    200: {1e-4:"+", 3e-5: "*", 1e-5:"x"},
    400: {1e-4:"1", 3e-5: "3", 1e-5:"2"},
    800: {1e-4:"s", 3e-5: "D", 1e-5:"d"},
    }
#    50:  {1e-4:"h", 5e-5: ".", 2e-5:"o", 1e-5:"."},
#    100: {1e-4:">", 5e-5: "<", 2e-5:"v", 1e-5:"^"},
#    200: {1e-4:"_", 5e-5: "*", 2e-5:"+", 1e-5:"x"},
#    400: {1e-4:"4", 5e-5: "3", 2e-5:"1", 1e-5:"2"},
#    800: {1e-4:"p", 5e-5: "D", 2e-5:"s", 1e-5:"d"},
#    }

# [lint]
color = {
    1:   "gray",
#    2:   "navy",
    3:   "navy",
#    5:   "pink",
    10:  "red",
#    20:  "green",
    30:  "green",
#    50:  "orange",
    100: "blue",
#    200: "cyan",
    300: "cyan",
}

    
ymin, ymax = 1e-5, 1e-2

for n in range(nacc):
    ens, lint, lr = key[n]
    c = color[lint]
    m = mark[ens][lr]
    pl1.scatter([ens], [acc[n]], c=c, marker=m)
pl1.set_ylim(ymin,ymax)
pl1.set(xlabel="# of ensembles", ylabel="error", xscale="log", yscale="log")

for n in range(nacc):
    ens, lint, lr = key[n]
    c = color[lint]
    m = mark[ens][lr]
    pl2.scatter([lint], [acc[n]], c=c, marker=m)
pl2.set_ylim(ymin,ymax)
pl2.set(xlabel="L", ylabel="error", xscale="log", yscale="log")


for n in range(nacc):
    ens, lint, lr = key[n]
    c = color[lint]
    m = mark[ens][lr]
    pl3.scatter([lr], [acc[n]], c=c, marker=m)
pl3.set_ylim(ymin,ymax)
pl3.set(xlabel="learning rate", ylabel="error", xscale="log", yscale="log")


for n in range(nacc):
    ens, lint, lr = key[n]
    c = color[lint]
    m = mark[ens][lr]
#    pl4.scatter([lint], [nit[n]], c=c, marker=m)
    pl4.scatter([ens], [nit[n]], c=c, marker=m)
pl4.set_ylim(20,1e4)
pl4.set(xlabel="L", ylabel="# of iteration", xscale="log", yscale="log")





#plt.title = "Learning curve"

fig.tight_layout()

plt.show()


#fig.savefig("accuracy_4dvar.png")
