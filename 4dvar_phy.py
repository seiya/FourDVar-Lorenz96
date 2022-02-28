import numpy as np
import lorenz96


import sys
import math
import random


args = sys.argv
argn = len(args)

random.seed(0)
np.random.seed(0)


k = 40
f = 8.0
dt = 0.01

sigma = 0.1
obs_sigma = 0.1



fact0 = 1.0

gfact = ( 1.0 + math.sqrt(5) ) * 0.5
gfact1 = gfact + 1.0


nt = 50
int_obs = 5
nobs = int(nt/int_obs) + 1

nstep = 1000
#nstep = 100
int_out = 10


#nseeds = 50
nseeds = 20


double = False
#double = True

nobsloc = 10
l = list(range(k))
idx = random.sample(l,nobsloc)
idx.sort()
print("obsloc:", idx)

def phy(x, guess):
    xx = x - guess
    binvx = np.dot(Binv, xx)
    l = np.dot(xx, binvx) * 0.5
    #l = 0.0
    for n in range(nt):
        x = model.forward(x)
        if (n+1)%int_obs == 0:
            #diff = x - xt_obs[(n+1)//int_obs,:]
            #l += (diff**2).sum() / k
            diff = x[idx] - xt_obs[(n+1)//int_obs,idx]
            l += (diff**2).sum() * Rinv * 0.5
    return l

def min_exp(x0, dldx, fact, l, met, guess):
    fact = min(fact, 1.0/abs(dldx).max())
    f = [0.0, fact/gfact1, fact]
    ll = [l, 0.0, 0.0]
    for i in range(2):
        x = x0 - dldx * f[i+1]
        ll[i+1] = met(x, guess)
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
            ll[1] = met(x, guess)
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
            ll[2] = met(x, guess)
            if ll[2] > ll[1]:
                break
    if ll[2] < ll[1] and ll[2] < ll[0]:
        print("fact is quite large", ll, f)
        exit()

    for i in range(100):
        if f[1] - f[0] > f[2] - f[1]:
            ff = f[1] - ( f[2] - f[1] ) / gfact
            x = x0 - dldx * ff
            lll = met(x, guess)
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
            lll = met(x, guess)
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


if double:
    rp = np.float64
else:
    rp = np.float32



model = lorenz96.Lorenz96(k, f, dt)
x0 = model.init(f, 0.01)

seeds = list( range(100,100+nseeds) )


x_gues = np.empty([nseeds,k]).astype(rp)
cost_phy = np.empty([nseeds,nstep])

xt = x0
# spinup
for n in range(100):
    xt = model.forward(xt)
x_true = xt
xt_obs = np.zeros([nobs,k])
xt_obs[0,:] = xt


for n in range(nt):
    xt = model.forward(xt)
    if (n+1)%int_obs == 0:
        #xt_obs[(n+1)//int_obs,:] = xt
        xt_obs[(n+1)//int_obs,:] = xt + np.random.randn(k) * obs_sigma

xt_obs2 = xt_obs.astype(rp)

Rinv = 1.0 / obs_sigma**2
npz = np.load("data/init_norm.npz")
Binv = npz['arr_1']


nout = nstep//int_out + 1
x_out = np.empty([nseeds,nout,k])

for s in range(nseeds):
    seed = seeds[s]
    print("seed=",seed)

    np.random.seed(seed)

    x = x0 + np.random.randn(k) * sigma
    x = x.astype(rp)
    # spinup
    for n in range(100):
        x = model.forward(x)

    fact = fact0

    xa = np.empty([nt+1,k], dtype=rp)
    dldx = np.empty(k, dtype=rp)

    x = x.astype(rp)
    guess = x

    x_out[s,0,:] = x
    for m in range(nstep):

        x00 = x
        xa[0,:] = x
        for n in range(nt):
            x = model.forward(x)
            xa[n+1,:] = x

        dldx[:] = 0.0
        #cost = 0.0
        xx = x00 - guess
        binvx = np.dot(Binv, xx)
        cost = np.dot(xx, binvx) * 0.5
        #print(cost)
        for nn in range(nobs-1):
            n = nobs - nn - 1
            na = n * int_obs
            #tmp = xa[na,:] - xt_obs2[n,:]
            #cost += ( tmp**2 ).sum() / k
            #dldx += 2.0 * tmp / k
            tmp = xa[na,idx] - xt_obs2[n,idx]
            cost += ( tmp**2 ).sum() * Rinv * 0.5
            #print(nn, cost, abs(tmp*Rinv).max())
            dldx[idx] += tmp * Rinv
            for nnn in range(int_obs):
                dldx = model.adjoint(dldx, xa[na-nnn-1,:])
        dldx += binvx

        if math.isnan(cost):
            print(cost)
            exit()
            break

        cost_phy[s,m] = cost / (nobs-1)

        if m==0 or (m+1)%50 == 0:
            norm = ( ( x00 - x_true )**2 ).mean()
            print( f"[%04d]: %e, %e"%(m+1, cost_phy[s,m], norm) )

        fact, _ = min_exp(x00, dldx, fact, cost, phy, guess)
        #print(fact)
        x = x00 - dldx * fact

        if (m+1)%int_out == 0:
            x_out[s,(m+1)//int_out,:] = x

        fact = max(fact * gfact, fact0)


    x_gues[s,:] = x



fname = f"data/4dvar_phy"
if double:
    fname = fname + "_double"
np.savez(fname, cost_phy, x_true, x_gues, x_out)


exit()


import matplotlib as mpl
import matplotlib.pyplot as plt


fig = plt.figure()

xx = np.arange(nstep)



plt.plot(xx, cost_phy, color="red", label="physical")
plt.yscale("log")
plt.title("error")
#plt.xlim([0,max_epoch+1])
#plt.ylim([0.0008,0.1])
plt.xlabel("iteration")
plt.ylabel("loss")
plt.legend(loc="upper right")
plt.show()

