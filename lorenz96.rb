require "numru/dcl"
require "npy"
include NumRu
include NMath


npz = Npy.load_npz("lorenz96.npz")
err = npz["arr_0"]
err = NArray.to_na(err.to_binary, "float")

dt = 0.01
t = NArray.sfloat(err.length).indgen * dt

xm = t.mean
xx = t - xm
yy = log(err)
ym = yy.mean
yy = yy - ym

a = (xx*yy).sum / (xx**2).sum
b = ym - xm * a

p a

#iws=1
iws=2
DCL.gropn(iws)
DCL.sglset("lfull", true)
DCL.swlset("lsysfnt", true)

DCL.grfrm
DCL.grsvpt(0.1, 0.95, 0.1, 0.65)
DCL.grswnd(0, t.max, 0.1, 10)
DCL.grstrn(2)
DCL.grstrf

DCL.usdaxs
DCL.uxsttl("b", "time", 0)
DCL.sgplzu(t, err, 1, 3)
DCL.sgplzu(t, exp(a*t+b), 3, 2)

DCL.grcls

