require "numru/dcl"
require "npy"
include NumRu
include NMath



ens = [50, 100, 200, 400, 800]

iws = ARGV.delete("-pdf") ? 2 : 1
#iws = 2
DCL.swcset("fname", "learning")
DCL.gropn(-iws)
DCL.sglset("lfull", true)
DCL.swlset("lsysfnt", true)
DCL.uzfact(0.7)

label = "a".ord

vxmin0 = 0.09



err1 = {}
ens.each do |en|
  err1[en] = {}
  for f in Dir["./train_1step_ens#{en}_*.npz"]
    unless /ens#{en}_bsize(\d+)\.npz/ =~ f
      raise "invalid filename: ", f
    end
    nb = $1.to_i
    npz = Npy.load_npz(f)
    l_e = npz['arr_1']
    mask = l_e > 0
    l_e = l_e[mask]
    lmin = l_e.min
    err1[en][nb] = lmin
  end
end

err10 = {}
ens.each do |en|
  err10[en] = {}
  for f in Dir["./train_10step_ens#{en}_*_initFalse_1_fin.npz"]
    unless /ens#{en}_bsize(\d+)_/ =~ f
      raise "invalid filename: ", f
    end
    nb = $1.to_i
    npz = Npy.load_npz(f)
    l_e = npz['arr_1']
    mask = l_e > 0
    l_e = l_e[mask]
    lmin = l_e.min
    err10[en][nb] = lmin
  end
end

#color1 = {125=>1, 250=>2, 500=>3, 1000=>4, 2000=>78, 4000=>7, 8000=>8, 16000=>9}
color1 = {2.5=>1, 5.0=>2, 10.0=>3, 20.0=>9}
mark1 = {2.5=>2, 5.0=>4, 10.0=>5, 20.0=>6}
color10 = {2.2=>1, 5.5=>2, 11.0=>3}
mark10 = {2.2=>2, 5.5=>4, 11.0=>5}


xmin, xmax = [40, 1000]

[
  [err1, color1, mark1, 3e-4, 1e-2],
  [err10, color10, mark10, 2e-3, 6e-2],
].each_with_index do |ary,n|
  err, color, mark, ymin, ymax = ary


  vxmin = vxmin0 + 0.34 * n
  vxmax = vxmin + 0.27
  vymin = 0.552
  vymax = 0.952

  n==0 ? DCL.grfrm : DCL.grfig
  DCL.grsvpt(vxmin, vxmax, vymin, vymax)
  DCL.grswnd(xmin, xmax, ymin, ymax)
  DCL.grstrn(4)
  DCL.grstrf
  DCL.usdaxs
  DCL.uxsttl("t", "#{[1,10][n]}-step learning", 0)
  DCL.uxsttl("b", "ensemble size", 0)
  DCL.uysttl("l", "loss", 0) if n==0
  DCL.sgtxzr(vxmin, vymax+0.03, "(#{label.chr})", 0.02, 0, 1, 3)
  DCL.swlset("lsysfnt", false)
  ens.each do |en|
    err[en].sort.each do |nb,er|
      f = nb.to_f/en
      DCL.sgpmzu([en], [er], mark[f], color[f]*10+3, 0.018)
    end
  end
  color.keys.each_with_index do |f,i|
    vy = vymax-0.03*(i+1)
    DCL.swlset("lsysfnt", false)
    DCL.sgpmzr([vxmax-0.07], [vy], mark[f], color[f]*10+3, 0.015)
    DCL.swlset("lsysfnt", true)
    s =  f.to_i.to_f==f ? f.to_i.to_s : f.to_s
    DCL.sgtxzr(vxmax-0.055, vy+0.002, ": #{s}", 0.015, 0, -1, 3)
  end

  label += 1

end






dat1 = Npy.load_npz("run_err_1step_ens800_bsize4000.npz")
dat1 = dat1["arr_0"]
dat1 = NArray.to_na(dat1.to_binary, "float")

dat10 = Npy.load_npz("run_err_10step_ens800_bsize8800_initFalse_1.npz")
dat10 = dat10["arr_0"]
dat10 = NArray.to_na(dat10.to_binary, "float")

dt = 0.05
t = NArray.sfloat(dat1.length).indgen * dt

ymax = 1e-1
ymin = 1e-4

a = 2.14
b = log(ymin)
#b = log(ymax) - a * t[-1]

vxmin = vxmin0
vxmax = 0.70
vymin = 0.06
vymax = 0.44
DCL.grfig
DCL.grsvpt(vxmin, vxmax, vymin, vymax)
DCL.grswnd(0, t.max, ymin, ymax)
DCL.grstrn(2)
DCL.grstrf

DCL.swlset("lsysfnt", true)
DCL.uzlset("labelyl", true)
DCL.usdaxs
DCL.uxsttl("b", "time", 0)
  DCL.uysttl("l", "MSE", 0)
DCL.sgplzu(t[1..-1], dat1[1..-1], 1, 23)
DCL.sgplzu(t[1..-1], dat10[1..-1], 1, 43)
DCL.sgplzu(t, exp(a*t+b), 3, 2)
DCL.sgtxzr(vxmin, vymax+0.03, "(#{label.chr})", 0.02, 0, 1, 3)
DCL.sgplzr([vxmin+0.02,vxmin+0.07],[vymax-0.03]*2, 1, 23)
DCL.sgtxzr(vxmin+0.08, vymax-0.03, "1-step learning", 0.015, 0, -1, 3)
DCL.sgplzr([vxmin+0.02,vxmin+0.07],[vymax-0.06]*2, 1, 43)
DCL.sgtxzr(vxmin+0.08, vymax-0.06, "10-step learning", 0.015, 0, -1, 3)


p [dat1[-1], dat10[-1], dat1[-1]/dat10[-1]]

DCL.grcls
