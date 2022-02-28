# coding: utf-8
require "numru/dcl"
require "./lorenz96_model"
require "npy"
include NumRu
include NMath



K = 40
F = 8.0
dt = 0.01
#nt = 50
nt = 100

ens = [50, 100, 200, 400, 800]
#ens = [100, 200, 400]

li  = [1, 3, 10, 30, 100, 300, 10000]
#li  = [3, 10, 30]

lr = %w(0.0001 3e-05 1e-05)
#lr = %w(1e-05)


nseeds = 10
seeds = (100...(100+nseeds)).to_a


#rmse_th = 0.25
rmse_th = 0.45


len_max = 1000

nout = 101
out_int = 10


model = Lorenz96.new(K,F,dt,nt)





f = "data/4dvar_phy.npz"
npz = Npy.load_npz(f)
x0 = npz['arr_1'].to_binary
x0 = NArray.to_na(x0, "float", K)
x_true = model.step(x0)



lmin = {}
rmse = {}
count = {}
nitr = {}
ens.each do |en|
  p en

  lmin[en] = {}
  rmse[en] = {}
  nitr[en] = {}

  li.each do |k|

    lr.each do |l|

      ext = NArray.sfloat(nout)
      cnt = 0
      seeds.each do |seed|
        f = "data/4dvar_err_ens#{en}_bsize#{en*11}_lint#{k}_lr#{l}_seed#{seed}.npz"
        next unless File.exist?(f)
        npz = Npy.load_npz(f)
        l_p = npz['arr_1']
        x_out = npz['arr_4']
#        if l_p[-1] < 1000
          xo = NArray.to_na(x_out.to_binary, "float", K, nout)
          nout.times do |n|
            x = model.step(xo[true,n])
            ext[n] += sqrt( ((x - x_true)**2).mean(0) )
          end
          cnt += 1
#        end
      end

      if cnt > nseeds/2
        ext /= cnt
        min = ext[-1]
        #min = cp[-1]
        #p ext[0]
        if lmin[en][k].nil? || min < lmin[en][k]
          lmin[en][k] = min
          rmse[en][k] = ext
          idx = ext.le(rmse_th)
          if idx.any?
            nitr[en][k] = idx.where[0] * out_int
          else
            nitr[en][k] = 999e10
          end
        end
      end

    end


  end
end





iws = ARGV.delete("-pdf") ? 2 : 1
#iws=1
DCL.swcset("fname", "4dvar_rmse")
#DCL.gropn(-iws)
DCL.gropn(iws)
DCL.sglset("lfull", true)
DCL.sglset("lclip", true)
DCL.swlset("lsysfnt", true)
#DCL.uzfact(0.7)
DCL.uzfact(0.8)

#vxmin0 = 0.09
vxmin0 = 0.105
#vxmin0 = 0.11



mark = { 50=>2, 100=>3, 200=>4, 400=>5, 800=>7 }
color = {1=>1, 3=>2, 10=>10, 30=>3, 100=>78, 300=>4, 10000=>9}

color2 = [1, 2, 4, 78, 7]

label = "a".ord



=begin
x = NArray.sint(nout).indgen * out_int
[
#  [ens, li[2,1]],
  [[50,200,800], li[2,1]],
#  [[100, 200, 800], [10]],
#  [[100, 200, 400], [10]],
#  [ens[2,1], li],
  [ens[2,1], [1,10,100]],
#  [[200], [1, 10, 30]],
#  [[200], [1, 10, 100]],
].each_with_index do |ary, n|
  e = ary[0]
  l = ary[1]

  vxmin = vxmin0 + 0.325 * n
  vxmax = vxmin + 0.285
  vymin = 0.552
  vymax = 0.952

  n==0 ? DCL.grfrm : DCL.grfig
  DCL.grsvpt(vxmin, vxmax, vymin, vymax)
  #DCL.grswnd(0, 150, 1e-4, 0.4)
  DCL.grswnd(0, 500, 0.4, 1.0)
#  DCL.grswnd(0, 1000, 0.18, 0.30)
#  DCL.grswnd(0, 1000, 0.35, 1.59)
#  DCL.grstrn(2)
  DCL.grstrn(1)
  DCL.grstrf
  DCL.uzlset("labelyl", n==0)
  DCL.usdaxs
  DCL.uxsttl("b", "iteration", 0)
  DCL.uysttl("l", "RMSE", 0) if n==0
  DCL.sgtxzr(vxmin+0.030, vymax-0.03, ["K=#{l[0]}", "M=#{e[0]}"][n], 0.015, 0, -1, 3)
  DCL.sgtxzr(vxmin, vymax+0.03, "(#{label.chr})", 0.02, 0, -1, 3)
  label += 1
  m = 0
  e.each do |en|
    l.each do |k|
      p [en,k]
      err = rmse[en][k]
      next unless err
      p [err.min, err.max]
      DCL.sgplzu(x, err, 1, color2[m]*10+3)
      m += 1
    end
  end
  [e,l][n].each_with_index do |z,m|
    vy = vymax-0.03-0.02*m
    DCL.sgplzr([vxmax-0.15,vxmax-0.1],[vy]*2, 1, color2[m]*10+3)
    DCL.sgtxzr(vxmax-0.09, vy, ["M","K"][n] + "=#{z}", 0.015, 0, -1, 3)
  end
end
=end



2.times do |n|

  xmin, xmax = [ [40, 1000], [0.7, 500] ][n]

=begin
  vxmin = vxmin0 + 0.325 * n
  vxmax = vxmin + 0.285
#  vymin = 0.552
#  vymax = 0.952
  vymin = 0.0612
  vymax = 0.4612
=end

  vxmin = vxmin0 + 0.445 * n
  vxmax = vxmin + 0.425
#  vymin = 0.552
#  vymax = 0.952
  vymin = 0.075
  vymax = 0.68

  n==0 ? DCL.grfrm : DCL.grfig
#  DCL.grfig
  DCL.grsvpt(vxmin, vxmax, vymin, vymax)
#  DCL.grswnd(xmin, xmax, 10, 10000)
#  DCL.grswnd(xmin, xmax, 0.16, 0.26)
  DCL.grswnd(xmin, xmax, 0.35, 0.55)
#  DCL.grstrn(4)
  DCL.grstrn(3)
  DCL.grstrf
  DCL.uzlset("labelyl", n==0)
  DCL.usdaxs
  DCL.uxsttl("b", ["ensemble size", "update interval"][n], 0)
  DCL.uysttl("l", "RMSE", 0) if n==0
  DCL.sgtxzr(vxmax+0.002, vymin-0.015, "∞", 0.025, 0, -1, 3) if n==1
  DCL.sgtxzr(vxmin, vymax+0.03, "(#{label.chr})", 0.02, 0, -1, 3)
  label += 1
  DCL.swlset("lsysfnt", false)
  ens.each do |en|
    li.each do |k|
      x = n==0 ? en : k
      if rmse[en][k]
        p [k, rmse[en][k][-1]]
        if n==1 && k==10000
          DCL.sglset("lclip", false)
          DCL.sgpmzu([xmax*1.25], [rmse[en][k][-1]], mark[en], color[k]*10+3, 0.018)
          DCL.sglset("lclip", true)
        else
          DCL.sgpmzu([x], [rmse[en][k][-1]], mark[en], color[k]*10+3, 0.018)
        end
      end
    end
  end
  DCL.swlset("lsysfnt", true)

end


=begin

2.times do |n|

  xmin, xmax = [ [40, 1000], [0.7, 150] ][n]

  vxmin = vxmin0 + 0.325 * n
  vxmax = vxmin + 0.285
  vymin = 0.0612
  vymax = 0.4612

  DCL.grfig
  DCL.grsvpt(vxmin, vxmax, vymin, vymax)
#  DCL.grswnd(xmin, xmax, 80, 720)
  DCL.grswnd(xmin, xmax, 150, 730)
#  DCL.grstrn(4)
  DCL.grstrn(3)
  DCL.grstrf
  DCL.uzlset("labelyl", n==0)
  DCL.usdaxs
  DCL.uxsttl("b", ["ensemble size", "update interval"][n], 0)
  DCL.uysttl("l", "iteration", 0) if n==0
  DCL.sgtxzr(vxmin, vymax+0.03, "(#{label.chr})", 0.02, 0, -1, 3)
  label += 1
  DCL.swlset("lsysfnt", false)
  ens.each_with_index do |en|
    li.each_with_index do |k,m|
      if n==0
        x = en * 1.02**(m%2-0.5)
        #p [m,x,en]
      else
        x = k
      end
      if nitr[en][k] && nitr[en][k] < 10000
        p nitr[en][k]
        DCL.sgpmzu([x], [nitr[en][k]], mark[en], color[k]*10+3, 0.018)
      end
    end
  end
  DCL.swlset("lsysfnt", true)

end

=end


DCL.grcls


