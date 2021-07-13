require "numru/dcl"
require "npy"
include NumRu


ens = [50, 100, 200, 400, 800]
li  = [1, 3, 10, 30, 100, 300]
lr = %w(0.0001 3e-05 1e-05)


lmin = {}
cost_e = {}
cost_p = {}
ens.each do |en|
  lmin[en] = {}
  cost_e[en] = {}
  cost_p[en] = {}
  li.each do |k|

    lr.each do |l|
      f = "4dvar_ens#{en}_bsize#{en*11}_lint#{k}_lr#{l}.npz"
      npz = Npy.load_npz(f)
      l_e = npz['arr_0']
      l_p = npz['arr_1']
      min = l_p[-1]
      if lmin[en][k].nil? || lmin[en][k] > min
        lmin[en][k] = min
        cost_e[en][k] = l_e
        cost_p[en][k] = l_p
      end
    end

  end
end


nstep = 10000
x = NArray.sint(nstep).indgen(1)



iws = ARGV.delete("-pdf") ? 2 : 1
#iws=1
DCL.swcset("fname", "4dvar")
DCL.gropn(-iws)
DCL.sglset("lfull", true)
DCL.sglset("lclip", true)
DCL.swlset("lsysfnt", true)
DCL.uzfact(0.7)

vxmin0 = 0.09



color = [1, 2, 4, 78, 7]

label = "a".ord
[
  [[50, 200, 800], [10]],
  [[200], [1, 10, 100]],
#  [[100, 200, 400], [10]]
].each_with_index do |ary, n|
  e = ary[0]
  l = ary[1]

  vxmin = vxmin0 + 0.325 * n
  vxmax = vxmin + 0.285
  vymin = 0.552
  vymax = 0.952

  n==0 ? DCL.grfrm : DCL.grfig
  DCL.grsvpt(vxmin, vxmax, vymin, vymax)
  DCL.grswnd(0, 150, 1e-4, 0.4)
  DCL.grstrn(2)
  DCL.grstrf
  DCL.uzlset("labelyl", n==0)
  DCL.usdaxs
  DCL.uxsttl("b", "iteration", 0)
  DCL.uysttl("l", "cost", 0) if n==0
  DCL.sgtxzr(vxmin+0.015, vymax-0.03, ["K=#{l[0]}", "M=#{e[0]}"][n], 0.015, 0, -1, 3)
  DCL.sgtxzr(vxmin, vymax+0.03, "(#{(label+n).chr})", 0.02, 0, -1, 3)
  m = 0
  e.each do |en|
    l.each do |k|
      p [en,k]
      y_e = NArray.to_na(cost_e[en][k].to_binary, "sfloat")
      y_p = NArray.to_na(cost_p[en][k].to_binary, "sfloat")
      DCL.sgplzu(x, y_e, 3, color[m]*10+3)
      DCL.sgplzu(x, y_p, 1, color[m]*10+3)
      m += 1
    end
  end
  [e,l][n].each_with_index do |z,m|
    vy = vymax-0.03-0.02*m
    DCL.sgplzr([vxmax-0.15,vxmax-0.1],[vy]*2, 1, color[m]*10+3)
    DCL.sgtxzr(vxmax-0.09, vy, ["M","K"][n] + "=#{z}", 0.015, 0, -1, 3)
  end
end



ens = [50, 100, 200, 400, 800]
li  = [1, 3, 10, 30, 100, 300]
lr = %w(0.0001 3e-05 1e-05)


nitr = {}
ens.each do |en|
  nitr[en] = {}
  li.each do |k|
    nitr[en][k] = []

    lr.each do |l|
      f = "4dvar_ens#{en}_bsize#{en*11}_lint#{k}_lr#{l}.npz"
      npz = Npy.load_npz(f)
      l_e = npz['arr_1']
      idx = l_e.le(1e-3)
      if idx.any?
        nitr[en][k].push idx.where[0]
      else
        nitr[en][k].push 999e10
      end
    end

    nitr[en][k] = nitr[en][k].min
  end
end



mark = { 50=>2, 100=>3, 200=>4, 400=>5, 800=>7 }
color = {1=>1, 3=>2, 10=>3, 30=>4, 100=>78, 300=>7}

2.times do |n|

  xmin, xmax = [ [40, 1000], [0.7, 500] ][n]

  vxmin = vxmin0 + 0.325 * n
  vxmax = vxmin + 0.285
  vymin = 0.0612
  vymax = 0.4612

  DCL.grfig
  DCL.grsvpt(vxmin, vxmax, vymin, vymax)
  DCL.grswnd(xmin, xmax, 10, 10000)
  DCL.grstrn(4)
  DCL.grstrf
  DCL.uzlset("labelyl", n==0)
  DCL.usdaxs
  DCL.uxsttl("b", ["ensemble size", "update interval"][n], 0)
  DCL.uysttl("l", "iteration", 0) if n==0
  DCL.sgtxzr(vxmin, vymax+0.03, "(#{(label+n+2).chr})", 0.02, 0, -1, 3)
  DCL.swlset("lsysfnt", false)
  ens.each do |en|
    li.each do |k|
      x = n==0 ? en : k
      if nitr[en][k] < 10000
        DCL.sgpmzu([x], [nitr[en][k]], mark[en], color[k]*10+3, 0.018)
      end
    end
  end
  DCL.swlset("lsysfnt", true)

end




DCL.grcls


