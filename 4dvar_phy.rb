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


nseeds = 10


#rmse_th = 0.25
rmse_th = 0.45




nout = 101
out_int = 10


model = Lorenz96.new(K,F,dt,nt)





f = "data/4dvar_phy.npz"
npz = Npy.load_npz(f)
x0 = npz['arr_1'].to_binary
xout = npz['arr_3'].to_binary
x0 = NArray.to_na(x0, "float", K)
xout = NArray.to_na(xout, "float", K, 101, 20)[true,true,0...nseeds]
x_true = model.step(x0)


rmse = NArray.float(nout)
nseeds.times do |m|
  nout.times do |n|
    x = model.step(xout[true,n,m])
    rmse[n] += sqrt( ((x - x_true)**2).mean(0) )
  end
end

rmse /= nseeds

idx = rmse.le(rmse_th).where[0]

p rmse.to_a
p idx * out_int
