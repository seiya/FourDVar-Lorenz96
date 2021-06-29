import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP



import sys
import time
import math
import os
import random
import argparse

import numpy as np


torch.manual_seed(0)


class DataSet:
    def __init__(self, data, len: int, nobs: int, nobs2: int, k: int):
        self.data = data
        self.len = len
        self.nobs = nobs
        self.nobs2 = nobs2
        self.k = k
        self.nf = nobs2 - nobs + 1

    def __len__(self):
        return self.len * self.nf

    def __getitem__(self, index):
        i = int(index/self.nf)
        n = index % self.nf
        return self.data[i,n:n+self.nobs,:]




def save(fname, state, loss_t, loss_e):
    path = fname+".pth"
    torch.save(state, path)

    np.savez(fname, loss_t, loss_e)

def average_gradients(model, size):
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def average_loss(loss, size):
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    loss /= size
    return loss





def training(myrank, rank_size, k, nobs, nobs2, data_t, data_e, ndata_t, ndata_e, batch_size_t, init, cp, double, debug, fix):


    if debug:
        print(myrank, "start training", flush=True)

    nf = nobs2 - nobs + 1



    nitr = 2

    #path_net = "./train_1step_ens200_bsize4000.pth"
    #path_net = "./train_1step_ens400_bsize4000.pth"

    fname_pre = f"train_10step_ens{ndata_t}_bsize{batch_size_t}_init{init}"
    if cp > 1:
        path_net = fname_pre + f"_{cp-1}.pth"
    else:
        bs = {50: 125, 100: 250, 200: 2000, 400: 4000, 800: 4000}[ndata_t]
        path_net = f"./train_1step_ens{ndata_t}_bsize{bs}.pth"


    if myrank==0:
        print(f"pth file is {path_net}")
        print(f"# of ensembles is {ndata_t}")
        print(f"# of batch_size is {batch_size_t}")
        print(f"init is {init}")
        print(f"checkpoint count is {cp}")
        print(f"rank_size is {rank_size}")





    if debug:
        fname = "test"
    else:
        fname = fname_pre + f"_{cp}"



    max_norm = 0.01
    max_grad_norm = 0.01

    if debug:
        max_epoch = 10
    else:
        max_epoch = 50000
    #max_epoch = 1000
    #max_epoch = 500
    #max_epoch = 1




    batch_size_e = ndata_e * nf

    batch_num_t = ndata_t * nf / batch_size_t
    batch_num_e = ndata_e * nf / batch_size_e


    loader_t = torch.utils.data.DataLoader(data_t, batch_size=batch_size_t//rank_size, shuffle=True)
    loader_e = torch.utils.data.DataLoader(data_e, batch_size=batch_size_e//rank_size)





    stat = torch.load(path_net)


    if debug:
        nint = 1
        nint2 = 1
    else:
        nint = 10
        nint2 = 500
        #nint2 = 1000




    if cp > 1:
        path = fname_pre + f"_{cp-1}.npz"
        npz = np.load(path)
        loss_t = npz['arr_0']
        loss_e = npz['arr_1']
    else:
        loss_t = np.zeros(int(max_epoch/nint+1))
        loss_e = np.zeros(int(max_epoch/nint+1))


    large = 999.9e5
    if (cp > 1) and ('min' in stat.keys()):
        min0 = stat['min']
    else:
        min0 = [large, 0, 0]

    if (cp > 1) and ('epoch' in stat.keys()):
        epoch_min = stat['epoch'] + 1
    else:
        epoch_min = 0





    if torch.cuda.is_available():
        device = torch.device(f"cuda:{myrank}")
    else:
        device = None

    import net
    net = net.Net(k, nitr, device=device, rank_size=rank_size)
    if not init:
        net.load_state_dict(stat['net'])
    if double:
        net = net.double()
    if device:
        net = net.to(device)
    if rank_size > 1 and init:
        net = DDP(net, device_ids=[myrank])

    criterion = nn.MSELoss()

    if init:
        lr = 0.01 * batch_size_t / 1000
    else:
        lr = 0.001 * batch_size_t / 1000
#        lr = 0.0002 * batch_size_t / 1000

    optimizer = optim.Adam(net.parameters(), lr=lr)
    #optimizer = optim.Adam(net.parameters(), lr=0.01)
    #optimizer = optim.Adam(net.parameters(), lr=0.0001)
    if cp > 1:
        optimizer.load_state_dict(stat['opt'])

    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    if cp > 1:
        scheduler.load_state_dict(stat['sch'])

    if myrank==0:
        print("start training", flush=True)

    start = time.time()


    min_tmp = large
    unchange = 0
    min = min0
    for epoch in range(epoch_min, max_epoch):

        if init:
            net.train()
            net.drop = True
        elif fix:
#            net.train()
#            net.drop = False
            net.eval()
            net.drop = True
        else:
            net.eval()
            net.drop = False

        running_loss_t = 0.0
        for data in loader_t:

            #if debug and myrank==0:
            #    print("forward", epoch, flush=True)

            optimizer.zero_grad()
            if device:
                data = data.to(device)
            out = data[:,0,:]
            #out.requires_grad = True
            #tmp = out
            loss = 0.0
            lmsg = True
            for n in range(nobs-1):
                out = net(out)
                target = data[:,n+1,:]
                norm = criterion(out, target)
                loss += norm
                norm = norm.item()
                #if debug:
                #    print(epoch, n, norm)
                if norm >= max_norm:
                    if ( epoch > 10000 or debug ) and lmsg:
                        print("reducing norm", myrank, n, norm, max_norm, flush=True)
                        lmsg = False
                    out = target + ( out - target ) * ( max_norm / norm )

            #if debug and myrank==0:
            #    print("backward", epoch, flush=True)

            loss.backward()
            if rank_size > 1:
                #if debug and myrank==0:
                #    print("all reduce", epoch, flush=True)
                if not init:
                    average_gradients(net, rank_size)
                #print(epoch, myrank, loss.item())
                loss = average_loss(loss, rank_size)

            #if debug and myrank==0:
            #    print("optimizer", epoch, flush=True)

            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()
            running_loss_t += loss.item()

        scheduler.step()

        if (epoch+1)%nint == 0 or epoch==0:

            #print(torch.norm(tmp).item(), torch.norm(tmp.grad).item())

            net.eval()
            net.drop = False

            #if debug and myrank==0:
            #    print("eval", epoch, flush=True)

            with torch.no_grad():
                running_loss_e = 0.0
                for data in loader_e:
                    if device:
                        data = data.to(device)
                    loss = 0.0
                    out = data[:,0,:]
                    for n in range(nobs-1):
                        out = net(out)
                        norm = criterion(out, data[:,n+1,:])
                        loss += norm
                        #if debug:
                        #    print(epoch, n, norm.item())
                    running_loss_e += loss.item()

            if rank_size > 1:
                running_loss_e = average_loss(torch.tensor(running_loss_e, device=device), rank_size).item()

            l_t = running_loss_t / ( batch_num_t * (nobs-1) )
            l_e = running_loss_e / ( batch_num_e * (nobs-1) )
            if myrank == 0:
                loss_t[int((epoch+1)/nint)] = l_t
                loss_e[int((epoch+1)/nint)] = l_e

            if epoch > 0 and l_e < min[0]:
                min = [l_e, l_t, epoch+1]
                unchange = 0
                if myrank == 0:
                    state = {
                        'net': net.state_dict(),
                        'opt': optimizer.state_dict(),
                        'sch': scheduler.state_dict(),
                        'epoch': epoch,
                        'min': min,
                        'elapse': time.time() - start,
                    }
                    save(fname, state, loss_t, loss_e)

            if (epoch+1)%(max_epoch/10) == 0 and myrank==0:
                st = {
                    'net': net.state_dict(),
                    'opt': optimizer.state_dict(),
                    'sch': scheduler.state_dict(),
                    'epoch': epoch,
                    'min': min,
                    'elapse': time.time() - start,
                }
                save(fname+"_fin", st, loss_t, loss_e)

            if l_e < min_tmp:
                min_tmp = l_e

            if (epoch+1)%nint2 == 0 or epoch == 0:
                if myrank == 0:
                    print('[%d] lr: %.2e, training: %.6f, eval: %.6f (%.6f, %.6f)' % (epoch + 1, scheduler.get_last_lr()[0], l_t, l_e, min_tmp, min[0]), flush=True)
                if min_tmp > min[0]:
                    unchange += 1
                if ( epoch > 10000 and min_tmp > min[0] * 1.5 ) or unchange >= 20:
                    break
                min_tmp = large


    if myrank == 0:
        state = {
            'net': net.state_dict(),
            'opt': optimizer.state_dict(),
            'sch': scheduler.state_dict(),
            'epoch': epoch,
            'min': min,
            'elapse': time.time() - start,
        }

        if (cp > 1) and ('elapse' in stat.keys()):
            elapse = stat['elapse']
        else:
            elapse = 0

        print("minimam loss: %.6f, %.6f, %d"%(min[0], min[1], min[2]))
        print(f"elapsed time: %d sec"%(time.time() - start + elapse))

        save(fname+"_fin", state, loss_t, loss_e)






def init_process(myrank, rank_size, k, nobs, nobs2, data_t, data_e, ndata_t, ndata_e, batch_size_t, init, cp, double, debug, fix):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    port = batch_size_t + ndata_t + 10000
    if init:
        port += 1
    if myrank==0:
        print("port: ", port)
    os.environ["MASTER_PORT"] = f"{port}"
    #backend = "gloo"
    backend = "nccl"
    dist.init_process_group(backend, rank=myrank, world_size=rank_size)
    training(myrank, rank_size, k, nobs, nobs2, data_t, data_e, ndata_t, ndata_e, batch_size_t, init, cp, double, debug, fix)
    dist.destroy_process_group()



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("ndata", type=int)
    parser.add_argument("batch_size", type=int)
    parser.add_argument("init")
    parser.add_argument("--checkpoint", type=int, default=1)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-f", "--fix", action="store_true")
    args = parser.parse_args()
    ndata_t = args.ndata
    batch_size_t = args.batch_size
    init = args.init == "True"
    cp = args.checkpoint
    debug = args.debug
    fix = args.fix



#    args = sys.argv
#    argn = len(args)
#    if argn==0:
#        print("Usage: train_10step.py [ndata] [batch_size] [init] [checkpoint] [-d]")
#        exit()
#    ndata_t = int(args[1]) if argn>1 else 100
#    batch_size_t = int(args[2]) if argn>2 else ndata_t * nf
#    init = args[3]=="True" if argn>3 else False
#    cp = int(args[4]) if argn>4 else 1
#    debug = args[5]=="-d" if argn>5 else False



    #ndata_e = 1
    ndata_e = 100


    nt = 50
    nt2 = 100
    #nt2 = 200


    int_obs = 5
    nobs = int(nt/int_obs) + 1
    nobs2 = int(nt2/int_obs) + 1


    double = False
    #double = True



    np.random.seed(0)

    import lorenz96
    k = 40
    f = 8.0
    dt = 0.01

    sigma = 1e-1


    model = lorenz96.Lorenz96(k, f, dt)
    x0 = model.init(f, 0.01)

    #training data
    print("prepare training data")
    if double:
        data_t = np.zeros([ndata_t,nobs2,k], dtype="float64")
    else:
        data_t = np.zeros([ndata_t,nobs2,k], dtype="float32")
    for m in range(ndata_t):
        x = x0 + np.random.randn(k) * sigma
        # spinup
        for n in range(100):
            x = model.forward(x)

        data_t[m,0,:] = x
        for n in range(nt2):
            x = model.forward(x)
            if (n+1)%int_obs == 0:
                data_t[m,(n+1)//int_obs,:] = x


    # evaluation data
    print("prepare evaluation data")
    if double:
        data_e = np.zeros([ndata_e,nobs2,k], dtype="float64")
    else:
        data_e = np.zeros([ndata_e,nobs2,k], dtype="float32")
    for m in range(ndata_e):
        x = x0 + np.random.randn(k) * sigma
        # spinup
        for n in range(100):
            x = model.forward(x)

        data_e[m,0,:] = x
        for n in range(nt2):
            x = model.forward(x)
            if (n+1)%int_obs == 0:
                data_e[m,(n+1)//int_obs,:] = x


    rank_size = torch.cuda.device_count()


    if rank_size == 1:
        data_t = DataSet(data_t, ndata_t, nobs, nobs2, k)
        data_e = DataSet(data_e, ndata_e, nobs, nobs2, k)
        training(0, 1, k, nobs, nobs2, data_t, data_e, ndata_t, ndata_e, batch_size_t, init, cp, double, debug, fix)

    else:

        if ndata_t % rank_size > 0:
            print("ndata_t % rank_size is not 0: ", ndata_t, rank_size)
            exit()
        if ndata_e % rank_size > 0:
            print("ndata_e % rank_size is not 0: ", ndata_e, rank_size)
            exit()

        if batch_size_t % rank_size > 0:
            print("batch_size_t % rank_size is not 0: ", batch_size_t, rank_size)
            exit()

        import torch.multiprocessing as mp

        processes = []

        lt = ndata_t // rank_size
        le = ndata_e // rank_size
        mp.set_start_method("spawn")
        for myrank in range(rank_size):
            data_ts = DataSet(data_t[lt*myrank:lt*(myrank+1)], lt, nobs, nobs2, k)
            data_es = DataSet(data_e[le*myrank:le*(myrank+1)], le, nobs, nobs2, k)

            p = mp.Process(target=init_process, args=(myrank, rank_size, k, nobs, nobs2, data_ts, data_es, ndata_t, ndata_e, batch_size_t, init, cp, double, debug, fix))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


