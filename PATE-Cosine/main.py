import numpy as np
import sys
from matplotlib import pyplot as plt
import time
sys.path.insert(0, '/home/angelo/Documents/Matthieu_project/mpc/mpc')
from data import data


def sig(x):
    a = x<2
    b = x<-2
    return (1-b)*(a*(0.25*x+0.5)+(1-a))


def function(prot, v):
    Acc_clear = []
    Acc_mpc = []
    EV_clear = []
    EV_mpc = []
    np.set_printoptions(suppress=True)
    f = data.Unpack().get_array('target.classification')
    #print(f)
    voters, N = v.shape
    v_clear = prot.reconstruct(v)
    #print(v_clear)
    v2 = prot.mul_const(prot.mul(v, v), 1/2)
    v_2 = prot.mul_const(v, 1/2)

    ev = prot.div(prot.sum(v2 + v_2, 1) + prot.sum(v2 - v_2, 1) , prot.add_const(prot.sum(prot.add_const(prot.mul_const(v2, 2) , -1), 1), N))

    ev.shape = (voters, 1)
    EV_mpc.append(prot.reconstruct(ev))
    wf = np.ones(f.shape)
    wf_repeated = np.tile(wf, (voters, 1))
    F = prot.mul_const(v, wf_repeated)

    eta = 0.1
    print(np.mean(np.abs((f + 1) / 2 - (wf > 0))))
    Acc_mpc.append(wf)

    for iteration in range(10):
        t0 = time.process_time()
        t1 = time.time()
        print(iteration, "------------")

        posF = prot.sum(prot.mul(F, prot.add(v2, v_2)), 1)
        negF = prot.sum(prot.mul(F, prot.subs(v2, v_2)), 1)
        zeros = prot.add_const(prot.sum(prot.add_const(prot.mul_const(v2, 2) , -1), 1), N)
        F_ = prot.square(prot.mul(F, prot.mul_const(v2, 2)))
        F2 = prot.sum(F_, 1)
        norm = prot.mul(zeros, F2)
        norm = prot.sqrt(norm)
        posF.shape, negF.shape, norm.shape = ev.shape, ev.shape, ev.shape

        old_ev = prot.mul_const(ev, np.tile((1 - eta), ev.shape))
        new_ev = prot.mul_const((posF + negF), np.tile(eta, ev.shape))
        ev = prot.add(old_ev, new_ev)

        ev = prot.div(ev, norm)
        EV_mpc.append(prot.reconstruct(ev))

        V = prot.mul(v, np.tile(ev, (1, N)))

        Vt = V.transpose()
        v2t = v2.transpose()
        v_2t = v_2.transpose()

        posV = prot.sum(prot.cube(prot.mul(Vt, v2t+v_2t)), 1)
        negV = prot.sum(prot.cube(prot.mul(Vt, v2t-v_2t)), 1)
        norm = prot.subs(posV, negV)
        posV.shape, negV.shape, norm.shape = wf.shape, wf.shape, wf.shape
        wf = prot.add(posV, negV)
        wf = prot.div(wf, norm)

        wf_clear = prot.reconstruct(wf)
        #print('wf', wf_clear)
        print(np.mean(np.abs((f + 1) / 2 - (wf_clear > 0)*1)))
        Acc_mpc.append(wf_clear)
        wf_repeated = np.tile(wf, (voters, 1))
        F = prot.mul(v, wf_repeated)
        print('clock', time.time() - t1 )
        print('cpu', time.process_time()-t0)

    #In clear:
    v = v_clear

    v2 = v*v/2
    v_2 = v/2

    ev = (np.sum(v2 + v_2, 1) + np.sum(v2 - v_2, 1))/(np.sum(2*v2-1,1)+N)
    ev.shape = (voters, 1)
    EV_clear.append(ev)
    wf = np.ones(f.shape)
    F = v*wf

    eta = 0.1
    print(np.mean(np.abs((f + 1) / 2 - (wf > 0))))
    Acc_clear.append(wf)
    for iteration in range(10):
        print(iteration, "------------")

        posF = np.sum(F*(v2+v_2), 1)
        negF = np.sum(F*(v2-v_2), 1)
        zeros = (np.sum(2*v2-1, 1)+N)
        F_ = (F*2*v2)**2
        F2 = prot.sum(F_, 1)
        norm = zeros*F2
        norm = np.sqrt(norm)
        posF.shape, negF.shape, norm.shape = ev.shape, ev.shape, ev.shape

        old_ev = ev*(1-eta)
        new_ev = (posF + negF)*eta
        ev = old_ev + new_ev

        ev = ev / norm
        EV_clear.append(ev)
        V = v*ev

        Vt = V.transpose()
        v2t = v2.transpose()
        v_2t = v_2.transpose()

        posV = np.sum((Vt * (v2t + v_2t))**3, 1)
        negV = np.sum((Vt * (v2t - v_2t))**3, 1)
        norm = posV - negV
        posV.shape, negV.shape, norm.shape = wf.shape, wf.shape, wf.shape
        wf = posV + negV
        wf = wf / norm
        print(np.mean(np.abs((f + 1) / 2 - (wf> 0))))
        Acc_clear.append(wf)
        F = v*wf

    Acc_clear = np.array(Acc_clear)
    Acc_mpc = np.array(Acc_mpc)
    Acc_clear.shape = (11,100)
    Acc_mpc.shape = (11,100)
    EV_mpc = np.array(EV_mpc)
    EV_clear = np.array(EV_clear)
    EV_clear.shape = (11,10)
    EV_mpc.shape = (11,10)
    print(EV_clear.shape)
    print(EV_mpc.shape)
    if prot.identity == 'alice':
        plt.figure(figsize=(30, 10))
        plt.subplot(1, 2, 1)
        plt.gca().set_title('Prediction error of the truth value for each query', fontsize=30)
        plt.plot([i for i in range(1, 101)], np.abs(Acc_clear[-1] - Acc_mpc[-1]), 'kv')
        plt.plot([np.mean(np.abs(Acc_clear[-1] - Acc_mpc[-1]))] * 101, 'b--', label='Mean prediction error')
        plt.plot([np.median(np.abs(Acc_clear[-1] - Acc_mpc[-1]))] * 101, 'r--', label='Median prediction error')
        plt.ylabel("Prediction error", fontsize=30)
        plt.xlabel("Query ID", fontsize=30)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=30)
        plt.legend(fontsize=30)
        plt.subplot(1, 2, 2)
        plt.gca().set_title('Prediction error of the trust value for each voter', fontsize=30)
        plt.plot([i for i in range(1, 11)], np.abs(EV_clear[-1] - EV_mpc[-1]), 'kv')
        plt.plot([np.mean(np.abs(EV_clear[-1] - EV_mpc[-1]))] * 11, 'b--', label='Mean prediction error')
        plt.plot([np.median(np.abs(EV_clear[-1] - EV_mpc[-1]))] * 11, 'r--', label='Median prediction error')
        plt.ylabel("Prediction error", fontsize=30)
        plt.xlabel("Voter ID", fontsize=30)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=30)
        plt.legend(fontsize=30)
        plt.savefig('prediction-error.eps', format='eps')

        plt.clf()

        plt.figure(figsize=(30, 10))
        plt.subplot(1, 2, 1)
        plt.gca().set_title('Relative error of the truth value for each query', fontsize=30)
        a = np.abs((Acc_clear[-1] - Acc_mpc[-1])/Acc_clear[-1])
        plt.plot([i for i in range(1, 101)], a, 'v')
        plt.plot([np.mean(a)] * 101, 'orange', label='Mean relative error')
        plt.plot([np.median(a)] * 101, 'green', label='Median relative error')
        plt.ylabel("Relative error", fontsize=30)
        plt.xlabel("Query ID", fontsize=30)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=30)
        plt.legend(fontsize=30)
        plt.subplot(1, 2, 2)
        plt.gca().set_title('Relative error of the trust value for each voter', fontsize=30)
        b = np.abs((EV_clear[-1] - EV_mpc[-1])/EV_clear[-1])
        plt.plot([i for i in range(1, 11)], b , 'v')
        plt.plot([np.mean(b)] * 11, 'orange', label='Mean relative error')
        plt.plot([np.median(b)] * 11, 'green', label='Median relative error')
        plt.ylabel("Relative error", fontsize=30)
        plt.xlabel("Voter ID", fontsize=30)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=30)
        plt.legend(fontsize=30)
        plt.savefig('relative-error.eps', format='eps')
    return 0