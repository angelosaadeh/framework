import numpy as np
import sys
from matplotlib import pyplot as plt
import time
sys.path.insert(0, '/home/angelo/Documents/Matthieu_project/mpc/mpc')
from data import data
from time import process_time as pt


def norm(vector):
    return vector

def est3_clear(v, iterations, target):
    Y,T,D = [],[],[]

    nq, nv = v.shape

    trust = 0.1* np.ones((1, nv))  # voters trustworthiness
    diff = 0.1*np.ones((nq, 1))  # queries difficulty
    y = np.ones((nq, 1))
    target.shape = y.shape

    t = (v==1)
    t_ = (1-t)

    for it in range(iterations):
        n = nv
        pos = np.sum(t * (1 - np.matmul(diff, trust)), 1)
        neg = np.sum(t_ * np.matmul(diff, trust), 1)
        y = (pos + neg) / n
        y.shape = (nq, 1)
        y = norm(y)
        Y.append(y)

        n = nv
        pos = np.sum(t * np.matmul((1 - y), 1 / trust), 1)
        neg = np.sum(t_ * np.matmul(y, 1 / trust), 1)
        diff = (pos + neg) / n
        diff.shape = (nq, 1)
        diff = norm(diff)
        D.append(diff)

        n = nq
        pos = np.matmul(t.transpose(), (1 - y) / diff)
        neg = np.matmul(t_.transpose(), y / diff)
        trust = (pos + neg) / n
        trust.shape = (1, nv)
        trust = norm(trust)
        T.append(trust.transpose())

        print(np.mean(np.abs((np.array(y) > 0.5) - target)))

    return np.array(y), np.array(trust), Y,T,D


def est3_mpc(mpc, v, iterations, target):
    Y,D,T = [],[],[]
    nq, nv = v.shape


    if mpc.identity == 'alice':
        trust = 0.1*np.ones((1,nv))
        trust = mpc.secret_share(trust)
        time.sleep(1)
        diff = 0.1*np.ones((nq,1))
        diff = mpc.secret_share(diff)
        time.sleep(1)
        amin = np.array([0])
        amin = mpc.secret_share(amin)
        time.sleep(1)
        amax = np.array([1])
        amax = mpc.secret_share(amax)
    else:
        trust = mpc.receive_shares()
        diff = mpc.receive_shares()
        amin = mpc.receive_shares()
        amax = mpc.receive_shares()

    y = np.ones((nq,1))
    target.shape = y.shape

    ### Compute the truth matrix where v==1 knowing that there are no zeros in v
    v2 = mpc.mul_const(mpc.mul(v, v), 1/2)
    v_2 = mpc.mul_const(v, 1/2)
    t =  mpc.add(v2, v_2) #=(v==1)
    t_ = mpc.add_const(-t, 1, real=True) #(v==-1)
    ###

    for it in range(iterations):
        print('------------------',it,'---------------------')
        mpc.reset()
        n = nv
        pos = mpc.sum(mpc.mul(t, mpc.add_const(-mpc.matmul(diff, trust), 1)), 1)
        neg = mpc.sum(mpc.mul(t_, mpc.matmul(diff, trust)), 1)
        y = mpc.mul_const(pos + neg, 1/n)
        y.shape = (nq, 1)
        y = mpc.normalize(y, amin, amax)
        Y.append(mpc.reconstruct(y))

        mpc.reset()
        n = nv
        pos = np.sum(mpc.mul(t, mpc.matmul(mpc.add_const(-y, 1), mpc.inverse(trust))), 1)
        neg = mpc.sum(mpc.mul(t_, mpc.matmul(y, mpc.inverse(trust))), 1)
        diff = mpc.mul_const(pos + neg, 1/n)
        diff.shape = (nq, 1)
        diff = mpc.normalize(diff, amin, amax)
        D.append(mpc.reconstruct(diff))

        mpc.reset()
        n = nq
        pos = mpc.matmul(t.transpose(), mpc.div(mpc.add_const(-y, 1), diff))
        neg = mpc.matmul(mpc.add_const(-t.transpose(),1), mpc.div(y, diff))
        trust = mpc.mul_const(pos + neg, 1/n)
        trust.shape = (1, nv)
        trust = mpc.normalize(trust.transpose(), amin, amax).transpose()
        T.append(mpc.reconstruct(trust.transpose()))
        #print(mpc.reconstruct(trust))

        y = mpc.reconstruct(y)
        print(np.mean(np.abs((np.array(y) > 0.5) - target)))

    return np.array(y), np.array(mpc.reconstruct(trust)), Y,T,D


def function(prot, v):

    np.set_printoptions(suppress=True)
    v_clear = prot.reconstruct(v)
    target = data.Unpack().get_array('data/banknotes.testtarget')

    '''a = est3_clear(v_clear.transpose(), 1, target)
    b = est3_mpc(prot, v.transpose(), 1, target)
    if prot.identity == 'alice':
        print(np.max(np.abs(a-b)))
        print(np.median(a))
        print(np.median(b))
        #plt.imshow(np.abs(a-b), interpolation='none')
        #plt.colorbar()
        #plt.show()
        plt.plot(np.abs(a-b))
        plt.show()
    return 0'''

    pred = est3_mpc(prot, v.transpose(), 20, target)
    trust = pred[1]
    print(trust)
    print((trust - np.min(trust)) / (np.max(trust) - np.min(trust)))
    Y_mpc = np.array(pred[2])
    T_mpc = np.array(pred[3])
    D_mpc = np.array(pred[4])

    print('---clear---')

    pred = est3_clear(v_clear.transpose(), 20, target)
    trust = pred[1]
    print(trust)
    print((trust - np.min(trust)) / (np.max(trust) - np.min(trust)))
    Y_clear = np.array(pred[2])
    T_clear = np.array(pred[3])
    D_clear = np.array(pred[4])


    if prot.identity == 'alice':
        plt.figure(figsize=(60, 35))
        plt.subplot(2, 3, 1)
        plt.gca().set_title('Prediction error of the truth value for each query', fontsize=30)
        plt.plot([i for i in range(1, 101)], np.abs(Y_clear[-1] - Y_mpc[-1]), 'kv')
        plt.plot([np.mean(np.abs(Y_clear[-1] - Y_mpc[-1]))] * 101, 'b--', label='Mean prediction error')
        plt.plot([np.median(np.abs(Y_clear[-1] - Y_mpc[-1]))] * 101, 'r--', label='Median prediction error')
        plt.ylabel("Truth prediction error", fontsize=30)
        plt.xlabel("Query ID", fontsize=30)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=30)
        plt.legend(fontsize=30)
        plt.subplot(2, 3, 2)
        plt.gca().set_title('Prediction error of the difficulty factor for each query', fontsize=30)
        plt.plot([i for i in range(1, 101)], np.abs(D_clear[-1] - D_mpc[-1]), 'kv')
        plt.plot([np.mean(np.abs(D_clear[-1] - D_mpc[-1]))] * 101, 'b--', label='Mean prediction error')
        plt.plot([np.median(np.abs(D_clear[-1] - D_mpc[-1]))] * 101, 'r--', label='Median prediction error')
        plt.ylabel("Difficulty prediction error", fontsize=30)
        plt.xlabel("Query ID", fontsize=30)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=30)
        plt.legend(fontsize=30)
        plt.subplot(2, 3, 3)
        plt.gca().set_title('Prediction error of the trust value for each voter', fontsize=30)
        plt.plot([i for i in range(1, 7)], np.abs(T_clear[-1] - T_mpc[-1]), 'kv')
        plt.plot([np.mean(np.abs(T_clear[-1] - T_mpc[-1]))] * 7, 'b--', label='Mean prediction error')
        plt.plot([np.median(np.abs(T_clear[-1] - T_mpc[-1]))] * 7, 'r--', label='Median prediction error')
        plt.ylabel("Trust prediction error", fontsize=30)
        plt.xlabel("Voter ID", fontsize=30)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=30)
        plt.legend(fontsize=30)
        plt.subplot(2, 3, 4)
        plt.gca().set_title('Relative error of the truth value for each query', fontsize=30)
        plt.plot([i for i in range(1, 101)], 100*np.abs((Y_clear[-1] - Y_mpc[-1])/Y_clear[-1]), 'kv')
        plt.plot([100*np.mean(np.abs((Y_clear[-1] - Y_mpc[-1])/Y_clear[-1]))] * 101, 'b--', label='Mean relative error')
        plt.plot([100*np.median(np.abs((Y_clear[-1] - Y_mpc[-1])/Y_clear[-1]))] * 101, 'r--', label='Median relative error')
        plt.ylabel("Truth relative error (%)", fontsize=30)
        plt.xlabel("Query ID", fontsize=30)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=30)
        plt.legend(fontsize=30)
        plt.subplot(2, 3, 5)
        plt.gca().set_title('Relative error of the difficulty factor for each query', fontsize=30)
        plt.plot([i for i in range(1, 101)], 100*np.abs((D_clear[-1] - D_mpc[-1])/D_clear[-1]), 'kv')
        plt.plot([100*np.mean(np.abs((D_clear[-1] - D_mpc[-1])/D_clear[-1]))] * 101, 'b--', label='Mean relative error')
        plt.plot([100*np.median(np.abs((D_clear[-1] - D_mpc[-1])/D_clear[-1]))] * 101, 'r--', label='Median relative error')
        plt.ylabel("Difficulty relative error (%)", fontsize=30)
        plt.xlabel("Query ID", fontsize=30)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=30)
        plt.legend(fontsize=30)
        plt.subplot(2, 3, 6)
        plt.gca().set_title('Relative error of the trust value for each voter', fontsize=30)
        plt.plot([i for i in range(1, 7)], 100*np.abs((T_clear[-1] - T_mpc[-1])/T_clear[-1]), 'kv')
        plt.plot([100*np.mean(np.abs((T_clear[-1] - T_mpc[-1])/T_clear[-1]))] * 7, 'b--', label='Mean relative error')
        plt.plot([100*np.median(np.abs((T_clear[-1] - T_mpc[-1])/T_clear[-1]))] * 7, 'r--', label='Median relative error')
        plt.ylabel("Trust relative error (%)", fontsize=30)
        plt.xlabel("Voter ID", fontsize=30)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=30)
        plt.legend(fontsize=30)
        plt.savefig('bank-error2.eps', format='eps')

        return 0

        plt.subplot(1, 3, 1)
        plt.imshow(np.abs(Y_clear - Y_mpc), interpolation='none')
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.imshow(np.abs(D_clear - D_mpc), interpolation='none')
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.imshow(np.abs(T_clear - T_mpc), interpolation='none')
        plt.colorbar()

        plt.show()

        plt.subplot(1, 3, 1)
        plt.plot(np.abs(Y_clear - Y_mpc)[-1])
        plt.subplot(1, 3, 2)
        plt.plot(np.abs(D_clear - D_mpc)[-1])
        plt.subplot(1, 3, 3)
        plt.plot(np.abs(T_clear - T_mpc)[-1])
        plt.show()

    return 0