import numpy as np
import sys
from matplotlib import pyplot as plt
import time
sys.path.insert(0, '/home/angelo/Documents/Matthieu_project/mpc/mpc')
from data import data


def cosine(votes, iterations, target):
    v = votes.transpose()
    t = (v == 1)
    t_ = (v == -1)
    nv, nq = v.shape
    error = []

    N = np.sum(v != 0, 1)  # Number of {votes!=0}
    trust = (np.sum(t, 1) - np.sum(t_, 1)) / N
    trust.shape = (nv, 1)
    N.shape = trust.shape
    y = np.ones((nq, 1))
    target.shape = y.shape
    error.append(np.mean(np.abs(target - (y > 0))))
    eta = 0.2

    for iteration in range(iterations):
        pos = np.matmul(t, y)
        neg = np.matmul(t_, y)
        norm = np.matmul((v != 0), y ** 2) * N
        norm = np.sqrt(norm)
        new_trust = (pos - neg) / norm
        trust = trust * (1 - eta) + new_trust * eta

        pos = np.matmul(t.transpose(), (trust**3))
        neg = np.matmul(t_.transpose(), (trust**3))
        norm = pos + neg
        y = (pos - neg) / norm

        error.append(np.mean(np.abs(target - (y > 0))))
    return y, error

def cosine_mpc(mpc, votes, iterations, target):
    v = votes.transpose()
    v2 = mpc.mul(v,v)

    t = mpc.mul_const(mpc.add(v2,v), 0.5)
    t_ = mpc.mul_const(mpc.subs(v2,v), 0.5)

    nv, nq = v.shape
    error = []

    N = mpc.sum(v2, 1) # Number of {votes!=0}
    srN = mpc.sqrt(N, big=True, n=10)
    trust = mpc.div(mpc.subs(mpc.sum(t, 1), mpc.sum(t_, 1)), N, big=True)
    trust.shape = (nv, 1)
    N.shape = trust.shape

    if mpc.identity == 'alice':
        y = np.ones((nq,1))
        y = mpc.secret_share(y)
        time.sleep(1)
    else:
        y = mpc.receive_shares()
    target.shape = y.shape

    e = np.mean(np.abs(target - (mpc.reconstruct(y)> 0)))
    print(e)
    error.append(e)

    eta = 0.2

    for iteration in range(iterations):
        #print('-----', iteration, '------')
        pos = mpc.matmul(t,y)
        neg = mpc.matmul(t_,y)
        pre_norm = mpc.sqrt(mpc.matmul(v2, mpc.mul(y, y)), big=True, n=13)
        srN.shape = pre_norm.shape
        norm = mpc.mul(pre_norm, srN)
        new_trust = mpc.div(mpc.subs(pos, neg), norm, big=True, n=10)
        trust = mpc.add(mpc.mul_const(trust, 1-eta) , mpc.mul_const(new_trust, eta))

        t3 = mpc.cube(trust)
        num = mpc.matmul(v.transpose(), t3)
        norm = mpc.matmul(v2.transpose(), t3)
        y = mpc.div(mpc.mul(num, norm), mpc.square(norm))

        e = np.mean(np.abs(target - (mpc.reconstruct(y) > 0)))
        error.append(e)

    return y, np.array(error)


def function(mpc, v):


    np.set_printoptions(suppress=True)
    v_clear = mpc.reconstruct(v)
    target = data.Unpack().get_array('data/hubdub.target')

    n = cosine(v_clear, 10, target)
    m = cosine_mpc(mpc, v, 10, target)

    if mpc.identity == 'alice':
        plt.plot(n[-1], 'r--', m[-1], 'b--')
        plt.show()

    '''
    if mpc.identity == 'alice':
        y = np.array([100,1000,10**4,10**5,10**6])
        y = mpc.secret_share(y)
        time.sleep(1)
    else:
        y = mpc.receive_shares()

    z = mpc.sqrt(y)
    z = mpc.reconstruct(z)
    print(z)

    z = mpc.log_high(y)
    z = mpc.reconstruct(z)
    print(z)
    '''
    return 0