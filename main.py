import numpy as np
from matplotlib import pyplot as plt
import time


def small_exp(mpc, r):
    r2 = mpc.mul(r, r)
    r3 = mpc.mul(r2, r)
    r4 = mpc.mul(r2, r2)
    expr = (mpc.mul_const(r4, 0.0591278145247190) + mpc.mul_const(r3, 0.153732317764270) + mpc.mul_const(r2,0.504655896968477) + mpc.mul_const(r, 0.999171373745965)) % mpc.mod
    expr = mpc.add_const(expr, 1.00005866051268)
    return expr


def function(mpc, x, k_enc):
    x_c = mpc.reconstruct(x)
    y_c = np.exp(x_c)
    ###########
    t = time.time()
    exp = mpc.exp(x)
    exp = mpc.reconstruct(exp)

    print(time.time()-t)

    t = time.time()
    c = 1/np.log(2)
    xc = mpc.mul_const(x, c, real=True)
    k = np.array([[i for i in range(-16, 17)]])
    if mpc.identity == 'alice':
        a = (xc.transpose()-mpc.map_to_ring(k)) % mpc.mod
    else:
        a = xc.transpose()-(k-k)
    print('-')
    a = mpc.rabbit_compare(a, 0)
    print('-')
    b = np.delete(a, 0, 1)
    zeros = np.array([[0]*b.shape[0]]).transpose()
    b = np.append(b, zeros, 1)
    k_enc = np.tile(k_enc,(b.shape[0],1))
    k = np.tile(k, (b.shape[0], 1))
    res = mpc.subs((a + b) % mpc.mod, mpc.mul_const(mpc.mul(a, b), 2))
    res_ = mpc.mul(res, k_enc)
    res_ = mpc.sum(res_, 1)
    res_.shape = x.shape
    print(mpc.reconstruct(res_))
    r = mpc.subs(x, mpc.mul_const(res_, np.log(2)))
    print(mpc.reconstruct(r))
    k2 = 2.**k
    print(k2)
    print(mpc.reconstruct(res))
    y = mpc.mul_const(res, k2)
    print(mpc.reconstruct(y))
    y = mpc.sum(y,1)
    expr = small_exp(mpc,r)
    y.shape = expr.shape
    y = mpc.mul(y, expr)
    y = mpc.reconstruct(y)
    x_c.shape = y[0].shape
    print(time.time()-t)
    exp.shape = x_c.shape
    if mpc.identity == 'alice':
        plt.plot(x_c,y[0],'blue', x_c, y_c[0],'black',x_c, exp, 'green',x_c,100*(y[0]-y_c[0]),'red',x_c,(exp-y_c[0]),'yellow')
        plt.show()

    return 0