import numpy as np
import sys
sys.path.insert(0, '/home/angelo/Documents/Matthieu_project/mpc/mpc')
from data import data


def sig(x):
    a = x<2
    b = x<-2
    return (1-b)*(a*(0.25*x+0.5)+(1-a))


def function(prot, x_a, x_b, w, b):
    # For Testing the accuracy in clear:
    x_a_clear = data.Unpack().get_file('reg_alice.csv')
    x_b_clear = data.Unpack().get_file('reg_bob.csv')
    d = np.concatenate((x_a_clear,x_b_clear), 1)
    x_clear = d[:,:3]
    y_clear = d[:,3:]
    #---

    d_clear = np.concatenate((x_a, x_b), 1)
    x = d_clear[:,:3]
    y = d_clear[:,3:]

    np.set_printoptions(suppress=True)
    W = []
    B = []

    for i in range(5):
        y_hat = prot.sigmoid(prot.add(prot.matmul(x, w), b))
        dw = prot.mul_const(prot.matmul(x.transpose(), prot.subs(y_hat , y)), 1/100)
        db = prot.mul_const(prot.sum_all(prot.subs(y_hat, y)), 1/100)

        print(" At iteration ", i+1," the gradients are:")
        print(prot.reconstruct(dw))
        print(prot.reconstruct(db))

        w = prot.subs(w, dw)
        b = prot.subs(b, db)

        #in clear:
        w_clear = prot.reconstruct(w)
        b_clear = prot.reconstruct(b)

        print("the updated weights are: ")
        print(w_clear)
        print(b_clear)

        y_hat = sig(np.matmul(x_clear, w_clear) + b_clear)
        acc = 1 - np.mean(np.abs(np.round(y_hat) - y_clear))

        print("and the accuracy on the training data is:")
        print(acc)

    return 0