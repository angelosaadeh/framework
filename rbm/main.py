import numpy as np
import sys
sys.path.insert(0, '/home/angelo/Documents/Matthieu_project/mpc/mpc')
from data import data


def sig(x):
    a = x<2
    b = x<-2
    return (1-b)*(a*(0.25*x+0.5)+(1-a))


def function(prot, train, test, W, a, b):
    np.set_printoptions(suppress=True)
    nb_epoch = 10
    nb_users = 943
    batch_size = 100

    for epoch in range(1, nb_epoch + 1):
        train_loss = 0
        s = 0
        print("epoch: ", epoch)
        for id_user in range(0, nb_users - batch_size, batch_size):
            vk = train[id_user:id_user + batch_size]
            v0 = train[id_user:id_user + batch_size]
            ph0 = prot.sigmoid(prot.add(prot.matmul(v0, W.transpose()), a))
            for k in range(10):
                print("iteration: ", k)
                hk = prot.bernoulli(prot.sigmoid(prot.add(prot.matmul(vk, W.transpose()), a)))
                vk = prot.bernoulli(prot.sigmoid(prot.add(prot.matmul(hk, W), b)))
                vk = prot.reset_rbm(vk,v0)
            phk = prot.sigmoid(prot.add(prot.matmul(vk, W.transpose()), a))

            dW = prot.subs(prot.matmul(v0.transpose(), ph0), prot.matmul(vk.transpose(), phk))
            dW.shape = W.shape
            W = prot.subs(W, dW)
            b = prot.subs(b, prot.sum(prot.subs(v0, vk), 0))
            a = prot.subs(a, prot.sum(prot.subs(ph0, phk), 0))

            vk_clear, v0_clear = prot.reconstruct(vk),prot.reconstruct(v0)
            train_loss += np.mean(np.abs(v0[v0 >= 0] - vk[v0 >= 0]))
            s += 1.
        print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s))