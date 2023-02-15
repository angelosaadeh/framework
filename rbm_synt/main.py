import numpy as np
import sys
sys.path.insert(0, '/home/angelo/Documents/Matthieu_project/mpc/mpc')
from sklearn.linear_model import LogisticRegression


def sig(x):
    a = x<2
    b = x<-2
    return (1-b)*(a*(0.25*x+0.5)+(1-a))


def function(prot, train, test, train_labels, test_labels, W, a, b):

    np.set_printoptions(suppress=True)
    nb_epoch = 10
    nb_users = 800
    batch_size = 16
    alpha = 0.001

    for epoch in range(1, nb_epoch + 1):
        train_loss = 0
        for id_user in range(0, nb_users - batch_size, batch_size):
            vk = train[id_user:id_user + batch_size]
            v0 = train[id_user:id_user + batch_size]
            ph0 = prot.sigmoid(prot.add(prot.matmul(v0, W.transpose()), a))
            for k in range(3):
                hk = prot.bernoulli(prot.sigmoid(prot.add(prot.matmul(vk, W.transpose()), a)))
                vk = prot.bernoulli(prot.sigmoid(prot.add(prot.matmul(hk, W), b)))
            phk = prot.sigmoid(prot.add(prot.matmul(vk, W.transpose()), a))

            dW = prot.subs(prot.matmul(v0.transpose(), ph0), prot.matmul(vk.transpose(), phk))
            dW.shape = W.shape
            W = prot.add(W, prot.mul_const(dW, alpha/batch_size))
            b = prot.add(b, prot.mul_const(prot.sum(prot.subs(v0, vk), 0), alpha/batch_size))
            a = prot.add(a, prot.mul_const(prot.sum(prot.subs(ph0, phk), 0), alpha/batch_size))

            vk_clear, v0_clear = prot.reconstruct(vk),prot.reconstruct(v0)
            train_loss += np.sum((v0_clear - vk_clear)**2)
        print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss))

        W_c = prot.reconstruct(W)
        a_c = prot.reconstruct(a)
        train_c, test_c, train_labels_c, test_labels_c = prot.reconstruct(train), prot.reconstruct(test), prot.reconstruct(train_labels), prot.reconstruct(test_labels)

        train_features = sig(np.matmul(train_c, W_c.transpose())+a_c)
        test_features = sig(np.matmul(test_c, W_c.transpose()) + a_c)

        clf = LogisticRegression()
        clf.fit(train_features, train_labels_c)
        predictions = clf.predict(test_features)
        print(np.sum(predictions == test_labels_c), '/', test_labels_c.shape[0])
