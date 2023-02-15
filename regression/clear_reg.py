import sys
import numpy as np
sys.path.insert(0, '/home/angelo/Documents/Matthieu_project/mpc/mpc')
from data import data


def sig(x):
    a = x<2
    b = x<-2
    return (1-b)*(a*(0.25*x+0.5)+(1-a))


if __name__ == "__main__":

    x_a = data.Unpack().get_file('reg_alice.csv')
    x_b = data.Unpack().get_file('reg_bob.csv')

    d = np.concatenate((x_a, x_b), 1)
    x = d[:, :3]
    y = d[:, 3:]

    w = np.zeros((3, 1))
    b = -0

    for i in range(5):
        y_hat = sig(np.matmul(x, w) + b)
        dw = np.matmul(x.transpose(), y_hat - y) / 100
        db = np.mean(y_hat - y)
        print(" At iteration ", i+1," the gradients are:")

        print(dw)
        print(db)

        w = w - dw
        b = b - db

        print("the updated weights are: ")
        print(w)
        print(b)

        y_hat = sig(np.matmul(x, w) + b)
        acc = 1 - np.mean(np.abs(np.round(y_hat) - y))

        print("and the accuracy on the training data is:")
        print(acc)