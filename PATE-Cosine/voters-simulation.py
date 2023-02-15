import sys
import numpy as np
sys.path.insert(0, '/home/angelo/Documents/Matthieu_project/mpc/mpc')
from data import data
import mpc

def sig(x):
    a = x<2
    b = x<-2
    return (1-b)*(a*(0.25*x+0.5)+(1-a))


if __name__ == "__main__":

    dataset_length = 100
    features = 10
    voters = 10
    epochs = 10

    W = np.random.uniform(-1, 1, (features, 1))
    W_ = []
    Loss = []
    W_noise = [ np.random.normal(1, 0.01, W.shape) for i in range(voters)]

    for voter in range(voters):
        x = np.random.normal(0, 1, (dataset_length, features))
        y_ = np.matmul(x, W * W_noise[voter])
        y = (y_ > np.median(y_))

        w = np.zeros(W.shape)
        for epoch in range(epochs):
            y_hat = sig(np.matmul(x, w))
            dw = np.matmul(x.transpose(), y_hat - y)
            w = w - dw / dataset_length
        W_.append(w) #Each voter saves their model to classify the queries (facts)
        y_hat = (sig(np.matmul(x, w)) > 0.5) * 1
        Loss.append(np.mean(np.abs(y_hat - y)))

    W_ = np.array(W_)
    W_.shape = (voters, features) #Each voter

    N = 100 #number of facts / querries to be answered

    x = np.random.normal(0, 1, (features, N)) #The facts i.e. vectors to be classified

    v = sig(np.matmul(W_, x)) #Classified facts from each voter
    v = -1*(v<0.45)+(v>0.55) #Only submit votes with higher probability

    f = (np.matmul(W.transpose(), x) > 0) * 2 - 1  # Classification of the facts by the original model (label/target)

    data_functions = data.Unpack()
    data_functions.secret_share_onfile(v, 'alice.data', 'bob.data')
    data_functions.write(f, 'target.classification')

