import main
import numpy as np
import time
import sys
sys.path.insert(0, '/home/angelo/Documents/Matthieu_project/mpc/mpc')
import mpc
from data import data


if __name__ == "__main__":
    prot = mpc.Run(True, 5001, '', 'alice', 2**60, 20, 2**32)

    train = data.Unpack().get_array('alice/alice.train')
    test = data.Unpack().get_array('alice/alice.test')
    train_labels = data.Unpack().get_array('alice/alice.trainlabel')
    test_labels = data.Unpack().get_array('alice/alice.testlabel')

    nh = 3
    nv = 10
    W = np.random.normal(0, 1, (nh, nv))*0.01
    a = np.zeros((1, nh))
    b = np.ones((1, nv))/2

    W = prot.secret_share(W)
    time.sleep(1)
    a = prot.secret_share(a)
    time.sleep(1)
    b = prot.secret_share(b)

    main.function(prot, train, test, train_labels, test_labels, W,a,b)
    
    time.sleep(3)