import main
import numpy as np
import time
import sys
sys.path.insert(0, '/home/angelo/Documents/Matthieu_project/mpc/mpc')
import mpc
from data import data


if __name__ == "__main__":
    prot = mpc.Run(True, 5001, '', 'alice', 2**60, 20, 2**32)

    train = data.Unpack().get_array('bob.train')
    test = data.Unpack().get_array('bob.test')

    nh = 100
    nv = 1682
    W = np.random.normal(0, 1, (nh, nv))
    a = np.random.normal(0, 1, (1, nh))
    b = np.random.normal(0, 1, (1, nv))

    W = prot.secret_share(W)
    time.sleep(1)
    a = prot.secret_share(a)
    time.sleep(1)
    b = prot.secret_share(b)

    main.function(prot, train, test,W,a,b)
    
    time.sleep(3)