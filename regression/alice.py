import main
import numpy as np
import time
import sys
sys.path.insert(0, '/home/angelo/Documents/Matthieu_project/mpc/mpc')
import mpc
from data import data



def sig(x):
    a = x<2
    b = x<-2
    return (1-b)*(a*(0.25*x+0.5)+(1-a))


if __name__ == "__main__":
    prot = mpc.Run(True, 5001, '', 'alice', 2**60, 20, 2**32)

    x_a = data.Unpack().get_file('reg_alice.csv')
    x_b = data.Unpack().get_file('reg_bob.csv')

    x_a = prot.secret_share(x_a)
    x_b = prot.receive_shares()

    d = np.concatenate((x_a, x_b), 1)
    x = d[:,:3]
    y = d[:,3:]

    w = prot.secret_share(np.zeros((3,1)))
    time.sleep(1)
    b = prot.secret_share(0)

    main.function(prot, x, y,w, b)
    
    time.sleep(3)