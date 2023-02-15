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

    v = data.Unpack().get_array('alice.data')

    main.function(prot, v)
    
    time.sleep(3)