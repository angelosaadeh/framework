import main
import sys
import numpy as np
sys.path.insert(0, '/home/angelo/Documents/Matthieu_project/mpc/mpc')
import mpc
from data import data


if __name__ == "__main__":
    prot = mpc.Run(False, 5001, '', 'bob', 2**60, 20, 2**32)

    v = data.Unpack().get_array('data/mnist.bob')
    main.function(prot, v)

    exit()
