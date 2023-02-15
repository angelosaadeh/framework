import main
import numpy as np
import time
import sys
sys.path.insert(0, '/home/angelo/Documents/Matthieu_project/mpc/mpc')
import mpc
from data import data


if __name__ == "__main__":
    prot = mpc.Run(True, 5001, '', 'alice', 2**60, 20, 2**32)

    v = data.Unpack().get_array('data/hubdub.alice')

    main.function(prot, v)
    
    time.sleep(3)