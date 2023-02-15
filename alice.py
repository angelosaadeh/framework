import sys
import main
import numpy as np
import time
sys.path.insert(0, '/home/angelo/Documents/Matthieu_project/mpc/mpc')
import mpc



if __name__ == "__main__":

    prot = mpc.Run(True, 5002, '', 'alice', 2**60, 20, 2**32)

    x = np.array([[(i-100)/10 for i in range(201)]])
    time.sleep(1)
    x = prot.secret_share(x)
    time.sleep(1)
    k = np.array([[i for i in range(-16, 17)]])
    k = prot.secret_share(k)

    main.function(prot, x, k)
    time.sleep(5)
