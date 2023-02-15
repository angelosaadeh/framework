import sys
import main
import numpy as np
sys.path.insert(0, '/home/angelo/Documents/Matthieu_project/mpc/mpc')
import mpc
import time



if __name__ == "__main__":
    prot = mpc.Run(True, 5001, '', 'alice', 2**60, 15, 2**32)

    x = np.array([(i-50)/10 for i in range(101)])
    time.sleep(1)
    x = prot.secret_share(x)
    time.sleep(1)
    main.function(prot, x)
