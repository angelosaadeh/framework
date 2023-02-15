import sys
import main
import numpy as np
sys.path.insert(0, '/home/angelo/Documents/Matthieu_project/mpc/mpc')
import mpc


if __name__ == "__main__":
    prot = mpc.Run(False, 5001, '', 'bob', 2**60, 20, 2**32)

    x = prot.receive_shares()
    y = np.array([[-1.1,0],[0,-1.1]])
    y = prot.secret_share(y)

    main.function(prot, x, y)

