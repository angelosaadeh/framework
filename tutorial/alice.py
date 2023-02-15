import sys
import main
import numpy as np
sys.path.insert(0, '/home/angelo/Documents/Matthieu_project/mpc/mpc')
import mpc



if __name__ == "__main__":
    prot = mpc.Run(True, 5001, '', 'alice', 2**60, 20, 2**32)

    x = np.array([[1,2],[3,6]])
    x = prot.secret_share(x)
    y = prot.receive_shares()

    main.function(prot, x, y)
