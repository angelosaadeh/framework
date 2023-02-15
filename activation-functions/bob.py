import sys
import main
import numpy as np
sys.path.insert(0, '/home/angelo/Documents/Matthieu_project/mpc/mpc')
import mpc


if __name__ == "__main__":
    prot = mpc.Run(False, 5001, '', 'bob', 2**60, 15, 2**32)

    x = prot.receive_shares()

    main.function(prot, x)
