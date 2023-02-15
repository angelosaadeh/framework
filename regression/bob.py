import main
import sys
import numpy as np
sys.path.insert(0, '/home/angelo/Documents/Matthieu_project/mpc/mpc')
import mpc
from data import data


if __name__ == "__main__":
    prot = mpc.Run(False, 5001, '', 'bob', 2**60, 20, 2**32)

    x_b = data.Unpack().get_file('reg_bob.csv')
    x_a = prot.receive_shares()
    x_b = prot.secret_share(x_b)
    w = prot.receive_shares()
    b = prot.receive_shares()
    main.function(prot, x_a, x_b, w, b)

    exit()
