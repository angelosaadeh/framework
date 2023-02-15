import main
import sys
import numpy as np
sys.path.insert(0, '/home/angelo/Documents/Matthieu_project/mpc/mpc')
import mpc
from data import data


if __name__ == "__main__":
    prot = mpc.Run(False, 5001, '', 'bob', 2**60, 20, 2**32)

    train = data.Unpack().get_array('bob/bob.train')
    test = data.Unpack().get_array('bob/bob.test')
    train_labels = data.Unpack().get_array('bob/bob.trainlabel')
    test_labels = data.Unpack().get_array('bob/bob.testlabel')

    W = prot.receive_shares()
    a = prot.receive_shares()
    b = prot.receive_shares()

    main.function(prot, train, test, train_labels, test_labels, W, a, b)

    exit()
