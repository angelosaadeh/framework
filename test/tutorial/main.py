import numpy as np


def function(prot, x, y):

    np.set_printoptions(suppress=True)

    a = prot.add(x, y)
    print("The addition is :", prot.reconstruct(a))

    m = prot.mul(x, y)
    print("The multiplication is :", prot.reconstruct(m))

    p = prot.rabbit_compare(m, 0)
    print("The positivity of the multiplication is :", prot.reconstruct(p))

    s = prot.sigmoid(m)
    print("The sigmoid of the multiplication is :", prot.reconstruct(s))

    prot.reset()
    l = prot.log(x)
    print("The natural logarithm of x is:", prot.reconstruct(l))
    prot.bilan()

    prot.reset()
    l = prot.exp(x)
    print("The exp of x is:", prot.reconstruct(l))
    prot.bilan()

    prot.reset()
    l = prot.sqrt(x)
    print("The sqrt of x is:", prot.reconstruct(l))
    prot.bilan()

    prot.reset()
    l = prot.inverse(x)
    print("The inverse of x is:", prot.reconstruct(l))
    prot.bilan()