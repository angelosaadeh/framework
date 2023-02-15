"""This class contains the MPC functions that are common to all users"""
import pickle
import numpy as np
import math
import sys
sys.path.insert(0, '/home/angelo/Documents/Matthieu_project/mpc/mpc')
from communications import communications
from randomness import generate
import time


class Protocols(communications.Communication):
    """Protocols for MPC"""
    def __init__(self, identity, partner, mod, precision, length):
        """Identity is a string equals to alice or bob"""
        super().__init__(length)
        self.identity = identity
        self.partner = partner
        self.mod = mod
        self.precision = precision
        self.triple, self.triple_bin, self.eda, self.eda_bin = self.get_crypto()
        self.interactions = 0
        self.number_mul_bin = 0
        self.number_mul = 0
        self.qty_mul_bin = 0
        self.qty_mul = 0

    def get_crypto(self):
        if self.identity == 'alice':
            file = open('/home/angelo/Documents/Matthieu_project/mpc/mpc/randomness/alice_rand.p', 'rb')
        else:
            file = open('/home/angelo/Documents/Matthieu_project/mpc/mpc/randomness/bob_rand.p', 'rb')
        triple, triple_bin, eda, eda_bin = pickle.load(file)
        file.close()
        return triple, np.array(triple_bin), eda, eda_bin


    def get_specific_crypto(self):
        if self.identity == 'alice':
            file = open('alice_spec_rand.p', 'rb')
        else:
            file = open('bob_spec_rand.p', 'rb')
        crypto = pickle.load(file)[0]
        file.close()
        return crypto

    def map_to_ring(self, x):
        """Maps the real elements to a ring or field"""
        a = 2**self.precision
        a = a * x
        a = np.rint(a)
        a = a.astype(int)
        a = a % self.mod
        return a

    def secret_share(self, x, real=True, modulo=False):
        """Secret share the secrets"""
        if not modulo:
            modulo = self.mod
        if real:
            x = self.map_to_ring(x)
        shares_partner = np.random.randint(0, modulo, x.shape)
        shares_identity = (x - shares_partner).astype(int) % modulo
        self.send(self.partner, shares_partner)
        return shares_identity

    def secret_share_bin(self, x):
        """Secret share the secrets"""
        shares_partner = np.random.randint(0, 2, x.shape)
        shares_identity = (x - shares_partner).astype(int) % 2
        self.send(self.partner, shares_partner)
        return shares_identity

    def receive_shares(self):
        """To be used after every secret_share on the receiver's side"""
        return self.get(self.partner)

    def reconstruct(self, x):
        """Reconstruct a secret"""
        if self.identity == 'alice':
            self.send(self.partner, x)
            y = self.get(self.partner)
        if self.identity == 'bob':
            y = self.get(self.partner)
            self.send(self.partner, x)
        z = (x+y) % self.mod
        return (z > (self.mod/2)) * (-self.mod / 2**self.precision) + (z/2**self.precision)

    def reveal(self, x):
        """Open an element without converting it to a real number, ie keep it as a ring element"""
        if self.identity == 'alice':
            self.send(self.partner, x)
            y = self.get(self.partner)
        if self.identity == 'bob':
            y = self.get(self.partner)
            self.send(self.partner, x)
        return (x+y) % self.mod

    def reveal_bin(self, x):
        """Open an element without converting it to a real number, ie keep it as a ring element"""
        if self.identity == 'alice':
            self.send(self.partner, x)
            y = self.get(self.partner)
        if self.identity == 'bob':
            y = self.get(self.partner)
            self.send(self.partner, x)
        return (x+y) % 2

    def add(self, x, y):
        """Addition of two secrets"""
        return (x+y) % self.mod

    @staticmethod
    def add_bin(x, y):
        return (x+y) % 2

    def subs(self, x, y):
        """Subtraction of two secrets"""
        return (x-y) % self.mod

    def add_const(self, x, k, real=True):
        """Addition with a constant that should be mapped to the field if it is real"""
        if self.identity == 'alice':
            if real:
                k = self.map_to_ring(k)
            x = (x+k) % self.mod
        return x % self.mod

    def add_const_bin(self, x, k):
        if self.identity == 'alice':
            x = (x+k) % 2
        return x % 2

    def truncate_bit(self, x):
        """Truncate one bit"""
        if self.identity == 'alice':
            return np.rint(x/2).astype(int)
        return np.rint(self.mod - (self.mod-x)/2).astype(int)

    def new_truncate_bit(self, x):
        """Truncate one bit"""
        if self.identity == 'alice':
            return x >> self.precision
        return self.mod - ((self.mod-x) >> self.precision)

    def truncate(self, x, bits, old):
        """Truncate the required number of bits"""
        if old:
            for i in range(bits):
                    x = self.truncate_bit(x)
        else:
            x = self.new_truncate_bit(x)
        return x

    def get_triple_bin(self, x):
        quantity = np.prod(x.shape)
        self.qty_mul_bin += quantity
        if quantity > len(self.triple_bin[0]):
            print("-")
            if self.identity == 'alice':
                generate.generate(name='triple_bin')
                self.send(self.partner, 'go')
                self.triple_bin = np.array(self.get_specific_crypto())
            if self.identity == 'bob':
                _ = self.get(self.partner)
                self.triple_bin = np.array(self.get_specific_crypto())
        a = self.triple_bin[0][0:quantity]
        b = self.triple_bin[1][0:quantity]
        c = self.triple_bin[2][0:quantity]
        a.shape = x.shape
        b.shape = x.shape
        c.shape = x.shape
        #self.triple_bin = [vector[quantity:] for vector in self.triple_bin]
        return a, b, c

    def get_triple(self, x):
        quantity = np.prod(x.shape)
        self.qty_mul += quantity
        if quantity > len(self.triple[0]):
            print("--")
            if self.identity == 'alice':
                generate.generate(name='triple')
                self.send(self.partner, 'go')
                self.triple = self.get_specific_crypto()
            if self.identity == 'bob':
                _ = self.get(self.partner)
                self.triple = self.get_specific_crypto()
        a = self.triple[0][0:quantity]
        b = self.triple[1][0:quantity]
        c = self.triple[2][0:quantity]
        a.shape = x.shape
        b.shape = x.shape
        c.shape = x.shape
        #self.triple = [vector[quantity:] for vector in self.triple]
        return a, b, c

    def get_eda(self, x):
        quantity = np.prod(x.shape)
        r = self.eda[0][0:quantity]
        r_bin = self.eda[1][0:quantity]
        r_bin = self.eda[1][0:quantity]
        r.shape = x.shape
        r_bin.shape = x.shape + tuple([math.floor(math.log(self.mod, 2)) + 2])
        #self.eda = [vector[quantity:] for vector in self.eda]
        return r, r_bin

    def get_eda_bin(self, x):
        quantity = np.prod(x.shape)
        r = self.eda_bin[0][0:quantity]
        r_bin = self.eda_bin[1][0:quantity]
        r.shape = x.shape
        r_bin.shape = x.shape
        #self.eda_bin = [vector[quantity:] for vector in self.eda_bin]
        return r, r_bin

    def mul_bin(self, x, y):
        """Element-wise multiplication of secrets"""
        self.number_mul_bin += 1
        a, b, c = self.get_triple_bin(x)
        d = self.add_bin(x, a)
        e = self.add_bin(y, b)
        d, e = self.reveal_bin(np.array([d, e]))
        z = self.add_const_bin(c + d * b + e * a, e * d)
        return z

    def mul(self, x, y, real=True, old=False):
        """Element-wise multiplication of secrets"""
        self.number_mul += 1
        a, b, c = self.get_triple(x)
        d = self.subs(x, a)
        e = self.subs(y, b)
        d, e = self.reveal(np.array([d, e]))
        z = self.add_const(c + d * b + e * a, e * d, real=False)
        if real:
            z = self.truncate(z, self.precision, old)
        return z

    def matmul(self, x, y):
        """Multiplication of two matrices"""
        if len(x.shape) != len(y.shape) or len(x.shape) != 2:
            print('ERROR: They should be 2x2 matrices')
            return 0
        x_i, x_j = x.shape
        y_i, y_j = y.shape
        if x_j != y_i:
            print('ERROR: Shapes do not match')
            return 0
        x_repeated = np.tile(x, (1, y_j))
        x_repeated.shape = (x_i, y_j, x_j)
        y_repeated = np.tile(y.transpose(), (x_i, 1))
        y_repeated.shape = (x_i, y_j, x_j)
        z_repeated = self.mul(x_repeated, y_repeated)
        z = np.sum(z_repeated, 2) % self.mod
        return z

    @staticmethod
    def convert_to_bin(x, width):
        numbers = np.copy(x)
        length = np.prod(numbers.shape)
        original_shape = tuple(list(numbers.shape)+[width])
        numbers.shape = (1, length)
        numbers = numbers[0]
        array = [[int(b) for b in f'{int(num):0{width}b}'[::-1]] for num in numbers]
        array = np.array(array)
        array.shape = original_shape
        return array

    def convert_from_bin(self, x):
        r, r_bin = self.get_eda_bin(x)
        r_bin.shape = x.shape
        y = self.reveal_bin(self.add_bin(x, r_bin))
        z = self.add_const(r - 2 * y * r, y, real=False)
        return z

    def or_bin(self, x, y):
        z = self.add_bin(x, y)
        m = self.mul_bin(x, y)
        return self.add_bin(z, m)

    def prefix_or(self, x):
        l = math.log2(len(x[0]))
        while l != int(l):
            x = np.concatenate((x, np.zeros((x.shape[0], 1))), axis=1)
            l = math.log2(len(x[0]))
        k = np.prod(x.shape)
        l = int(len(x[0]) / 2)
        if l == 1:
            a = x[:, :l]
            b = x[:, l:]
            c = self.or_bin(a, b)
            r = np.concatenate((c, b), axis=1)
            return r
        else:
            x.shape = (int(k / l), l)
            r = self.prefix_or(x)
            a = np.array([r[2 * i] for i in range(int(k / l / 2))])
            b = np.array([r[2 * i + 1] for i in range(int(k / l / 2))])
            b_0 = np.tile(np.array([b[:, 0]]).transpose(), (1, l))
            c = self.or_bin(b_0, a)
            return np.concatenate((c, b), axis=1)

    def less_than(self, c, x): #renvoie c<x
        """c_copy is public, x is private binary shares"""
        c_copy = np.copy(c)
        y = self.add_const_bin(x, c_copy)
        shape = c_copy.shape
        final_shape = tuple(shape[:-1])
        y.shape = (np.prod(final_shape), shape[-1])
        z = self.prefix_or(y)
        z = z[:, :shape[-1]]
        z = z.transpose()
        w = (z[0]+z[1]) % 2
        w.shape = (1, len(z[0]))
        for i in range(1, len(z) - 1):
            w = np.insert(w, i, (z[i] + z[i+1]) % 2, axis=0)
        w = np.insert(w, i, z[i], axis=0)
        w = w.transpose()
        c_copy.shape = w.shape
        w = np.sum(w * (1-c_copy), 1) % 2
        w.shape = final_shape
        return w.astype(int)

    def less_than_old(self, c, x):
        c_copy = np.copy(c)
        y = self.add_const_bin(x, c_copy)
        shape = c_copy.shape
        final_shape = tuple(shape[:-1])
        y.shape = (np.prod(final_shape), shape[-1])
        y = y.transpose()
        z = y[-1]
        z.shape = (1, len(y[0]))
        w = z[0]
        w.shape = (1, len(y[0]))
        for i in range(len(y) - 1):
            bib = self.or_bin(y[len(y) - i - 2], z[0])
            z = np.insert(z, 0, bib , axis=0)
            w = np.insert(w, 0, (z[0] + z[1]) % 2, axis=0)
        c_copy.shape = w.transpose().shape
        w = np.sum(w.transpose() * (1-c_copy), 1) % 2
        w.shape = final_shape
        return w

    def mul_const(self, x, c, real=True):
        if real:
            c = self.map_to_ring(c)
        y = (c*x) % self.mod
        if real:
            y = self.truncate(y, self.precision, old=False)
        return y

    def sigmoid(self, x):
        """Prend 2xlog(taille de anneau) interations. Solution combiner les operations pour faire en parallele."""
        x_a = self.rabbit_compare(x, -2)
        x_b = self.rabbit_compare(x, 2)
        y = self.add_const(self.mul_const(x, 0.25), 0.5)
        a = self.add_const(self.mul_const(y, -1), 1)
        b = self.mul(x_b, a)
        c = self.add(b, y)
        sig = self.mul(x_a, c)
        return sig

    def rabbit_compare(self, x, c, real=True, old=True): # returns x-c>=0
        """ Add description here """
        if real:
            x = self.add_const(x, -c)
        else:
            x = self.add_const(x, -c, real=False)
        c = self.mod/2
        r, r_bin = self.get_eda(x)
        a = self.add(x, r)
        a = self.reveal(a)
        b = self.subs(a, c)
        k = math.floor(math.log(self.mod, 2)) + 2
        b_bin = self.convert_to_bin(b, k)
        a_bin = self.convert_to_bin(a, k)
        # w1 = self.less_than(a_bin, r_bin) #on veut a<r
        # w2 = self.less_than(b_bin, r_bin) #on veut b<r
        if old==True:
            w = self.less_than_old(np.concatenate((a_bin, b_bin)), np.concatenate((r_bin, r_bin)))  # on veut a<r et b<r
        else:
            w = self.less_than(np.concatenate((a_bin, b_bin)), np.concatenate((r_bin, r_bin)))  # on veut a<r et b<r
        w1 = w[:a_bin.shape[0]]
        w2 = w[a_bin.shape[0]:]
        w3 = b < (self.mod-c)
        w_bin = self.add_const_bin(w1-w2, w3)
        w_bin = self.add_const_bin(-w_bin, 1)
        w = self.convert_from_bin(w_bin)
        w = self.map_to_ring(w)
        return w

    def random_uniform_bits(self, shape):
        if self.identity == 'alice':
            a = np.random.randint(0, 2, shape+(self.precision,))
            a = self.secret_share(a)
            b = self.receive_shares()
        if self.identity == 'bob':
            a = self.receive_shares()
            b = np.random.randint(0, 2, shape + (self.precision,))
            b = self.secret_share(b)
        c = self.add(a, b)
        k = self.mul(a, b)
        k = self.mul_const(k, 2)
        c = self.subs(c, k)
        c = self.mul_const(c, np.array([2**i for i in range(self.precision)]))
        c = np.sum(c, 2) % self.mod
        c = self.mul_const(c, 2**(-self.precision))
        return c

    def old_random_uniform(self, shape):
        if self.identity == 'alice':
            a = np.random.uniform(0, 1, shape)
            a = self.secret_share(a)
            b = self.receive_shares()
        if self.identity == 'bob':
            a = self.receive_shares()
            b = np.random.uniform(0, 1, shape)
            b = self.secret_share(b)
        c = self.add(a, b)
        sign = self.rabbit_compare(self.mul_const(c, -1), -1)
        c = self.add_const(self.add(sign, c), -1)
        return c

    def exp(self, x, iterations=8):
        a = self.add_const(self.mul_const(x, 2**-iterations), 1)
        for i in range(iterations):
            a = self.mul(a, a)
        return a

    def log(self, x):
        a = self.add_const(self.mul_const(x, 1/31), 1.59)
        b = self.mul_const(self.exp(self.add_const(self.mul_const(x, -2), -1.4)), 20)
        y = self.subs(a, b)

        for i in range(2):
            h = [self.exp(-y)]
            h[0] = self.add_const(self.mul(-x, h[0]), 1)
            for j in range(1, 5):
                h.append(self.mul(h[-1], h[0]))
            k = self.add_const(self.mul_const(h[0], 1 / 2) + self.mul_const(h[1], 1 / 3) + self.mul_const(h[2], 1 / 4) +
                               self.mul_const(h[3], 1 / 5) + self.mul_const(h[4], 1 / 6), 1)
            y = self.subs(y, self.mul(h[0], k))
        return y

    def random_laplace(self, scale, shape):
        u, s = self.random_uniform(shape)
        l = self.log(u)
        k = self.log(self.add_const(-u,1))
        res = self.add(k, self.mul(-s, self.add(l,k), real=False))
        res = self.mul_const(res, -scale)
        return res

    def old_random_laplace(self, scale, shape):
        u = self.old_random_uniform(shape)
        s = self.add_const(self.mul_const(self.rabbit_compare(self.add_const(u, -0.5), 0), 2), -1)
        abs_u = self.mul(s, self.add_const(u, -0.5))
        res = self.mul(s, self.log(self.add_const(self.mul_const(abs_u, -2), 1)))
        res = self.mul_const(res, -scale)
        return res

    def random_uniform(self, shape):
        if self.identity == 'alice':
            a = np.random.randint(0, 2, shape+(self.precision+1,))
            a = self.secret_share(a, real=False)
            b = self.receive_shares()
        if self.identity == 'bob':
            a = self.receive_shares()
            b = np.random.randint(0, 2, shape+(self.precision+1,))
            b = self.secret_share(b, real=False)
        c = self.add(a, b)
        k = self.mul(a, b, real=False)
        k = self.mul_const(k, 2, real=False)
        c = self.subs(c, k)
        c.shape = (np.prod(shape), self.precision+1)
        u = c[:, :-1]
        u.shape = shape + (self.precision,)
        s = c[:, -1]
        s.shape = shape
        b = np.array([2 ** i for i in range(self.precision)]*np.prod(shape))
        b.shape = u.shape
        u = self.mul_const(u, b, real=False)
        u = np.sum(u, -1) % self.mod
        return u, s

    def bilan(self):
        print("Interactions: ", self.interactions)
        print("Multiplications: ", self.number_mul)
        print("Multiplications binaires: ", self.number_mul_bin)
        print("Triplets de multiplications utilisés: ", self.qty_mul)
        print("Triplets de multiplications binaires utilisés: ", self.qty_mul_bin)

    def reset(self):
        self.interactions = 0
        self.number_mul_bin = 0
        self.number_mul = 0
        self.qty_mul_bin = 0
        self.qty_mul = 0

    def numerical_compare(self, x):
        """Comparision >0 for real numbers between -1 and 1, it is faster for matrices of size at least 1000"""
        y = x
        for i in range(20):
            y2 = self.mul(y,y)
            y3 = self.mul(y2,y)
            y = self.add(self.mul_const(y3, -1/2), self.mul_const(y, 1.5))
        y = self.mul_const(self.mul(y, self.add_const(y, 1)), 0.5)
        return y

    def inverse(self, x):
        """inverse for positive real numbers, correct for numbers between 1/1000 and 100"""
        return self.exp(self.mul_const(self.log(x), -1))

    def div(self, x, y, small=False, big=False, n=1):
        if small:
            return self.mul(x, self.small_inverse(y, n))
        if big:
            return self.mul(x, self.big_inverse(y, n))
        return self.mul(x, self.inverse(y))

    def reset_rbm(self, x, y):
        "si y=0 return x, si y=1 return x, si y=-1 return -1"
        return self.add(self.mul(self.mul_const(self.add_const(-x,-1), 1/2), self.add(self.mul(y,y), -y)), x)

    def bernoulli(self, k):
        u = np.random.uniform(0, 1, k.shape)
        #return self.numerical_compare(self.add_const(k, -u))
        return self.rabbit_compare(k, u, real=True)

    def sum(self, x, b):
        return np.sum(x, b) % self.mod

    def sum_all(self, x):
        return np.sum(x) % self.mod

    def easy_sigmoid(self, x):
        return self.add_const(self.mul_const(x, 0.25), 0.5)

    def power_8(self, x):
        a = np.array([x**i for i in range(1,8)])
        if self.identity == 'alice':
            a = np.flip(a)
            b = self.receive_shares()
            a = self.secret_share(a, real=False)
        if self.identity == 'bob':
            b = self.secret_share(a, real=False)
            a = self.receive_shares()
        c = self.mul(a, b, real=False)
        return 0

    def compare_newton(self, x):
        y = self.mul_const(x,2**-10)
        for j in range(30):
            y = self.subs(self.mul_const(y, 2), self.mul(self.mul(y, y), x))
            x = self.mul_const(x+y, 1/2)
        return self.mul_const(self.mul(x, self.add_const(x, 1)), 0.5)

    def matmul_const(self,a,x,const_position):
        if const_position == 1:
            return 0

    def matmul_new(self, x, y):
        """Multiplication of two matrices"""
        # DOES NOT WORK YET
        if self.identity=='alice':
            a = np.random.randint(0, self.mod, x.shape)
            b = np.random.randint(0, self.mod, y.shape)
            c = np.matmul(x,y) % self.mod
            a = self.secret_share(a, real=False)
            time.sleep(1)
            b = self.secret_share(b, real=False)
            time.sleep(1)
            c = self.secret_share(c, real=False)
        if self.identity == 'bob':
            a = self.receive_shares()
            b = self.receive_shares()
            c = self.receive_shares()
        d = self.subs(x, a)
        e = self.subs(y, b)
        d, e = self.reveal(np.array([d, e]))
        z = self.add_const(c - np.matmul(d, b) - np.matmul(a, e), np.matmul(d, e), real=False)
        #z = self.truncate(z, self.precision, old=False)
        return z

    def log_high(self, x, n):
        x = self.mul_const(x,2**(-n))
        return self.add_const(self.log(x), n*np.log(2))

    def sqrt(self, x, big=True, n=1):
        if big:
            return self.exp(self.mul_const(self.log_high(x, n), 1/2))
        return self.exp(self.mul_const(self.log(x), 1 / 2))

    def inverse_negative(self, x):
        """Use """

    def fast_positive_inverse(self, x):
        """untested in MPC, but is correct in clear"""
        y = (2**self.precision-1)*int(np.ones(x.shape))
        if self.identity == 'bob':
            y = int(np.zeroz(x.shape))
        for i in range(10):
            y = np.ceil(y*2 - self.mul(self.mul(y,y,real=False), x, real = False) * 2 ^ (-2 * self.precision))
        return y

    def cube(self, x):
        y = self.mul(x, x)
        return self.mul(y, x)

    def square(self, x):
        return self.mul(x, x)

    def small_inverse(self, x, n):
        return self.mul_const(self.inverse(self.mul_const(x, 2**n)), 2**n)

    def big_inverse(self, x, n):
        return self.exp(self.mul_const(self.log_high(x, n), -1))

    def abs(self, x):
        return self.mul(self.add_const(self.mul_const(self.rabbit_compare(x, 0), 2), -1), x)

    def num_truncation(self, x):
        a = -2 / 1329227995784915872903807060280344576
        b = 6 / 2305843009213693952
        for i in range(3):
            x2 = self.mul(x,x)
            x3 = self.mul(x2,x)
            x = b * x2 + a * x3
        return x/2
        print("Does not work unless operations are in a ring of size 2^180 (huge)")

    def softmax(self, x):
        e = self.exp(x)
        s = self.sum_all(e)
        c = self.log(self.subs(s, e))
        return self.sigmoid(self.subs(x, c))

    def tanh(self, x):
        return self.add_const(self.mul_const(self.sigmoid(self.mul_const(x,2,real=True)),2,real=True), -1)

    def softplus(self, x):
        s = self.rabbit_compare(x, 3)
        a = self.mul(s,x)
        y = self.log(self.add_const(self.exp(x), 1, real=True))
        b = self.mul(self.add_const(-s,1),y)
        return self.add(a,b)

    def relu(self, x):
        s = self.rabbit_compare(x, 0)
        return self.mul(s,x)

    def leaky_relu(self, x, alpha):
        s = self.rabbit_compare(x, 0)
        a = self.mul(s, x)
        y = self.mul_const(x, alpha)
        b = self.mul(self.add_const(-s,1),y)
        return self.add(a, b)

    def sigmoid_linunit(self, x):
        x_a = self.rabbit_compare(x, -2.5)
        x_b = self.rabbit_compare(x, 2.5)
        x_1 = self.mul_const(x, 0.5)
        x_2 = self.mul_const(self.mul(x, x), 0.25)
        x_4 = self.mul_const(self.mul(x_2, x_2), -1/3)
        x_6 = self.mul_const(self.mul(x_4, x_2), -1/2.5)
        y = self.add(x_1, x_2)
        y = self.add(y, x_4)
        y = self.add(y, x_6)
        a = self.add(self.mul_const(y, -1), x)
        b = self.mul(x_b, a)
        c = self.add(b, y)
        sig = self.mul(x_a, c)
        return sig

    def gaussian(self, x):
        y = self.mul(x, x)
        y = self.mul_const(y, -1)
        return self.exp(y)

    def minmax(self, array, amin=None, amax=None):
        '''The array should be of shape (n,1)'''
        if amin==None:
            amin = array[0]
        if amax==None:
            amax = array[0]
        for i in range(len(array)):
            c = self.rabbit_compare(self.subs(amax, array[i]), 0)
            amax = self.add(self.mul(c, self.subs(amax, array[i])), array[i])
            c = self.rabbit_compare(self.subs(amin, array[i]), 0)
            amin = self.add(self.mul(c, self.subs(array[i], amin)), amin)
        return amin, amax

    def normalize(self, array, amin=None, amax=None):
        '''The array should be of shape (n,1)'''
        #return self.add_const(self.mul_const(array, 0.5), 0.25, real=True)
        amin, amax = self.minmax(array, amin, amax)
        denom = self.new_inverse(self.subs(amax, amin))
        temp = self.subs(array, amin)
        temp = self.mul(temp, np.tile(denom,(temp.shape[0],1)))
        #return temp  # between 0 and 1
        #return self.add_const(self.mul_const(temp, 2), -1) #between -1 and 1
        return self.add_const(self.mul_const(temp, 2), -1)

    def new_inverse(self, x):
        y = self.exp(self.add_const(-x,0.5))
        y = self.add_const(self.mul_const(y,3),0.003)
        for i in range(13):
            y = self.mul(y, self.add_const(self.mul(-x,y), 2))
        return y

    def new_1_sqrt(self, x):
        y = self.exp(self.add_const(self.mul_const(x,-0.7),-0.6))
        y = self.add_const(self.mul_const(y,5),0.003)
        for i in range(13):
            y = self.mul(self.mul_const(y, 0.5), self.add_const(self.mul(-x,self.mul(y,y)), 3))
        return y