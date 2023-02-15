"""All the classes and functions that are common to all users"""
import socket
import pickle
import numpy as np
import pandas as pd
import math
import threading
import generate


class Unpack:
    """The user's local directory"""
    def __init__(self):
        pass

    @staticmethod
    def get_file(file):
        """Imports a file as secret"""
        print('A file is being imported...')
        return pd.read_csv(file, header=None).values


class Network:
    """Set up the network"""
    def __init__(self):
        pass

    @staticmethod
    def serve(ip, port):
        """Create a server"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((ip, port))
        print('A server is created')
        return server

    @staticmethod
    def connect(ip, port):
        """Connect to a server"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.connect((ip, port))
        print('You are connected to a server')
        return server

    @staticmethod
    def accept_connections(server):
        """Accept a connection"""
        server.listen()
        print('The server is listening to accept connections')
        client, _ = server.accept()
        print('A client just got connected')
        return client


class Communication:
    """Send and receive messages"""
    def __init__(self, length, header=10000):
        """The maximum length of the messages that are to be sent"""
        self.length = length
        self.header = header

    def send(self, receiver, message, print_receipt=False):
        """Send a message"""
        if print_receipt:
            print(f'A message is being prepared to be sent.')
        message_s = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL) #todo verifier les niveaux de protocoles
        if print_receipt:
            print(f'The message has been pickled.')
        message_s = bytes(f'{len(message_s):<{self.header}}', "utf-8") + message_s
        receiver.send(message_s)
        if print_receipt:
            print(f'The message {message} has been sent.')

    def get(self, sender, print_receipt=False):
        """Receive a message in packets"""
        if print_receipt:
            print(f'Waiting for the message...')
        full_message = b''
        new_message = True
        while True:
            if print_receipt:
                print(f'A packet is received')
            message = sender.recv(self.length)
            if new_message:
                message_len = int(message[:self.header])
                new_message = False
                print(message_len)
            full_message += message

            if len(full_message) - self.header == message_len:
                message = pickle.loads(full_message[self.header:])
                if print_receipt:
                    print(f'The message {message} has been well received')
                return message

    def broadcast(self, receivers, message, print_receipt=False):
        """Broadcast the same message to everyone"""
        threads = [threading.Thread(target=self.send, args=[receiver, message, print_receipt]) for receiver in receivers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def send_shares(self, receivers, shares, print_receipt=False):
        """Broadcast one message for every player, not necessarily the same message"""
        threads = [threading.Thread(target=self.send, args=[receivers[i], shares[i], print_receipt]) for i in range(len(receivers))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

class Protocols(Communication):
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
            file = open('alice_rand.p', 'rb')
        else:
            file = open('bob_rand.p', 'rb')
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
        self.interactions += 1
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
        self.interactions += 1
        shares_partner = np.random.randint(0, 2, x.shape)
        shares_identity = (x - shares_partner).astype(int) % 2
        self.send(self.partner, shares_partner)
        return shares_identity

    def receive_shares(self):
        """To be used after every secret_share on the receiver's side"""
        self.interactions += 1
        return self.get(self.partner, print_receipt=True)

    def reconstruct(self, x):
        """Reconstruct a secret"""
        self.interactions += 1
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
        self.interactions += 1
        if self.identity == 'alice':
            self.send(self.partner, x)
            y = self.get(self.partner)
        if self.identity == 'bob':
            y = self.get(self.partner)
            self.send(self.partner, x)
        return (x+y) % self.mod

    def reveal_bin(self, x):
        """Open an element without converting it to a real number, ie keep it as a ring element"""
        self.interactions += 1
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

    def truncate(self, x, bits):
        """Truncate the required number of bits"""
        for i in range(bits):
            x = self.truncate_bit(x)
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

    def mul(self, x, y, real=True):
        """Element-wise multiplication of secrets"""
        self.number_mul += 1
        a, b, c = self.get_triple(x)
        d = self.subs(x, a)
        e = self.subs(y, b)
        d, e = self.reveal(np.array([d, e]))
        z = self.add_const(c + d * b + e * a, e * d, real=False)
        if real:
            z = self.truncate(z, self.precision)
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
            z = np.insert(z, 0, self.or_bin(y[len(y) - i - 2], z[0]), axis=0)
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
            y = self.truncate(y, self.precision)
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

    def inverse(self, x):
        y = self.exp(-self.log(x))
        return y

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
        y=x
        for i in range(35):
            y2 = self.mul(y,y)
            y3 = self.mul(y2,y)
            y = self.add_const(self.add(self.mul_const(y3, -0.25), self.mul_const(y, 0.75)), 0.5)
            y = self.add_const(self.mul_const(y, 2), -1)
        y = self.mul_const(self.add_const(y,1),0.5)
        return y

    def inverse(self, x):
        """inverse for positive real numbers, correct for numbers between 1/1000 and 100"""
        return self.exp(self.mul_const(self.log(x), -1))

    def reset_rbm(self, x, y):
        return self.add(self.mul(self.mul_const(self.add_const(-x, - 1), 1 / 2), self.add(self.mul(y, y) , - y)) , x)

    def bernoulli(self, k):
        u = np.random.uniform(0, 1, k.shape)
        return self.rabbit_compare(self.add(-u, k), 0)

    def sum(self, x, y, b):
        return np.sum(x,y,b) % self.mod
