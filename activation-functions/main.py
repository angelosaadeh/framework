import numpy as np
from matplotlib import pyplot as plt


def function(prot, x):
    np.set_printoptions(suppress=True)
    a = np.array([(i - 50) / 10 for i in range(101)])

    b = np.exp(-a**2)
    y = prot.reconstruct(prot.gaussian(x))
    e1 = (y-b)*100
    plt.plot(b,'black', e1,'red')
    plt.show()

    b = (np.exp(a)-np.exp(-a))/(np.exp(a)+np.exp(-a))
    y = prot.reconstruct(prot.tanh(x))
    e2 = np.abs(y-b)

    b = np.log(1+np.exp(a))
    y = prot.reconstruct(prot.softplus(x))
    e3 = np.abs(y-b)

    b = (a>0)*a
    y = prot.reconstruct(prot.relu(x))
    e4 = np.abs(y-b)

    b = (a>0)*a + (1-(a>0))*0.01*a
    y = prot.reconstruct(prot.leaky_relu(x, 0.01))
    e5 = np.abs(y-b)

    b = a/(1+np.exp(-a))
    y = prot.reconstruct(prot.sigmoid_linunit(x))
    e6 = np.abs(y-b)

    b = 1/(1+np.exp(-a))
    y = prot.reconstruct(prot.sigmoid(x))
    e7 = np.abs(y-b)

    if prot.identity == 'alice':
        data = [e2, e3, e6, e7]

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)

        bp = ax.boxplot(data, patch_artist=True,
                        notch='True', vert=0)

        colors = ['#5F9EA0']*len(data) #, '#EE7621', '#E3CF57', '#6E8B3D', '#009ACD', '#5D478B', '#4B0082']

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        for whisker in bp['whiskers']:
            whisker.set(color='#292421',
                        linewidth=1.5,
                        linestyle=":")

        for cap in bp['caps']:
            cap.set(color='#292421',
                    linewidth=2)

        for median in bp['medians']:
            median.set(color='red',
                       linewidth=3)

        # changing style of fliers
        for flier in bp['fliers']:
            flier.set(marker='D',
                      color='#292421',
                      alpha=0.5)

        ax.set_yticklabels(['TanH', 'Softplus', 'Sigmoid LU', 'Sigmoid'])

        plt.title("Box plots of activation functions errors")

        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        plt.show()

    if prot.identity == 'bob':
        data = [e1, e4, e5]

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)

        bp = ax.boxplot(data, patch_artist=True,
                        notch='True', vert=0)

        colors = ['#5F9EA0'] * len(data)  # , '#EE7621', '#E3CF57', '#6E8B3D', '#009ACD', '#5D478B', '#4B0082']

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        for whisker in bp['whiskers']:
            whisker.set(color='#292421',
                        linewidth=1.5,
                        linestyle=":")

        for cap in bp['caps']:
            cap.set(color='#292421',
                    linewidth=2)

        for median in bp['medians']:
            median.set(color='red',
                       linewidth=3)

        # changing style of fliers
        for flier in bp['fliers']:
            flier.set(marker='D',
                      color='#292421',
                      alpha=0.5)

        ax.set_yticklabels(['Gaussian',  'Relu', 'Leaky Relu'])

        plt.title("Box plots of activation functions errors")

        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        plt.show()