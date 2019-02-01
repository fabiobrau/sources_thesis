##
# Define deep neural network class and other tools
# Using Torch for automatic differentiation
import numpy as np
import torch as tr
import struct as st
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt




class Layer:
    def __init__(self, signal, neuron, kind='rand', requires_grad=False, device=tr.device("cpu")):
        #   neuron   : number of neuron, i.e. row of the layer
        #   signal   : number of input signals, i.e column of layer
        self.device = device
        self.requires_grad = requires_grad
        if kind == 'void':
            self.weight = tr.zeros((neuron, signal), requires_grad=requires_grad, device=device)
            self.bias = tr.zeros(neuron, requires_grad=requires_grad, device=device)
        if kind == 'rand':
            self.weight = tr.randn((neuron, signal), device=device)*np.sqrt(2/signal)
            self.bias = tr.randn(neuron, device=device)*np.sqrt(2/signal)
            self.weight.requires_grad_(requires_grad)
            self.bias.requires_grad_(requires_grad)
        if kind == 'ones':
            self.weight = tr.ones((neuron, signal), requires_grad=requires_grad, device=device)
            self.bias = tr.ones(neuron, requires_grad=requires_grad, device=device)
        if kind == 'empty':
            self.weight = tr.empty((neuron, signal), requires_grad=requires_grad, device=device)
            self.bias = tr.empty(neuron, requires_grad=requires_grad, device=device)

    def __add__(self, b):
        # return the sum of two same-dimension layer
        dev = self.device
        c = Layer(self.weight.shape[1], self.weight.shape[0], kind='void', device=dev)
        c.weight = self.weight+b.weight
        c.bias = self.bias+b.bias
        return c

    def __sub__(self, b):
        # return the difference between two layer
        dev = self.device
        c = Layer(self.weight.shape[1], self.weight.shape[0], kind='void', device=dev)
        c.weight = self.weight-b.weight
        c.bias = self.bias-b.bias
        return c

    def __rmul__(self, x):
        # return the product for a scalar
        dev = self.device
        c = Layer(self.weight.shape[1], self.weight.shape[0], kind='void', device=dev)
        c.weight = x*self.weight
        c.bias = x*self.bias
        return c

    def __neg__(self):
        # return the layer with weight and bias of negative sign
        dev = self.device
        c = Layer(self.weight.shape[1], self.weight.shape[0], kind='void', device=dev)
        c.weight = -self.weight
        c.bias = -self.bias
        return c

    def detach(self):
        # detach a layer
        self.weight = self.weight.detach()
        self.bias = self.bias.detach()
        return self

    def grad(self):
        # compute grad respect to the layer
        dev = self.device
        c = Layer(self.weight.shape[1], self.weight.shape[0], requires_grad=False, device=dev)
        c.weight = self.weight.grad
        c.bias = self.bias.grad
        return c


class Network:
    def __init__(self, typ, iterations=10, learn=1e-5, mom=0.9, kind='rand', requires_grad=False, device=tr.device('cpu')):
        # typ   : integer vector of the type of the neural network
        # iterations : maximum number of iterations
        # learn : learn parameter in gradient correction
        # mom : momentum hyper-parameter
        # kind  : initialize layers with specific kind
        # requires_grad : initialize layers with gradient tracking option enable
        # device : ambient of operations, CPU or GPU
        self.device = device
        self.requires_grad = requires_grad
        self.tol = 1e-10
        self.iterations = iterations
        self.learn = learn
        self.mom = mom
        self.type = typ
        self.depth = np.size(typ)-1
        self.layer = []
        for i in range(self.depth):
            self.layer.append(Layer(self.type[i], self.type[i+1], kind=kind, requires_grad=requires_grad, device=device))

    def __call__(self, x):
        # Evaluate the network in the input x
        # x : input tensor
        for i in range(self.depth):
            if i < self.depth-1:
                x = tr.matmul(self.layer[i].weight, x)+self.layer[i].bias
                x.relu_()
            else:
                x = tr.matmul(self.layer[i].weight, x)+self.layer[i].bias
        return x

    def loss(self, x, y):
        z = self.__call__(x)
        N = 1
        if len(x.shape) > 1:
            N = x.shape[1]
        return 1/(2*N)*tr.norm(y-z, 2, 0).sum()

    def train_online(self, obj):
        # obj   : list of training object, input + labels
        # The meaning function, trains the neural network
        N = np.size(obj[0], 1)
        n = np.size(obj[0], 0)
        m = np.size(obj[1], 0)
        # Enable gradient tracking
        for i in range(self.depth):
            self.layer[i].weight.requires_grad_()
            self.layer[i].bias.requires_grad_()
        for k in range(self.iterations):
            glob_error = 0
            x = tr.tensor(obj[0], dtype=tr.float32, device=self.device)
            y = tr.tensor(obj[1], dtype=tr.float32, device=self.device)
            for j in range(N):
                    loss_error = self.loss(x[:, j], y[:, j])
                    loss_error.backward()
                    glob_error += loss_error.item()
                    print('Epoch :'+str(k)+' Train : '+str(j)+' error : '+str(loss_error.item()), end='\r')
                    for i in range(self.depth):
                        diff = self.layer[i].grad()
                        with tr.no_grad():
                            self.layer[i].weight -= self.learn*diff.weight
                            self.layer[i].bias -= self.learn*diff.bias
                        self.layer[i].weight.grad.zero_()
                        self.layer[i].bias.grad.zero_()
            print('Epoch : ' + str(k) + ', Global error : ' + str(1 / N * glob_error))
        # Disable gradient tracking at the end of the procedure
        for i in range(self.depth):
            self.layer[i].weight.requires_grad_()
            self.layer[i].bias.requires_grad()
        return self

    def train_batch(self, obj, dev=tr.device('cpu')):
        # obj   : list of training object, input + labels
        # The meaning function, trains the neural network
        N = np.size(obj[0], 1)
        n = np.size(obj[0], 0)
        m = np.size(obj[1], 0)
        # Enable gradient tracking
        for i in range(self.depth):
            self.layer[i].weight.requires_grad_()
            self.layer[i].bias.requires_grad_()
        x = tr.tensor(obj[0], dtype=tr.float32, device=dev)
        y = tr.tensor(obj[1], dtype=tr.float32, device=dev)
        for k in range(self.iterations):
            glob_error = self.loss(x, y)
            print('Epoch :'+str(k)+' Train : '+str(j)+' error : '+str(glob_error.item()), end='\r')
            glob_error.backward()
            for i in range(self.depth):
                diff_weight = self.layer[i].weight.grad
                diff_bias = self.layer[i].bias.grad
                with tr.no_grad():
                    self.layer[i].weight += -self.learn*diff_weight
                    self.layer[i].bias += -self.learn*diff_bias
                self.layer[i].weight.grad.zero_()
                self.layer[i].bias.grad.zero_()
        # Disable gradient tracking at the end of the procedure
        for i in range(self.depth):
            self.layer[i].weight.requires_grad_(False)
            self.layer[i].bias.requires_grad_(False)
        return self

    def classify(self, x):
        x = tr.tensor(x, dtype=tr.float32)
        out = self.__call__(x)
        out = np.argmax(np.array(out))
        return out


def vectorize(x):
    # to use with torchvision.transforms, convert to tensor
    # return the vectorized form of the photo
    y = transforms.functional.to_tensor(x)
    return y.view(np.prod(y.shape))


def vectorize_target(x):
    # to use with torchvision.transforms, convert to tensor
    # return the vectorized form of the target
    categories = 10
    y = tr.zeros(categories)
    y[x] = 1
    return y.float()


def mnist_load(batch_size=1, transform=transforms.ToTensor(), target_transform=None):
    train_buff = datasets.MNIST('.//data//MNIST', train=True, download=True, transform=transform,
                                target_transform=target_transform)
    test_buff = datasets.MNIST(".//data/MNIST", train=False, download=True, transform=transform,
                               target_transform=target_transform)
    train = tr.utils.data.DataLoader(train_buff, batch_size=batch_size, shuffle=True, pin_memory=True)
    test = tr.utils.data.DataLoader(test_buff, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train, test


def fashion_mnist_load(batch_size=1, transform=transforms.ToTensor()):
    train_buff = datasets.FashionMNIST('.//data//FashionMNIST', train=True, download=True, transform=transform)
    test_buff = datasets.FashionMNIST(".//data/FashionMNIST", train=False, download=True, transform=transform)
    train = tr.utils.data.DataLoader(train_buff, batch_size=batch_size, shuffle=True, pin_memory=True)
    test = tr.utils.data.DataLoader(test_buff, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train, test


def cifar10_load(batch_size=1, transform=transforms.ToTensor(), target_transform=None):
    train_buff = datasets.CIFAR10('.//data//CIFAR10', train=True, download=True, transform=transform,
                                target_transform=target_transform)
    test_buff = datasets.CIFAR10(".//data/CIFAR10", train=False, download=True, transform=transform,
                               target_transform=target_transform)
    train = tr.utils.data.DataLoader(train_buff, batch_size=batch_size, shuffle=True, pin_memory=True)
    test = tr.utils.data.DataLoader(test_buff, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train, test


def cifar100_load(batch_size=1, transform=transforms.ToTensor(), target_transform=None):
    train_buff = datasets.CIFAR100('.//data//CIFAR100', train=True, download=True, transform=transform,
                                target_transform=target_transform)
    test_buff = datasets.CIFAR100(".//data/CIFAR100", train=False, download=True, transform=transform,
                               target_transform=target_transform)
    train = tr.utils.data.DataLoader(train_buff, batch_size=batch_size, shuffle=True, pin_memory=True)
    test = tr.utils.data.DataLoader(test_buff, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train, test


def mnist(mode='train'):
    filename = {'images' : "..//MNIST//t10k-images.idx3-ubyte",
        'images_label' : "..//MNIST//t10k-labels.idx1-ubyte",
        'train' : "..//MNIST//train-images.idx3-ubyte",
        'train_label' : "..//MNIST//train-labels.idx1-ubyte"
        }
    # Reading training images
    print('Reading training images... ')
    train_buff = open(filename['train'],'br')
    train_buff.seek(0) 
    magic = st.unpack('>I', train_buff.read(4))[0]
    nimg = st.unpack('>I', train_buff.read(4))[0]
    nrow = st.unpack('>I', train_buff.read(4))[0]
    ncol = st.unpack('>I', train_buff.read(4))[0]
    print('magic number : ', magic)
    print('number of images : ', nimg)
    print('size of each image : ', nrow, 'x', ncol)
    elem = nimg*nrow*ncol
    train = np.asarray(st.unpack('>'+'B'*elem, train_buff.read(elem))).reshape((nimg, nrow*ncol))
    train = np.transpose(train)/(28*28)
    # Reading training images labels
    print('Reading training labels... ')
    lab_buff = open(filename['train_label'],'br')
    magic = st.unpack('>I', lab_buff.read(4))[0]
    nlab = st.unpack('>I', lab_buff.read(4))[0]
    print('magic number : ', magic)
    print('number of items : ', nlab)
    train_aux = np.zeros((1, nlab))
    train_aux = np.asarray(st.unpack('>'+'B'*nlab,lab_buff.read(nlab))).reshape((1,nlab))
    train_lab = np.zeros((10,nlab))
    for i in range(nlab):
        train_lab[train_aux[0,i],i]=1
    return train, train_lab
