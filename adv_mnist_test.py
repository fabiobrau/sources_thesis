import torch as tr
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from dnn import mnist_load, vectorize
import numpy as np
from convex_adversarial import DualNetwork
import matplotlib.pyplot as plt
import imageio


transform = transforms.Lambda(vectorize)
batch_size = 4
trainloader, testloader = mnist_load(batch_size, transform=transform)
tr.manual_seed(3)
net = nn.Sequential(
    nn.Linear(28*28, 100),
    nn.ReLU(),
    nn.Linear(100, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 10)
)
print('Network : ')
print(net)
lossy_fn = nn.CrossEntropyLoss(reduction='elementwise_mean')
opt = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

batch_size = 4
print('Optimization method : ')
print(opt)
epochs = 3
print('Start Learning Process...')
for epoch in range(epochs):
    total_loss = 0.0
    train_iter = iter(trainloader)
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # Training relu network
        out = net(inputs)
        loss = lossy_fn(out, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 20 == 19:
            print(' Epoch : %d, training object : %5d, local error  : %.2e '
                  % (epoch + 1, (i + 1)*batch_size, loss.item()), end='\r')
        total_loss += loss.item()
    print()
    print(' Epoch : %d, main error  : %e' % (epoch + 1, total_loss/((i+1)*batch_size)))
print('End of Training.')
print(' ')
print('Start of Testing...')

corr_unit = list(0. for i in range(10))
corr_unit_adv = list(0. for i in range(10))
corr_unit_dual = list(0. for i in range(10))
total = list(0. for i in range(10))
total_adv = list(0. for i in range(10))
total_dual = list(0. for i in range(10))

eps = 0.03

# Build an auxiliar list of possible vector c to apply dual bound
Adv_Label = tr.zeros(10, 9, 10)
Iaux = tr.eye(10)
for i in range(10):
    Adv_Label[i, :, i] = 1
    Adv_Label[i, :i, :] += - Iaux[:i, :]
    Adv_Label[i, i:, :] += -Iaux[(i+1):, :]

tr.manual_seed(1)
for data in testloader:
    images, labels = data
    # Testing relu network
    images.requires_grad_()
    out = net(images)
    _, pred = tr.max(out, 1)
    c = (pred == labels).squeeze()
    for i in range(batch_size):
        label = labels[i]
        corr_unit[label] += c[i].item()
        total[label] += 1
    # Creating Network's Adversarial Examples
    loss = lossy_fn(out, labels)
    loss.backward()
    perturbation = images.grad.sign()
    images_adv = images.detach() + eps*perturbation
    images_adv = tr.clamp(images_adv, 0, 1)
    out = net(images_adv)
    _, pred = tr.max(out, 1)
    c = (pred == labels).squeeze()
    for i in range(batch_size):
        label = labels[i].item()
        corr_unit_adv[label] += c[i].item()
        total_adv[label] += 1
    # Compute dual bound (As minimum of all possible adv examples
    dual_net = DualNetwork(net, images, eps)
    Labtest = tr.zeros(batch_size, 9, 10)
    for i in range(batch_size):
        label = labels[i]
        Labtest[i, :, :] = Adv_Label[label, :, :]
    test, _ = dual_net(Labtest).min(1)
    print(test)
    c = (test > 0).squeeze()
    for i in range(batch_size):
        label = labels[i].item()
        corr_unit_dual[label] += c[i].item()
        total_dual[label] += 1

for i in range(10):
    print('Accuracy of predicting %2d | Predected correctly  : %2d %% | Adversarial Examples: %2d %% | '
          'Proved Robust %2d %%'
          % (i, 100 * corr_unit[i] / total[i], 100 - 100 * corr_unit_adv[i] / total_adv[i],
             100*corr_unit_dual[i]/total_dual[i]))
total = np.array(total).sum()
total_adv = np.array(total_adv).sum()
total_dual = np.array(total_dual).sum()
correct = np.array(corr_unit).sum()
correct_adv = np.array(corr_unit_adv).sum()
corr_dual = np.array(corr_unit_dual).sum()
print('Global Accuracy for Clean Images: %d %% | '
      'Global Amount of Adversarial Examples %d %% | ' 
      'Global Proved Robust Testing Object %d %%'
      % (100*correct/total, 100 - 100 * correct_adv / total_adv, 100*corr_dual/total_dual))
