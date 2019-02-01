import torch as tr
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from dnn import mnist_load, vectorize
import numpy as np
import matplotlib.pyplot as plt
import imageio


transform = transforms.Lambda(vectorize)
batch_size = 4
trainloader, testloader = mnist_load(batch_size, transform=transform)

tr.manual_seed(3)
net = nn.Sequential(
    nn.Linear(28*28, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
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
epochs = 2
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
total = list(0. for i in range(10))
total_adv = list(0. for i in range(10))
eps = 0.07
for data in testloader:
    images, labels = data
    # Testing relu netowrk
    images.requires_grad_()
    out = net(images)
    _, pred = tr.max(out, 1)
    c = (pred == labels).squeeze()
    for i in range(batch_size):
        label = labels[i]
        corr_unit[label] += c[i].item()
        total[label] += 1
    # Testing Network over Adversarial Examples
    loss = lossy_fn(out, labels)
    loss.backward()
    perturbation = images.grad.sign()
    images_adv = images.detach() + eps*perturbation
    out = net(images_adv)
    _, pred = tr.max(out, 1)
    c = (pred == labels).squeeze()
    for i in range(batch_size):
        label = labels[i]
        corr_unit_adv[label] += c[i].item()
        total_adv[label] += 1
for i in range(10):
    print('Accuracy of predicting %2d | Clean   : %2d %% | Perturbed: %2d %% '
          % (i, 100 * corr_unit[i] / total[i], 100 * corr_unit_adv[i] / total_adv[i]))
total = np.array(total).sum()
total_adv = np.array(total_adv).sum()
correct = np.array(corr_unit).sum()
correct_adv = np.array(corr_unit_adv).sum()
print('Global Accuracy for Clean Images: %d %% | '
      'Global Accuracy for Adversarial Examples %d %%' %
      (100*correct/total, 100 * correct_adv / total_adv))
