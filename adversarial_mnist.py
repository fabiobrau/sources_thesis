# Questo script istruisce una rete neurala sequenziale lineare
# genera con il metodo del fast gradient alcuni esempi avversariali e permatte
# Di salvarli ATTENZIONE non salvare a caso che viene usato nel pdf latex

import torch as tr
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from dnn import mnist_load, vectorize
import numpy as np
import matplotlib.pyplot as plt


transform = transforms.Lambda(vectorize)
batch_size = 4
trainloader, _ = mnist_load(batch_size, transform=transform)

tr.manual_seed(3)
net = nn.Sequential(
    nn.Linear(28*28, 100),
    nn.ReLU(),
#    nn.Linear(100, 16),
#    nn.ReLU(),
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

print('Start of Testing...')
batch_size = 1
_, testloader = mnist_load(batch_size, transform=transform)
eps = 0.07
tr.manual_seed(3)
test_iter = iter(testloader)
get_out = ''
while get_out != 'q':
    images, labels = test_iter.next()
    samples = torchvision.utils.make_grid(images.view(batch_size, 1, 28, 28))
    out = net(images)
    _, pred = out.max(1)
    plt.subplot(131)
    plt.imshow(samples[1, :, :], cmap='gray')
    plt.axis('off')
    plt.title(pred.item())
    print('Ground Truth :', labels.tolist(), 'Predicted : ', pred.tolist())
    # Making Adversarial Example
    images.requires_grad_()
    out = net(images)
    loss = lossy_fn(out, labels)
    loss.backward()
    correction = images.grad.sign()
    adver_examples = images.detach() + eps*correction.detach()
    adver_examples = tr.clamp(adver_examples, 0, 1)
    out = net(adver_examples)
    _, pred_adv = out.max(1)
    print('Ground Truth :', labels.tolist(), 'Predicted : ', pred_adv.tolist())
    # Display adversarial examples
    adver_samples = torchvision.utils.make_grid(adver_examples.view(batch_size, 1, 28, 28))
    plt.subplot(132)
    plt.imshow(adver_samples[1, :, :], cmap='gray')
    plt.axis('off')
    plt.title(pred_adv.item())
    plt.pause(0.5)
    # Making noise example
    noise = 2*eps*tr.randn(images.shape)
    noise_examples = images.detach() + noise
    noise_examples = tr.clamp(noise_examples, 0, 1)
    out = net(noise_examples)
    _, pred_noise = out.max(1)
    print('Ground Truth :', labels.tolist(), 'Predicted : ', pred_noise.tolist())
    # Display noise examples
    noise_samples = torchvision.utils.make_grid(noise_examples.view(batch_size, 1, 28, 28))
    plt.subplot(133)
    plt.imshow(noise_samples[1, :, :], cmap='gray')
    plt.axis('off')
    plt.title(pred_noise.item())
    plt.pause(0.5)
    # Quitting from choice
    get_out = input('Press q to exit, s to save')
    if get_out == 's':
        plt.savefig('./images/adversal'+str(pred_adv.item())+'noise'+str(pred_noise.item())+'.png')

