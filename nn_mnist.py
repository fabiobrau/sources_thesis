import torch as tr
import torch.nn as nn
import torch.optim as optim
from dnn import mnist_load, vectorize, mnist_imshow
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import imageio

transform = transforms.Lambda(vectorize)

trainloader, testloader = mnist_load(4, transform)

print('Initialize Network...')
tr.manual_seed(1)
net = nn.Sequential(
    nn.Linear(28*28, 100),
    nn.ReLU(),
    nn.Linear(100, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 10)
)
print(net)
print('Done...')
criterion = nn.CrossEntropyLoss(reduction='elementwise_mean')
learning = 1e-3
optimizer = optim.SGD(net.parameters(), lr=learning, momentum=0.9)
print('Selected lossy function : ', criterion)
print('Selected optimization methods : ', optimizer)
max_iteration = 2
print('Start Learning Process...')
for epoch in range(max_iteration):
    current_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        out = net(inputs)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        current_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3e2' % (epoch + 1, i + 1, current_loss / 2000))
            current_loss = 0.0
print('End of Training.')
print('Start of Testing')
dataiter = iter(testloader)
images, labels = dataiter.next()
image = images[0, :].view(28, 28)
# print images
plt.imshow(image.numpy())
print('GroundTruth: ', ' '.join('%5s' % labels[j].item() for j in range(4)))
out = net(images)
_, pred = tr.max(out, 1)
print('Predicted :', ' '.join('%5s' % pred[j].item() for j in range(4)))

correct = 0
total = 0
with tr.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = tr.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the %d test images: %d %%' % (total,
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with tr.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = tr.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(10):
    print('Accuracy of %2d : %2d %%' % (
        i, 100 * class_correct[i] / class_total[i]))

print('Test over handwritten number seven')
img = imageio.imread('.//images//7amano.png')
img = tr.tensor(img).float().view(1, 784)/255
_, handpred = tr.max(net(img), 1)
print('Predicted : %d' % handpred )