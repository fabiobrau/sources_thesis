import torch as tr
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn
import matplotlib.patches as patches
from scipy.spatial import HalfspaceIntersection
import numpy as np
import numpy.random as random
import sys
from convex_adversarial import robust_loss, DualNetwork, RobustBounds
sys.path.append("../")


seaborn.set(font_scale=2)
seaborn.set_style("white")

# generiamo m punti random nel piano a distanza 3r
m = 30
np.random.seed(19)
x = [random.rand(2)]
r = 0.04
while len(x) < m:
    p = random.rand(2)
    if min(np.max(np.abs(p-a)) for a in x) > 3*r:
        x.append(p)
X = tr.tensor(np.array(x), dtype=tr.float32)
tr.manual_seed(1)
y = (tr.rand(m)+0.5).long()

# Inizializzaimo una rete neurale
tr.manual_seed(1)
net = nn.Sequential(
    nn.Linear(2, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 2)
)
epoch = 1000
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
opt = optim.Adam(net.parameters(), lr=learning_rate)
for i in range(epoch):
    out = net(X)
    loss = loss_fn(out, y)
    print('Epoch : %d, error : %e' % (i+1, loss.item()), end='\r')
#    err = (out.max(1)[1].item != y).float().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

print('Error : ' + str(loss.item()))
print('Training :' + str(x))
print('Image of Training :' + str(out))
print('Targets :' + str(y))
l = 1
for par in net.parameters():
    if len(par.data.shape) > 1:
        l = l*par.data.norm(1, 1).max()

print('Lipschitz norm of the network : '+str(l.item()))
# Construct the robust test function
test_net = RobustBounds(net, r)

XX, YY = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
X0 = np.zeros((np.size(XX), 2))
X0[:, 0] = XX.reshape(np.size(XX))
X0[:, 1] = YY.reshape(np.size(YY))
X0 = tr.tensor(X0, dtype=tr.float32)
y0 = net(X0)
y0 = y0.detach()
ZZ = (y0[:, 0] - y0[:, 1]).view(100, 100).numpy()

fig, ax = plt.subplots(figsize=(8, 8))
ax.contourf(XX, YY, -ZZ, cmap="Spectral", levels=np.linspace(-1000, 1000, 3))
ax.scatter(X.numpy()[:, 0], X.numpy()[:, 1], c=y.numpy(), cmap="Spectral", s=70)
ax.axis("equal")
ax.axis([0, 1, 0, 1])
plt.axis('off')
plt.savefig('./images/2Dpoints'+str(m)+'.png')

#testing the net
dual_net = DualNetwork(net, X, r)
Y = tr.zeros(m, 1, 2)
for i in range(m):
    aux_i = y[i].item()
    Y[i, 0, aux_i] = 1
    Y[i, 0, 1-aux_i] = -1
test = dual_net(Y)
for i in range(m):
    ax.annotate(str(round(test[i, 0].item(), 2)), X[i, :])
plt.savefig('./images/2Dpoints'+str(m)+'testing.png')
for i in range(m):
    ax.add_patch(patches.Rectangle((X[i, 0]-r, X[i, 1]-r), 2*r, 2*r, fill=False))
plt.savefig('./images/2Dpoints'+str(m)+'tested.png')
