import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from models import QAlexNetMNIST
from filters import filter_state_dict
from mnist import TransformMNIST

class Net(nn.Module):
    def __init__(self, x_dim=1, y_dim=1):
        super(Net, self).__init__()
        H = 30
        self.fc1 = nn.Linear(x_dim, H)
        self.fc2 = nn.Linear(y_dim, H)
        self.fc3 = nn.Linear(H, 1)

    def forward(self, x, y):
        h1 = F.relu(self.fc1(x)+self.fc2(y))
        h2 = self.fc3(h1)
        return h2

def estimate_mi(X, Y, batch_size=100, num_batch=10000, net=None):
    
    y_shuffle = np.random.permutation(Y)
    print('estimate_mi', y_shuffle.shape)
    print(X.shape)
    start_idx = 0
    opt = optim.Adam(net.parameters(), lr=0.001, amsgrad=True)
    all_loss = []
    for i in range(num_batch):
        end_idx = start_idx + batch_size
        if end_idx >= X.shape[0]:
            #start_idx = 0
            #end_idx = start_idx + batch_size
            end_idx = X.shape[0]
        x_sample = Variable(torch.from_numpy(X[start_idx:end_idx,:]).float().cuda())
        y_sample = Variable(torch.from_numpy(Y[start_idx:end_idx,:]).float().cuda())
        ys_sample = Variable(torch.from_numpy(y_shuffle[start_idx:end_idx,:]).float().cuda())
        pred_xy = net(x_sample,y_sample)
        pred_x_y = net(x_sample, ys_sample)
        loss = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        loss = -1.0 * loss
        net.zero_grad()
        loss.backward()
        opt.step()
        print(loss)
        all_loss.append(float(-1.0*loss.detach().data.cpu().numpy()))
        start_idx += batch_size
    return all_loss

def calc_MI(X,Y,bins):
    c_xy = np.histogram2d(X,Y,bins)[0]
    c_x = np.histogram(X, bins)[0]
    c_y = np.histogram(Y, bins)[0]
    h_x = shan_entropy(c_x)
    h_y = shan_entropy(c_y)
    h_xy = shan_entropy(c_xy)
    mi = h_x + h_y - h_xy
    return mi

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -np.sum(c_normalized* np.log(c_normalized))  
    return H



if __name__ == "__main__":
    main()
    rangeNum = 1000
    dimOfInput = 5
    x_mi = []
    y_mi = []
    testarray1 = []
    testarray2 = []
    for dim in range(rangeNum):
        temp = dim
        while temp != 0:
            testarray1 = testarray1.append[0]
            testarray2 = testarray2.append[1]
            temp=temp-1 

        x = np.random.multivariate_normal(mean=np.array(testarray1),cov=np.diag(np.array(testarray2)),size=10000000).reshape((10000000,dimOfInput))
        y = np.random.multivariate_normal(mean=np.array(testarray1),cov=np.diag(np.array(testarray2)),size=10000000).reshape((10000000,dimOfInput))
        y2 = x * 2
        net1 = Net(dimOfInput,dimOfInput).float().cuda()
        net2 = Net(dimOfInput,dimOfInput).float().cuda()
        mi1 = estimate_mi(x, y, net=net1)
        mi2 = estimate_mi(x, y2, net=net2)
        x_mi.append(mi1)
        y_mi.append(mi2)


    plt.plot(x_mi,y_mi)
