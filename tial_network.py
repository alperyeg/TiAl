import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum
from dataloader import load_data
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, batch_size):
        super(Net, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(batch_size * 2, 100),
            nn.ELU(),
            nn.Linear(100, 10),
            nn.ELU(),
            nn.Linear(10, 3),
        )

    def forward(self, x):
        x = x.flatten()
        x = self.block(x)
        return x


class LossFunctions(Enum):
    """
    Enumeration for (custom) losses.

    Used by :function:loss_decision function.
    """
    BCE = 'BCE'
    CE = 'CE'
    MAE = 'MAE'
    MSE = 'MSE'
    NORM = 'NORM'
    customLossFunction = 'customLossFunction'

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


def loss_decision(x, target, epsilon, loss_name=LossFunctions.customLossFunction):
    """
    Method for defining losses
    """
    if loss_name == LossFunctions.BCE:
        loss = F.binary_cross_entropy(x, target.to(torch.float),
                                      reduction='sum')
    elif loss_name == LossFunctions.MSE:
        loss = F.mse_loss(x, target, reduction='sum')
    elif loss_name == LossFunctions.CE:
        loss = F.crossEntropyLoss(x, target)
    elif loss_name == LossFunctions.customLossFunction:
        temp = x[0] + x[1] * torch.pow(epsilon, x[2])
        loss = F.mse_loss(target.reshape(-1), temp, reduction='sum')
        # loss = torch.pow(target - temp, 2).sum()
    else:
        raise KeyError('Not a known loss name: {}'.format(loss_name))
    return loss


def train(net, epoch, train_loader, batch_size):
    net.train()
    train_loss = 0
    for idx, (img, target) in enumerate(train_loader):
        if img.size() != (batch_size, 3):
            continue
        optimizer.zero_grad()
        # network prediction for the image
        output = net(img[:, :2])
        # calculate the loss
        loss = loss_decision(output, target, img[:, 2],
                             LossFunctions.customLossFunction)
        # backprop
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if idx % 10 == 0:
            print('Loss {} in epoch {}, idx {}'.format(
                loss.item(), epoch, idx))

    print('Average loss: {} epoch:{}'.format(
        train_loss / len(train_loader.dataset), epoch))


def test(net, epoch, test_loader, batch_size):
    """
    Accuracy on the the test dataset
    """
    net.eval()
    test_accuracy = 0
    test_loss = 0
    with torch.no_grad():
        for idx, (img, target) in enumerate(test_loader):
            if img.size() != (batch_size, 3):
                continue
            output = net(img[:, :2])
            epsilon = img[:, 2]
            loss = loss_decision(output, target, epsilon,
                                 LossFunctions.customLossFunction)
            test_loss += loss.item()
            # network prediction
            pred = output[0] + output[1] * torch.pow(epsilon, output[2])
            # how many are correctly classified, compare with targets
            # test_accuracy += pred.eq(target.view_as(pred)).sum().item()
            # print(abs(target-pred))
            # print(sum(target - pred))
            if idx % 30 == 0 and ep == 3:
                plt.plot(target.reshape(-1).numpy(), label='target')
                plt.plot(pred.numpy(), label='prediction')
                plt.legend()
                # plt.savefig('res_ep{}_idx{}.eps'.format(ep, idx), format='eps')
                plt.show()
                print('Test Loss {} in epoch {}, idx {}'.format(
                    loss.item(), epoch, idx))

        print('Test accuracy: {} Average test loss: {} epoch:{}'.format(
            100 * 1 / len(test_loader.dataset),
            test_loss / len(test_loader.dataset), epoch))


if __name__ == "__main__":
    torch.manual_seed(0)
    batch_s = 32
    network = Net(batch_s).to(device)
    print(network)
    # Adam optimizer
    optimizer = optim.Adam(network.parameters(), lr=1e-3)
    dat = load_data('TNM.wfl', names=(
        'T', 'epunkt', 'e', 'sigma'), batch_size=batch_s, standardize=True)
    train_data = dat[0]
    test_data = dat[1]
    for ep in range(1, 4):
        train(network, ep, train_data, batch_s)
        print('training done')
        test(network, ep, test_data, batch_s)
        print('test done')
