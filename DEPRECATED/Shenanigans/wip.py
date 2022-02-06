import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# Plan:
# 1. Make simple MNIST CNN work (don't use the general Net from below, use something more hard-coded)
# 2. Make multiple copies of it with different layer widths and stitch those together, get stitching to work
# 3. Make a simple single axis of variation and see if we can do optimal stitches for these different lengths

# A simple CNN network for MNIST that expects in MNIST digits in a single band channel
# and also expects a tuple layer_sizes that declares a sequence of layers such that the beginning
# if the sequence is comprised of convolutions and the end is comprised of FCs. Between every FC
# there is a ReLU and a dropout, but not for the last FC. We use zero padding. Maxpool after the last conv layer.
# We have dropout before the last fc layer.
class Net(nn.Module):
    # Expect a tuple of layer sizes in the format
    #   (conv_layer_sizes, fc_layer_sizes)
    # where conv_layer_sizes has format
    #   [(channels_in, channels_out, kernel_size, stride)]
    # and fc_layer_sizes has format
    #   [(in_dims, out_dims)]
    # where all those root values are ints
    def __init__(self, layer_sizes, img_dims=(28, 28), maxpool_size=2, dropout_p1=0.25, dropout_p2=0.5):
        super(Net, self).__init__()

        # We do not support RGB images yet
        assert(len(img_dims) == 2)
        img_width, img_height = img_dims

        # Ensure right format
        assert(not layer_sizes is None)
        assert(type(layer_sizes) == tuple)
        assert(len(layer_sizes) == 2)

        conv_layer_sizes, fc_layer_sizes = layer_sizes

        # Make sure the proper formats
        assert(not conv_layer_sizes is None)
        assert(not fc_layer_sizes is None)
        assert(type(conv_layer_sizes) == list or type(conv_layer_sizes) == tuple)
        assert(type(fc_layer_sizes) == list or type(fc_layer_sizes) == tuple)
        assert(len(fc_layer_sizes) > 0)
        assert(len(conv_layer_sizes) > 0)

        assert(max([len(conv) for conv in conv_layer_sizes]) == 4)
        assert(max([len(conv) for conv in conv_layer_sizes]) == 4)
        assert(sum([(1 if (sum([(1 if (type(conv[i]) == int) else 0) for i in range(len(conv))]) == 4) else 0) for conv in conv_layer_sizes]) == len(conv_layer_sizes))
        assert(max([len(fc) for fc in fc_layer_sizes]) == 2)
        assert(max([len(fc) for fc in fc_layer_sizes]) == 2)
        assert(sum([(1 if (type(fc[0]) == int and type(fc[1]) == int) else 0) for fc in fc_layer_sizes]) == len(fc_layer_sizes))

        # Make sure the sizes match
        assert(conv_layer_sizes[0][0] == 1)
        assert(max([abs(conv_layer_sizes[i][1] - conv_layer_sizes[i+1][0]) for i in range(0, len(conv_layer_sizes - 1))]) == 0)
        assert(fc_layer_sizes[0][0] == image_width * image_height * conv_layer_sizes[-1][1])
        assert(max([abs(fc_layer_sizes[i][out_dim] - fc_layer_sizes[i+1][in_dim]) for i in range(0, len(fc_layer_sizes) - 1)]) == 0)

        convs = [nn.Conv2d(in_chan, out_chan, kern_size, stride) for in_chan, out_chan, kern_size, stride in conv_layer_sizes]
        conv_with_relus = [val for val in [convs[i], nn.ReLU()] for i in range(len(conv_layer_sizes))]
        fcs = [nn.Linear(in_size, out_size) for in_size, out_size in fc_layer_sizes]
        fc_with_relus = [val for val in [fcs[i], nn.ReLU()] for i in range(len(fc_layer_sizes) - 1)]
        self.seq = nn.Sequantial(*(
            convs_with_relus +
            [nn.MaxPool2d(maxpool_size), nn.Dropout(dropout_p1), nn.Flatten(1)] +
            fcs_with_relus +
            [nn.Dropout(drouput_p2), nn.Linear(*fc_layer_sizes[-1])]
        ))
    
    def forward(self, x):
        # conv 1
        # relu
        # conv 2
        # relu
        # ...
        # max pool 2d
        # dropout
        # flatten
        # fc 1
        # relu
        # fc 2
        # relu
        # ...
        # dropout
        # fc final
        # softmax classifies
        x = self.seq(x)
        output = F.log_softmax(, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    pass

# stitch on a simple hardcoded model
def simple_stitch(model1, model2):
    pass

# stitch two layers and return the accuracy (i.e. a measure of similarity)
def stitch(model1, model2, layer1, layer2):
    return 0

# return a tuple of pairings of the best stitches between two models
def optimal_stitchs(model1, model2, layers1, layers2, seq=False):
    pairings = []
    start = 0
    for i in range(0, layers1):
        find = max(i for i in range(start, layers2), key=lambda l1, l2: stitch(model1, model2, layer1, layer2))
        if seq:
            start = find + 1
        pairings.append((i, find))
    return pairings

if __name__ == "__main__":
    pass