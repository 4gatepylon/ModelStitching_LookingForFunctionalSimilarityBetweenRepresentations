from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from enum import Enum
from tensorboardX import SummaryWriter
import yaml
import argparse

class Net(nn.Module):
    # Note that MNIST has 28x28 images
    # convolutional depth is the number of convolutions (and they are put in the depth direction)
    # The width is the number of neurons of that layer (corresponds to the height of the weight
    # matrix that outputs that layer, and width of the matrix that takes that as an input)
    def __init__(self, conv1_depth=32, conv2_depth=64, hidden_width=128):
        super(Net, self).__init__()
        # MNIST is not rgb, but instead black and weight and so starts out with a depth of 1
        # We can mess with stride and/or convolutional window size later
        self.conv1 = nn.Conv2d(1, conv1_depth, 3, 1)
        self.conv2 = nn.Conv2d(conv1_depth, conv2_depth, 3, 1)

        # We can mess with these parameters later
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # We use 3x3 convolutions so the width and height lose 2 each convolution
        # meaning we go from 28 x 28 -> 26 x 26 -> 24 x 24. Subsequently, we half each
        # by using the max pool, yielding 12 x 12 (depth conv2_depth, still). Thus we end up with
        # 12 x 12 x conv2_depth = 144 * conv2_depth
        fc1_width = 144 * conv2_depth
        self.fc1 = nn.Linear(fc1_width, hidden_width)
        self.fc2 = nn.Linear(hidden_width, 10)

        # Store these to enable stitching later
        self.conv1_depth = conv1_depth
        self.conv2_depth = conv2_depth
        self.fc1_width = fc1_width

        # In the future we may want to use this
        self.hidden_width = hidden_width

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        latent = F.relu(x)
        return latent
    
    def classify(self, latent, use_dropout=False):
        if use_dropout:
            latent = self.dropout2(latent)
        x = self.fc2(latent)
        output = F.log_softmax(x, dim=1)
        return output

class StitchMode(Enum):
    CONV2 = 1 # Stitch before conv2
    FC1 = 2 # Stitch before fc1
    FC2 = 3 # Stitch before fc2

def linear_stitch_seq(starter_out, ender_in):
    return nn.Sequential(
        # Batch norm 1d subtracts the expected value (average), divides by the
        # standard deviation, and then rescales and biases that vector (once per mini-batch)
        nn.BatchNorm1d(starter_out),
        nn.Linear(starter_out, ender_in),
        nn.BatchNorm1d(ender_in),
    )

def cnn_stitch_seq(starter_out, ender_in):
    # Note that connect_units is the output size of the previous layer (i.e. if we modify conv2
    # this would be the output size of conv1 and should yield the same output size, since conv2
    # is expecting that as input)
    return nn.Sequential(
        # Batch norm 2d does the same to batch-norm 1d but for a 2d tensor (i.e. a grid); I think you can
        # think of it as doing it on a flattened version, per mini-batch
        nn.BatchNorm2d(starter_out),
        # kernel size 1, stride 1
        nn.Conv2d(starter_out, ender_in, 1, 1),
        nn.BatchNorm2d(ender_in),
    )

# Stitch net is very slow because of all the CPU logic, I think... 
# TODO we need a way to ascertain that starter and ender are, in fact,
# frozen
class StitchNet(nn.Module):
    def __init__(self, starter, ender, stitch_mode=None):
        super(StitchNet, self).__init__()
        if stitch_mode == StitchMode.CONV2:
            self.stitch_mode = StitchMode.CONV2
            # Second one will expect input to conv2 to have the output of its own convolution
            self.stitch = cnn_stitch_seq(starter.conv1_depth, ender.conv1_depth)
        elif stitch_mode == StitchMode.FC1:
            self.stitch_mode = StitchMode.FC1
            # Same as before second one will expect the input to have the dimensions of its own
            # inputs
            self.stitch = linear_stitch_seq(starter.fc1_width, ender.fc1_width)
        elif stitch_mode == StitchMode.FC2:
            self.stitch_mode = StitchMode.FC2
            # Once again we need to make sure to pass the inputs in the proper sizes that
            # the ender network expects
            self.stitch = linear_stitch_seq(starter.hidden_width, ender.hidden_width)
        else:
            raise NotImplementedError

        # Freeze the networks and store the frozen weights; careful with aliasing
        for param in starter.parameters():
            param.requires_grad = False
        for param in ender.parameters():
            param.requires_grad = False

        self.starter = starter
        self.ender = ender
    
    # Whether to forward and do the stitch right before conv2, right before fc1,
    # or whether to do it later for fc2 (or wherever), using only the starter for forward
    def forward_conv2(self, x):
        # Starter
        x = self.starter.conv1(x)
        x = F.relu(x)
        # Stitch
        x = self.stitch(x)
        # Ender
        x = self.ender.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.ender.dropout1(x)
        x = torch.flatten(x , 1)
        x = self.ender.fc1(x)
        latent = F.relu(x)
        return latent
    def forward_fc1(self, x):
        # Starter
        x = self.starter.conv1(x)
        x = F.relu(x)
        x = self.starter.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.starter.dropout1(x)
        x = torch.flatten(x, 1)
        # Stitch
        x = self.stitch(x)
        # Ender
        x = self.ender.fc1(x)
        latent = F.relu(x)
        return latent
    def forward(self, x):
        if self.stitch_mode == StitchMode.CONV2:
            return self.forward_conv2(x)
        elif self.stitch_mode == StitchMode.FC1:
            return self.forward_fc1(x)
        else:
            # Simply use the starter's forward to get the latent
            return self.starter(x)
    
    # Here we choose whether the starter or ender are giving us the latent for this classification
    def classify_starter(self, latent, use_dropout=False):
        if use_dropout:
            latent = self.starter.dropout2(latent)
        x = self.stitch(latent)
        x = self.ender.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    def classify(self, latent, use_dropout=False):
        if self.stitch_mode == StitchMode.FC2:
            return self.classify_starter(latent, use_dropout=use_dropout)
        return self.ender.classify(latent, use_dropout=use_dropout)

# Copied from the MNIST example present in Pytorch's code base
def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        latent = model(data)
        output = model.classify(latent, use_dropout=True)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            # TODO use tensorboard
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            latent = model(data)
            # In the future we may want to avoid using dropout, but for consistency
            # I've chosen to use it for all such examples
            output = model.classify(latent, use_dropout=True)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# The idea is to train two networks with slightly different starting parameters
DEFAULT_EPOCHS = 1
DEFAULT_TRAIN_BATCH_SIZE = 64
DEFAULT_TEST_BATCH_SIZE = 1000
DEFAULT_LR = 1.0
DEFAULT_LR_EXP_DROP = 0.7 # also called 'gamma'
PRINT_EVERY = 100
SAVE_MODEL_FILE_TEMPLATE = "mnist_cnn_{}.pt" # use .format(name here) to give it a name

# We'll store layer settings, etc... later here and do various stitching tests
SETTINGS_FILE = "mnist_fixed.yaml"

# Here are some testing constants that we are using to make sure we are able to stich,
# obviously this is later going to come via the yaml file and also some hyperparameter tuning, etc...
# TODO start using the YAML file instead; potentially before that actually enable longer network
# stitching
DEFAULT_NET_KWARGS = {
    "conv1_depth": 32,
    "conv2_depth": 64,
    "hidden_width": 128,
}

FAT_CONV2_NET_KWARGS = {
    "conv1_depth": 32,
    "conv2_depth": 128,
    "hidden_width": 128,
}

FAT_FC_NET_KWARGS = {
    "conv1_depth": 32,
    "conv2_depth": 64,
    "hidden_width": 256,
}

def train_test_save_models(models_and_names, device, train_loader, test_loader):
    for model, model_name in models_and_names:
        print("Training {} for {} epochs".format(model_name, DEFAULT_EPOCHS))
        # Each model gets its own optimizer and scheduler since we may want to vary across them later
        optimizer = optim.Adadelta(model.parameters(), lr=DEFAULT_LR)
        scheduler = StepLR(optimizer, step_size=1, gamma=DEFAULT_LR_EXP_DROP)

        # Train for the epochs for that one model
        for epoch in range(1, DEFAULT_EPOCHS + 1):
            train(model, device, train_loader, optimizer, epoch, PRINT_EVERY)
            test(model, device, test_loader)
            scheduler.step()
        
        # Save that specific model, TODO save a tensorboard plot somewhere and print the command to use to show it
        torch.save(model.state_dict(), SAVE_MODEL_FILE_TEMPLATE.format(model_name))

# I think this is a greyscale thing, and it is shared across both
# main and __main__
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

# Stitch from/to FC will be a linear layer
# Stitch from/to CNN will be a 1x1 cnn that maintains the number of channels
def main():
    # Set up the settings based on the requested ones (in arguments)
    use_cuda = torch.cuda.is_available()
    train_batch_size = DEFAULT_TRAIN_BATCH_SIZE
    test_batch_size = DEFAULT_TEST_BATCH_SIZE

    # TODO we are going to want to modify the seed here to be a different (possibly manual)
    # seed for each of the differnet networks. For now we use a random seed, but having manual
    # capabilities may be helpful
    # seed = 1
    # torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': train_batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    # load the dataset for test and train from a local location (supercloud has to access to the internet)
    dataset1 = datasets.MNIST('./data', train=True, download=False, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)
    
    # these are basically iterators that give us utility functionality
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # We have three models that represent three variations on the same network, and we will attempt to stitch them
    default_model = Net(**DEFAULT_NET_KWARGS).to(device)
    fat_fc_model = Net(**FAT_FC_NET_KWARGS).to(device)
    fat_conv2_model = Net(**FAT_CONV2_NET_KWARGS).to(device)

    # Run main training loop to train all the models
    original_models = ((default_model, "default"), (fat_fc_model, "fat_fc1"), (fat_conv2_model, "fat_conv2"))
    train_test_save_models(original_models, device, train_loader, test_loader)

    # Try stithing from smaller models to larger models; expect better performance
    # TODO measure stitching accuracy (i.e. whether the stitch worked or not)
    default_to_fat_fc = StitchNet(default_model, fat_fc_model, StitchMode.FC1)
    default_to_fat_conv2  = StitchNet(default_model, fat_conv2_model, StitchMode.CONV2)
    stitched_models = ((default_to_fat_fc, "default_to_fat_fc1"), (default_to_fat_conv2, "default_to_fat_conv2"))
    train_test_save_models(stitched_models, device, train_loader, test_loader)
    
if __name__ == '__main__':
    if torch.cuda.is_available():
        print("CUDA IS AVAILABLE!")
    # We allow users to pick whether they want to download or not, because in supercloud you cannot use
    # the internet, so you'd want to download before, then activate your node (etc) in, say, interactive mode
    parser = argparse.ArgumentParser(description='Decide whether to download the dataset (MNIST) or run training.')
    parser.add_argument('--d', dest='d', action='store_true')
    parser.set_defaults(d=False)
    args = parser.parse_args()

    if args.d:
        # Simply download everything
        datasets.MNIST('./data', train=True, download=True, transform=transform)
        datasets.MNIST('./data', train=False, transform=transform)
    else:
        main()
