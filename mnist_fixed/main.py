from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
import argparse

# Import our networks
from net import Net, StitchNet, StitchMode
from yaml_reader import Experiment

# TODO please make a printf utility when you make the tensorboard shit (and make it
# print/log to a file as necessary or whatever)

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
DEFAULT_TRAIN_BATCH_SIZE = 64
DEFAULT_TEST_BATCH_SIZE = 1000
DEFAULT_LR = 1.0
DEFAULT_LR_EXP_DROP = 0.7 # also called 'gamma'
PRINT_EVERY = 100
SAVE_MODEL_FILE_TEMPLATE = "mnist_cnn_{}.pt" # use .format(name here) to give it a name


def train_test_save_models(models_and_names, device, train_loader, test_loader, epochs):
    for model_name, model in models_and_names.items():
        print("Training {} for {} epochs".format(model_name, epochs))
        # Each model gets its own optimizer and scheduler since we may want to vary across them later
        optimizer = optim.Adadelta(model.parameters(), lr=DEFAULT_LR)
        scheduler = StepLR(optimizer, step_size=1, gamma=DEFAULT_LR_EXP_DROP)

        # Train for the epochs for that one model
        for epoch in range(1, epochs + 1):
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
def main(epochs=1):
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
    
    # Default filename is experiment.yaml in this same folder
    experiment = Experiment()
    net_init = lambda **kwargs: Net(**kwargs).to(device)
    stitch_init = lambda starter, ender, stitch_mode: StitchNet(starter, ender, stitch_mode, device=device)
    train_func = lambda models: train_test_save_models(models, device, train_loader, test_loader, epochs)

    # TODO we need an evaluator than can measure accuracies and do some plots ideally also in log-space
    eval_func = lambda models: None

    experiment.load_yaml()
    experiment.init_nets(net_init)
    experiment.train_nets(train_func)
    experiment.init_stitch_nets(stitch_init)
    experiment.train_stitch_nets(train_func)
    experiment.evaluate_stitches(eval_func)
    
if __name__ == '__main__':
    if torch.cuda.is_available():
        print("CUDA IS AVAILABLE!")
    else:
        print("RUNNING ON CPU")
    
    # We allow users to pick whether they want to download or not, because in supercloud you cannot use
    # the internet, so you'd want to download before, then activate your node (etc) in, say, interactive mode
    parser = argparse.ArgumentParser(description='Decide whether to download the dataset (MNIST) or run training.')
    parser.add_argument('--d', dest='d', action='store_true')
    parser.add_argument('--e', default=1, type=int, help="Number of epochs")
    parser.set_defaults(d=False)
    args = parser.parse_args()

    if args.d:
        # Simply download everything
        datasets.MNIST('./data', train=True, download=True, transform=transform)
        datasets.MNIST('./data', train=False, transform=transform)
    else:
        assert(type(args.e) == int)
        assert(args.e >= 1)
        main(epochs=args.e)
