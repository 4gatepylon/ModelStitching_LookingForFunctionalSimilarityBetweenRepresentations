from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from datetime import datetime
import argparse

# Import our networks
from net import Net, StitchNet, StitchMode
from yaml_reader import Experiment

# TODO we need a way to ascertain that starter and ender are, in fact, frozen in net.py (or somewhere else)
# TODO we need a way to ascertain that stitching correctly flows the data as necessary

# For this experiment. We assume one experiment per run of main, store it in a log-dir with year, month, day,
# hour, and minute (we assume that here is at most one experiment per minute).
tb_writer = SummaryWriter(logdir="experiment-tensorboard-{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M")))

# Copied from the MNIST example present in Pytorch's code base
def train(model, device, train_loader, optimizer, epoch, log_interval, verbose=True):
    model.train()

    avg_loss = 0.0
    num_batches = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        latent = model(data)
        output = model.classify(latent, use_dropout=True)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if verbose and (batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        
        avg_loss += torch.sum(loss).item()
        num_batches += 1
    
    num_batches = max(float(num_batches), 1.0)
    avg_loss /= num_batches
    return avg_loss


def test(model, device, test_loader, verbose=True):
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
    percent_correct = 100. * correct / len(test_loader.dataset)

    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), percent_correct))
    
    return test_loss, percent_correct


def eval_stitches(model_tuples, device, test_loader, stitch_acc_per_epoch, verbose=True):
    for stitch_name, (starter, ender, stitch) in model_tuples.items():
        avg_starter_loss, pc_start = test(starter, device, test_loader, verbose=False)
        avg_ender_loss, pc_end = test(ender, device, test_loader, verbose=False)
        avg_stitched_loss, pc_stitch = test(stitch, device, test_loader, verbose=False)
        if verbose:
            print("Stitch {} losses: {}, {} -> {}".format(stitch_name, avg_starter_loss, avg_ender_loss, avg_stitched_loss))
            print("Stitch {} correct rates {}, {} -> {}".format(stitch_name, pc_start, pc_end, pc_stitch))

        # Often we will be increasing the sizes for the ender, so we will want to see a negative or very low
        # penalty (improvement). Note that these are averages and we may want to look example by example.
        starter_stitch_penalty = pc_start - pc_stitch
        ender_stitch_penalty = pc_end - pc_stitch
        if verbose:
            print("Stitching penalty for starter: {}".format(starter_stitch_penalty))
            print("Stitching penalty for ender: {}".format(ender_stitch_penalty))
        
        # Calculate the stitching penalty over time (i.e. by epoch of stitch training)
        for epoch, acc in stitch_acc_per_epoch[stitch_name]:
            tb_writer.add_scalar("{}_starter_stitching_penalty".format(stitch_name), pc_start - acc, epoch + 1)
            tb_writer.add_scalar("{}_ender_stitching_penalty".format(stitch_name), pc_end - acc, epoch + 1)
            # Acc obviously in [0, 100] so roughly this is shifted by 2 in the log space
            tb_writer.add_scalar("{}_log_starter_stitching_penalty".format(stitch_name), torch.log10(pc_start - acc).item(), epoch + 1)
            tb_writer.add_scalar("{}_log_ender_stitching_penalty".format(stitch_name), torch.log10(pc_end - acc).item(), epoch + 1)

# The idea is to train two networks with slightly different starting parameters
DEFAULT_TRAIN_BATCH_SIZE = 64
DEFAULT_TEST_BATCH_SIZE = 1000
DEFAULT_LR = 1.0
DEFAULT_LR_EXP_DROP = 0.7 # also called 'gamma'
PRINT_EVERY = 100
SAVE_MODEL_FILE_TEMPLATE = "mnist_cnn_{}.pt" # use .format(name here) to give it a name

def train_test_save_models(models_and_names, device, train_loader, test_loader, epochs, acc_per_epoch=None):
    assert(acc_per_epoch is None or type(acc_per_epoch) == dict)
    for model_name, model in models_and_names.items():
        if not acc_per_epoch is None:
                acc_per_epoch[model_name] = []
        
        print("Training {} for {} epochs".format(model_name, epochs))
        # Each model gets its own optimizer and scheduler since we may want to vary across them later
        optimizer = optim.Adadelta(model.parameters(), lr=DEFAULT_LR)
        scheduler = StepLR(optimizer, step_size=1, gamma=DEFAULT_LR_EXP_DROP)

        # Train for the epochs for that one model
        for epoch in range(1, epochs + 1):
            # Train accuracy is not totally meaningful since
            # the model is training across batches, meaning that
            # the accuracy is not the same per epoch. We decide
            # to look at per-epoch metrics to avoid testing too much.
            train_loss = train(model, device, train_loader, optimizer, epoch, PRINT_EVERY)
            test_loss, test_acc = test(model, device, test_loader)
            scheduler.step()

            # Log to tensorboard
            tb_writer.add_scalars("{}_loss".format(model_name), {"test": test_loss, "train": train_loss}, epoch)
            tb_writer.add_scalar("{}_test_acc".format(model_name), test_acc, epoch)

            if not acc_per_epoch is None:
                acc_per_epoch[model_name].append(test_acc)
        
        torch.save(model.state_dict(), SAVE_MODEL_FILE_TEMPLATE.format(model_name))

# I think this is a greyscale thing, and it is shared across both
# main and __main__
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

# Stitch from/to FC will be a linear layer
# Stitch from/to CNN will be a 1x1 cnn that maintains the number of channels
def main(epochs=1, shuffle_data=False):
    # Set up the settings based on the requested ones (in arguments)
    use_cuda = torch.cuda.is_available()
    train_batch_size = DEFAULT_TRAIN_BATCH_SIZE
    test_batch_size = DEFAULT_TEST_BATCH_SIZE

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': train_batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    train_kwargs['shuffle'] = shuffle_data
    test_kwargs['shuffle'] = shuffle_data
    
    # load the dataset for test and train from a local location (supercloud has to access to the internet)
    dataset1 = datasets.MNIST('./data', train=True, download=False, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)
    
    # these are basically iterators that give us utility functionality
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    
    # Default filename is experiment.yaml in this same folder
    stitch_acc_per_epoch = {}
    experiment = Experiment()
    net_init = lambda **kwargs: Net(**kwargs).to(device)
    stitch_init = lambda starter, ender, stitch_mode: StitchNet(starter, ender, stitch_mode, device=device)
    train_func = lambda models: train_test_save_models(models, device, train_loader, test_loader, epochs)
    stitch_train_func = lambda models: train_test_save_models(models, device, train_loader, test_loader, epochs, stitch_acc_per_epoch)
    eval_func = lambda models: eval_stitches(models, device, test_loader, stitch_acc_per_epoch, verbose=True)

    experiment.load_yaml()
    experiment.init_nets(net_init)
    experiment.train_nets(train_func)
    experiment.init_stitch_nets(stitch_init)
    experiment.train_stitch_nets(train_func)
    experiment.evaluate_stitches(eval_func)
    
if __name__ == '__main__':
    if torch.cuda.is_available():
        print("### CUDA IS AVAILABLE!")
    else:
        print("### RUNNING ON CPU")
    
    # We allow users to pick whether they want to download or not, because in supercloud you cannot use
    # the internet, so you'd want to download before, then activate your node (etc) in, say, interactive mode
    parser = argparse.ArgumentParser(description='Decide whether to download the dataset (MNIST) or run training.')
    parser.add_argument('--d', dest='d', action='store_true', help="Download dataset or run experiment")
    parser.add_argument('--e', default=1, type=int, help="Number of epochs")
    parser.add_argument('--drng', dest="drng", action="store_true", help="Randomize data")
    parser.set_defaults(d=False)
    parser.set_defaults(drng=False)
    args = parser.parse_args()

    if args.d:
        # Simply download everything
        datasets.MNIST('./data', train=True, download=True, transform=transform)
        datasets.MNIST('./data', train=False, transform=transform)
    else:
        assert(type(args.e) == int)
        assert(args.e >= 1)
        main(epochs=args.e, shuffle_data=args.drng)
