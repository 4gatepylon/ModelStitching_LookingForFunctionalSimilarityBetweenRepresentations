import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from hyperparams import (
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_TEST_BATCH_SIZE,
    DOWNLOAD_DATASET,
)

# Simple train function to run a single epoch of training
def train1epoch(model, device, train_loader, optimizer):
    model.train()

    avg_loss = 0.0
    num_batches = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        avg_loss += torch.sum(loss).item()
        num_batches += 1
    
    num_batches = max(float(num_batches), 1.0)
    avg_loss /= num_batches
    return avg_loss

# Simple testing function (can be used every epoch, at the end, or whenever)
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
    percent_correct = 100. * correct / len(test_loader.dataset)
    
    return test_loss, percent_correct

# Function initialize datasets/dataloaders, return the correct device to use (i.e. cuda if it exists), and set up testing params
def init():
    # Initialize the datasets (assuming already downloaded)
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device is {}".format("cuda" if use_cuda else "cpu"))

    train_kwargs = {'batch_size': DEFAULT_TRAIN_BATCH_SIZE}
    test_kwargs = {'batch_size': DEFAULT_TEST_BATCH_SIZE}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    train_kwargs['shuffle'] = True
    test_kwargs['shuffle'] = True
    
    # load the dataset for test and train from a local location (supercloud has to access to the internet)
    dataset1 = datasets.MNIST('./data', train=True, download=DOWNLOAD_DATASET, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    return train_loader, test_loader, device