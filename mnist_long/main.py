import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# This is for the overnight experiment
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from datetime import datetime

# These are two hard-coded examples with 3 and 10 convolutions respectively
# of sizes hard-coded by me. They both have 2 FCs followed by an FC classifier
# of the same shape/size
from examples_cnn import (
    NET_3_2 as C32,
    NET_4_2 as C42,
    NET_10_2 as C102,
    NET_3_2_TO_NET_3_2_STITCHES as C32T32,
    NET_4_2_TO_NET_4_2_STITCHES as C42T42,
    NET_10_2_TO_NET_10_2_STITCHES as C102T102,
    NET_3_2_TO_NET_4_2_STITCHES as C32T42,
    NET_3_2_TO_NET_10_2_STITCHES as C32T102,
)
from examples_fc import (
    NET_3 as F3,
    NET_5 as F5,
    NET_8 as F8,
    NET_3_TO_NET_3_STITCHES as F3T3,
    NET_5_TO_NET_5_STITCHES as F5T5,
    NET_8_TO_NET_8_STITCHES as F8T8,
    NET_3_TO_NET_5_STITCHES as F3T5,
    NET_3_TO_NET_8_STITCHES as F3T8,
)
from hyperparams import (
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_TEST_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_LR_EXP_DROP,
    DEFAULT_EPOCHS_OG,
    DEFAULT_EPOCHS_STITCH,
    NUM_EXPERIMENTS,
    NUM_STITCH_EXPERIMENTS,
    DOWNLOAD_DATASET,
    MODE
)

from net import (
    Net,
    get_stitches,
    StitchedForward,   
)

# Simple train and test functions to run a single train or test run in a single epoch
def train(model, device, train_loader, optimizer):
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

# Run train and test for each epoch that exists after initializing an optimizer and log to a file for
# further analysis in a future time
def train_og_model(model, model_name, model_directory, device, train_loader, test_loader, epochs, logfile):        
    # Each model gets its own optimizer and scheduler since we may want to vary across them later
    optimizer = optim.Adadelta(model.parameters(), lr=DEFAULT_LR)
    scheduler = StepLR(optimizer, step_size=1, gamma=DEFAULT_LR_EXP_DROP)

    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer)
        test_loss, test_acc = test(model, device, test_loader)
        scheduler.step()

        # Log to the file
        with open(logfile, "a") as f:
            f.write("model {}\n\tepoch {}\n\ttrain loss {}\n\ttest loss {}\n\ttest acc {}\n".format(
                model_name, epoch, train_loss, test_loss, test_acc))
    # Make sure to save the model for further analysis
    torch.save(model.state_dict(), "{}/{}.pt".format(model_directory, model_name))

def train_stitch(model1, model2, stitch,
    model1_name, model2_name,
    idx1, idx2,
    device, train_loader, test_loader, epochs,
    logfile, directory, name_prefix):
    stitched_model = StitchedForward(model1, model2, stitch, idx1, idx2)

    # NOTE how we only optimize the parameters in the stitch!
    for p in model1.parameters():
        p.requires_grad = False
    for p in model2.parameters():
        p.requires_grad = False

    optimizer = optim.Adadelta(stitch.parameters(), lr=DEFAULT_LR)
    scheduler = StepLR(optimizer, step_size=1, gamma=DEFAULT_LR_EXP_DROP)

    for epoch in range(1, epochs + 1):
        train_loss = train(stitched_model, device, train_loader, optimizer)
        test_loss, test_acc = test(stitched_model, device, test_loader)
        scheduler.step()
        # Log to the file
        with open("{}/{}_{}.txt".format(directory, name_prefix, logfile), "a") as f:
            f.write("model1 {} model2 {}\nidx1 {} idx2 {}\n\tepoch {}\n\ttrain loss {}\n\ttest loss{}\n\ttest acc{}\n".format(
                model1_name, model2_name, idx1, idx2, epoch, train_loss, test_loss, test_acc))
    # Make sure to save the model for further analysis
    torch.save(stitch.state_dict(), "{}/{}_l{}_{}_l{}.pt".format(
        directory, name_prefix,
        model1_name, idx1, model2_name, idx2))

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

# Run a number of experiments where for each we run a number of stitch experiments per stitch
# The directory of the experiment dir will be flat
def run_experiments(
    experiments_prefix,
    shortnet_name_original, shortnet_layers,
    longnet_name_original, longnet_layers,
    stitch_idx_dict,
    num_experiments, num_stitch_experiments,
    experiments_dir,
    train_loader, test_loader,
    device):
    # You gigve a directory in which to put experiments
    # We create:
    #   exp_i (folder)
    #     stitch_exp_j pts (files)
    #     stitch_exp_j txts (files)
    #     exp_i txts (files)
    for experiment_num in range(num_experiments):
        shortnet_name = "{}_{}".format(shortnet_name_original, experiment_num)
        longnet_name = "{}_{}".format(longnet_name_original, experiment_num)

        print("*** Initializing experiment {}/{} in {} for {}***".format(experiment_num, num_experiments, experiments_dir, experiments_prefix))
        exp_name = "{}/exp_{}_{}".format(experiments_dir, experiments_prefix, experiment_num)
        os.mkdir(exp_name)
        shortnet = Net(layers=shortnet_layers).to(device)
        longnet = Net(layers=longnet_layers).to(device)

        print("*** Training OG Models ***")

        # Note that the logfile is not put in the directory automatically (even though the
        # pt file is put, in this case, in exp_name)
        train_og_model(
            shortnet, shortnet_name, exp_name,
            device, train_loader, test_loader, DEFAULT_EPOCHS_OG, "{}/{}_train.txt".format(exp_name, shortnet_name))
        train_og_model(
            longnet, longnet_name, exp_name,
            device, train_loader, test_loader, DEFAULT_EPOCHS_OG, "{}/{}_train.txt".format(exp_name, longnet_name))
        
        # Do regular stitch experiments
        for stitch_experiment_num in range(num_stitch_experiments):
            print("*** Trying Stitch Experiment {}/{} for {} ***".format(stitch_experiment_num, num_stitch_experiments, exp_name))
            stitch_exp_name = "stitch_exp_{}".format(stitch_experiment_num)
    
            stitches = get_stitches(shortnet, longnet, stitch_idx_dict)
            for idx1, idx2s in stitches.items():
                for idx2, stitch in idx2s.items():
                    stitch = stitch.to(device)
                    train_stitch(
                        shortnet, longnet, stitch,
                        shortnet_name, longnet_name,
                        idx1, idx2, device,
                        # We store each stitch experiment as it's own file
                        train_loader, test_loader, DEFAULT_EPOCHS_STITCH,
                        "{}_l{}_{}_l{}.txt".format(
                            # The stitch experiment name includes the directory path
                            stitch_exp_name,
                            shortnet_name, idx1, longnet_name, idx2),
                        exp_name, stitch_exp_name)

        # For the control we only do one experiment because it'll be cheaper and it should be "good enough"
        # to simply do one since we expect it to "fail" anyways. Note that we don't save the weights of
        # the random network since once again we assume they are not totally meaningful (the chance that
        # we can extract some pattern and find it to have impacted the result is astronomically low)
        print("*** Setting up Untrained Control for experiment {} ***".format(exp_name))
        shortnet = Net(layers=shortnet_layers).to(device)
        longnet = Net(layers=longnet_layers).to(device)
        untrained_shortnet_name = "untrained_{}".format(shortnet_name)
        untrained_longnet_name = "untrained_{}".format(longnet_name)

        print("*** Trying Stitch Untrained Control for experiment {} ***".format(exp_name))
        for idx1, idx2s in stitches.items():
            for idx2, stitch in idx2s.items():
                stitch_exp_name = "stitch_untrained_exp_{}".format(stitch_experiment_num)
                stitch = stitch.to(device)
                train_stitch(
                    shortnet, longnet, stitch,
                    untrained_shortnet_name, untrained_longnet_name,
                    idx1, idx2, device,
                    train_loader, test_loader, DEFAULT_EPOCHS_STITCH,
                    "{}_{}_l{}_{}_l{}.txt".format(
                        stitch_exp_name,
                        untrained_shortnet_name, idx1, untrained_longnet_name, idx2),
                    exp_name, stitch_exp_name)


if __name__ == "__main__":
    intras = [
        # Convolutional Intra-network
        ("C32", C32, "C32", C32, C32T32, "C32T32"),
        ("C42", C42, "C42", C42, C42T42, "C42T42"),
        ("C102", C102, "C102", C102, C102T102, "C102T102"),
        # FC Intra-network
        ("F3", F3, "F3", F5, F3T3, "F3T3"),
        ("F5", F5, "F5", F5, F5T5, "F5T5"),
        ("F8", F8, "F8", F5, F8T8, "F8T8"),
    ]
    inters = [
        # Convolutional Inter-network
        ("C32", C32, "C42", C42, C32T42, "C32T42"),
        ("C32", C32, "C42", C102, C32T102, "C32T102"),
        # FC Inter-network
        ("F3", F3, "F5", F5, F3T5, "F3T5"),
        ("F3", F3, "F8", F8, F3T8, "F3T8")
    ]
    do = None
    do_str = None
    if MODE == INTER:
        do = inters
        do_str = "inter"
    if MODE == INTRA:
        do = intras
        do_mode = "intra"
    else:
        raise ValueError("Mode was {}, but it should be either {} (inters) or {} (intras)".format(MODE, INTER, INTRA))
    
    # Make a folder for these experiments
    # You will not be running more than one set of experiments per minute...
    date_prefix = datetime.now().strftime("%Y-%m-%d-%m")
    if not os.path.isdir("experiments"):
        print("Making experiments folder")
        os.mkdir("experiments")
    if os.path.isdir("experiments/{}".format(date_prefix)):
        raise RuntimeError("An experiment is already running for {}".format(date_prefix))
    print("Making experiment for date {}".format(date_prefix))
    experiments_dir = "experiments/{}_{}".format(do_str, date_prefix)
    os.mkdir(experiments_dir)
    

    train_loader, test_loader, device = init()

    for model_pair in do:
        shortnet_name, shortnet_layers, longnet_name, longnet_layers, stitch_idx_dict, exp_prefix = model_pair
        run_experiments(
            exp_prefix,
            shortnet_name, shortnet_layers,
            longnet_name, longnet_layers,
            stitch_idx_dict,
            NUM_EXPERIMENTS, NUM_STITCH_EXPERIMENTS,
            experiments_dir,
            train_loader, test_loader,
            device)
