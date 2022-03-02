import requests
import os

import torch
import torch.nn as nn

from resnet import (
    model_urls, 
    RESNETS_FOLDER,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    resnext50_32x4d,
    resnext101_32x8d,
    wide_resnet50_2,
    wide_resnet101_2,
)

model_constructors = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "resnext50_32x4d": resnext50_32x4d,
    "resnext101_32x8d": resnext101_32x8d,
    "wide_resnet50_2": wide_resnet50_2,
    "wide_resnet101_2": wide_resnet101_2,
}

# generate works once you have downloaded
def generate(name):
    fname = f"{name}_imagenet.pt"
    fname = os.path.join(RESNETS_FOLDER, fname)
    
    model = model_constructors[name](pretrained=True, progress=True)

    stateDict= torch.load(fname)
    # NOTE this is a hack to avoid problems with the state dict
    # These remove the final linear layer according to "CIFAR10 modification"
    del stateDict['fc.weight']
    del stateDict['fc.bias']
    # This is to fix the problem that arises from the fact that conv1 for imagenet is
    # a 7x7. Note how in the resnet.py code, self.inplanes is 64 at the start. This
    # works because the state dict is literally just matrices, and they are added
    # as nn.Parameter objects later inside the nn.Conv2d and other modules.
    del stateDict['conv1.weight']
    stateDict['conv1.weight'] = torch.rand((64, 3, 3, 3), requires_grad=True)
    model.load_state_dict(stateDict)
    return model

FINETUNE_EPOCHS = 40
if __name__ == "__main__":
    if not os.path.isdir(RESNETS_FOLDER):
        os.mkdir(RESNETS_FOLDER)
    # Download the imagenet version of the resnets
    for name, url in model_urls.items():
        fname = f"{name}_imagenet.pt"
        fname = os.path.join(RESNETS_FOLDER, fname)
        if not os.path.isfile(fname):
            print(f"Downloading ImageNet model {name} from {url} to {fname}")
            r = requests.get(url)
            with open(fname, "wb") as f:
                f.write(r.content)
    # Finetune and save CIFAR-10 modification
    for name, url in model_urls.items():
        fname = f"{name}.pt"
        fname = os.path.join(RESNETS_FOLDER, fname)
        if not os.path.isfile(fname):
            model = generate(name)
            train()
