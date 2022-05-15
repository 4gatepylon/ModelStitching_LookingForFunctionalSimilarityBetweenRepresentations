import numpy as np
import torch
import torch.nn as nn

from mega_resnet import Resnet

class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)

# NOTE this is modified to work with the mega_resnet Resnet
def convert_model_to_Seq(model):
    assert(type(model) == Resnet, "Only resnets are supported in `yamini.convert_model_to_Seq`")
    blocksets = model.blocksets
    layer1 = blocksets[0]
    layer2 = blocksets[1]
    layer3 = blocksets[2]
    layer4 = blocksets[3]
    return [model.conv1, model.bn1, model.relu] + list(layer1) + list(layer2) + list(layer3) + list(layer4) + [model.avgpool, Flatten(), model.fc]

# NOTE model 1 and model 2 should have the same architecture
# NOTE you will have to pass in pre-trained already-loaded models if you want to stitch them
def model_stitch(model1, model2, conv_layer_num, kernel_size=1, stitch_depth=1, stitch_no_bn=False, conv_layer_num_top=None):
    # NOTE that this is modified (or will be) to work with the mega_resnet Resnet

    assert(type(model1) == Resnet, "Only resnets are supported in `yamini.model_stitch`")
    assert(type(model2) == Resnet, "Only resnets are supported in `yamini.model_stitch`")

    # If the top layer is not different from the bottom layer, just
    # use the same layer for each.
    if conv_layer_num_top is None:
        conv_layer_num_top = conv_layer_num

    ##### ResNet ########
    new_model1 = convert_model_to_Seq(model1)
    new_model2 = convert_model_to_Seq(model2)

    # NOTE this is possible to calculate, but since they do it this way, you MUST run it on Cuda
    x = torch.randn(2, 3, 32, 32).cuda()
    # These are probably depth
    # NOTE this is in (sender, expected sender) format, which is DIFFERENT from (sender, reciever)
    # ... relative to my system, this is (_, -1), but seems to have a different conv_layer_num
    connect_units_in = new_model1[:conv_layer_num].cuda()(x).shape[1]
    connect_units_out = new_model2[:conv_layer_num_top].cuda()(x).shape[1]

    if stitch_depth==1:
        if stitch_no_bn:
            connection_layer = [
                nn.Conv2d(connect_units_in, connect_units_out, kernel_size=(kernel_size, kernel_size), padding=int(kernel_size/2)),
            ]
        else:
            connection_layer = [
                nn.BatchNorm2d(connect_units_in),
                nn.Conv2d(connect_units_in, connect_units_out, kernel_size=(kernel_size, kernel_size), padding=int(kernel_size/2)),
                nn.BatchNorm2d(connect_units_out),
            ]

        # TODO wtf is this
        # if 'resnet18k' in model_name and conv_layer_num==11:
        #     connection_layer = [
        #         nn.BatchNorm1d(connect_units_in),
        #         nn.Linear(in_features = connect_units_in, out_features=connect_units_out, bias=True),
        #         nn.BatchNorm1d(connect_units_out),
        #     ]
        
    elif stitch_depth==2:
        connection_layer = [
            nn.Conv2d(connect_units_in, connect_units_out, kernel_size=(kernel_size, kernel_size), padding=int(kernel_size/2)),
            nn.BatchNorm2d(connect_units_in),
            nn.ReLU(),
            nn.Conv2d(connect_units_out, connect_units_out, kernel_size=(kernel_size, kernel_size), padding=int(kernel_size/2)),
            nn.BatchNorm2d(connect_units_out),
        ]

    else:
        raise NotImplementedError

    # Stitched model
    model_centaur = nn.Sequential(*(
        list(new_model1[:conv_layer_num]) + 
        connection_layer + 
        list(new_model2[conv_layer_num_top:])
    ))

    # Freeze
    for i in range(conv_layer_num):
        for param in model_centaur[i].parameters():
            param.requires_grad = False
    for i in np.arange(conv_layer_num + len(connection_layer), len(model_centaur)):
        for param in model_centaur[i].parameters():
            param.requires_grad = False

    return model_centaur