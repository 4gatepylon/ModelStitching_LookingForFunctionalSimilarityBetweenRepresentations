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
    # NOTE this is in (pre-sender, reciever) format, which is DIFFERENT from (sender, reciever)
    # ... you need to pass in the index AFTER the pre-sender if you want to include the sender
    # (as I do)
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

# NOTE I added this to be able to convert labels to yamini's indices
# NOTE resnet_10 is input -> [ conv1 -> bn1 -> relu -> blocksets[0] -> ... -> blocksets[3] -> avgpool -> flatten -> fc ]
# which means:
# conv1        = 0
# bn1          = 1
# relu         = 2
# blocksets[0] = 3
# blocksets[1] = 4
# blocksets[2] = 5
# blocksets[3] = 6
# avgpool      = 7
# flatten      = 8
# fc           = 9
def resnet10_label2idx(label):
    if type(label) == str:
        if label == "conv1":
            return 0
        if label == "bn1":
            return 1
        if label == "relu":
            return 2
        if label == "avgpool":
            return 7
        if label == "flatten":
            return 8
        if label == "fc":
            return 9
    elif type(label) == tuple and len(label) == 2 and type(label[0]) == int and type(label[1]) == int:
        blockset, block = label
        assert(1 <= blockset and blockset <= 4)
        assert(block == 0)
        blockset_idx = blockset - 1
        return 3 + blockset_idx
    else:
        raise NotImplementedError

# NOTE I added this for cross-compatibility between Yamini's code and my own
def supported_output_label(label):
    if type(label) == str:
        return label in ["conv1", "bn1", "relu", "avgpool", "flatten"]
    elif type(label) == tuple and len(label) == 2 and type(label[0]) == int and type(label[1]) == int:
        blockset, block = label
        assert(block == 0)
        return 1 <= blockset and blockset <= 4 and block == 0
    else:
        raise NotImplementedError

# NOTE only supports resnet 10 (which is why we assert that the block is zero)
def next_label(label):
    if type(label) == str:
        if label == "conv1":
            return "bn1"
        if label == "bn1":
            return "relu"
        if label == "relu":
            return (1, 0)
        if label == "avgpool":
            return "flatten"
        if label == "flatten":
            return "fc"
        if label == "fc":
            raise NotImplementedError
    elif type(label) == tuple and len(label) == 2 and type(label[0]) == int and type(label[1]) == int:
        blockset, block = label
        assert(1 <= blockset and blockset <= 4)
        assert(block == 0)
        if blockset == 4:
            return "relu"
        else:
            return (blockset + 1, 0)
    else:
        raise NotImplementedError