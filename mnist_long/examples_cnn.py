# A classification and FC head used by both NET_3_2 and NET_10_2
CLASSIFIER_2 = [
    # Flatten
    # n

    # First FC
    {
        # n + 1
        "layer_type": "Linear",
        "output_width": 128,
    },
    {
        # n + 2
        "layer_type": "ReLU",
    },

    # Second FC (for classification)
    {
        # n + 3
        "layer_type": "Linear",
        "output_width": 10,
    },
    {
        # n + 6
        "layer_type": "LogSoftmax",
    },
]

# An example network with three convolutional layers followed by one maxpool
# followed by two linear layers (using the CLASSIFIER_2 from above)
NET_3_2 = [
    # First convolution
    {
        # 0
        "layer_type": "Conv2d",
        "kernel_size": 3,
        "stride": 1,
        "output_depth": 32,
    },
    {
        # 1
        "layer_type": "ReLU",
    },

    # Second convolution
    {
        # 2
        "layer_type": "Conv2d",
        "kernel_size": 3,
        "stride": 1,
        "output_depth": 64,
    },
    {
        # 3
        "layer_type": "ReLU",
    },

    # Third convolution
    {
        # 4
        "layer_type": "Conv2d",
        "kernel_size": 3,
        "stride": 1,
        "output_depth": 128,
    },
    {
        # 5
        "layer_type": "ReLU",
    },

    # MaxPool
    {
        # 6
        "layer_type": "MaxPool2d",
        "kernel_size": 2,
        "stride": 2,
    },
    
    # Flatten: 7

    # First FC
    # Linear: 8
    # ReLU: 9

    # Second FC + Classification
    # Linear: 10
    # LogSoftmax: 11

] + CLASSIFIER_2

# An example network with 4 convolutional layers followed by a maxpool
# followed by two linear layers (second one which is for classification)
NET_4_2 = [
    # First convolution
    {
        # 0
        "layer_type": "Conv2d",
        "kernel_size": 3,
        "stride": 1,
        "output_depth": 32,
    },
    {
        # 1
        "layer_type": "ReLU",
    },
    # Second convolution
    {
        # 2
        "layer_type": "Conv2d",
        "kernel_size": 3,
        "stride": 1,
        "output_depth": 64,
    },
    {
        # 3
        "layer_type": "ReLU",
    },
    # Third convolution
    {
        # 4
        "layer_type": "Conv2d",
        "kernel_size": 3,
        "stride": 1,
        "output_depth": 128,
    },
    {
        # 5
        "layer_type": "ReLU",
    },
    # Fourth convolution
    {
        # 6
        "layer_type": "Conv2d",
        "kernel_size": 3,
        "stride": 1,
        "output_depth": 256,
    },
    {
        # 7
        "layer_type": "ReLU",
    },

    # MaxPool
    {
        # 8
        "layer_type": "MaxPool2d",
        "kernel_size": 2,
        "stride": 2,
    },

    # Flatten: 9

    # First FC
    # Linear: 10
    # ReLU: 11

    # Second FC + Classification
    # Linear: 12
    # LogSoftmax: 13
    
] + CLASSIFIER_2

# An example network with 10 convolutional layers followed by a maxpool
# followed by two linear layers and a classifier

NET_10_2 = [
    # First convolution
    {
        # 0
        "layer_type": "Conv2d",
        "kernel_size": 3,
        "stride": 1,
        "output_depth": 32,
    },
    {
        # 1
        "layer_type": "ReLU",
    },
    # Second convolution
    {
        # 2
        "layer_type": "Conv2d",
        "kernel_size": 3,
        "stride": 1,
        "output_depth": 64,
    },
    {
        # 3
        "layer_type": "ReLU",
    },
    # Third convolution
    {
        # 4
        "layer_type": "Conv2d",
        "kernel_size": 3,
        "stride": 1,
        "output_depth": 128,
    },
    {
        # 5
        "layer_type": "ReLU",
    },
    # Fourth convolution
    {
        # 6
        "layer_type": "Conv2d",
        "kernel_size": 3,
        "stride": 1,
        "output_depth": 256,
    },
    {
        # 7
        "layer_type": "ReLU",
    },
    # Fifth convolution
    {
        # 8
        "layer_type": "Conv2d",
        "kernel_size": 3,
        "stride": 1,
        "output_depth": 256,
    },
    {
        # 9
        "layer_type": "ReLU",
    },
    # Sixth convolution
    {
        # 10
        "layer_type": "Conv2d",
        "kernel_size": 3,
        "stride": 1,
        "output_depth": 256,
    },
    {
        # 11
        "layer_type": "ReLU",
    },
    # Seventh convolution
    {
        # 12
        "layer_type": "Conv2d",
        "kernel_size": 3,
        "stride": 1,
        "output_depth": 128,
    },
    {
        # 13
        "layer_type": "ReLU",
    },
    # Eighth convolution
    {
        # 14
        "layer_type": "Conv2d",
        "kernel_size": 3,
        "stride": 1,
        "output_depth": 64,
    },
    {
        # 15
        "layer_type": "ReLU",
    },
    # Ninth convolution
    {
        # 16
        "layer_type": "Conv2d",
        "kernel_size": 3,
        "stride": 1,
        "output_depth": 64,
    },
    {
        # 17
        "layer_type": "ReLU",
    },
    # Tenth convolution
    {
        # 18
        "layer_type": "Conv2d",
        "kernel_size": 3,
        "stride": 1,
        "output_depth": 64,
    },
    {
        # 19
        "layer_type": "ReLU",
    },


    # MaxPool
    {
        # 20
        "layer_type": "MaxPool2d",
        "kernel_size": 2,
        "stride": 2,
    },

    # Flatten: 21

    # First FC
    # Linear: 22
    # ReLU: 23

    # Second FC + Classification
    # Linear: 24
    # LogSoftmax: 25

] + CLASSIFIER_2

# Intra-network stitches try to see if two versions of the same
# network are similar (expect the case to be true for 1:1 mapping
# of the layers)
NET_3_2_TO_NET_3_2_STITCHES = {
    # Convs
    2: [2, 4, 6],
    4: [2, 4, 6],
    6: [2, 4, 6],
    # FCs
    10: [10, 11],
    11: [10, 11],
}
NET_4_2_TO_NET_4_2_STITCHES = {
    # Convs
    2: [2, 4, 6, 8],
    4: [2, 4, 6, 8],
    6: [2, 4, 6, 8],
    8: [2, 4, 6, 8],
    # FCs
    12: [12, 13],
    13: [12, 13],
}
NET_10_2_TO_NET_10_2_STITCHES = {
    # Convs
    2: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    4: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    6: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    8: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    10: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    12: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    14: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    16: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    18: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    20: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    # FCs
    24: [24, 25],
    25: [24, 25],
}

# Inter-network stitches try to see if different networks are similar
NET_3_2_TO_NET_4_2_STITCHES = {
    # Convolutional layers after the ReLU into 
    
    # Compare the outputs of different layers. This specifies (using zero indexxing) the layer that
    # would take the desired output as an input (i.e. for 2 it's 1) and then the layer in the second
    # neural network to actually take the input through the stitch.
    2: [2, 4, 6, 8],
    4: [2, 4, 6, 8],
    6: [2, 4, 6, 8],

    # Linear layers (avoid the flatten since it's so big)
    10: [12, 13],
    11: [12, 13],
}

NET_3_2_TO_NET_10_2_STITCHES = {
    # Similar to case from 3_2 to 4_2
    2: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    4: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    6: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],

    # Linear layers
    10: [24, 25],
    11: [24, 25],
}