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

NET_3_2_TO_NET_4_2_STITCHES = {
    # Convolutional layers after the ReLU into 
    
    # Compare the outputs of different layers. This specifies (using zero indexxing) the layer that
    # would take the desired output as an input (i.e. for 2 it's 1) and then the layer in the second
    # neural network to actually take the input through the stitch.
    2: [2, 4, 6, 8],
    4: [2, 4, 6, 8],

    # Linear layers (avoid the flatten since it's so big)
    10: [12, 13],
    11: [12, 13],
}