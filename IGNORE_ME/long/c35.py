NET_3_5 = [
    # First convolution
    {
        # 0
        "layer_type": "Conv2d",
        "kernel_size": 3,
        "stride": 1,
        "output_depth": 64,
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
        "output_depth": 128,
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
        "output_depth": 256,
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
    # First FC
    {
        "layer_type": "Linear",
        "output_width": 512,
    },
    {
        "layer_type": "ReLU",
    },
    
    # Second FC
    # Linear: 10
    # ReLU: 11
    {
        "layer_type": "Linear",
        "output_width": 256,
    },
    {
        "layer_type": "ReLU",
    },

    # Third FC
    # Linear: 12
    # ReLU: 13
    {
        "layer_type": "Linear",
        "output_width": 128,
    },
    {
        "layer_type": "ReLU",
    },

    # Fourth FC
    # Linear: 14
    # ReLU: 15
    {
        "layer_type": "Linear",
        "output_width": 64,
    },
    {
        "layer_type": "ReLU",
    },

    # Last FC + Classification
    # Linear: 16
    # LogSoftmax: 17
    {
        "layer_type": "Linear",
        "output_width": 10,
    },
    {
        "layer_type": "LogSoftmax",
    },
]

# This specifies (using zero indexxing) the layer that
# would take the desired output as an input (i.e. for 2 it's 1) and then the layer in the second
# neural network to actually take the input through the stitch.
NET_3_5_TO_NET_3_5_STITCHES_INTO_FMT = {
    # Convs
    2: [2, 4, 6],
    4: [2, 4, 6],
    6: [2, 4, 6],
    # FCs
    10: [10, 12, 14, 16],
    12: [10, 12, 14, 16],
    14: [10, 12, 14, 16],
    16: [10, 12, 14, 16],
}