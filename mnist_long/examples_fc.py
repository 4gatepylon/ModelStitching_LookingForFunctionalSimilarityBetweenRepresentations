# A set of fully connected network examples

CLASSIFIER = [
    {
        "layer_type": "Linear",
        "output_width": 10,
    },
    {
        "layer_type": "LogSoftmax",
    },
]

NET_3 = [
    # Flatten 0

    #  First FC
    {
        # 1
        "layer_type": "Linear",
        "output_width": 256,
    },
    {
        # 2
        "layer_type": "ReLU",
    },
    #  Second FC
    {
        # 3
        "layer_type": "Linear",
        "output_width": 128,
    },
    {
        # 4
        "layer_type": "ReLU",
    },

    # Third FC is in the classifier
    # Linear: 5
    # LogSoftmax 6
] + CLASSIFIER

NET_5 = [
    #  First FC
    {
        # 1
        "layer_type": "Linear",
        "output_width": 256,
    },
    {
        # 2
        "layer_type": "ReLU",
    },
    #  Second FC
    {
        # 3
        "layer_type": "Linear",
        "output_width": 128,
    },
    {
        # 4
        "layer_type": "ReLU",
    },
    #  Third FC
    {
        # 5
        "layer_type": "Linear",
        "output_width": 64,
    },
    {
        # 6
        "layer_type": "ReLU",
    },
    #  Fourth FC
    {
        # 7
        "layer_type": "Linear",
        "output_width": 64,
    },
    {
        # 8
        "layer_type": "ReLU",
    },

    # Fifth FC is in the classifier
    # Linear: 9
    # LogSoftmax 10
] + CLASSIFIER
NET_8 = [ 
    #  First FC
    {
        # 1
        "layer_type": "Linear",
        "output_width": 256,
    },
    {
        # 2
        "layer_type": "ReLU",
    },
    #  Second FC
    {
        # 3
        "layer_type": "Linear",
        "output_width": 128,
    },
    {
        # 4
        "layer_type": "ReLU",
    },
    #  Third FC
    {
        # 5
        "layer_type": "Linear",
        "output_width": 64,
    },
    {
        # 6
        "layer_type": "ReLU",
    },
    #  Fourth FC
    {
        # 7
        "layer_type": "Linear",
        "output_width": 64,
    },
    {
        # 8
        "layer_type": "ReLU",
    },
    #  Fifth FC
    {
        # 9
        "layer_type": "Linear",
        "output_width": 32,
    },
    {
        # 10
        "layer_type": "ReLU",
    },
    # Sixth FC
    {
        # 11
        "layer_type": "Linear",
        "output_width": 32,
    },
    {
        # 12
        "layer_type": "ReLU",
    },
    #  Seventh FC
    {
        # 13
        "layer_type": "Linear",
        "output_width": 16,
    },
    {
        # 14
        "layer_type": "ReLU",
    },

    # Eighth FC is in the classifier
    # Linear: 15
    # LogSoftmax 16
] + CLASSIFIER

NET_3_TO_NET_5_STITCHES = {
    3: [3, 5, 7, 9, 10],
    5: [3, 5, 7, 9, 10],
    6: [3, 5, 7, 9, 10],
}

NET_3_TO_NET_8_STITCHES = {
    3: [3, 5, 7, 9, 11, 13, 15, 16],
    5: [3, 5, 7, 9, 11, 13, 15, 16],
    6: [3, 5, 7, 9, 11, 13, 15, 16],
}