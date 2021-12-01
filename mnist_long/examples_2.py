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

    # Second FC
    {
        # n + 3
        "layer_type": "Linear",
        "output_width": 64,
    },
    {
        # n + 4
        "layer_type": "ReLU",
    },

    # Classification
    {
        # n + 5
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

    # Second FC
    # Linear: 10
    # ReLU: 11

    # Classification
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

    # Second FC
    # Linear: 24
    # ReLU: 25

    # Classification
    # Linear: 26
    # LogSoftmax: 27
] + CLASSIFIER_2

NET_3_2_TO_NET_10_2_STITCHES = {
    # Convolutional layers after the ReLU into 
    1 : [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    3: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    5: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],

    # Linear layers
    9: [24, 26],
    11: [24, 26],
}