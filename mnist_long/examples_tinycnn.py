
# An example network with two convolutional layers and one classifier. The classifiers cannot be
# compared using stitching but the convolutional layers can be interoperably switched. They have
# the smallest valid kernel size and depth so that compute will be fast. We don't optimize stride
# for compute yet, because it's easier for width/height.
NET_CTINY = [
    # First convolution
    {
        # 0
        "layer_type": "Conv2d",
        "kernel_size": 1,
        "stride": 1,
        "output_depth": 1,
    },
    {
        # 1
        "layer_type": "ReLU",
    },

    {
        # 2
        "layer_type": "Conv2d",
        "kernel_size": 1,
        "stride": 1,
        "output_depth": 1,
    },
    {
        # 3
        "layer_type": "ReLU",
    },

    # MaxPool
    {
        # 4
        "layer_type": "MaxPool2d",
        "kernel_size": 2,
        "stride": 2,
    },

    # Flatten
    # 5
    
    # Classifier
    {
        # 6
        "layer_type": "Linear",
        "output_width": 10,
    },
    {
        # 7
        "layer_type": "LogSoftmax",
    },
]

# Remember that this is in reciever format (i.e. the sender of
# 2, which is the reicever, can send into 2 or 4, and the same for
# the sender of 4).
NET_CTINY_TO_NET_CTINY_STITCHES = {
    # Convs
    2: [2, 4],
    4: [2, 4],
}