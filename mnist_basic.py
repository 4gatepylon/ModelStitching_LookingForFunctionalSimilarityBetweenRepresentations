# A simple CNN network for MNIST that expects in MNIST digits in a single band channel
# and also expects a tuple layer_sizes that declares a sequence of layers such that the beginning
# if the sequence is comprised of convolutions and the end is comprised of FCs. Between every FC
# there is a ReLU and a dropout, but not for the last FC. We use zero padding. Maxpool after the last conv layer.
# We have dropout before the last fc layer.
class Net(nn.Module):
    # Expect a tuple of layer sizes in the format
    #   (conv_layer_sizes, fc_layer_sizes)
    # where conv_layer_sizes has format
    #   [(channels_in, channels_out, kernel_size, stride)]
    # and fc_layer_sizes has format
    #   [(in_dims, out_dims)]
    # where all those root values are ints
    def __init__(self, layer_sizes, img_dims=(28, 28), maxpool_size=2, dropout_p1=0.25, dropout_p2=0.5):
        super(Net, self).__init__()

        # We do not support RGB images yet
        assert(len(img_dims) == 2)
        img_width, img_height = img_dims

        # Ensure right format
        assert(not layer_sizes is None)
        assert(type(layer_sizes) == tuple)
        assert(len(layer_sizes) == 2)

        conv_layer_sizes, fc_layer_sizes = layer_sizes

        # Make sure the proper formats
        assert(not conv_layer_sizes is None)
        assert(not fc_layer_sizes is None)
        assert(type(conv_layer_sizes) == list or type(conv_layer_sizes) == tuple)
        assert(type(fc_layer_sizes) == list or type(fc_layer_sizes) == tuple)
        assert(len(fc_layer_sizes) > 0)
        assert(len(conv_layer_sizes) > 0)

        assert(max([len(conv) for conv in conv_layer_sizes]) == 4)
        assert(max([len(conv) for conv in conv_layer_sizes]) == 4)
        assert(sum([(1 if (sum([(1 if (type(conv[i]) == int) else 0) for i in range(len(conv))]) == 4) else 0) for conv in conv_layer_sizes]) == len(conv_layer_sizes))
        assert(max([len(fc) for fc in fc_layer_sizes]) == 2)
        assert(max([len(fc) for fc in fc_layer_sizes]) == 2)
        assert(sum([(1 if (type(fc[0]) == int and type(fc[1]) == int) else 0) for fc in fc_layer_sizes]) == len(fc_layer_sizes))

        # Make sure the sizes match
        assert(conv_layer_sizes[0][0] == 1)
        assert(max([abs(conv_layer_sizes[i][1] - conv_layer_sizes[i+1][0]) for i in range(0, len(conv_layer_sizes - 1))]) == 0)
        assert(fc_layer_sizes[0][0] == image_width * image_height * conv_layer_sizes[-1][1])
        assert(max([abs(fc_layer_sizes[i][out_dim] - fc_layer_sizes[i+1][in_dim]) for i in range(0, len(fc_layer_sizes) - 1)]) == 0)

        convs = [nn.Conv2d(in_chan, out_chan, kern_size, stride) for in_chan, out_chan, kern_size, stride in conv_layer_sizes]
        conv_with_relus = [val for val in [convs[i], nn.ReLU()] for i in range(len(conv_layer_sizes))]
        fcs = [nn.Linear(in_size, out_size) for in_size, out_size in fc_layer_sizes]
        fc_with_relus = [val for val in [fcs[i], nn.ReLU()] for i in range(len(fc_layer_sizes) - 1)]
        self.seq = nn.Sequantial(*(
            convs_with_relus +
            [nn.MaxPool2d(maxpool_size), nn.Dropout(dropout_p1), nn.Flatten(1)] +
            fcs_with_relus +
            [nn.Dropout(drouput_p2), nn.Linear(*fc_layer_sizes[-1])]
        ))
    
    def forward(self, x):
        # conv 1
        # relu
        # conv 2
        # relu
        # ...
        # max pool 2d
        # dropout
        # flatten
        # fc 1
        # relu
        # fc 2
        # relu
        # ...
        # dropout
        # fc final
        # softmax classifies
        x = self.seq(x)
        output = F.log_softmax(, dim=1)
        return output