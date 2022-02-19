import math

DATA_FOLDER = "../data"

# Calculate the width and height of a convolutional layer
# from the input width and height as well as the (assumed symmetric)
# width and stride. Assume no padding. Same for pools.
def conv_output_dims(height, width, kernel, stride):
    # NOTE this is currently unused
    # We take the floor since there is no padding
    output_height = math.floor((height - kernel + 1) / stride)
    output_width = math.floor((width - kernel + 1) / stride)
    return output_height, output_width
def pool_output_dims(height, width, kernel, stride):
    # NOTE this is currently unused
    if stride != kernel:
        raise NotImplementedError("Tried to use pooling with stride {} != kernel {}".format(stride, kernel))
    # It's the same due to lack of padding
    return math.floor(height / stride), math.floor(width / stride)

def kwargs(**kwargs):
    return kwargs

def ensure_in_dict(dictionary, *args):
    for arg in args:
        if not arg in dictionary:
            raise ValueError("Dictionary {} missing {}".format(dictionary, arg))
def ensure_not_in_dict(dictionary, *args):
    for arg in args:
        if arg in dictionary:
            raise ValueError("Dictionary {} should not have {}".format(dictionary, arg))

def ensure_not_none(*args):
    for arg in args:
        if arg is None:
            raise ValueError("Argument was None")
def ensure_none(*args):
    for arg in args:
        if not arg is None:
            raise ValueError("Argument was not None")

def stitching_penalty(acc1, acc2, st_acc):
    # We look at the improvement
    imp1 = st_acc - acc1
    imp2 = st_acc - acc2

    # Return the minimum improvment
    return min(imp1, imp2)