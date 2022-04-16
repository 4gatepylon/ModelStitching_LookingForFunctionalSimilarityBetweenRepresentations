from torch.utils.data import Dataset, DataLoader

# Not clear if this is going to be used but at a high level this is something we will want


class RepDataset(Dataset):
    """ The representation dataset is the dataset of intermediate outputs. You choose it by specifying
    a layer for whom the outputs are the target of the dataset."""

    def __init__(self):
        raise NotImplementedError


class RepDataloader(DataLoader):
    def __init__(self):
        raise NotImplementedError
