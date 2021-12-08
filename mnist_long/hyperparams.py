# Info per experiment
DEFAULT_TRAIN_BATCH_SIZE = 64
DEFAULT_TEST_BATCH_SIZE = 1000

# NOTE we may want to optimize different for stitch vs. not given this
DEFAULT_LR = 1.0
DEFAULT_LR_EXP_DROP = 0.7 # also called 'gamma'

# NOTE change this to test (by scaling down) or scale it up!
DEFAULT_EPOCHS_OG = 40
DEFAULT_EPOCHS_STITCH = 20

# Run 10 experiments
NUM_EXPERIMENTS = 10
NUM_STITCH_EXPERIMENTS = 10

DOWNLOAD_DATASET = False