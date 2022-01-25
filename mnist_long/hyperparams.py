# Switch to let you debug more easily!
DEBUG = True          # Run a fast test
MEMORY_CAREFUL = True  # Run only one of each test


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
NUM_EXPERIMENTS = 5
NUM_STITCH_EXPERIMENTS = 5

DOWNLOAD_DATASET = False

if MEMORY_CAREFUL:
    print("*** Memory Careful ***")
    NUM_EXPERIMENTS = 1
    NUM_STITCH_EXPERIMENTS = 1

if DEBUG:
    print("*** Debugging Mode ***")
    DEFAULT_EPOCHS_OG = 1
    # We use 2 because we want to grab a chunk of code inside the loop
    DEFAULT_EPOCHS_STITCH = 2
    NUM_EXPERIMENTS = 1
    NUM_STITCH_EXPERIMENTS = 1