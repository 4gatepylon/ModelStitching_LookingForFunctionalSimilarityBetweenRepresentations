import yaml

# So that we can do the experiment
from net import StitchMode

DEFAULT_FILENAME = "experiment.yaml"

# Experiment flow is simple
# You:
# (1) init your networks (pass in a lambda)
# (2) train your networks (pass in a lambda)
# (3) create your stitches (pass in a lambda)
# (4) train your stitches (pass in a lambda)
# (5) evaluate how the stitches went (pass in a lambda)
# The network initializers and the training functions and the evaluator functions are expected to store all
# the state they need in themselves. The only thing the Experiment gives them is the kwargs as well as the
# starter and ender and stitch_mode parameters when initializing networks.

# The train and evaluator functions are expected to take in either a map(model_name : model) 
# for train or a map(stitch_name: list[starting model, ending model, stitch_model]) in the case of 
# a stitch evaluation.
class Experiment:
    def __init__(self, filename=DEFAULT_FILENAME):
        self.filename = filename
        self.kwargs = None
        self.stitches = None
        self.source_nets = None
        self.stitch_nets = None
    # Load the yaml into maps of strings and integers
    def load_yaml(self):
        with open(self.filename) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            sources = data["sources"]
            stitches = data["stitches"]
            self.kwargs = {model_name : params["kwargs"] for model_name, params in sources.items()}
            self.stitches = {stitch_name : stitch_params for stitch_name, stitch_params in stitches.items()}
    # Given a network initializer function, create the networks
    # (we may want to add an easier way to deal with different seeds)
    def init_nets(self, net_init):
        self.source_nets = {model_name: net_init(**kwargs) for model_name, kwargs in self.kwargs.items()}
    # Use a training function they provide
    def train_nets(self, train_function):
        train_function(self.source_nets)
    def init_stitch_nets(self, stitch_net_init):
        # Stitch net ender must at least take in (starter, ender, stitch_mode)
        self.stitch_nets = {
            stitch_model_name : stitch_net_init(
                self.source_nets[stitch["starter"]],
                self.source_nets[stitch["ender"]],
                StitchMode(stitch["stitch_mode"])) for stitch_model_name, stitch in self.stitches.items()}
    def train_stitch_nets(self, stitch_train_function):
        stitch_train_function(self.stitch_nets)
    def evaluate_stitches(self, evaluate_function):
        # The evaluate function should take in tuples of (starter, ender, stitch) models
        evaluate_function({
            stitch_name: [
                self.source_nets[stitch["starter"]],
                self.source_nets[stitch["ender"]],
                self.stitch_nets[stitch_name]
            ] for stitch_name, stitch in self.stitches.items() })
