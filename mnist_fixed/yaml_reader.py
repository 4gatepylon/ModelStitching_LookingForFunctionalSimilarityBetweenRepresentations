import torch
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
# a stitch evaluation. Experiment also lets you pass in a seed in the yaml file so you don't need
# to worry about reproducibility. You may not use the same name for a stitch as for a source. Note
# that there is no random number seeding for the optimizer nor for the data. If you don't give
# a seed then it picks a random one using the default generator.

# TODO we may want to be able to control randomness of the data as well as of the optimizer.

# Yaml format is
# sources:
#   <source name>:
#     seed: <seed val for the layers of the network>
#     kwargs:
#       <kwargs map>
#   ...
# stitches:
#   <sitch name>:
#     starter: <starter name in sources>
#     ender: <ender name in sources>
#     stitch_mode: <stitch mode as a string>
#     seed: <seed val for the stitch layer>
class Experiment:
    def __init__(self, filename=DEFAULT_FILENAME):
        self.filename = filename
        self.kwargs = None
        self.seeds = None
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

            self.seeds = {}
            for items in (sources.items(), stitches.items()):
                for model_name, params in items:
                    if "seed" in params:
                        assert(not model_name in self.seeds)
                        self.seeds[model_name] = params["seed"]
    # Given a network initializer function, create the networks
    # (we may want to add an easier way to deal with different seeds)
    def init_nets(self, net_init):
        self.source_nets = {}
        for model_name, kwargs in self.kwargs.items():
            seed = self.seeds[model_name] if model_name in self.seeds else torch.seed()
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            self.source_nets[model_name] = net_init(**kwargs)
    
    # Use a training function they provide
    def train_nets(self, train_function):
        train_function(self.source_nets)
    
    def init_stitch_nets(self, stitch_net_init):
        # Stitch net ender must at least take in (starter, ender, stitch_mode)
        self.stitch_nets = {}
        for stitch_model_name, stitch in self.stitches.items():
            seed = self.seeds[model_name] if model_name in self.seeds else torch.seed()
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            self.stitch_nets[stitch_model_name] = stitch_net_init(
                self.source_nets[stitch["starter"]],
                self.source_nets[stitch["ender"]],
                StitchMode(stitch["stitch_mode"]))
        
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
