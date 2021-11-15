import yaml

DEFAULT_FILENAME = "experiment.yaml"

# In the future we are going to want some additional flexibility!
def load_models_kwargs(filename=DEFAULT_FILENAME):
    with open(filename) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        return data