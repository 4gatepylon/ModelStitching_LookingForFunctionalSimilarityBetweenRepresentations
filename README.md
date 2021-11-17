# Plato
Contrastive Learning SuperUROP Fall 2021 - Spring 2022

Right now most of the code is in `mnist_fixed` which simply does stitching for the MNIST tutorial network from pytorch.

## Game Plan and Status
1. Show that stitching works on MNIST with CNNs `In Progress`: `Awaiting Testing w/ Tensorboard; sanity checks`
2. Show that it works on MNIST with CNNs of different sizes (i.e. low stitching penalty)
3. Ideally show that it works on MNIST with Resnets or FCs (or that it doesn't)

Selling point here is that this is primarily an empirial paper. We are going to see if modularity exists or not. If it does not that's totally fine. It's also possible that the experiment has problems. We will theorize about why what happened, happened after we are done.