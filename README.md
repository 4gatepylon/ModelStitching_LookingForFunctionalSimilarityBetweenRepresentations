# Plato
Contrastive Learning SuperUROP Fall 2021 - Spring 2022

Right now most of the code is in `mnist_fixed` which simply does stitching for the MNIST tutorial network from pytorch.

## Game Plan and Status
1. Show that stitching works on MNIST with CNNs `In Progress`: `Awaiting Sanity Checks, Then Testing`
2. Show that it works on MNIST with CNNs/Resnets/FCs of different sizes (i.e. low stitching penalty) `In Progress` : `Awaiting Feature Requests`

Selling point here is that this is primarily an empirial paper. We are going to see if modularity exists or not. If it does not that's totally fine. It's also possible that the experiment has problems. We will theorize about why what happened, happened after we are done.

To get rid of something named `-*` do `rm -rf -- name`.