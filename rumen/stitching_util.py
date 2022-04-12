import torch
import torch.nn as nn
import torch.nn.functional as F


class Stitched(nn.Module):
    def __init__(self, net1, net2, snd_label, rcv_label, stitch):
        super(Stitched, self).__init__()
        self.sender = net1
        self.reciever = net2
        self.snd_lbl = snd_label
        self.rcv_lbl = rcv_label
        self.stitch = stitch

    def forward(self, x):
        h = self.sender(x, vent=self.snd_lbl, into=False)
        h = self.stitch(h)
        h = self.reciever(h, vent=self.rcv_lbl, into=True)
        return h
