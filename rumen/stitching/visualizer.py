import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import os

import unittest

class Visualizer(object):
    """ Class to encapsulate our visualization methods."""

    @staticmethod
    def matrix_heatmap(input_file_name: str, output_file_name: str):
        mat = torch.load(input_file_name)
        assert type(mat) == torch.Tensor or type(mat) == np.ndarray
        if type(mat) == torch.Tensor:
            mat = mat.numpy()
        assert len(mat.shape) == 2

        mat_height, mat_width = mat.shape

        mat = np.round(mat, decimals=2)
        yticks, xticks = np.arange(mat_height), np.arange(mat_width)

        fig, ax = plt.subplots()
        im = ax.imshow(mat)
        ax.set_yticks(yticks)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
        ax.set_yticklabels(yticks)
        plt.setp(ax.get_xticklabels(), rotation=45,
                 ha="right", rotation_mode="anchor")

        # This is inserting the text into the boxes so that we can compare easily
        for i in range(len(yticks)):
            for j in range(len(xticks)):
                text = ax.text(j, i, mat[i, j],
                               ha="center", va="center", color="w")

        title = input_file_name.split("/")[-1].split('.')[0]
        ax.set_title(f"{title}")
        fig.tight_layout()
        plt.savefig(output_file_name)
        plt.clf()

    @staticmethod
    def images(model: nn.Module, test_loader: DataLoader, output_folder: str):
        raise NotImplementedError

class VisualizerTester(unittest.TestCase):
    def test_matrix_heatmap(self):
        # Ideally this would be set up using a testing folder
        # and would involve a fixture to do preparation beforehand
        # but this is probably good enough for now.
        test_file = "__tmp.pt"
        test_output = "__tmp.png"
        self.assertFalse(os.path.exists(test_file))
        self.assertFalse(os.path.exists(test_output))

        tensor = torch.Tensor([[0.25,0.5,0.25], [0.25,0.5,0.25], [0.25,0.5,0.25]])
        torch.save(tensor, test_file)
        Visualizer.matrix_heatmap(test_file, test_output)

        # There is no good way to test other than visually
        # NOTE in the future we may want to have a more robust
        # testing infrastructure!

        self.assertTrue(os.path.exists(test_file))
        self.assertTrue(os.path.exists(test_output))
        os.remove(test_file)
        os.remove(test_output)


if __name__ == "__main__":
    unittest.main(verbosity=2)