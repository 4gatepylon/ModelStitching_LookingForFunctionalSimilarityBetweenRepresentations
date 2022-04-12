import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from utils import combos

def matrix_heatmap(input_file_name=None, output_file_name=None):
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
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # This is inserting the text into the boxes so that we can compare easily
    for i in range(len(yticks)):
        for j in range(len(xticks)):
            text = ax.text(j, i, mat[i, j], ha="center", va="center", color="w")

    title = input_file_name.split("/")[-1].split('.')[0]
    ax.set_title(f"{title}")
    fig.tight_layout()
    plt.savefig(output_file_name)
    plt.clf()

# TODO averging as well plz if you see differing results
if __name__ == "__main__":
    # c_1 = combos(4, [1, 2])
    # c_2 = list(map(lambda combo: "".join(map(str, combo)), c_1))
    # c_3  = list(map(lambda combo: f"resnet_{combo}", c_2))
    # combinations = [(c1, c2) for c1 in c_3 for c2 in c_3]
    combinations = [("resnet_1111","resnet_1111")]
    foldersi = list(map(lambda tp: f"_{tp[0]}_{tp[1]}", combinations))
    for i in foldersi:
        print(f"Trying for `sims{i}`")
        folder = f"sims{i}" # created by experiment

        inner_folder = "heatmaps"
        files = os.listdir(folder)
        # files = list(filter(lambda f: "sims.pt" in f, files))
        def isfile(f):
            # print(f)
            # print(os.path.isfile(os.path.join(folder, f)))
            return os.path.isfile(os.path.join(folder, f))
        files = list(filter(isfile, files))
        if not os.path.exists(os.path.join(folder, inner_folder)):
            os.mkdir(os.path.join(folder, inner_folder))
        print(f"############ THERE ARE {len(files)} FILES")
        # print("\n".join(map(lambda f: "\t" + f, files)))
        for input_filename in files:
            output_filename = os.path.join(folder, inner_folder, input_filename.replace(".pt", ".png"))
            input_filename = os.path.join(folder, input_filename)

            print(f"\tGenerating {input_filename}->{output_filename}")
            matrix_heatmap(input_file_name=input_filename, output_file_name=output_filename)
        break # TODO