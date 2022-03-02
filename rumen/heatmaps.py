import numpy as np
import torch
import matplotlib.pyplot as plt

def matrix_heatmap(input_file_name=None):
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
    ax.set_title(f"Self Similarity of {title}")
    fig.tight_layout()

    output_file_name = (input_file_name.split("/")[-1]).split(".")[0] + ".png"
    plt.savefig(output_file_name)
    plt.clf()

if __name__ == "__main__":
    for input_file_name in [
        "resnet18_resnet18_sims.pt",
        "resnet18_rand_resnet18_sims.pt",
        "resnet18_resnet34_sims.pt",
        "resnet18_rand_resnet34_sims.pt"]:
        matrix_heatmap(input_file_name=input_file_name)