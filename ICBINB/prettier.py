import numpy as np
import matplotlib.pyplot as plt

from typing import List

# Shift left
# Comparing outputs:
# 0, 1, 2, 3, 4 -> ID
TBL_r1111_r1111 = np.array(
    [
        [0.91, 0.54, 0.39, 0.34, 0.32],
        [0.89, 0.91, 0.8, 0.56, 0.58],
        [0.89, 0.9, 0.88, 0.79, 0.65],
        [0.87, 0.89, 0.92, 0.91, 0.85],
        [0.92, 0.92, 0.92, 0.92, 0.93]
    ]
)

# 0, 1, 2, 3, 4, 5 -> 0, 1, 2, 3, 4, 5, 6
TBL_r1112_r1122 = np.array(
    [
        [0.92, 0.66, 0.38, 0.47, 0.34, 0.33, 0.39],
        [0.92, 0.93, 0.89, 0.72, 0.54, 0.55, 0.57],
        [0.9, 0.91, 0.92, 0.83, 0.74, 0.66, 0.62],
        [0.87, 0.88, 0.9, 0.9, 0.89, 0.83, 0.76],
        [0.91, 0.92, 0.93, 0.92, 0.93, 0.93, 0.92],
        [0.92, 0.92, 0.93, 0.93, 0.94, 0.94, 0.93]
    ]
)

# 0, 1, 2, 3, 4 -> 0, 1, 2, 3, 4, 5
TBL_r1111rand_r1112 = np.array(
    [
        [0.85, 0.5, 0.32, 0.34, 33, 0.3],
        [0.82, 0.7, 0.54, 0.45, 0.41, 0.42],
        [0.71, 0.65, 0.44, 0.44, 0.43, 0.43],
        [0.48, 0.41, 0.41, 0.4, 0.41, 0.47],
        [0.32, 0.29, 0.37, 0.39, 0.37, 0.42]
    ]
)

# 0, 1, 2, 3, 4, 5 -> 0, 1, 2, 3, 4
# This one is kind of bs
TBL_r1112_r1111rand = np.array(
    [
        [0.33, 0.29, 0.3, 0.34, 0.36],
        [0.35, 0.35, 0.42, 0.4, 0.49],
        [0.35, 0.39, 0.4, 0.3, 0.28],
        [0.44, 0.42, 0.4, 0.41, 0.38],
        [0.57, 0.79, 0.9, 0.92, 0.92],
        [0.73, 0.87, 0.93, 0.93, 0.94]
    ]
)

# This one is for a little bit of extrapolation
TBL_r18_r34 = np.array(
    [
        [95, 94, 91, 84, 64, 37, 50, 29, 32, 38, 39, 38, 36, 28, 32, 28, 31],
        [95, 95, 94, 93, 86, 49, 66, 71, 35, 42, 34, 34, 23, 37, 39, 38, 44],
        [94, 94, 94, 94, 92, 92, 88, 87, 52, 51, 46, 37, 42, 45, 47, 43, 51],
        [92, 93, 93, 95, 93, 93, 93, 92, 56, 77, 72, 72, 62, 46, 64, 52, 60],
        [92, 92, 93, 94, 93, 93, 93, 94, 86, 89, 80, 74, 67, 73, 73, 64, 66],
        [82, 88, 88, 92, 71, 79, 80, 94, 85, 77, 81, 77, 69, 85, 84, 75, 74],
        [86, 88, 88, 91, 92, 93, 92, 94, 91, 91, 90, 93, 93, 94, 94, 92, 89],
        [93, 94, 94, 94, 94, 94, 95, 95, 94, 94, 95, 95, 95, 95, 95, 95, 95],
        [95, 95, 94, 94, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95]
    ]
) / 100.0

TBL_r1111_r1111_naive = np.array(
    [
        [0.25, 0.1, 0.09, 0.1, 0.12],
        [0.11, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.1, 0.1],
        [0.11, 0.09, 0.1, 0.09, 0.1]
    ]
)

OUTPUT_FILENAME = 'test.png'
def matrix_heatmap(mats: List[List[np.array]], names: List[List[str]], output_file_name: str):
    dims = [[mat.shape for mat in row] for row in mats]
    ticks = [[(np.arange(dim[0]), np.arange(dim[1])) for dim in row] for row in dims]

    # Create subplots grid with 1 row and 2 columns
    # You must pass in a rectangular list of lists
    fig, axs = plt.subplots(len(mats), len(mats[0]))
    if type(axs) != list:
        axs = [[axs]]
    for _, (ax_row, mat_row, ticks_row, name_row) in enumerate(zip(axs, mats, ticks, names)):
        for _, (ax, mat, ticks, name) in enumerate(zip(ax_row, mat_row, ticks_row, name_row)):
            yticks, xticks = ticks
            # These are empirical... sort of sus but ok
            ax.imshow(mat, vmin=0.2, vmax=0.95)
            ax.set_yticks(yticks)
            ax.set_xticks(xticks)
            ax.set_yticklabels(yticks)
            ax.set_xticklabels(xticks)
            plt.setp(ax.get_xticklabels(), rotation=45,
                    ha="right", rotation_mode="anchor")

            # This is inserting the text into the boxes so that we can compare easily
            for i in range(len(yticks)):
                for j in range(len(xticks)):
                    ax.text(j, i, mat[i, j], ha="center", va="center", color="w", fontsize=7)

            ax.set_title(f"{name}")
    fig.tight_layout(h_pad=2, w_pad=0)
    plt.savefig(output_file_name, dpi=900)
    plt.clf()

if __name__ == '__main__':
    # Comment out what you really want to use
    # heatmaps = [[TBL_r1111_r1111, TBL_r1112_r1122], [TBL_r1111rand_r1112, TBL_r1112_r1111rand]]
    # names = [['$R_{1111}$ to $R_{1111}$', '$R_{1112}$ to $R_{1122}$'], ['Random $R_{1111}$ to $R_{1112}$', '$R_{1112}$ to Random $R_{1111}$']]
    # heatmaps = [[TBL_r18_r34]]
    # names = [['Resnet18 to Resnet34']]
    heatmaps = [[TBL_r1111_r1111_naive]]
    names = [['$R_{1111}$ to $R_{1111}$ using Self-Similarity Stitch']]
    matrix_heatmap(heatmaps, names, OUTPUT_FILENAME)
