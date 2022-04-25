import torchvision
import torchvision.transforms.functional as TF
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.cuda.amp import autocast

import cv2
import os
from typing import List, Tuple, Dict, Callable, Any

from layer_label import LayerLabel

FFCV_CIFAR_MEAN = [125.307, 122.961, 113.8575]
FFCV_CIFAR_STD = [51.5865, 50.847, 51.255]
FFCV_INV_CIFAR_MEAN = [-ans for ans in FFCV_CIFAR_MEAN]
FFCV_INV_CIFAR_STD = [1.0/ans for ans in FFCV_CIFAR_STD]
FFCV_NORMALIZE_TRANSFORM = transforms.Normalize(
    FFCV_CIFAR_MEAN, FFCV_CIFAR_STD)
FFCV_INV_NORMALIZE_TRANSFORM = transforms.Compose([
    transforms.Normalize(mean=[0, 0, 0], std=FFCV_INV_CIFAR_STD),
    transforms.Normalize(mean=FFCV_INV_CIFAR_MEAN, std=[1, 1, 1])
])

NO_FFCV_CIFAR_MEAN = [0.1307, ]
NO_FFCV_CIFAR_STD = [0.3081, ]
NO_FFCV_INV_CIFAR_MEAN = [-ans for ans in NO_FFCV_CIFAR_MEAN]
NO_FFCV_INV_CIFAR_STD = [1.0/ans for ans in NO_FFCV_CIFAR_STD]
NO_FFCV_NORMALIZE_TRANSFORM = transforms.Normalize(
    NO_FFCV_CIFAR_MEAN, NO_FFCV_CIFAR_STD)
NO_FFCV_INV_NORMALIZE_TRANSFORM = transforms.Compose([
    transforms.Normalize(mean=[0, 0, 0], std=NO_FFCV_INV_CIFAR_STD),
    transforms.Normalize(mean=NO_FFCV_INV_CIFAR_MEAN, std=[1, ])
])


def pclone(model) -> List[torch.Tensor]:
    return [p.data.detach().clone() for p in model.parameters()]


def listeq(l1, l2) -> bool:
    return min((torch.eq(a, b).int().min().item() for a, b in zip(l1, l2))) == 1


def mapeq(n2l1: Dict[str, List[nn.Module]], n2l2: Dict[str, List[nn.Module]]) -> bool:
    assert n2l1.keys() == n2l2.keys()
    # `all` is a function which returns True iff all elements of the iterable are True.
    return all((listeq(n2l1[n], n2l2[n]) for n in n2l1.keys()))


def mapneq(n2l1: Dict[str, List[nn.Module]], n2l2: Dict[str, List[nn.Module]]):
    # Slightly stronger than "not mapeq"
    # check that they are ALL FALSE
    assert n2l1.keys() == n2l2.keys()
    return all(
        map(
            lambda x: not x,
            (listeq(n2l1[n], n2l2[n]) for n in n2l1.keys())
        ),
    )

def flattened_table(l: List[List[Any]]) -> List[Any]:
    vec = []
    for row in l:
        vec += row
    return vec

# def tensor_normalized2rgb(x: torch.Tensor):
#     f = x.float()
#     y = INV_NORMALIZE_TRANSFORM(f)
#     return y


# def tensor_normalized2pil_img(x: torch.Tensor):
#     rgb_tensor = tensor_normalized2rgb(x)
#     return TF.to_pil_image(rgb_tensor)


# def tensor_normalized2nd_array(x: torch.Tensor):
#     rgb_tensor = tensor_normalized2rgb(x)
#     return rgb_tensor.numpy()


# This will also be helpful!
# https://pytorch.org/vision/master/auto_examples/plot_visualization_utils.html
# https://medium.com/analytics-vidhya/read-image-using-cv2-imread-opencv-python-idiot-developer-4b7401e76c20
# https://www.tutorialkart.com/opencv/python/opencv-python-save-image-example/
# IMG_FILE = "human_interpretable_img.png"


# def show_tensor_normalized(x: torch.Tensor, file=IMG_FILE):
#     nd_array = tensor_normalized2nd_array(x)
#     cv2.imwrite(file, nd_array)
#     pass


# def get_n_inputs(n, loader):
#     k = 0
#     for x, _ in loader:
#         if k > n:
#             break
#         batch_size, _, _, _ = x.size()
#         # print(f"batch size {batch_size}")
#         for i in range(min(batch_size, n - k)):
#             # Output as a 4D tensor so that the network can take this as input
#             y = x[i, :, :, :].flatten(end_dim=0).unflatten(0, (1, -1))
#             # print(y.size())
#             yield y
#         k += batch_size

# def save_random_image_pairs(st, sender, snd_label, num_pairs, foldername_images, train_loader):
#     original_tensors = list(get_n_inputs(num_pairs, train_loader))
#     for i in range(num_pairs):
#         # Pick the filenames
#         original_filename = os.path.join(
#             foldername_images, f"original_{i}.png")
#         generated_filename = os.path.join(
#             foldername_images, f"generated_{i}.png")

#         with autocast():
#             original_tensor = original_tensors[i]
#             print(f"\tog tensor shape is {original_tensor.size()}")
#             generated_tensor_pre = sender(
#                 original_tensor, vent=snd_label, into=False)
#             generated_tensor = st(generated_tensor_pre)

#         # Save the images
#         original_tensor_flat = original_tensor.flatten(end_dim=0)
#         generated_tensor_flat = generated_tensor.flatten(end_dim=0)
#         original_np = original_tensor_flat.cpu().numpy()
#         generated_np = generated_tensor_flat.cpu().numpy()
#         cv2.imwrite(original_np, original_filename)
#         cv2.imwrite(generated_np, generated_filename)
#         # TDOO
