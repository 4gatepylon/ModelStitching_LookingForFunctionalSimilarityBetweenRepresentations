import os
import cv2
import torch
import torchvision
import numpy as np
from torchvision import datasets

IMAGES_FOLDER = "../../images/"
NO_FFCV_FOLDER = "../../data_no_ffcv/"
NO_FFCV_CIFAR_MEAN = [0.1307, ]
NO_FFCV_CIFAR_STD = [0.3081, ]
INV_NO_FFCV_CIFAR_MEAN = [-ans for ans in NO_FFCV_CIFAR_MEAN]
INV_NO_FFCV_CIFAR_STD = [1.0/ans for ans in NO_FFCV_CIFAR_STD]

transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(
        #     NO_FFCV_CIFAR_MEAN, NO_FFCV_CIFAR_STD)
    ])

inv_transform = torchvision.transforms.Compose([
    # torchvision.transforms.Normalize(mean=[0,], std=INV_NO_FFCV_CIFAR_STD),
    # torchvision.transforms.Normalize(mean=INV_NO_FFCV_CIFAR_MEAN, std=[1,]),
    # torchvision.transforms.Normalize(mean=[0,], std=[1.0/255.0,])
    torchvision.transforms.ToPILImage(mode="RGB"),
])

if __name__ == "__main__":
    dataset_original = datasets.CIFAR10(
        os.path.join(NO_FFCV_FOLDER, "no_transform/"),
        train=True,
        download=True, 
        transform=torchvision.transforms.PILToTensor(),
    )

    dataset = datasets.CIFAR10(
        NO_FFCV_FOLDER, train=True, download=True, transform=transform)

    image, _ = dataset[0]
    print(f"Image shape (transform) is {image.shape}")
    print("Image\n*************************\n")
    print(image)

    original_image = inv_transform(image)

    # print(f"Image shape (inv transform) is {original_image.shape}")
    print("Original Image\n*************************\n")
    # to pil image returns (H x W x D) ~ 32 x 32 x 3, but tensor is (D x H x W) ~ 3 x 32 x 32
    og = np.array(original_image).transpose(2, 0, 1)
    print(og)

    original_dataset_image, _ = dataset_original[0]
    print(f"Image shape (original) is {original_dataset_image.shape}")
    print("Original Dataset Image\n*************************\n")
    print(original_dataset_image)

    ogdi = np.array(original_dataset_image)
    print(f"og shape is {og.shape}")
    print(f"ogdi shape is {ogdi.shape}")
    assert og.shape == (3, 32, 32)
    assert ogdi.shape == (3, 32, 32)
    assert (og == ogdi).all(), f"OG Image was {og}\nOriginal dataset image is {ogdi}"
    print("OK!")

