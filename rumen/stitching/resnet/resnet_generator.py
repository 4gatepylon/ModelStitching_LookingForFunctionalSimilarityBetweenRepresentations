from typing import Type, Any, Callable, Union, List, Optional, Tuple

# Debugging quick hack
from resnet.resnet import BasicBlock, Bottleneck  # , Resnet
from resnet.resnet_og import ResNet as Resnet


class ResnetGenerator(object):
    """ An object to encapsulate function to create new resnets """

    @staticmethod
    def generate(
            arch: str,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            pretrained: bool,
            progress: bool,
            **kwargs: Any,
    ) -> Resnet:
        model = Resnet(block, layers, **kwargs)
        return model

    @staticmethod
    def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> Resnet:
        r"""Resnet-18 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        return ResnetGenerator.generate("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

    @staticmethod
    def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> Resnet:
        r"""Resnet-34 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        return ResnetGenerator.generate("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)

    @staticmethod
    def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> Resnet:
        r"""Resnet-50 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        return ResnetGenerator.generate("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

    @staticmethod
    def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> Resnet:
        r"""Resnet-101 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        return ResnetGenerator.generate("resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

    @staticmethod
    def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> Resnet:
        r"""Resnet-152 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        return ResnetGenerator.generate("resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)

    @staticmethod
    def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> Resnet:
        r"""ResNeXt-50 32x4d model from
        `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        kwargs["groups"] = 32
        kwargs["width_per_group"] = 4
        return ResnetGenerator.generate("resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

    @staticmethod
    def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> Resnet:
        r"""ResNeXt-101 32x8d model from
        `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        kwargs["groups"] = 32
        kwargs["width_per_group"] = 8
        return ResnetGenerator.generate("resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

    @staticmethod
    def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> Resnet:
        r"""Wide Resnet-50-2 model from
        `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
        The model is the same as Resnet except for the bottleneck number of channels
        which is twice larger in every block. The number of channels in outer 1x1
        convolutions is the same, e.g. last block in Resnet-50 has 2048-512-2048
        channels, and in Wide Resnet-50-2 has 2048-1024-2048.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        kwargs["width_per_group"] = 64 * 2
        return ResnetGenerator.generate("wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

    @staticmethod
    def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> Resnet:
        r"""Wide Resnet-101-2 model from
        `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
        The model is the same as Resnet except for the bottleneck number of channels
        which is twice larger in every block. The number of channels in outer 1x1
        convolutions is the same, e.g. last block in Resnet-50 has 2048-512-2048
        channels, and in Wide Resnet-50-2 has 2048-1024-2048.
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        kwargs["width_per_group"] = 64 * 2
        return ResnetGenerator.generate("wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

# model_constructors = {
#     "resnet18": resnet18,
#     "resnet34": resnet34,
#     "resnet50": resnet50,
#     "resnet101": resnet101,
#     "resnet152": resnet152,
#     "resnext50_32x4d": resnext50_32x4d,
#     "resnext101_32x8d": resnext101_32x8d,
#     "wide_resnet50_2": wide_resnet50_2,
#     "wide_resnet101_2": wide_resnet101_2,
# }


# def generate(name):
#     fname = f"{name}_imagenet.pt"
#     fname = os.path.join(RESNETS_FOLDER, fname)

#     model = model_constructors[name](pretrained=True, progress=True)

#     stateDict = torch.load(fname)
#     # NOTE this is a hack to avoid problems with the state dict
#     # This is to fix the problem that arises from the fact that conv1 for imagenet is
#     # a 7x7. Note how in the resnet.py code, self.inplanes is 64 at the start. This
#     # works because the state dict is literally just matrices, and they are added
#     # as nn.Parameter objects later inside the nn.Conv2d and other modules.
#     del stateDict['conv1.weight']
#     stateDict['conv1.weight'] = torch.rand((64, 3, 3, 3), requires_grad=True)
#     model.load_state_dict(stateDict)
#     return model


# def download_imagenet_models():
#     for name, url in model_urls.items():
#         fname = f"{name}_imagenet.pt"
#         fname = os.path.join(RESNETS_FOLDER, fname)
#         if not os.path.isfile(fname):
#             print(f"Downloading ImageNet model {name} from {url} to {fname}")
#             r = requests.get(url)
#             with open(fname, "wb") as f:
#                 f.write(r.content)


# def cifar_model_from_imagenet_model(name):
#     fname = f"{name}.pt"
#     fname = os.path.join(RESNETS_FOLDER, fname)
#     if not os.path.isfile(fname):
#         model = generate(name)
#         return model, False
#     else:
#         # NOTE this is not the same statedict as the "imagenet" one
#         model = model_constructors[name](pretrained=True, progress=True)
#         model.load_state_dict(torch.load(fname))
#         return model, True


# def cifar_models_from_imagenet_models():
#     return {name: cifar_model_from_imagenet_model(name) for name, url in model_urls.items()}


# if __name__ == "__main__":
#     if not os.path.isdir(RESNETS_FOLDER):
#         os.mkdir(RESNETS_FOLDER)
#     download_imagenet_models()
#     models = cifar_models_from_imagenet_models()
#     # NOTE here you might want to finetune; we do it inside cifar_supervised.py
