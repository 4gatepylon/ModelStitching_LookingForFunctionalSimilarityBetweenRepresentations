# Things talked on April 13th, 2022
BatchNorm should be in eval mode (fies the variance and mean)
FFCV could introduce an error: try without FFCV
Make sure to lay it out in a minimalistic way
Be Modular (what I want)
Reuse code (what I want) instead of rewriting
In theory setting requires_grad to False should work: if it doesn't... why?

# Things Talked on April 6th, 2022
- we either have some very interesting behavior or a bug, so... plz try hard to find bug!
- when is similarity small? (that is mean squared error or cca): i.e. look at the lower left hand triangle; WHY IS IT THAT BIG RESNETS HAVE A DIAGONAL WHILE THE SMALL ONES HAVE A TRIANGLE?
- right after you finish images go see what size resnets start to "phase transition" (those that go from triangles to diagonals)
- Look into CCA for similarity loss, also maybe correlation? random trials
- Tianhong thought we should look at densenet (vary skips: to the end, and/or none)

If I want to submit to MurJ I need to get my shit together!

# Resnet34 dimensions for batch size 1:
```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ResNet                                   --                        --
├─Conv2d: 1-1                            [1, 64, 32, 32]           1,728
├─BatchNorm2d: 1-2                       [1, 64, 32, 32]           128
├─ReLU: 1-3                              [1, 64, 32, 32]           --
├─Sequential: 1-4                        [1, 64, 32, 32]           --
│    └─BasicBlock: 2-1                   [1, 64, 32, 32]           --
│    │    └─Conv2d: 3-1                  [1, 64, 32, 32]           36,864
│    │    └─BatchNorm2d: 3-2             [1, 64, 32, 32]           128
│    │    └─ReLU: 3-3                    [1, 64, 32, 32]           --
│    │    └─Conv2d: 3-4                  [1, 64, 32, 32]           36,864
│    │    └─BatchNorm2d: 3-5             [1, 64, 32, 32]           128
│    │    └─ReLU: 3-6                    [1, 64, 32, 32]           --
│    └─BasicBlock: 2-2                   [1, 64, 32, 32]           --
│    │    └─Conv2d: 3-7                  [1, 64, 32, 32]           36,864
│    │    └─BatchNorm2d: 3-8             [1, 64, 32, 32]           128
│    │    └─ReLU: 3-9                    [1, 64, 32, 32]           --
│    │    └─Conv2d: 3-10                 [1, 64, 32, 32]           36,864
│    │    └─BatchNorm2d: 3-11            [1, 64, 32, 32]           128
│    │    └─ReLU: 3-12                   [1, 64, 32, 32]           --
│    └─BasicBlock: 2-3                   [1, 64, 32, 32]           --
│    │    └─Conv2d: 3-13                 [1, 64, 32, 32]           36,864
│    │    └─BatchNorm2d: 3-14            [1, 64, 32, 32]           128
│    │    └─ReLU: 3-15                   [1, 64, 32, 32]           --
│    │    └─Conv2d: 3-16                 [1, 64, 32, 32]           36,864
│    │    └─BatchNorm2d: 3-17            [1, 64, 32, 32]           128
│    │    └─ReLU: 3-18                   [1, 64, 32, 32]           --
├─Sequential: 1-5                        [1, 128, 16, 16]          --
│    └─BasicBlock: 2-4                   [1, 128, 16, 16]          --
│    │    └─Conv2d: 3-19                 [1, 128, 16, 16]          73,728
│    │    └─BatchNorm2d: 3-20            [1, 128, 16, 16]          256
│    │    └─ReLU: 3-21                   [1, 128, 16, 16]          --
│    │    └─Conv2d: 3-22                 [1, 128, 16, 16]          147,456
│    │    └─BatchNorm2d: 3-23            [1, 128, 16, 16]          256
│    │    └─Sequential: 3-24             [1, 128, 16, 16]          8,448
│    │    └─ReLU: 3-25                   [1, 128, 16, 16]          --
│    └─BasicBlock: 2-5                   [1, 128, 16, 16]          --
│    │    └─Conv2d: 3-26                 [1, 128, 16, 16]          147,456
│    │    └─BatchNorm2d: 3-27            [1, 128, 16, 16]          256
│    │    └─ReLU: 3-28                   [1, 128, 16, 16]          --
│    │    └─Conv2d: 3-29                 [1, 128, 16, 16]          147,456
│    │    └─BatchNorm2d: 3-30            [1, 128, 16, 16]          256
│    │    └─ReLU: 3-31                   [1, 128, 16, 16]          --
│    └─BasicBlock: 2-6                   [1, 128, 16, 16]          --
│    │    └─Conv2d: 3-32                 [1, 128, 16, 16]          147,456
│    │    └─BatchNorm2d: 3-33            [1, 128, 16, 16]          256
│    │    └─ReLU: 3-34                   [1, 128, 16, 16]          --
│    │    └─Conv2d: 3-35                 [1, 128, 16, 16]          147,456
│    │    └─BatchNorm2d: 3-36            [1, 128, 16, 16]          256
│    │    └─ReLU: 3-37                   [1, 128, 16, 16]          --
│    └─BasicBlock: 2-7                   [1, 128, 16, 16]          --
│    │    └─Conv2d: 3-38                 [1, 128, 16, 16]          147,456
│    │    └─BatchNorm2d: 3-39            [1, 128, 16, 16]          256
│    │    └─ReLU: 3-40                   [1, 128, 16, 16]          --
│    │    └─Conv2d: 3-41                 [1, 128, 16, 16]          147,456
│    │    └─BatchNorm2d: 3-42            [1, 128, 16, 16]          256
│    │    └─ReLU: 3-43                   [1, 128, 16, 16]          --
├─Sequential: 1-6                        [1, 256, 8, 8]            --
│    └─BasicBlock: 2-8                   [1, 256, 8, 8]            --
│    │    └─Conv2d: 3-44                 [1, 256, 8, 8]            294,912
│    │    └─BatchNorm2d: 3-45            [1, 256, 8, 8]            512
│    │    └─ReLU: 3-46                   [1, 256, 8, 8]            --
│    │    └─Conv2d: 3-47                 [1, 256, 8, 8]            589,824
│    │    └─BatchNorm2d: 3-48            [1, 256, 8, 8]            512
│    │    └─Sequential: 3-49             [1, 256, 8, 8]            33,280
│    │    └─ReLU: 3-50                   [1, 256, 8, 8]            --
│    └─BasicBlock: 2-9                   [1, 256, 8, 8]            --
│    │    └─Conv2d: 3-51                 [1, 256, 8, 8]            589,824
│    │    └─BatchNorm2d: 3-52            [1, 256, 8, 8]            512
│    │    └─ReLU: 3-53                   [1, 256, 8, 8]            --
│    │    └─Conv2d: 3-54                 [1, 256, 8, 8]            589,824
│    │    └─BatchNorm2d: 3-55            [1, 256, 8, 8]            512
│    │    └─ReLU: 3-56                   [1, 256, 8, 8]            --
│    └─BasicBlock: 2-10                  [1, 256, 8, 8]            --
│    │    └─Conv2d: 3-57                 [1, 256, 8, 8]            589,824
│    │    └─BatchNorm2d: 3-58            [1, 256, 8, 8]            512
│    │    └─ReLU: 3-59                   [1, 256, 8, 8]            --
│    │    └─Conv2d: 3-60                 [1, 256, 8, 8]            589,824
│    │    └─BatchNorm2d: 3-61            [1, 256, 8, 8]            512
│    │    └─ReLU: 3-62                   [1, 256, 8, 8]            --
│    └─BasicBlock: 2-11                  [1, 256, 8, 8]            --
│    │    └─Conv2d: 3-63                 [1, 256, 8, 8]            589,824
│    │    └─BatchNorm2d: 3-64            [1, 256, 8, 8]            512
│    │    └─ReLU: 3-65                   [1, 256, 8, 8]            --
│    │    └─Conv2d: 3-66                 [1, 256, 8, 8]            589,824
│    │    └─BatchNorm2d: 3-67            [1, 256, 8, 8]            512
│    │    └─ReLU: 3-68                   [1, 256, 8, 8]            --
│    └─BasicBlock: 2-12                  [1, 256, 8, 8]            --
│    │    └─Conv2d: 3-69                 [1, 256, 8, 8]            589,824
│    │    └─BatchNorm2d: 3-70            [1, 256, 8, 8]            512
│    │    └─ReLU: 3-71                   [1, 256, 8, 8]            --
│    │    └─Conv2d: 3-72                 [1, 256, 8, 8]            589,824
│    │    └─BatchNorm2d: 3-73            [1, 256, 8, 8]            512
│    │    └─ReLU: 3-74                   [1, 256, 8, 8]            --
│    └─BasicBlock: 2-13                  [1, 256, 8, 8]            --
│    │    └─Conv2d: 3-75                 [1, 256, 8, 8]            589,824
│    │    └─BatchNorm2d: 3-76            [1, 256, 8, 8]            512
│    │    └─ReLU: 3-77                   [1, 256, 8, 8]            --
│    │    └─Conv2d: 3-78                 [1, 256, 8, 8]            589,824
│    │    └─BatchNorm2d: 3-79            [1, 256, 8, 8]            512
│    │    └─ReLU: 3-80                   [1, 256, 8, 8]            --
├─Sequential: 1-7                        [1, 512, 4, 4]            --
│    └─BasicBlock: 2-14                  [1, 512, 4, 4]            --
│    │    └─Conv2d: 3-81                 [1, 512, 4, 4]            1,179,648
│    │    └─BatchNorm2d: 3-82            [1, 512, 4, 4]            1,024
│    │    └─ReLU: 3-83                   [1, 512, 4, 4]            --
│    │    └─Conv2d: 3-84                 [1, 512, 4, 4]            2,359,296
│    │    └─BatchNorm2d: 3-85            [1, 512, 4, 4]            1,024
│    │    └─Sequential: 3-86             [1, 512, 4, 4]            132,096
│    │    └─ReLU: 3-87                   [1, 512, 4, 4]            --
│    └─BasicBlock: 2-15                  [1, 512, 4, 4]            --
│    │    └─Conv2d: 3-88                 [1, 512, 4, 4]            2,359,296
│    │    └─BatchNorm2d: 3-89            [1, 512, 4, 4]            1,024
│    │    └─ReLU: 3-90                   [1, 512, 4, 4]            --
│    │    └─Conv2d: 3-91                 [1, 512, 4, 4]            2,359,296
│    │    └─BatchNorm2d: 3-92            [1, 512, 4, 4]            1,024
│    │    └─ReLU: 3-93                   [1, 512, 4, 4]            --
│    └─BasicBlock: 2-16                  [1, 512, 4, 4]            --
│    │    └─Conv2d: 3-94                 [1, 512, 4, 4]            2,359,296
│    │    └─BatchNorm2d: 3-95            [1, 512, 4, 4]            1,024
│    │    └─ReLU: 3-96                   [1, 512, 4, 4]            --
│    │    └─Conv2d: 3-97                 [1, 512, 4, 4]            2,359,296
│    │    └─BatchNorm2d: 3-98            [1, 512, 4, 4]            1,024
│    │    └─ReLU: 3-99                   [1, 512, 4, 4]            --
├─AdaptiveAvgPool2d: 1-8                 [1, 512, 1, 1]            --
==========================================================================================
Total params: 21,276,992
Trainable params: 21,276,992
Non-trainable params: 0
Total mult-adds (G): 1.16
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 16.38
Params size (MB): 85.11
Estimated Total Size (MB): 101.50
==========================================================================================
```